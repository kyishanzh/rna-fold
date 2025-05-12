import os
import sys
import argparse
import logging
from tqdm import tqdm
import time
import random
import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.distributed as dist
import wandb
import numpy as np
from Bio.PDB import PDBParser

# Add openfold to path
openfold_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "openfold"))
if openfold_path not in sys.path:
    sys.path.insert(0, openfold_path)

from rhofold.rhofold import RhoFold
from rhofold.config import rhofold_config
from rhofold.utils.alphabet import get_features, read_fas
from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.utils.loss import compute_fape

# Import evaluation utilities
from utils import eval_model, tm_score, all_seq_ids

# Import distributed utilities to ensure compatibility with eval_model
from distributed_utils import get_world_size, get_rank, is_main_process

# Training configuration
CHECKPOINT_DIR = "checkpoints"
USE_EVO2 = True
BATCH_SIZE = 1  # Keep batch size at 1 for each GPU
NUM_EPOCHS = 20
LEARNING_RATE = 2e-7
EVO2_LR = 1e-4  # Higher learning rate for evo2_head
EVAL_PER_EPOCH = 10  # Number of evaluations per epoch
CHECKPOINT_EVERY = 1
WARMUP_STEPS = 1000
SKIP_SHORT_SEQS = True  # Skip sequences with length > MAX_SEQ_LENGTH
GRAD_ACCUM_STEPS = 4   # Number of steps to accumulate gradients
MAX_SEQ_LENGTH = 200    # Maximum sequence length to avoid OOM errors
WEIGHT_DECAY = 0.01    # Weight decay for regularization 
DROPOUT_RATE = 0.1     # Dropout rate for regularization

# Set to your specific project/entity here or use environment variables
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "rhofold")

USE_RHOFOLD_DATA = True

def get_evo2head_norm(model):
    """Get L2 norm of the evo2_head parameters"""
    if isinstance(model, RhoFoldLightningModule):
        evo2_head = model.model.evo2_head
    else:
        evo2_head = model.module.evo2_head
    
    param_count = sum(p.numel() for p in evo2_head.parameters())
    param_sum = sum((p**2).sum() for p in evo2_head.parameters())
    return param_sum / param_count

def get_mask(tru_seq, pdb_seq):
    # Given true seq and seq of residues covered by pdb, find mask of true seq that can match to pdb
    assert len(tru_seq) >= len(pdb_seq), f"True sequence length {len(tru_seq)} is less than PDB sequence length {len(pdb_seq)}"
    mask = torch.zeros(len(tru_seq), dtype=torch.bool)
    pdb_ptr = 0
    for i, res in enumerate(tru_seq):
        if pdb_ptr < len(pdb_seq) and res == pdb_seq[pdb_ptr]:
            pdb_ptr += 1
            mask[i] = True
    return mask

def get_pdb_seq(structure):
    return "".join([res.get_resname() for res in structure.get_residues()])

def extract_atom_positions_from_pdb(pdb_path, tru_seq):
    """
    Extract atom positions from PDB file and apply sequence mask
    Returns dict with C1', P, C4, and N atom positions, plus seq mask and pdb_seq
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("gt", pdb_path)
    chain = next(structure.get_chains())
    
    # Extract atoms from the PDB structure
    gt_c1_positions = []
    gt_p_positions = []
    gt_c4_positions = []
    gt_n_positions = []
    
    for res in chain:
        try:
            gt_c1_positions.append(res["C1'"].get_coord())
        except:
            gt_c1_positions.append([0, 0, 0])
        
        try:
            gt_p_positions.append(res["p"].get_coord())
        except:
            gt_p_positions.append(gt_p_positions[-1] if gt_p_positions else [0, 0, 0])
        
        try:
            gt_c4_positions.append(res["c4_"].get_coord())
        except:
            gt_c4_positions.append(gt_c4_positions[-1] if gt_c4_positions else [0, 0, 0])
        
        try:
            gt_n_positions.append(res["n"].get_coord())
        except:
            gt_n_positions.append(gt_n_positions[-1] if gt_n_positions else [0, 0, 0])
    
    # Convert to numpy arrays
    gt_c1_positions = np.array(gt_c1_positions)
    gt_dist_positions = np.array([gt_p_positions, gt_c4_positions, gt_n_positions])
    
    # Get PDB sequence and mask
    pdb_seq = get_pdb_seq(structure)
    mask = get_mask(tru_seq, pdb_seq)
    
    return {
        'c1_positions': gt_c1_positions,
        'dist_positions': gt_dist_positions,
        'mask': mask,
        'pdb_seq': pdb_seq
    }

class RNADataset(Dataset):
    def __init__(self, data_dir, use_evo2=True, max_seq_length=MAX_SEQ_LENGTH, preload_pdbs=True):
        self.data_dir = data_dir
        self.use_evo2 = use_evo2
        self.max_seq_length = max_seq_length
        self.preload_pdbs = preload_pdbs
        
        # For caching loaded data
        self.cached_features = {}
        self.cached_evo2_embeddings = {}
        self.cached_pdb_data = {}

        if not USE_RHOFOLD_DATA:
            self.seq_ids = all_seq_ids()
            return
        
        # Get sequence IDs from the directory
        seq_dir = os.path.join(data_dir, "RNA3D_DATA/seq")
        
        # Filter sequence IDs to only include those with both seq and a3m files
        self.seq_ids = []
        for f in os.listdir(seq_dir):
            if f.endswith('.seq'):
                seq_id = f.split('.')[0]
                self.seq_ids.append(seq_id)
        
        # Track filtered out sequences
        self.filtered_ids = []
        self.missing_pdb_ids = []
        self.skipped_long_seqs = []
        
        # Filter out sequences with missing PDB files
        filtered_seq_ids = []
        for seq_id in self.seq_ids:
            if seq_id == '4v9r_BB':
                continue
            pdb_path = os.path.join(data_dir, f"RNA3D_DATA/pdb/{seq_id}.pdb")
            input_fas = os.path.join(data_dir, f"RNA3D_DATA/seq/{seq_id}.seq")
            # Read sequence to check length
            seq = read_fas(input_fas)[0][1]
            if not os.path.exists(pdb_path):
                self.missing_pdb_ids.append(seq_id)
                continue
            try:
                f = open(pdb_path, "r")
            except:
                self.missing_pdb_ids.append(seq_id)
                continue
            lines = f.readlines()
            f.close()
            if len(lines) < len(seq) / 2:
                self.missing_pdb_ids.append(seq_id)
                continue
            if not lines[0].startswith("ATOM"):
                self.missing_pdb_ids.append(seq_id)
                continue
            # Skip sequences longer than MAX_SEQ_LENGTH if SKIP_SHORT_SEQS is enabled
            if SKIP_SHORT_SEQS:
                if len(seq) <= MAX_SEQ_LENGTH:
                    filtered_seq_ids.append(seq_id)
                else:
                    self.skipped_long_seqs.append(seq_id)
            else:
                filtered_seq_ids.append(seq_id)
            if len(filtered_seq_ids) >= 100: break
        
        self.seq_ids = filtered_seq_ids
        
        logging.info(f"Found {len(self.seq_ids)} RNA sequences with valid PDB files for training")
        if self.missing_pdb_ids:
            logging.info(f"Filtered out {len(self.missing_pdb_ids)} sequences with missing PDB files")
        if self.skipped_long_seqs:
            logging.info(f"Skipped {len(self.skipped_long_seqs)} sequences with length > {MAX_SEQ_LENGTH}")
            
        # Preload data if enabled
        if self.preload_pdbs:
            self._preload_data()
    
    def _preload_data(self):
        """Preload all sequence data, evo2 embeddings, and PDB data to avoid disk I/O during training"""
        logging.info(f"Preloading data for {len(self.seq_ids)} sequences...")
        
        for idx, seq_id in enumerate(tqdm(self.seq_ids, desc="Preloading data")):
            # Load sequence and MSA features
            if USE_RHOFOLD_DATA:
                input_fas = os.path.join(self.data_dir, f"RNA3D_DATA/seq/{seq_id}.seq")
                input_a3m = os.path.join(self.data_dir, f"RNA3D_DATA/rMSA/{seq_id}.a3m")
                data_dict = get_features(input_fas, input_a3m)
            else:
                input_fas = os.path.join(self.data_dir, f"MSA/{seq_id}.fasta")
                input_a3m = os.path.join(self.data_dir, f"MSA/{seq_id}.MSA.fasta")
                data_dict = get_features(input_fas, input_a3m)
            
            # Skip if token dimensions don't match
            if data_dict['tokens'].shape[-1] != data_dict['rna_fm_tokens'].shape[-1]:
                if seq_id not in self.filtered_ids:
                    self.filtered_ids.append(seq_id)
                continue
            
            # Load evo2 embedding
            evo2_embedding = None
            if self.use_evo2:
                if USE_RHOFOLD_DATA:
                    embeddings_dir = os.path.join(self.data_dir, "RNA3D_DATA/evo2_embeddings")
                else:
                    embeddings_dir = os.path.join(self.data_dir, "evo2_embeddings")
                embedding_path = os.path.join(embeddings_dir, f"{seq_id}.pt")
                evo2_embedding = torch.load(embedding_path)
                self.cached_evo2_embeddings[seq_id] = evo2_embedding
            
            # Load PDB data
            if USE_RHOFOLD_DATA:
                pdb_path = os.path.join(self.data_dir, f"RNA3D_DATA/pdb/{seq_id}.pdb")
            else:
                pdb_path = os.path.join(self.data_dir, f"pdb/{seq_id}.pdb")
            
            # Extract atom positions from PDB
            pdb_data = extract_atom_positions_from_pdb(pdb_path, data_dict['seq'])
            
            # Cache all data
            self.cached_features[seq_id] = {
                'tokens': data_dict['tokens'],
                'rna_fm_tokens': data_dict['rna_fm_tokens'],
                'seq': data_dict['seq']
            }
            self.cached_pdb_data[seq_id] = pdb_data
            
        logging.info(f"Preloaded data for {len(self.cached_features)} sequences")
    
    def __len__(self):
        return len(self.seq_ids)
    
    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]
        
        # Return cached data if available
        if self.preload_pdbs and seq_id in self.cached_features:
            data_dict = self.cached_features[seq_id]
            evo2_embedding = self.cached_evo2_embeddings.get(seq_id, None)
            pdb_data = self.cached_pdb_data[seq_id]
            
            return {
                'seq_id': seq_id,
                'tokens': data_dict['tokens'],
                'rna_fm_tokens': data_dict['rna_fm_tokens'],
                'seq': data_dict['seq'],
                'evo2_fea': evo2_embedding,
                'pdb_data': pdb_data,
            }
        
        # Load data from disk if not preloaded
        # Load sequence and MSA features
        if USE_RHOFOLD_DATA:
            input_fas = os.path.join(self.data_dir, f"RNA3D_DATA/seq/{seq_id}.seq")
            input_a3m = os.path.join(self.data_dir, f"RNA3D_DATA/rMSA/{seq_id}.a3m")
            data_dict = get_features(input_fas, input_a3m)
        else:
            input_fas = os.path.join(self.data_dir, f"MSA/{seq_id}.fasta")
            input_a3m = os.path.join(self.data_dir, f"MSA/{seq_id}.MSA.fasta")
            data_dict = get_features(input_fas, input_a3m)
        
        # Check if tokens and rna_fm_tokens have the same last dimension
        if data_dict['tokens'].shape[-1] != data_dict['rna_fm_tokens'].shape[-1]:
            if seq_id not in self.filtered_ids:
                self.filtered_ids.append(seq_id)
            
            # Get new index by recursively calling __getitem__ with the next index
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)
        
        # Load evo2 embedding if enabled
        evo2_embedding = None
        if self.use_evo2:
            if USE_RHOFOLD_DATA:
                embeddings_dir = os.path.join(self.data_dir, "RNA3D_DATA/evo2_embeddings")
            else:
                embeddings_dir = os.path.join(self.data_dir, "evo2_embeddings")
            embedding_path = os.path.join(embeddings_dir, f"{seq_id}.pt")
            evo2_embedding = torch.load(embedding_path)
        
        # Get PDB path for ground truth
        if USE_RHOFOLD_DATA:
            pdb_path = os.path.join(self.data_dir, f"RNA3D_DATA/pdb/{seq_id}.pdb")
        else:
            pdb_path = os.path.join(self.data_dir, f"pdb/{seq_id}.pdb")
        
        # This check is redundant now, but kept for safety
        assert os.path.exists(pdb_path), f"PDB file not found: {pdb_path}"
        
        # Extract atom positions from PDB
        pdb_data = extract_atom_positions_from_pdb(pdb_path, data_dict['seq'])

        return {
            'seq_id': seq_id,
            'tokens': data_dict['tokens'],
            'rna_fm_tokens': data_dict['rna_fm_tokens'],
            'seq': data_dict['seq'],
            'evo2_fea': evo2_embedding,
            'pdb_data': pdb_data,
        }

def compute_tm_loss(output, pdb_data, device):
    """Compute TM score loss using preloaded PDB data"""
    pred_c1_positions = output["cords_c1'"][-1].squeeze(0)  # Shape [N, 3]
    
    # Get ground truth positions and mask from preloaded data
    gt_c1_positions = pdb_data['c1_positions'].to(device).float()
    mask = pdb_data['mask'].to(device)
    
    # Apply mask to predicted positions
    pred_c1_positions = pred_c1_positions[mask]
    
    # Compute TM score
    return tm_score(pred_c1_positions, gt_c1_positions)

def compute_fape_loss(output, pdb_data, device, length_scale=10.0, l1_clamp_distance=None):
    """Compute FAPE loss using preloaded PDB data"""
    # Extract predicted frames
    pred_frames_tensor = output["frames"]
    
    # Handle dimensions appropriately
    if isinstance(pred_frames_tensor, list):
        pred_frames_tensor = pred_frames_tensor[-1]  # Take last element if list
    
    # Extract the last recycle frame if there's a recycle dimension
    if pred_frames_tensor.dim() == 4:  # [recycle, batch, N, 7]
        pred_frames_tensor = pred_frames_tensor[-1]  # [batch, N, 7]
    
    # Remove batch dim if batch=1
    if pred_frames_tensor.dim() == 3 and pred_frames_tensor.shape[0] == 1:
        pred_frames_tensor = pred_frames_tensor.squeeze(0)  # [N, 7]
    
    # Convert to Rigid object
    pred_frames = Rigid.from_tensor_7(pred_frames_tensor)
    
    # Extract predicted C1' positions
    pred_c1_positions = output["cords_c1'"][-1].squeeze(0)  # Shape [N, 3]
    
    # Get ground truth positions and mask from preloaded data
    gt_c1_positions = pdb_data['c1_positions'].to(device).float()
    mask = pdb_data['mask'].to(device)
    
    # Apply mask to predictions
    pred_c1_positions = pred_c1_positions[mask]
    pred_frames = pred_frames[mask]
    
    # Create a mask for the positions (all 1s since we've truncated to match)
    pos_mask = torch.ones(len(pred_c1_positions), device=device)
    
    # Create target frames
    zeros = torch.zeros(len(pred_c1_positions), 3, device=device, dtype=pred_frames_tensor.dtype)
    ones = torch.ones(len(pred_c1_positions), 1, device=device, dtype=pred_frames_tensor.dtype)
    quats = torch.cat([ones, zeros], dim=-1)
    
    # Create target frames with identity rotations and ground truth C1' positions
    target_frames = Rigid(
        Rotation(quats=quats, rot_mats=None),
        gt_c1_positions
    )
    
    # Compute FAPE
    fape = compute_fape(
        pred_frames=pred_frames,
        target_frames=target_frames,
        frames_mask=pos_mask,
        pred_positions=pred_c1_positions,
        target_positions=gt_c1_positions,
        positions_mask=pos_mask,
        length_scale=length_scale,
        pair_mask=None,
        l1_clamp_distance=l1_clamp_distance,
    )
    
    return fape

def compute_dist_loss(output, pdb_data, device):
    """Compute distance loss using preloaded PDB data"""
    # Extract predicted distances
    pred_p_dist = output["p"][-1].squeeze(0)  # Shape [40, N, N]
    pred_c4_dist = output["c4_"][-1].squeeze(0)  # Shape [40, N, N]
    pred_n_dist = output["n"][-1].squeeze(0)  # Shape [40, N, N]
    pred_dist = torch.stack([pred_p_dist, pred_c4_dist, pred_n_dist], dim=0)
    
    # Get mask from preloaded data
    mask = pdb_data['mask'].to(device)
    
    # Apply mask to predicted distances
    pred_dist = pred_dist[..., mask, :][..., :, mask]
    
    # Get ground truth positions from preloaded data
    gt_dist_positions = pdb_data['dist_positions'].to(device).float()
    gt_dist = torch.norm(gt_dist_positions.unsqueeze(-2) - gt_dist_positions.unsqueeze(-3), dim=-1)
    
    # Get distance bin boundaries
    boundaries = torch.linspace(2, 40, 39, device=device)
    
    # Get bin indices for ground truth distances
    true_bins = torch.sum(gt_dist.unsqueeze(-1) > boundaries, dim=-1)
    
    # Compute loss
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fn(pred_dist, true_bins)
    
    return loss

# Custom Lightning DataModule for RNA data
class RNADataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=1, use_evo2=True, max_seq_length=200, preload_pdbs=True, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_evo2 = use_evo2
        self.max_seq_length = max_seq_length
        self.preload_pdbs = preload_pdbs
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        # Create full dataset - we'll use the same for train and validation
        self.dataset = RNADataset(
            self.data_dir, 
            use_evo2=self.use_evo2, 
            max_seq_length=self.max_seq_length, 
            preload_pdbs=self.preload_pdbs
        )
        
        # Print dataset stats
        print(f"Dataset loaded with {len(self.dataset)} sequences")
        
    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        # Use the same dataset for validation - in practice we'll use the EvaluationCallback
        # for more thorough evaluation
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
    def get_dataset_info(self):
        """Return information about filtered sequences, etc."""
        info = {}
        if hasattr(self.dataset, 'filtered_ids'):
            info['filtered_ids'] = self.dataset.filtered_ids
        if hasattr(self.dataset, 'missing_pdb_ids'):
            info['missing_pdb_ids'] = self.dataset.missing_pdb_ids
        if hasattr(self.dataset, 'skipped_long_seqs'):
            info['skipped_long_seqs'] = self.dataset.skipped_long_seqs
        return info

# PyTorch Lightning Module for RhoFold training
class RhoFoldLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = RhoFold(rhofold_config)
        self.automatic_optimization = False  # Handle optimization manually for more control
        
        # Set dropout rate
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = config["dropout_rate"]
        
        # Initialize validation TM scores list for later aggregation
        self.validation_tm_scores = []
    
    def forward(self, tokens, rna_fm_tokens, seq, evo2_fea=None, train=False):
        return self.model(tokens=tokens, rna_fm_tokens=rna_fm_tokens, seq=seq, evo2_fea=evo2_fea, train=train)
    
    def training_step(self, batch, batch_idx):
        # Get optimizer
        opt = self.optimizers()
        
        # Extract batch data
        seq_id = batch['seq_id'][0]
        tokens = batch['tokens'][0]
        rna_fm_tokens = batch['rna_fm_tokens'][0]
        seq = batch['seq'][0]
        pdb_data = batch['pdb_data']
        pdb_data = {key: val[0] for key, val in pdb_data.items()}
        
        # Add dimensions if needed
        while tokens.dim() < 3:
            tokens = tokens.unsqueeze(0)
        while rna_fm_tokens.dim() < 2:
            rna_fm_tokens = rna_fm_tokens.unsqueeze(0)
        
        # Handle evo2 features
        evo2_fea = None
        if self.hparams["use_evo2"] and 'evo2_fea' in batch and batch['evo2_fea'] is not None:
            evo2_fea = batch['evo2_fea'][0].to(torch.float32)
        
        # Zero gradients
        opt.zero_grad()
        
        # Run model forward pass
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.trainer.precision == "bf16-mixed"):
            outputs = self(tokens=tokens, rna_fm_tokens=rna_fm_tokens, seq=seq, evo2_fea=evo2_fea, train=True)
            
            # Take the last output from recycles
            output = outputs[-1]
            
            # Compute losses
            fape_loss = compute_fape_loss(output, pdb_data, self.device)
            tm_score_val = compute_tm_loss(output, pdb_data, self.device)
            dist_loss = compute_dist_loss(output, pdb_data, self.device)
            
            # Combined loss: FAPE - 100 * TM score + 0.3 * dist_loss
            loss = 2 * fape_loss - 100 * tm_score_val + 0.3 * dist_loss
            
            # Scale the loss for gradient accumulation
            loss = loss / self.hparams["grad_accum_steps"]
        
        # Backward pass
        self.manual_backward(loss)
        
        # Update weights if we've accumulated enough gradients
        if (batch_idx + 1) % self.hparams["grad_accum_steps"] == 0:
            # Clip gradients
            self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            
            # Check for NaN gradients and replace with zeros
            nan_count = 0
            for param in self.parameters():
                if param.grad is not None:
                    nan_mask = torch.isnan(param.grad)
                    if nan_mask.any():
                        nan_count += nan_mask.sum().item()
                        param.grad[nan_mask] = 0.0
            
            # Step the optimizer
            opt.step()
            opt.zero_grad()
            
            # Step the learning rate schedulers
            schedulers = self.lr_schedulers()
            
            # Step the warmup scheduler every step
            if isinstance(schedulers, list) and len(schedulers) > 0:
                # First scheduler is the warmup scheduler (step-based)
                if self.global_step < self.hparams["warmup_steps"]:
                    schedulers[0].step()
        
        # Log metrics
        self.log("train_loss", loss * self.hparams["grad_accum_steps"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_fape_loss", fape_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_tm_score", tm_score_val, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_dist_loss", dist_loss, on_step=True, on_epoch=True, sync_dist=True)
        
        if "plddt" in output:
            self.log("pLDDT", output["plddt"][1].item(), on_step=True, sync_dist=True)
        
        self.log("seq_length", len(seq), on_step=True, sync_dist=True)
        self.log("nan_count", float(nan_count), on_step=True, on_epoch=True, sync_dist=True)
        
        # Log evo2head norm
        evo2head_norm = get_evo2head_norm(self)
        self.log("evo2head_norm", evo2head_norm, on_step=True, on_epoch=True, sync_dist=True)
        
        # Log learning rates
        self.log("lr_evo2", opt.param_groups[0]['lr'], on_step=True, sync_dist=True)
        self.log("lr_main", opt.param_groups[1]['lr'], on_step=True, sync_dist=True)
        
        return {"loss": loss * self.hparams["grad_accum_steps"]}
    
    def validation_step(self, batch, batch_idx):
        # For proper validation, this should coordinate with the other processes
        # But since the EvaluationCallback will use the original eval_model function,
        # we can make this lightweight
        # Extract batch data for validation
        seq_id = batch['seq_id'][0]
        tokens = batch['tokens'][0]
        rna_fm_tokens = batch['rna_fm_tokens'][0]
        seq = batch['seq'][0]
        pdb_data = batch['pdb_data']
        pdb_data = {key: val[0] for key, val in pdb_data.items()}
        
        # Add dimensions if needed
        while tokens.dim() < 3:
            tokens = tokens.unsqueeze(0)
        while rna_fm_tokens.dim() < 2:
            rna_fm_tokens = rna_fm_tokens.unsqueeze(0)
        
        # Handle evo2 features
        evo2_fea = None
        if self.hparams["use_evo2"] and 'evo2_fea' in batch and batch['evo2_fea'] is not None:
            evo2_fea = batch['evo2_fea'][0].to(torch.float32)
        
        with torch.no_grad():
            # Run model forward pass
            outputs = self(tokens=tokens, rna_fm_tokens=rna_fm_tokens, seq=seq, evo2_fea=evo2_fea)
            
            # Take the last output from recycles
            output = outputs[-1]
            
            # Compute validation metrics
            fape_loss = compute_fape_loss(output, pdb_data, self.device)
            tm_score_val = compute_tm_loss(output, pdb_data, self.device)
            dist_loss = compute_dist_loss(output, pdb_data, self.device)
            
            # Log validation metrics
            self.log("val_fape_loss", fape_loss, sync_dist=True)
            self.log("val_tm_score", tm_score_val, sync_dist=True)
            self.log("val_dist_loss", dist_loss, sync_dist=True)
            
            # Store TM score for on_validation_epoch_end
            self.validation_tm_scores.append(tm_score_val)
    
    def on_validation_epoch_end(self):
        # Skip if no validation was performed
        if not self.validation_tm_scores:
            self.log("eval_tm_score", 0.0, prog_bar=True, sync_dist=True)
            return
        
        # Calculate average TM score
        avg_tm_score = torch.stack(self.validation_tm_scores).mean()
        
        # Log the aggregated score
        self.log("eval_tm_score", avg_tm_score, prog_bar=True, sync_dist=True)
        
        # Reset the list for next epoch
        self.validation_tm_scores = []
    
    def configure_optimizers(self):
        # Initialize optimizer with parameter groups
        # Group 1: evo2_head parameters with higher learning rate
        # Group 2: All other parameters with default learning rate
        evo2_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'evo2_head' in name:
                evo2_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {'params': evo2_params, 'lr': self.hparams["evo2_lr"]},
            {'params': other_params, 'lr': self.hparams["learning_rate"]}
        ]
        
        optimizer = torch.optim.Adam(param_groups, weight_decay=self.hparams["weight_decay"])
        
        # Warmup scheduler as the first scheduler
        warmup_scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(
                optimizer, 
                lr_lambda=lambda step: min(1.0, float(step) / float(self.hparams["warmup_steps"]))
            ),
            'interval': 'step',
            'frequency': 1,
            'name': 'warmup_scheduler',
            # Only apply warmup during the warmup period
            'monitor': 'step',
        }
        
        # Cosine annealing scheduler for after warmup
        cosine_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams["num_epochs"],
                eta_min=self.hparams["learning_rate"] * 0.01  # Minimum LR = 1% of initial rate
            ),
            'interval': 'epoch',
            'frequency': 1,
            'name': 'cosine_scheduler',
        }
        
        return [optimizer], [warmup_scheduler, cosine_scheduler]

# Custom callback for evaluation during training
class EvaluationCallback(pl.Callback):
    def __init__(self, eval_per_epoch=EVAL_PER_EPOCH):
        super().__init__()
        self.eval_per_epoch = eval_per_epoch
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Run evaluation at regular intervals
        total_batches = len(trainer.train_dataloader)
        eval_interval = max(1, total_batches // self.eval_per_epoch)
        
        if batch_idx % eval_interval == 0 and batch_idx > 0:
            # Temporarily set model to eval mode
            pl_module.eval()
            
            # Run evaluation
            try:
                self._run_evaluation(trainer, pl_module)
            except Exception as e:
                print(f"Evaluation error: {str(e)}")
            finally:
                # Return to train mode
                pl_module.train()
                torch.cuda.empty_cache()
    
    def _run_evaluation(self, trainer, pl_module):
        print(f"\n[Evaluation] Running during training...")
        
        # Create a generator function for eval_model that matches the expected interface
        def generator(features):
            with torch.no_grad():
                try:
                    if "evo2_fea" in features:
                        features["evo2_fea"] = features["evo2_fea"].to(torch.float32)
                    
                    outputs = pl_module(
                        tokens=features["tokens"].to(pl_module.device),
                        rna_fm_tokens=features["rna_fm_tokens"].to(pl_module.device),
                        seq=features["seq"],
                        evo2_fea=features["evo2_fea"].to(pl_module.device) if "evo2_fea" in features else None
                    )
                    
                    preds = []
                    for i in range(min(5, len(outputs))):
                        preds.append(outputs[i]["cords_c1'"][0][0].to(torch.float32))
                    return preds
                except Exception as e:
                    print(f"Error in generator: {str(e)}")
                    return []
        
        try:
            # Run evaluation using the original eval_model function
            # This function handles distributed evaluation internally
            eval_score = eval_model(generator)
            
            # Only log metrics on the main process as eval_model already handles aggregation
            if trainer.is_global_zero:
                trainer.logger.log_metrics({"eval_tm_score": eval_score})
                print(f"[Evaluation] TM Score: {eval_score:.4f}")
            
        except Exception as e:
            print(f"Evaluation block error: {str(e)}")

def train(args):
    """Main training function using PyTorch Lightning"""
    # Set seed for reproducibility
    pl.seed_everything(42)
    
    # Print the PyTorch version and available GPU info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU 0 name: {torch.cuda.get_device_name(0)}")
    
    # Check if bf16 is supported
    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"BF16 supported: {bf16_supported}")
    
    # Create configuration dict for the Lightning module
    config = {
        "learning_rate": LEARNING_RATE,
        "evo2_lr": EVO2_LR,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "use_evo2": USE_EVO2,
        "skip_short_seqs": SKIP_SHORT_SEQS,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "weight_decay": WEIGHT_DECAY,
        "dropout_rate": DROPOUT_RATE,
        "warmup_steps": WARMUP_STEPS,
    }
    
    # Create Lightning module
    model = RhoFoldLightningModule(config)
    
    # Apply torch.compile if requested and available
    if args.use_compile and hasattr(torch, 'compile'):
        try:
            print("Attempting to compile model with torch.compile()...")
            model.model = torch.compile(model.model, dynamic=True)
            print("Model compilation successful")
        except Exception as e:
            print(f"Model compilation failed, using eager mode: {str(e)}")
    
    # Create dataset
    dataset = RNADataset(args.data_dir, use_evo2=USE_EVO2, max_seq_length=MAX_SEQ_LENGTH, preload_pdbs=True)
    
    # Configure data loaders
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Configure loggers
    loggers = []
    if args.use_wandb:
        wandb_logger = WandbLogger(
            project=WANDB_PROJECT,
            name=f"rhofold_bf16_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            log_model=True
        )
        loggers.append(wandb_logger)
    
    # Configure callbacks
    callbacks = [
        # Checkpoint callback to save best models
        ModelCheckpoint(
            dirpath=CHECKPOINT_DIR,
            filename='rhofold-{epoch:02d}-{eval_tm_score:.4f}',
            monitor='eval_tm_score',
            mode='max',
            save_top_k=3,
            save_last=True,
            every_n_epochs=CHECKPOINT_EVERY
        ),
        # Learning rate monitor
        LearningRateMonitor(logging_interval='step'),
        # Custom evaluation callback
        EvaluationCallback(eval_per_epoch=EVAL_PER_EPOCH)
    ]
    
    # Configure DDP strategy
    strategy = DDPStrategy(
        find_unused_parameters=True,  # Required for RhoFold architecture
        static_graph=False
    )
    
    # Configure precision based on hardware support
    precision = "bf16-mixed" if bf16_supported else "32"
    
    # Create Lightning trainer
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        logger=loggers,
        callbacks=callbacks,
        accumulate_grad_batches=GRAD_ACCUM_STEPS,
        strategy=strategy,
        precision=precision,  # Use bf16 mixed precision when supported
        log_every_n_steps=10,
        default_root_dir=CHECKPOINT_DIR,
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        # This will only load the model parameters, not the optimizer state
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        
        # Handle model loading with compatibility for DDP and non-DDP checkpoints
        if 'state_dict' in checkpoint:
            if any(k.startswith('model.') for k in checkpoint['state_dict']):
                # DDP checkpoint
                state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()
                             if k.startswith('model.')}
            else:
                # Non-DDP checkpoint
                state_dict = checkpoint['state_dict']
        else:
            # Raw state dict
            state_dict = checkpoint
        
        # Load state dict into model
        model.model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully.")
    
    # Log dataset info if available
    if hasattr(dataset, 'filtered_ids'):
        print(f"Filtered out {len(dataset.filtered_ids)} sequences due to dimension mismatch")
    if hasattr(dataset, 'missing_pdb_ids'):
        print(f"Filtered out {len(dataset.missing_pdb_ids)} sequences with missing or invalid PDB files")
    if hasattr(dataset, 'skipped_long_seqs'):
        print(f"Skipped {len(dataset.skipped_long_seqs)} long sequences (> {MAX_SEQ_LENGTH})")
    
    # Train the model
    trainer.fit(model, train_loader)
    
    # Load and evaluate the best model
    best_checkpoint_path = trainer.checkpoint_callback.best_model_path
    if best_checkpoint_path:
        print(f"Best model checkpoint: {best_checkpoint_path}")
        
        # Load best model for final evaluation
        best_model = RhoFoldLightningModule.load_from_checkpoint(best_checkpoint_path)
        best_model.eval()
        
        # Create a generator function for eval_model
        def generator(features):
            with torch.no_grad():
                if "evo2_fea" in features:
                    features["evo2_fea"] = features["evo2_fea"].to(torch.float32)
                
                outputs = best_model(
                    tokens=features["tokens"].to(best_model.device),
                    rna_fm_tokens=features["rna_fm_tokens"].to(best_model.device),
                    seq=features["seq"],
                    evo2_fea=features["evo2_fea"].to(best_model.device) if "evo2_fea" in features else None
                )
                
                preds = []
                for i in range(min(5, len(outputs))):
                    preds.append(outputs[i]["cords_c1'"][0][0].to(torch.float32))
                return preds
        
        # Run final evaluation
        final_score = eval_model(generator)
        print(f"Final evaluation TM score: {final_score:.4f}")
        
        if args.use_wandb and trainer.is_global_zero:
            wandb.log({"final_tm_score": final_score})
    
    print("Training completed")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RhoFold model")
    parser.add_argument("--data_dir", type=str, default='/dev/shm', help="Path to RNA data directory")
    parser.add_argument("--checkpoint", type=str, default='pretrained/RhoFold_pretrained.pt', help="Path to checkpoint to resume from")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--use_compile", action="store_true", help="Use torch.compile for model compilation")
    
    args = parser.parse_args()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(CHECKPOINT_DIR, "training.log"))
        ]
    )
    
    # Train the model
    train(args)