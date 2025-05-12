import os
import sys
import argparse
import logging
from tqdm import tqdm
import time
import random
import socket
import multiprocessing as mp
import datetime

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
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
from rhofold.utils import get_device
from rhofold.utils.alphabet import get_features, read_fas
from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.utils.loss import compute_fape

# Import evaluation utilities
from utils import eval_model, tm_score

# Import distributed utilities to ensure compatibility with eval_model
from distributed_utils import get_world_size, get_rank, is_main_process

# Training configuration
CHECKPOINT_DIR = "checkpoints"
USE_EVO2 = True
BATCH_SIZE = 1  # Keep batch size at 1 for each GPU
NUM_EPOCHS = 20
LEARNING_RATE = 2e-7
EVO2_LR = 1e-4  # Higher learning rate for evo2_head
EVAL_PER_EPOCH = 4  # Number of evaluations per epoch
CHECKPOINT_EVERY = 1
WARMUP_STEPS = 1000
SKIP_SHORT_SEQS = True  # If True, skip sequences with length > MAX_SEQ_LENGTH
GRAD_ACCUM_STEPS = 4   # Number of steps to accumulate gradients
MAX_SEQ_LENGTH = 200    # Maximum sequence length to process to avoid OOM errors
WEIGHT_DECAY = 0.01    # Weight decay for regularization 
DROPOUT_RATE = 0.1     # Dropout rate for regularization

# Set to your specific project/entity here or use environment variables
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "rhofold")

def get_evo2head_norm(model):
    evo2_head = model.module.evo2_head
    param_count = sum(p.numel() for p in evo2_head.parameters())
    param_sum = sum((p**2).sum() for p in evo2_head.parameters())
    return param_sum / param_count
def setup_distributed():
    """
    Setup distributed training environment for torchrun
    
    Environment variables set by torchrun:
    - RANK: Global rank of the process
    - WORLD_SIZE: Total number of processes
    - LOCAL_RANK: Local rank of the process on the current node
    - MASTER_ADDR: Address of the master node
    - MASTER_PORT: Port of the master node
    """
    # Get local rank and world size from environment variables (set by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    # Enable NCCL debugging if needed (uncomment for debugging)
    # os.environ['NCCL_DEBUG'] = 'INFO'
    
    # Initialize the process group using env vars set by torchrun
    dist.init_process_group("nccl")
    
    # Set different seeds for different processes for proper randomization
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    
    # Make sure all processes are synchronized before proceeding
    dist.barrier()
    
    return local_rank, rank, world_size


def cleanup_distributed():
    """Clean up distributed training resources"""
    if dist.is_initialized():
        dist.destroy_process_group()


class RNADataset(Dataset):
    def __init__(self, data_dir, use_evo2=True, max_seq_length=MAX_SEQ_LENGTH):
        self.data_dir = data_dir
        self.use_evo2 = use_evo2
        self.max_seq_length = max_seq_length
        
        # Get sequence IDs from the directory
        seq_dir = os.path.join(data_dir, "RNA3D_DATA/seq")
        rMSA_dir = os.path.join(data_dir, "RNA3D_DATA/rMSA")
        
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
        
        tru_len = (len(filtered_seq_ids) // 8) * 8
        filtered_seq_ids = filtered_seq_ids[:tru_len]
        
        # Remove the requirement for dataset length to be a multiple of 8
        self.seq_ids = filtered_seq_ids
        
        logging.info(f"Found {len(self.seq_ids)} RNA sequences with valid PDB files for training")
        if self.missing_pdb_ids:
            logging.info(f"Filtered out {len(self.missing_pdb_ids)} sequences with missing PDB files")
        if self.skipped_long_seqs:
            logging.info(f"Skipped {len(self.skipped_long_seqs)} sequences with length > {MAX_SEQ_LENGTH}")
    
    def __len__(self):
        return len(self.seq_ids)
    
    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]
        
        # Load sequence and MSA features
        input_fas = os.path.join(self.data_dir, f"RNA3D_DATA/seq/{seq_id}.seq")
        input_a3m = os.path.join(self.data_dir, f"RNA3D_DATA/rMSA/{seq_id}.a3m")
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
            embeddings_dir = os.path.join(self.data_dir, "RNA3D_DATA/evo2_embeddings")
            embedding_path = os.path.join(embeddings_dir, f"{seq_id}.pt")
            evo2_embedding = torch.load(embedding_path)
        
        # Get PDB path for ground truth
        pdb_path = os.path.join(self.data_dir, f"RNA3D_DATA/pdb/{seq_id}.pdb")
        # This check is redundant now, but kept for safety
        assert os.path.exists(pdb_path), f"PDB file not found: {pdb_path}"

        assert None not in [seq_id, data_dict['tokens'], data_dict['rna_fm_tokens'], data_dict['seq'], evo2_embedding, pdb_path]

        return {
            'seq_id': seq_id,
            'tokens': data_dict['tokens'],
            'rna_fm_tokens': data_dict['rna_fm_tokens'],
            'seq': data_dict['seq'],
            'evo2_fea': evo2_embedding,
            'pdb_path': pdb_path
        }

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

def compute_tm_loss(output, tru_seq, pdb_path):
    pred_c1_positions = output["cords_c1'"][-1].squeeze(0)  # Shape [N, 3]
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("gt", pdb_path)
    chain = next(structure.get_chains())
    
    # Extract C1' atoms from the PDB structure
    gt_c1_positions = []
    for res in chain:
        try:
            gt_c1_positions.append(res["C1'"].get_coord())
        except:
            gt_c1_positions.append([0, 0, 0])
    gt_c1_positions = torch.tensor(np.array(gt_c1_positions), dtype=torch.float32, device=pred_c1_positions.device)
    pdb_seq = get_pdb_seq(structure)
    mask = get_mask(tru_seq, pdb_seq)
    pred_c1_positions = pred_c1_positions[mask]
    return tm_score(pred_c1_positions, gt_c1_positions)

def compute_fape_loss(output, tru_seq, pdb_path, length_scale=10.0, l1_clamp_distance=None):
    """
    Compute FAPE loss between model output and ground truth PDB
    
    Args:
        output: Dictionary containing model output predictions
        pdb_path: Path to ground truth PDB file
        chain_id: Optional chain ID to use from PDB
        length_scale: Scale factor for FAPE calculation
        l1_clamp_distance: Distance threshold for L1 clamping
        
    Returns:
        FAPE loss value
    """
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
    
    # Parse the PDB to get ground truth C1' positions
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("gt", pdb_path)
    chain = next(structure.get_chains())
    
    # Extract C1' atoms from the PDB structure
    gt_c1_positions = []
    for res in chain:
        try:
            gt_c1_positions.append(res["C1'"].get_coord())
        except:
            gt_c1_positions.append([0, 0, 0])
    gt_c1_positions = np.array(gt_c1_positions)
    
    # Convert to tensor with same dtype and device as predictions
    dtype = pred_frames_tensor.dtype
    device = pred_frames_tensor.device
    gt_c1_positions = torch.tensor(gt_c1_positions, dtype=dtype, device=device)
    
    # Create same-length arrays (truncate if necessary)
    pdb_seq = get_pdb_seq(structure)
    mask = get_mask(tru_seq, pdb_seq)
    pred_c1_positions = pred_c1_positions[mask]
    pred_frames = pred_frames[mask]
    
    # Create a mask for the positions (all 1s since we've truncated to match)
    mask = torch.ones(len(pred_c1_positions), device=device)
    
    # Create target frames
    zeros = torch.zeros(len(pred_c1_positions), 3, device=device, dtype=dtype)
    ones = torch.ones(len(pred_c1_positions), 1, device=device, dtype=dtype)
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
        frames_mask=mask,
        pred_positions=pred_c1_positions,
        target_positions=gt_c1_positions,
        positions_mask=mask,
        length_scale=length_scale,
        pair_mask=None,
        l1_clamp_distance=l1_clamp_distance,
    )
    
    return fape

def compute_dist_loss(output, tru_seq, pdb_path):
    # Extract predicted C1' positions
    pred_p_dist = output["p"][-1].squeeze(0)  # Shape [40, N, N]
    pred_c4_dist = output["c4_"][-1].squeeze(0)  # Shape [40, N, N]
    pred_n_dist = output["n"][-1].squeeze(0)  # Shape [40, N, N]
    pred_dist = torch.stack([pred_p_dist, pred_c4_dist, pred_n_dist], dim=0)
    
    # Parse the PDB to get ground truth p, c4, n positions
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("gt", pdb_path)
    chain = next(structure.get_chains())

    pdb_seq = get_pdb_seq(structure)
    mask = get_mask(tru_seq, pdb_seq)
    pred_dist = pred_dist[..., mask, :][..., :, mask]
    
    # Extract p, c4, n atoms from the PDB structure
    gt_p_positions = []
    gt_c4_positions = []
    gt_n_positions = []
    for res in chain:
        try:
            gt_p_positions.append(res["p"].get_coord())
            gt_c4_positions.append(res["c4_"].get_coord())
            gt_n_positions.append(res["n"].get_coord())
        except:
            gt_p_positions.append(gt_p_positions[-1] if gt_p_positions else [0, 0, 0])
            gt_c4_positions.append(gt_c4_positions[-1] if gt_c4_positions else [0, 0, 0])
            gt_n_positions.append(gt_n_positions[-1] if gt_n_positions else [0, 0, 0])

    positions = np.array([gt_p_positions, gt_c4_positions, gt_n_positions])

    gt_positions = torch.tensor(positions, device=pred_p_dist.device, dtype=torch.float32) # shape [3, N, 3]
    gt_dist = (gt_positions[..., None, :] - gt_positions[..., None, :, :]).norm(dim=-1) # shape [3, N, N]

    boundaries = torch.linspace(2, 40, 39, device=pred_p_dist.device)

    true_bins = torch.sum(gt_dist[..., None] > boundaries, dim=-1) # shape [3, N, N]

    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fn(pred_dist, true_bins)
    
    return loss

def load_checkpoint(model, checkpoint_path, rank):
    """
    Load model checkpoint with proper handling for DDP
    
    Args:
        model: DDP wrapped model
        checkpoint_path: Path to checkpoint file
        rank: Process rank
        
    Returns:
        Starting epoch number
    """
    # Load checkpoint using map_location to place tensors on the right device
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Handle module prefixes in state dict
    if any(k.startswith('module.') for k in checkpoint['model']):
        # Model was saved with DDP
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        # Model was saved without DDP, add module prefix
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            new_state_dict[f'module.{k}'] = v
        model.load_state_dict(new_state_dict, strict=False)
        
    start_epoch = checkpoint.get('epoch', 0)
    if rank == 0:
        print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch}")
    
    return start_epoch


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model checkpoint
    
    Args:
        model: DDP wrapped model
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        path: Path to save checkpoint
    """
    torch.save({
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
    }, path)
    logging.info(f"Checkpoint saved to {path}")

def evaluate_model(model, rank, world_size):
    logging.info(f"[Rank {rank}] Starting evaluation")
    model.eval()
    
    # No barrier here - we'll handle synchronization in the calling function
    
    def generator(features):
        # Use torch.no_grad() during evaluation
        with torch.no_grad():
            try:
                if "evo2_fea" in features:
                    features["evo2_fea"] = features["evo2_fea"].to(torch.float32)
                
                outputs = model.module(
                    tokens=features["tokens"].to(model.device),
                    rna_fm_tokens=features["rna_fm_tokens"].to(model.device),
                    seq=features["seq"],
                    evo2_fea=features["evo2_fea"].to(model.device) if "evo2_fea" in features else None
                )
                
                preds = []
                for i in range(min(5, len(outputs))):
                    preds.append(outputs[i]["cords_c1'"][0][0].to(torch.float32))
                return preds
            except Exception as e:
                logging.error(f"[Rank {rank}] Error in generator: {str(e)} on sequence {features.get('seq_id', 'unknown')}")
                return []
    
    try:
        # Add timeout to prevent indefinite hangs
        eval_score = eval_model(generator)
        
        # Synchronize scores across all processes
        score_tensor = torch.tensor([eval_score], device=model.device)
        dist.all_reduce(score_tensor, op=dist.ReduceOp.MAX)
        eval_score = score_tensor.item()
        
        return eval_score
    except Exception as e:
        logging.error(f"[Rank {rank}] Evaluation error: {str(e)}")
        # Ensure all processes continue even if evaluation fails on some
        error_tensor = torch.tensor([1.0], device=model.device)
        dist.all_reduce(error_tensor, op=dist.ReduceOp.MAX)
        return 0.0
    finally:
        # No barrier here - we'll handle synchronization in the calling function
        # Return to training mode
        model.train()
        # Free up memory
        torch.cuda.empty_cache()


def train_worker(args):
    """
    Training process for a single worker/GPU using torchrun
    
    Args:
        args: Command line arguments
    """
    # Setup distributed process (returns local_rank, global_rank, world_size)
    local_rank, rank, world_size = setup_distributed()
    
    # Set this process's device based on local_rank
    device = torch.device(f"cuda:{local_rank}")
    
    # Initialize wandb only on main process
    if rank == 0 and args.use_wandb:
        wandb.init(
            project=WANDB_PROJECT,
            config={
                "learning_rate": LEARNING_RATE,
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "use_evo2": USE_EVO2,
                "skip_short_seqs": SKIP_SHORT_SEQS,
                "precision": "fp32",
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "max_seq_length": MAX_SEQ_LENGTH,
                "world_size": world_size,
            }
        )
        logging.info(f"Initialized wandb with project={WANDB_PROJECT}")
    
    # Create checkpoint dir if it doesn't exist (only on main process)
    if rank == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Initialize model
    model = RhoFold(rhofold_config).to(device)
    
    # Set dropout rate in model
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = DROPOUT_RATE
    
    # Synchronize model parameters across processes
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    # Only use torch.compile if explicitly requested
    # Skip compilation by default as it may not work well with this complex model
    if args.use_compile and hasattr(torch, 'compile'):
        try:
            logging.info("Attempting to compile model with torch.compile()...")
            model = torch.compile(model, dynamic=True)
            logging.info("Model compilation successful")
        except Exception as e:
            logging.warning(f"Model compilation failed, using eager mode: {str(e)}")
    else:
        logging.info("Skipping model compilation, using eager mode")
    
    # Wrap model with DDP - find_unused_parameters needed for RhoFold architecture
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model initialized with {param_count:,} parameters")
    
    # Initialize optimizer with parameter groups
    # Group 1: evo2_head parameters with higher learning rate
    # Group 2: All other parameters with default learning rate
    evo2_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'evo2_head' in name:
            evo2_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {'params': evo2_params, 'lr': EVO2_LR},
        {'params': other_params, 'lr': LEARNING_RATE}
    ]
    
    optimizer = torch.optim.Adam(param_groups, weight_decay=WEIGHT_DECAY)
    
    # Create warmup scheduler
    def get_warmup_lr_lambda(current_step):
        if current_step < WARMUP_STEPS:
            return float(current_step) / float(max(1, WARMUP_STEPS))
        return 1.0
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_warmup_lr_lambda)
    
    # Learning rate scheduler - will be used after warmup
    # Replace ReduceLROnPlateau with CosineAnnealingLR for cosine decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=NUM_EPOCHS,  # Full cosine cycle over remaining epochs
        eta_min=LEARNING_RATE * 0.01  # Minimum learning rate will be 1% of initial rate
    )
    
    # If starting from a checkpoint, load it
    start_epoch = 0
    if args.checkpoint:
        start_epoch = load_checkpoint(model, args.checkpoint, rank)
    
    # Create dataset and dataloader
    dataset = RNADataset(args.data_dir, use_evo2=USE_EVO2, max_seq_length=MAX_SEQ_LENGTH)
    
    # Create distributed sampler - PyTorch's DistributedSampler automatically handles
    # ensuring all GPUs get the same number of batches, avoiding deadlocks
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=False,  # Keep sequential ordering
        drop_last=True  # Drop the last batch to ensure all GPUs have same number of batches
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop
    global_step = 0
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_loss = 0
        epoch_fape_loss = 0
        epoch_tm_score = 0
        epoch_dist_loss = 0
        optimizer.zero_grad()
        
        # Set epoch for sampler
        sampler.set_epoch(epoch)
        
        # Synchronize at the beginning of each epoch
        dist.barrier()
        
        # Create progress bar on main process only
        if rank == 0:
            pbar = tqdm(total=len(sampler), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        processed_batches = 0
        total_batches = len(dataloader)
        eval_interval = max(1, total_batches // EVAL_PER_EPOCH)  # Evaluate every 1/EVAL_PER_EPOCH of an epoch
        
        for batch_idx, batch in enumerate(dataloader):
            # Run evaluation at regular intervals
            if batch_idx % eval_interval == 0:
                model.eval()
                if rank == 0:
                    print(f"\n[Evaluation] Running at {(batch_idx + 1) / total_batches * 100:.1f}% of epoch {epoch + 1}")
                
                # Make sure all processes are ready to evaluate
                try:
                    # Single barrier before evaluation
                    dist.barrier()
                    
                    # Run evaluation
                    eval_score = evaluate_model(model, rank, world_size)
                    
                    # Log evaluation metrics on main process
                    if rank == 0 and args.use_wandb and eval_score > 0:
                        wandb.log({
                            "epoch": epoch + 1,
                            "progress": (batch_idx + 1) / total_batches,
                            "tm_score": eval_score
                        })
                        print(f"[Evaluation] TM Score: {eval_score:.4f}")
                    
                    # Make sure all processes are done with evaluation before proceeding
                    dist.barrier()
                except Exception as e:
                    logging.error(f"[Rank {rank}] Evaluation block error: {str(e)}")
                    # Try to synchronize processes even if evaluation fails
                    try:
                        dist.barrier()
                    except:
                        logging.error(f"[Rank {rank}] Failed to synchronize after evaluation error")
                
                # Return to train mode
                model.train()
                
                # Free memory after evaluation
                torch.cuda.empty_cache()
            
            # Skip sequences that are too long to avoid OOM
            seq = batch['seq'][0]
            assert len(seq) <= MAX_SEQ_LENGTH, f"Sequence length {len(seq)} > {MAX_SEQ_LENGTH}"
                
            # Process batch data
            seq_id = batch['seq_id'][0]
            tokens = batch['tokens'][0].to(device)
            rna_fm_tokens = batch['rna_fm_tokens'][0].to(device)
            pdb_path = batch['pdb_path'][0]
            while tokens.dim() < 3:
                tokens = tokens.unsqueeze(0)
            while rna_fm_tokens.dim() < 2:
                rna_fm_tokens = rna_fm_tokens.unsqueeze(0)
            
            # Handle evo2 features
            evo2_fea = None
            if USE_EVO2 and batch['evo2_fea'] is not None:
                evo2_fea = batch['evo2_fea'][0].to(device).to(torch.float32)
            
            # Run model forward pass
            outputs = model(tokens=tokens, rna_fm_tokens=rna_fm_tokens, seq=seq, evo2_fea=evo2_fea, train=True)
           
            # Take the last output from recycles
            output = outputs[-1]
            
            # Compute FAPE loss
            fape_loss = compute_fape_loss(output, seq, pdb_path)
            tm_score = compute_tm_loss(output, seq, pdb_path)
            dist_loss = compute_dist_loss(output, seq, pdb_path)
            
            # Combined loss: FAPE - 20 * TM score
            loss = 2 * fape_loss - 500 * tm_score + 0.3 * dist_loss
            
            # Scale the loss for gradient accumulation
            loss = loss / GRAD_ACCUM_STEPS
            
            # Backward pass
            loss.backward()
            
            # Print parameters with no gradients (rank 0 only)
            if False:
                if rank == 0 and (batch_idx == 0 or batch_idx % 20 == 0):  # Only print occasionally to avoid spam
                    params_with_no_grad = []
                    params_with_grad = []
                    for i, (name, param) in enumerate(model.named_parameters()):
                        if param.grad is None:
                            params_with_no_grad.append((i, name))
                        else:
                            params_with_grad.append((i, name))

                    
                    if params_with_no_grad:
                        print(f"\n[Rank 0] Parameters with no gradients after backward pass (batch {batch_idx}):")
                        for idx, name in params_with_no_grad:
                            print(f"  [{idx}] {name}")
                        print(f"Total: {len(params_with_no_grad)} parameters with no gradients")
            
            # Step if we've accumulated enough gradients
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(dataloader):
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Track loss
            epoch_loss += loss.item()
            epoch_fape_loss += fape_loss.item()
            epoch_tm_score += tm_score
            epoch_dist_loss += dist_loss.item()
            processed_batches += 1
            
            # Log batch metrics on main process
            if rank == 0 and args.use_wandb:
                evo2head_norm = get_evo2head_norm(model)
                wandb.log({
                    "head_lr": optimizer.param_groups[0]['lr'],
                    "batch_loss": loss.item(),
                    "fape_loss": fape_loss.item(),
                    "tm_score": tm_score,
                    "dist_loss": dist_loss.item(),
                    "pLDDT": output["plddt"][1].item() if "plddt" in output else 0.0,
                    "seq_id": seq_id,
                    "seq_length": len(seq),
                    "memory_usage_MB": torch.cuda.memory_allocated(device) / 1024**2,
                    "evo2head_norm": evo2head_norm,
                    "epoch_avg_loss": epoch_loss / processed_batches,
                    "epoch_avg_fape_loss": epoch_fape_loss / processed_batches,
                    "epoch_avg_tm_score": epoch_tm_score / processed_batches,
                    "epoch_avg_dist_loss": epoch_dist_loss / processed_batches,
                })
            
            # Update progress bar on main process
            if rank == 0:
                pbar.update(1)
                
            # Update global step
            global_step += 1
            
            # Apply warmup scheduler if still in warmup phase
            if global_step <= WARMUP_STEPS:
                warmup_scheduler.step()
        
        # Close progress bar on main process
        if rank == 0:
            pbar.close()
        
        # Calculate average epoch loss
        if processed_batches > 0:
            avg_loss = epoch_loss / processed_batches
            avg_fape_loss = epoch_fape_loss / processed_batches
            avg_tm_score = epoch_tm_score / processed_batches
        else:
            avg_loss = 0
            avg_fape_loss = 0
            avg_tm_score = 0
            
        logging.info(f"Rank {rank} - epoch {epoch+1} completed")
        # Average loss across all processes
        avg_loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / world_size
        
        # Average FAPE loss across all processes
        avg_fape_loss_tensor = torch.tensor([avg_fape_loss], device=device)
        dist.all_reduce(avg_fape_loss_tensor, op=dist.ReduceOp.SUM)
        avg_fape_loss = avg_fape_loss_tensor.item() / world_size
        
        # Average TM score across all processes
        avg_tm_score_tensor = torch.tensor([avg_tm_score], device=device)
        dist.all_reduce(avg_tm_score_tensor, op=dist.ReduceOp.SUM)
        avg_tm_score = avg_tm_score_tensor.item() / world_size
        
        # Get total processed batches from all GPUs
        processed_tensor = torch.tensor([processed_batches], device=device)
        dist.all_reduce(processed_tensor, op=dist.ReduceOp.SUM)
        total_processed = processed_tensor.item()
        
        # Log epoch metrics on main process
        if rank == 0:
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - FAPE Loss: {avg_fape_loss:.4f} - TM Score: {avg_tm_score:.4f} - Processed {total_processed} sequences")
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "train_fape_loss": avg_fape_loss,
                    "train_tm_score": avg_tm_score,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "processed_sequences": total_processed
                })
        
        logging.info(f"Rank {rank} - epoch {epoch+1} logging completed")
        # Save checkpoint (on main process only)
        if (epoch + 1) % CHECKPOINT_EVERY == 0 and rank == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"rhofold_epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
        logging.info(f"Rank {rank} - epoch {epoch+1} checkpoint saved")
    
    # Save final model (on main process only)
    if rank == 0:
        final_checkpoint_path = os.path.join(CHECKPOINT_DIR, "rhofold_final.pt")
        save_checkpoint(model, optimizer, NUM_EPOCHS - 1, avg_loss, final_checkpoint_path)
        
        # Print summary of filtered sequences
        if hasattr(dataset, 'filtered_ids') and dataset.filtered_ids:
            logging.info(f"Filtered {len(dataset.filtered_ids)} sequences due to token shape mismatch")
        
        # Print summary of sequences with missing PDB files
        if hasattr(dataset, 'missing_pdb_ids') and dataset.missing_pdb_ids:
            logging.info(f"Filtered {len(dataset.missing_pdb_ids)} sequences due to missing PDB files")
            
        # Print summary of skipped long sequences
        if hasattr(dataset, 'skipped_long_seqs') and dataset.skipped_long_seqs:
            logging.info(f"Skipped {len(dataset.skipped_long_seqs)} sequences with length > {MAX_SEQ_LENGTH}")
        
        if args.use_wandb:
            wandb.finish()
        print(f"Training completed. Final model saved to {final_checkpoint_path}")
    
    # Clean up distributed process
    cleanup_distributed()


def train(args):
    """
    Main training function that works with torchrun
    
    For torchrun, this function directly calls train_worker
    instead of spawning processes, as torchrun handles process creation
    
    Args:
        args: Command line arguments
    """
    try:
        # When using torchrun, we directly call train_worker 
        # as the processes are already spawned
        train_worker(args)
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        # Make sure to clean up in case of error
        if dist.is_initialized():
            dist.destroy_process_group()
        
        # Clean up any leaked semaphores
        try:
            import subprocess
            # Find semaphores created by this user
            result = subprocess.run(["ipcs", "-s"], capture_output=True, text=True)
            semaphore_lines = result.stdout.strip().split("\n")
            
            # Get username for filtering
            import getpass
            username = getpass.getuser()
            
            # Extract semaphore IDs owned by this user
            semaphore_ids = []
            for line in semaphore_lines:
                if username in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        semaphore_ids.append(parts[1])
            
            # Remove each semaphore
            for sem_id in semaphore_ids:
                subprocess.run(["ipcrm", "-s", sem_id])
                logging.info(f"Cleaned up semaphore {sem_id}")
        except Exception as cleanup_error:
            logging.warning(f"Failed to clean up semaphores: {str(cleanup_error)}")
        
        # Re-raise the original exception
        raise


def main():
    # Set environment variable to help debug DDP unused parameters
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    
    parser = argparse.ArgumentParser(description="Train RhoFold with FAPE loss")
    parser.add_argument("--data_dir", type=str, default="/dev/shm", 
                        help="Directory containing RNA data")
    parser.add_argument("--checkpoint", type=str, default="./pretrained/RhoFold_pretrained.pt", 
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--use_wandb", action="store_true", default=False,
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug mode with more verbose logging")
    parser.add_argument("--use_compile", action="store_true", default=False,
                        help="Attempt to use torch.compile() to optimize model (may not work for all models)")
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    # Include local rank in log format when running with torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    log_format = f"%(asctime)s [Rank {local_rank}] [%(levelname)s] %(message)s"
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"rhofold_training_rank{local_rank}.log")
        ]
    )
    
    # Set deterministic training for reproducibility
    # Base seed is the same, but each rank gets a different derived seed
    base_seed = 42
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    random.seed(base_seed)
    
    # Start the training process
    train(args)


if __name__ == "__main__":
    main()