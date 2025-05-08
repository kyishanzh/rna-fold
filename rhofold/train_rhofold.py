import os
import sys
import argparse
import logging
from tqdm import tqdm
import time
import random
import socket
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import wandb
import numpy as np
from Bio.PDB import PDBParser
from torch.cuda.amp import autocast, GradScaler

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
from utils import eval_model, tm_score as calculate_tm_score

# Import distributed utilities to ensure compatibility with eval_model
from distributed_utils import get_world_size, get_rank, is_main_process

# Training configuration
CHECKPOINT_DIR = "checkpoints"
USE_EVO2 = True
BATCH_SIZE = 1  # Keep batch size at 1 for each GPU
NUM_EPOCHS = 20
LEARNING_RATE = 4e-4
CHECKPOINT_EVERY = 2
WARMUP_STEPS = 100
SKIP_SHORT_SEQS = True  # If True, skip sequences with length > MAX_SEQ_LENGTH
GRAD_ACCUM_STEPS = 4   # Number of steps to accumulate gradients
MAX_SEQ_LENGTH = 45    # Maximum sequence length to process to avoid OOM errors
USE_FP16 = True        # Enable FP16 (half-precision) training

# Set to your specific project/entity here or use environment variables
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "rhofold")
WANDB_DISABLED = False


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


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that doesn't shuffle, ensuring each GPU gets
    unique samples in a deterministic order
    """
    def __init__(self, dataset, world_size, rank):
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.num_samples = len(dataset) // world_size + int(len(dataset) % world_size > rank)
        self.total_size = len(dataset)

    def __iter__(self):
        # Determine the indices for this process
        indices = list(range(self.total_size))
        # Subsample
        indices = indices[self.rank:self.total_size:self.world_size]
        return iter(indices)

    def __len__(self):
        return self.num_samples
        
    def set_epoch(self, epoch):
        # This method is needed for compatibility with DDP training
        # Since this sampler is sequential and doesn't shuffle, we don't
        # need to do anything with the epoch number
        pass


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
            pdb_path = os.path.join(data_dir, f"RNA3D_DATA/pdb/{seq_id}.pdb")
            if not os.path.exists(pdb_path):
                self.missing_pdb_ids.append(seq_id)
                continue
                
            # Skip sequences longer than MAX_SEQ_LENGTH if SKIP_SHORT_SEQS is enabled
            if SKIP_SHORT_SEQS:
                input_fas = os.path.join(data_dir, f"RNA3D_DATA/seq/{seq_id}.seq")
                # Read sequence to check length
                seq = read_fas(input_fas)[0][1]
                    
                if len(seq) <= MAX_SEQ_LENGTH:
                    filtered_seq_ids.append(seq_id)
                else:
                    self.skipped_long_seqs.append(seq_id)
            else:
                filtered_seq_ids.append(seq_id)
        
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
            if os.path.exists(embedding_path):
                evo2_embedding = torch.load(embedding_path)
        
        # Get PDB path for ground truth
        pdb_path = os.path.join(self.data_dir, f"RNA3D_DATA/pdb/{seq_id}.pdb")
        # This check is redundant now, but kept for safety
        assert os.path.exists(pdb_path), f"PDB file not found: {pdb_path}"

        return {
            'seq_id': seq_id,
            'tokens': data_dict['tokens'],
            'rna_fm_tokens': data_dict['rna_fm_tokens'],
            'seq': data_dict['seq'],
            'evo2_fea': evo2_embedding,
            'pdb_path': pdb_path
        }


def compute_fape_loss(output, pdb_path, chain_id=None, length_scale=10.0, l1_clamp_distance=None):
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
    if chain_id is None:
        chain = next(structure.get_chains())
    else:
        chain = structure[0][chain_id]
    
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
    n_residues = min(pred_c1_positions.shape[0], gt_c1_positions.shape[0])
    pred_c1_positions = pred_c1_positions[:n_residues]
    gt_c1_positions = gt_c1_positions[:n_residues]
    pred_frames = pred_frames[:n_residues]
    
    # Create a mask for the positions (all 1s since we've truncated to match)
    mask = torch.ones(n_residues, device=device)
    
    # Create target frames
    zeros = torch.zeros(n_residues, 3, device=device, dtype=dtype)
    ones = torch.ones(n_residues, 1, device=device, dtype=dtype)
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


def evaluate_model(model, rank, world_size, use_fp16=USE_FP16):
    """
    Run evaluation on the model during training
    
    Args:
        model: DDP wrapped model
        rank: Process rank
        world_size: Total number of processes
        use_fp16: Whether to use mixed precision
        
    Returns:
        Evaluation score (TM score)
    """
    # Switch model to eval mode
    model.eval()
    
    # Define generator function for eval_model
    def generator(features):
        with torch.no_grad():
            # Convert input tensors to float32 to ensure consistent dtype
            # Convert evolutionary features to float32 for consistent precision
            if "evo2_fea" in features:
                features["evo2_fea"] = features["evo2_fea"].to(torch.float32)
            
            # Disable autocast to avoid mixed precision
            outputs = model.module(
                tokens=features["tokens"].to(model.device), 
                rna_fm_tokens=features["rna_fm_tokens"].to(model.device), 
                seq=features["seq"],
                evo2_fea=features["evo2_fea"].to(model.device)
            )
            
            # Convert all output tensors to float32
            preds = []
            for i in range(min(5, len(outputs))):
                # Ensure all tensors are float32
                preds.append(outputs[i]["cords_c1'"][0][0].to(torch.float32))
            return preds
    
    try:
        # Run evaluation
        eval_score = eval_model(generator)
        return eval_score
    except Exception as e:
        if rank == 0:
            # More detailed error reporting
            import traceback
            logging.error(f"Evaluation error: {str(e)}")
            logging.error(traceback.format_exc())
        return 0.0


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
                "precision": "fp16" if USE_FP16 else "fp32",
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "max_seq_length": MAX_SEQ_LENGTH,
                "world_size": world_size,
            }
        )
        logging.info(f"Initialized wandb with project={WANDB_PROJECT}")
    
    # Create checkpoint dir if it doesn't exist (only on main process)
    if rank == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Wait for rank 0 to create checkpoint dir
    dist.barrier()
    
    # Initialize model
    model = RhoFold(rhofold_config).to(device)
    
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
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler(enabled=USE_FP16)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # If starting from a checkpoint, load it
    start_epoch = 0
    if args.checkpoint:
        start_epoch = load_checkpoint(model, args.checkpoint, rank)
    
    # Create dataset and dataloader
    dataset = RNADataset(args.data_dir, use_evo2=USE_EVO2, max_seq_length=MAX_SEQ_LENGTH)
    
    # Create distributed sampler
    sampler = SequentialDistributedSampler(dataset, world_size, rank)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        # Set epoch for sampler
        sampler.set_epoch(epoch)
        
        # Ensure all processes are synchronized before starting the epoch
        dist.barrier()
        
        # Create progress bar on main process only
        if rank == 0:
            pbar = tqdm(total=len(sampler), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        processed_batches = 0
        total_batches = len(dataloader)
        eval_interval = max(1, total_batches // 10)  # Evaluate every 10% of an epoch
        
        for batch_idx, batch in enumerate(dataloader):
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
            
            # Forward pass with mixed precision
            with autocast(enabled=USE_FP16, dtype=torch.float16):
                # Run model forward pass
                outputs = model(tokens=tokens, rna_fm_tokens=rna_fm_tokens, seq=seq, evo2_fea=evo2_fea)
                
                # Take the last output from recycles
                output = outputs[-1]
                
                # Compute FAPE loss
                loss = compute_fape_loss(output, pdb_path)
                # Scale the loss for gradient accumulation
                loss = loss / GRAD_ACCUM_STEPS

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Step if we've accumulated enough gradients
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(dataloader):
                # Unscale before gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Track loss
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
            processed_batches += 1
            
            # Log batch metrics on main process
            if rank == 0 and args.use_wandb:
                wandb.log({
                    "batch_loss": loss.item() * GRAD_ACCUM_STEPS,
                    "pLDDT": output["plddt"][1].item() if "plddt" in output else 0.0,
                    "seq_id": seq_id,
                    "seq_length": len(seq),
                    "memory_usage_MB": torch.cuda.memory_allocated(device) / 1024**2,
                })
            
            # Update progress bar on main process
            if rank == 0:
                pbar.update(1)
                
            # Run evaluation at regular intervals
            if batch_idx % eval_interval == 0:
                if rank == 0:
                    print(f"\n[Evaluation] Running at {(batch_idx + 1) / total_batches * 100:.1f}% of epoch {epoch + 1}")
                
                # Ensure all processes are synchronized before evaluation
                dist.barrier()
                
                # Run evaluation
                eval_score = evaluate_model(model, rank, world_size, use_fp16=USE_FP16)
                
                # Log evaluation metrics on main process
                if rank == 0 and args.use_wandb and eval_score > 0:
                    wandb.log({
                        "epoch": epoch + 1,
                        "progress": (batch_idx + 1) / total_batches,
                        "tm_score": eval_score
                    })
                    print(f"[Evaluation] TM Score: {eval_score:.4f}")
                
                # Ensure all processes are synchronized after evaluation
                dist.barrier()
                
                # Return to train mode
                model.train()
                
                # Free memory after evaluation
                torch.cuda.empty_cache()
        
        # Close progress bar on main process
        if rank == 0:
            pbar.close()
        
        # Wait for all processes to finish their epoch iterations
        dist.barrier()
        
        # Calculate average epoch loss
        if processed_batches > 0:
            avg_loss = epoch_loss / processed_batches
        else:
            avg_loss = 0
            
        # Average loss across all processes
        avg_loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / world_size
        
        # Get total processed batches from all GPUs
        processed_tensor = torch.tensor([processed_batches], device=device)
        dist.all_reduce(processed_tensor, op=dist.ReduceOp.SUM)
        total_processed = processed_tensor.item()
        
        # Log epoch metrics on main process
        if rank == 0:
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - Processed {total_processed} sequences")
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "processed_sequences": total_processed
                })
        
        # Update learning rate scheduler (on main process only)
        if rank == 0:
            scheduler.step(avg_loss)
            
            # Broadcast new learning rate to all processes
            for i, param_group in enumerate(optimizer.param_groups):
                lr = torch.tensor([param_group['lr']], device=device)
                dist.broadcast(lr, src=0)
                if rank != 0:
                    param_group['lr'] = lr.item()
        
        # Save checkpoint (on main process only)
        if (epoch + 1) % CHECKPOINT_EVERY == 0 and rank == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"rhofold_epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
        
        # Make sure all processes sync up before starting next epoch
        dist.barrier()
    
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
    dist.barrier()  # Final sync point
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
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--use_wandb", action="store_true", default=not WANDB_DISABLED,
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