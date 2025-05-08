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
from rhofold.utils.alphabet import get_features
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
SKIP_SHORT_SEQS = True  # If True, skip sequences with length > SEQ_CUTOFF
SEQ_CUTOFF = 73
GRAD_ACCUM_STEPS = 4   # Number of steps to accumulate gradients
MAX_SEQ_LENGTH = 45    # Maximum sequence length to process to avoid OOM errors
USE_FP16 = True        # Enable FP16 (half-precision) training

# Set to your specific project/entity here or use environment variables
os.environ["WANDB_API_KEY"] = "71278e965a6e50657c6b254d59ba8fac486c97dd"  # Uncomment and set your API key if needed
os.environ["WANDB_PROJECT"] = "rhofold"  # Uncomment to override default project

# Get wandb configuration from environment or use defaults
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "rhofold")
WANDB_DISABLED = False

def setup_distributed(rank, world_size, port):
    """
    Setup distributed training
    """
    # Use the provided port (must be the same across all processes)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set different seeds for different processes for proper randomization
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)
    
    # Make sure all processes are synchronized before proceeding
    torch.cuda.synchronize()
    dist.barrier()

def cleanup_distributed():
    """
    Clean up distributed training
    """
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
    def __init__(self, data_dir, use_evo2=True):
        self.data_dir = data_dir
        self.use_evo2 = use_evo2
        
        # Get sequence IDs from the directory
        seq_dir = os.path.join(data_dir, "RNA3D_DATA/seq")
        rMSA_dir = os.path.join(data_dir, "RNA3D_DATA/rMSA")
        
        # Filter sequence IDs to only include those with both seq and a3m files
        self.seq_ids = []
        for f in os.listdir(seq_dir):
            if f.endswith('.seq'):
                seq_id = f.split('.')[0]
                a3m_path = os.path.join(rMSA_dir, f"{seq_id}.a3m")
                
                # Only include if a3m file exists
                if os.path.exists(a3m_path):
                    self.seq_ids.append(seq_id)
        
        # Track filtered out sequences
        self.filtered_ids = []
        self.skipped_long_seqs = []
        
        # Skip sequences longer than SEQ_CUTOFF if SKIP_SHORT_SEQS is enabled
        if SKIP_SHORT_SEQS:
            filtered_seq_ids = []
            for seq_id in self.seq_ids:
                input_fas = os.path.join(data_dir, f"RNA3D_DATA/seq/{seq_id}.seq")
                # Read sequence to check length
                with open(input_fas, 'r') as f:
                    # Skip header line
                    f.readline()
                    seq = f.readline().strip()
                    
                if len(seq) > SEQ_CUTOFF:
                    self.skipped_long_seqs.append(seq_id)
                else:
                    filtered_seq_ids.append(seq_id)
            
            self.seq_ids = filtered_seq_ids
        
        logging.info(f"Found {len(self.seq_ids)} RNA sequences length at most {MAX_SEQ_LENGTH} with MSA data for training")
    
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
                # logging.warning(f"Skipping {seq_id}: tokens shape {data_dict['tokens'].shape} doesn't match rna_fm_tokens shape {data_dict['rna_fm_tokens'].shape}")
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
            else:
                logging.warning(f"Embedding file for {seq_id} not found at {embedding_path}")
        
        # Get PDB path for ground truth
        pdb_path = os.path.join(self.data_dir, f"RNA3D_DATA/pdb/{seq_id}.pdb")

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
    gt_c1_positions = np.array([res["C1'"].get_coord() for res in chain])
    
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

def train_worker(rank, world_size, args, port):
    """
    Training process for a single worker/GPU
    """
    # Setup distributed process
    setup_distributed(rank, world_size, port)
    
    # Set this process's device
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Initialize wandb only on main process
    if rank == 0 and args.use_wandb:
        if WANDB_API_KEY:
            wandb.login(key=WANDB_API_KEY)
            logging.info(f"Logged into wandb with API key from environment")
        
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
    torch.compile(model)
    
    # Important: synchronize model parameters across processes
    # before wrapping with DDP to ensure consistent initialization
    # Broadcast model parameters from rank 0 to all other ranks
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    # Wrap model with DDP - Enable find_unused_parameters to handle unused parameters in forward pass
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    if rank == 0:
        print(f"Number of params: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler(enabled=USE_FP16)
    
    # If starting from a checkpoint, load it
    start_epoch = 0
    if args.checkpoint:
        # Load checkpoint using map_location to place tensors on the right device
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(args.checkpoint, map_location=map_location)
        
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
            print(f"Loaded checkpoint from {args.checkpoint}, starting from epoch {start_epoch}")
    
    # Synchronize after checkpoint loading
    dist.barrier()
    
    # Create dataset
    dataset = RNADataset(args.data_dir, use_evo2=USE_EVO2)
    
    # Create distributed sampler to handle sharding
    sampler = SequentialDistributedSampler(dataset, world_size, rank)
    
    # Create dataloader with single batch size (no collation issues)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        # Set epoch for sampler
        sampler.set_epoch(epoch)
        
        # Make sure all processes are synchronized before starting the epoch
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
            if len(seq) > MAX_SEQ_LENGTH:
                if rank == 0:
                    print(f"Skipping sequence of length {len(seq)} > {MAX_SEQ_LENGTH}")
                continue
                
            # Move data to device
            seq_id = batch['seq_id'][0]
            tokens = batch['tokens'][0].to(device)
            rna_fm_tokens = batch['rna_fm_tokens'][0].to(device)
            pdb_path = batch['pdb_path'][0]
            
            # Get chain ID from seq_id if available
            chain_id = None
            
            # Handle evo2 features
            evo2_fea = None
            if USE_EVO2 and batch['evo2_fea'] is not None:
                evo2_fea = batch['evo2_fea'][0].to(device).to(torch.float32)
            
            # Print info only from rank 0 to avoid console spam
            if rank == 0:
                print(f"[GPU {rank}] Processing sequence {seq_id} (length {len(seq)})")
            
            # Synchronize before each forward pass to ensure all GPUs are ready
            torch.cuda.synchronize(device)
            
            # Forward pass with mixed precision
            with autocast(enabled=USE_FP16, dtype=torch.float16):
                # Run model forward pass
                outputs = model(tokens=tokens, rna_fm_tokens=rna_fm_tokens, seq=seq, evo2_fea=evo2_fea)
                
                # Take the last output from recycles
                output = outputs[-1]
                
                # Make sure to touch all output tensors to ensure proper DDP synchronization
                # This helps with the unused parameter issue
                for k, v in output.items():
                    if isinstance(v, torch.Tensor) and v.requires_grad:
                        v.sum()  # Just to make sure tensor is used
                
                # Compute FAPE loss
                loss = compute_fape_loss(output, pdb_path, chain_id=chain_id)
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
            
            # Synchronize after backward to ensure all processes have completed their backward pass
            torch.cuda.synchronize(device)
            
            # Print sequence length and VRAM usage information
            if rank == 0:
                print(f"[GPU {rank}] Took {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB of VRAM")
                # Force CUDA to free memory cache if needed
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
            
            # Track loss
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
            processed_batches += 1
            
            # Log batch metrics on main process
            if rank == 0 and args.use_wandb:
                wandb.log({
                    "batch_loss": loss.item() * GRAD_ACCUM_STEPS,
                    "pLDDT": output["plddt"][1].item(),
                    "seq_id": seq_id,
                    "seq_length": len(seq),
                    "memory_usage_MB": torch.cuda.memory_allocated(device) / 1024**2,
                    "gpu": rank
                })
            
            # Update progress bar on main process
            if rank == 0:
                pbar.update(1)
                
            # Run evaluation every 10% of an epoch
            if batch_idx % eval_interval == 0:
                if rank == 0:
                    print(f"\n[Evaluation] Running evaluation at {(batch_idx + 1) / total_batches * 100:.1f}% of epoch {epoch + 1}")
                
                # Make sure all processes are synchronized before evaluation
                dist.barrier()
                
                # Switch model to eval mode
                model.eval()
                
                try:
                    # Define generator function for eval_model
                    def generator(features):
                        with torch.no_grad():
                            with autocast(enabled=USE_FP16, dtype=torch.float16):
                                # Call the model's forward method directly through module to bypass DDP wrapper
                                outputs = model.module(
                                    tokens=features["tokens"], 
                                    rna_fm_tokens=features["rna_fm_tokens"], 
                                    seq=features["seq"],
                                    evo2_fea=features["evo2_fea"]
                                )
                                preds = []
                                for i in range(min(5, len(outputs))):
                                    preds.append(outputs[i]["cords_c1'"][0][0])
                                return preds
                    
                    # Run evaluation
                    eval_score = eval_model(generator)
                    
                    # Log evaluation metrics on main process
                    if rank == 0 and args.use_wandb:
                        wandb.log({
                            "epoch": epoch + 1,
                            "progress": (batch_idx + 1) / total_batches,
                            "tm_score": eval_score
                        })
                        print(f"[Evaluation] TM Score: {eval_score:.4f}")
                except Exception as e:
                    if rank == 0:
                        print(f"[Evaluation] Error during evaluation: {str(e)}")
                        logging.error(f"Evaluation error: {str(e)}")
                
                # Make sure all processes are synchronized after evaluation
                dist.barrier()
                
                # Switch back to train mode
                model.train()
                
                # Empty cuda cache after evaluation to free up memory
                torch.cuda.empty_cache()
        
        # Close progress bar on main process
        if rank == 0:
            pbar.close()
        
        # Wait for all processes to finish their epoch iterations
        dist.barrier()
        
        # Calculate average epoch loss
        # Gather losses from all GPUs
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
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Make sure all processes sync up before starting next epoch
        dist.barrier()
    
    # Save final model (on main process only)
    if rank == 0:
        final_checkpoint_path = os.path.join(CHECKPOINT_DIR, "rhofold_final.pt")
        torch.save({
            'epoch': NUM_EPOCHS,
            'model': model.state_dict(),
            'loss': avg_loss,
        }, final_checkpoint_path)
        
        # Print summary of filtered sequences
        if hasattr(dataset, 'filtered_ids') and dataset.filtered_ids:
            logging.info(f"Filtered {len(dataset.filtered_ids)} sequences due to token shape mismatch:")
            for seq_id in dataset.filtered_ids:
                logging.info(f"  - {seq_id}")
        
        # Print summary of skipped long sequences
        if hasattr(dataset, 'skipped_long_seqs') and dataset.skipped_long_seqs:
            logging.info(f"Skipped {len(dataset.skipped_long_seqs)} sequences with length > {SEQ_CUTOFF}:")
            for seq_id in dataset.skipped_long_seqs:
                logging.info(f"  - {seq_id}")
        
        if args.use_wandb:
            wandb.finish()
        print(f"Training completed. Final model saved to {final_checkpoint_path}")
    
    # Clean up distributed process
    dist.barrier()  # Final sync point
    cleanup_distributed()

def train(args):
    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    
    # Select a random port before spawning processes to ensure all processes use the same port
    port = 12355 + random.randint(0, 1000)
    
    if world_size > 1:
        logging.info(f"Found {world_size} GPUs, using DistributedDataParallel for training on port {port}")
        # Launch multiple processes, one per GPU
        mp.spawn(
            train_worker,
            args=(world_size, args, port),
            nprocs=world_size,
            join=True
        )
    else:
        logging.info("Found only 1 GPU, using single device training")
        # Just run on a single GPU (rank 0)
        train_worker(0, 1, args, port)

def main():
    # Set environment variable to help debug DDP unused parameters
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    
    parser = argparse.ArgumentParser(description="Train RhoFold with FAPE loss")
    parser.add_argument("--data_dir", type=str, default="/dev/shm", 
                        help="Directory containing RNA data")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to run training on (cuda or cpu)")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--use_wandb", action="store_true", default=not WANDB_DISABLED,
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug mode with more verbose logging")
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("rhofold_training.log")
        ]
    )
    
    # Set deterministic training for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    train(args)

if __name__ == "__main__":
    # Required for proper multiprocessing on Linux
    mp.set_start_method('spawn', force=True)
    main() 