import os
import sys
import argparse
import logging
import torch
from tqdm import tqdm
import wandb
import numpy as np
import random
from Bio.PDB import PDBParser
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Add openfold to path
openfold_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "openfold"))
if openfold_path not in sys.path:
    sys.path.insert(0, openfold_path)

from rhofold.rhofold import RhoFold
from rhofold.config import rhofold_config
from rhofold.utils import get_device
from rhofold.utils.alphabet import get_features
from train_rhofold import RNADataset, get_mask, get_pdb_seq, setup_distributed, cleanup_distributed, compute_tm_loss

def main():
    parser = argparse.ArgumentParser(description="Evaluate TM-score for RhoFold on the training set")
    parser.add_argument("--data_dir", type=str, default="/dev/shm", 
                        help="Directory containing RNA data")
    parser.add_argument("--checkpoint", type=str, default="./pretrained/RhoFold_pretrained.pt", 
                        help="Path to pretrained checkpoint")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per GPU")
    args = parser.parse_args()
    
    # Setup distributed environment
    local_rank, rank, world_size = setup_distributed()
    
    # Set this process's device based on local_rank
    device = torch.device(f"cuda:{local_rank}")
    
    # Set up logging - only on main process to avoid duplicate logs
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        logging.info(f"World size: {world_size} GPUs")
    
    # Initialize wandb only on main process
    if rank == 0:
        wandb.init(project="rhofold", config={
            "world_size": world_size,
            "checkpoint": args.checkpoint,
            "batch_size": args.batch_size
        })
    
    # Synchronize before model initialization
    dist.barrier()
    
    # Initialize model
    model = RhoFold(rhofold_config).to(device)
    
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    checkpoint = torch.load(args.checkpoint, map_location=map_location)
    model.load_state_dict(checkpoint['model'], strict=False)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    model.eval()
    
    # Create dataset
    dataset = RNADataset(args.data_dir, use_evo2=True)
    if rank == 0:
        logging.info(f"Dataset loaded with {len(dataset)} examples")
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Synchronize before evaluation
    dist.barrier()
    
    # Evaluate TM-score for each example
    total_tm_score = 0.0
    sample_count = 0
    
    with torch.no_grad():
        # Create progress bar only on rank 0
        if rank == 0:
            pbar = tqdm(total=len(dataloader))
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Process each item in the batch
                for i in range(len(batch['seq_id'])):
                    seq_id = batch['seq_id'][i]
                    tokens = batch['tokens'][i].to(device)
                    rna_fm_tokens = batch['rna_fm_tokens'][i].to(device)
                    seq = batch['seq'][i]
                    pdb_path = batch['pdb_path'][i]
                    evo2_fea = batch['evo2_fea'][i].to(device) if 'evo2_fea' in batch else None
                    
                    # Ensure dimensions are correct
                    while tokens.dim() < 3:
                        tokens = tokens.unsqueeze(0)
                    while rna_fm_tokens.dim() < 2:
                        rna_fm_tokens = rna_fm_tokens.unsqueeze(0)
                    
                    # Run model forward pass
                    outputs = model.module(tokens=tokens, rna_fm_tokens=rna_fm_tokens, seq=seq, evo2_fea=evo2_fea.to(torch.float32))
                    
                    # Take the last output from recycles
                    output = outputs[-1]
                    
                    # Compute TM-score
                    tm_score = compute_tm_loss(output, seq, pdb_path)
                    total_tm_score += tm_score
                    sample_count += 1
                    
                    # Log to wandb from rank 0 only
                    if rank == 0:
                        wandb.log({
                            "seq_id": seq_id,
                            "tm_score": tm_score,
                            "seq_length": len(seq),
                            "step": batch_idx * world_size + i
                        })
            
            except Exception as e:
                if rank == 0:
                    logging.error(f"Error processing batch {batch_idx}: {str(e)}")
            
            # Update progress bar on rank 0
            if rank == 0:
                pbar.update(1)
    
    # Close progress bar on rank 0
    if rank == 0:
        pbar.close()
    
    # Gather total scores and counts from all processes
    local_results = torch.tensor([total_tm_score, sample_count], dtype=torch.float32, device=device)
    global_results = torch.zeros_like(local_results)
    dist.all_reduce(local_results, op=dist.ReduceOp.SUM)
    
    # Calculate global average TM-score
    global_tm_score = local_results[0].item()
    global_sample_count = local_results[1].item()
    global_avg_tm_score = global_tm_score / global_sample_count if global_sample_count > 0 else 0.0
    
    # Log final results on rank 0
    if rank == 0:
        logging.info(f"Evaluation complete. Total samples: {global_sample_count}")
        logging.info(f"Average TM-score: {global_avg_tm_score:.4f}")
        wandb.log({
            "final_avg_tm_score": global_avg_tm_score,
            "total_samples": global_sample_count
        })
        wandb.finish()
    
    # Clean up distributed resources
    cleanup_distributed()

if __name__ == "__main__":
    main()
