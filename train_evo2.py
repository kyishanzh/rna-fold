import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from rhofold.rhofold import RhoFold
from rhofold.config import rhofold_config
from rhofold.utils import get_device
from rhofold.utils.alphabet import get_features
import wandb

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RNADataset(Dataset):
    """Dataset for RNA sequences with Evo2 embeddings"""
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        
        # Load sequence list
        with open(os.path.join(data_dir, f'{split}_list.txt'), 'r') as f:
            self.seq_ids = [line.strip() for line in f]
        
        logger.info(f"Loaded {len(self.seq_ids)} sequences for {split}")
    
    def __len__(self):
        return len(self.seq_ids)
    
    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]
        
        # Load sequence and MSA
        input_fas = os.path.join(self.data_dir, 'seq', f'{seq_id}.seq')
        input_a3m = os.path.join(self.data_dir, 'rMSA', f'{seq_id}.a3m')
        
        # Load Evo2 embedding
        evo2_path = os.path.join(self.data_dir, 'evo2_embeddings', f'{seq_id}.npy')
        evo2_embedding = torch.from_numpy(np.load(evo2_path))
        
        # Load ground truth structure
        pdb_path = os.path.join(self.data_dir, 'pdb', f'{seq_id}.pdb')
        
        # Get features using RhoFold's utility
        data_dict = get_features(input_fas, input_a3m)
        
        # Add Evo2 embedding to data_dict
        data_dict['evo2_fea'] = evo2_embedding
        
        return data_dict, pdb_path

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    data_dicts, pdb_paths = zip(*batch)
    
    # Find max sequence length
    max_len = max([d['tokens'].shape[1] for d in data_dicts])
    
    # Pad sequences
    tokens_batch = []
    rna_fm_tokens_batch = []
    seq_batch = []
    evo2_fea_batch = []
    
    for d in data_dicts:
        seq_len = d['tokens'].shape[1]
        
        # Pad tokens
        padded_tokens = torch.zeros((d['tokens'].shape[0], max_len), dtype=d['tokens'].dtype)
        padded_tokens[:, :seq_len] = d['tokens']
        tokens_batch.append(padded_tokens)
        
        # Pad RNA-FM tokens
        padded_rna_fm = torch.zeros(max_len, dtype=d['rna_fm_tokens'].dtype)
        padded_rna_fm[:seq_len] = d['rna_fm_tokens']
        rna_fm_tokens_batch.append(padded_rna_fm)
        
        # Add sequence
        seq_batch.append(d['seq'])
        
        # Pad Evo2 features
        padded_evo2 = torch.zeros((max_len, d['evo2_fea'].shape[1]), dtype=d['evo2_fea'].dtype)
        padded_evo2[:seq_len] = d['evo2_fea']
        evo2_fea_batch.append(padded_evo2)
    
    return {
        'tokens': torch.stack(tokens_batch),
        'rna_fm_tokens': torch.stack(rna_fm_tokens_batch),
        'seq': seq_batch,
        'evo2_fea': torch.stack(evo2_fea_batch)
    }, pdb_paths

def compute_loss(outputs, targets, native_structures=None, ss_matrices=None):
    """
    Compute the comprehensive loss function for RhoFold+ as described in the paper.
    
    Args:
        outputs: Model outputs containing predictions at different levels
        targets: Target coordinates
        native_structures: Ground truth structures (for FAPE loss)
        ss_matrices: Secondary structure matrices (L x L binary matrices where 1 indicates base pairs)
    
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary containing individual loss components
    """
    loss_dict = {}
    
    # Get the last output (final prediction)
    output = outputs[-1]
    
    # 1D level: Masked Language Modeling Loss (Lmlm)
    # Note: This would typically be computed during pre-training
    # We'll skip this for fine-tuning
    
    # 2D level: Distance Loss (Ldis)
    # Extract distance predictions for P, C4', and N atoms
    p_dist_pred = output['p']  # [batch, bins, L, L]
    c4_dist_pred = output['c4_']  # [batch, bins, L, L]
    n_dist_pred = output['n']  # [batch, bins, L, L]
    
    # Extract ground truth distance matrices from native structures
    # This is a placeholder - you'll need to implement distance binning from PDB
    if native_structures is not None:
        p_dist_true = native_structures['p_dist_bins']  # [batch, L, L]
        c4_dist_true = native_structures['c4_dist_bins']  # [batch, L, L]
        n_dist_true = native_structures['n_dist_bins']  # [batch, L, L]
        
        # Compute cross-entropy loss for each atom type
        loss_p = torch.nn.functional.cross_entropy(p_dist_pred, p_dist_true)
        loss_c4 = torch.nn.functional.cross_entropy(c4_dist_pred, c4_dist_true)
        loss_n = torch.nn.functional.cross_entropy(n_dist_pred, n_dist_true)
        
        # Combined distance loss
        loss_dist = (loss_p + loss_c4 + loss_n) / 3.0
        loss_dict['loss_dist'] = loss_dist
    else:
        loss_dist = torch.tensor(0.0, device=output['p'].device)
        loss_dict['loss_dist'] = loss_dist
    
    # 2D level: Secondary Structure Loss (Lss)
    ss_pred = output['ss']  # [batch, 1, L, L]
    if ss_matrices is not None:
        loss_ss = torch.nn.functional.binary_cross_entropy_with_logits(
            ss_pred.squeeze(1), ss_matrices.float()
        )
        loss_dict['loss_ss'] = loss_ss
    else:
        loss_ss = torch.tensor(0.0, device=output['ss'].device)
        loss_dict['loss_ss'] = loss_ss
    
    # 3D level: FAPE Loss (LFAPE)
    # This is a complex loss that requires frame alignment
    # For simplicity, we'll use a coordinate RMSD as an approximation
    pred_coords = output['cord_tns_pred'][-1]  # [batch, L*atoms_per_residue, 3]
    
    if targets is not None:
        # Simple RMSD loss as placeholder for FAPE
        loss_fape = torch.mean(torch.sqrt(torch.sum((pred_coords - targets)**2, dim=-1)))
        loss_dict['loss_fape'] = loss_fape
    else:
        loss_fape = torch.tensor(0.0, device=pred_coords.device)
        loss_dict['loss_fape'] = loss_fape
    
    # 3D level: Secondary Structure Constraint Loss (Lss3d)
    # This requires the pseudo-atoms T1-T4 for each base
    # For simplicity, we'll use a placeholder implementation
    if native_structures is not None and 'base_pairs' in native_structures:
        base_pairs = native_structures['base_pairs']  # List of (m, n) pairs that form base pairs
        
        loss_ss3d = torch.tensor(0.0, device=pred_coords.device)
        if len(base_pairs) > 0:
            # Extract the pseudo-atom coordinates for each base
            # This is a placeholder - you'll need to implement the actual extraction
            tau = 0.5  # Tolerance threshold
            
            for m, n in base_pairs:
                for i in range(4):  # T1-T4
                    for j in range(4):  # T1-T4
                        # Get predicted distance between pseudo-atoms
                        d_pred = torch.norm(
                            pred_coords[:, m*5+i, :] - pred_coords[:, n*5+j, :], dim=-1
                        )
                        
                        # Get standard distance from native structure
                        d_std = torch.norm(
                            targets[:, m*5+i, :] - targets[:, n*5+j, :], dim=-1
                        )
                        
                        # Compute loss using equation (2) from the paper
                        loss_ss3d += torch.mean(
                            torch.relu(torch.abs(d_pred - d_std) - tau)
                        )
            
            # Normalize by number of base pairs
            loss_ss3d = loss_ss3d / (len(base_pairs) * 16)  # 16 = 4*4 pseudo-atom pairs
        
        loss_dict['loss_ss3d'] = loss_ss3d
    else:
        loss_ss3d = torch.tensor(0.0, device=pred_coords.device)
        loss_dict['loss_ss3d'] = loss_ss3d
    
    # 3D level: Clash Violation Loss (Lclash)
    # This requires van der Waals radii for each atom
    # For simplicity, we'll use a placeholder implementation
    if native_structures is not None:
        # Get atom types and their van der Waals radii
        # This is a placeholder - you'll need to implement the actual clash detection
        loss_clash = torch.tensor(0.0, device=pred_coords.device)
        loss_dict['loss_clash'] = loss_clash
    else:
        loss_clash = torch.tensor(0.0, device=pred_coords.device)
        loss_dict['loss_clash'] = loss_clash
    
    # 3D level: pLDDT Loss (LpLDDT)
    # This trains the LDDT evaluator
    plddt_pred = output['plddt'][0]  # [batch, L]
    
    if native_structures is not None and 'plddt_true' in native_structures:
        plddt_true = native_structures['plddt_true']  # [batch, L]
        loss_plddt = torch.nn.functional.cross_entropy(plddt_pred, plddt_true)
        loss_dict['loss_plddt'] = loss_plddt
    else:
        loss_plddt = torch.tensor(0.0, device=pred_coords.device)
        loss_dict['loss_plddt'] = loss_plddt
    
    # Combine all losses according to equation (3) from the paper
    # You may need to adjust these weights based on your specific needs
    total_loss = (
        0.0 * loss_dist +  # Set to 0 for Evo2 head fine-tuning only
        0.0 * loss_ss +    # Set to 0 for Evo2 head fine-tuning only
        1.0 * loss_fape +  # Main 3D loss
        0.2 * loss_ss3d +  # Secondary structure in 3D
        0.1 * loss_clash + # Clash prevention
        0.1 * loss_plddt   # pLDDT prediction
    )
    
    loss_dict['total_loss'] = total_loss
    
    return total_loss, loss_dict

def freeze_model_except_evo2_head(model):
    """Freeze all parameters except the Evo2 head"""
    for name, param in model.named_parameters():
        if 'evo2_head' not in name:
            param.requires_grad = False
        else:
            logger.info(f"Keeping parameter trainable: {name}")
    
    return model

def train(args):
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(project="rhofold-evo2", name=args.run_name)
        
    # Set device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create datasets and dataloaders
    train_dataset = RNADataset(args.data_dir, split='train')
    val_dataset = RNADataset(args.data_dir, split='val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    # Initialize model
    logger.info("Initializing RhoFold model")
    model = RhoFold(rhofold_config)
    
    # Load pretrained weights
    logger.info(f"Loading pretrained weights from {args.ckpt}")
    model.load_state_dict(torch.load(args.ckpt, map_location=torch.device('cpu'))['model'])
    
    # Freeze all parameters except Evo2 head
    model = freeze_model_except_evo2_head(model)
    
    # Move model to device
    model = model.to(device)
    
    # Set up optimizer with polynomial decay scheduler
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Polynomial decay scheduler with warmup
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        else:
            return (1.0 - float(current_step - args.warmup_steps) / float(max(1, args.total_steps - args.warmup_steps))) ** 0.9
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    logger.info("Starting training")
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch_data, pdb_paths in pbar:
                # Move data to device
                tokens = batch_data['tokens'].to(device)
                rna_fm_tokens = batch_data['rna_fm_tokens'].to(device)
                seq = batch_data['seq']
                evo2_fea = batch_data['evo2_fea'].to(device)
                
                # Forward pass
                outputs = model(
                    tokens=tokens,
                    rna_fm_tokens=rna_fm_tokens,
                    seq=seq,
                    evo2_fea=evo2_fea
                )
                
                # Compute loss
                # Note: You'll need to implement target extraction from PDB files
                targets = torch.zeros_like(outputs[-1]['cord_tns_pred'][-1])  # Placeholder
                loss = compute_loss(outputs, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                train_loss += loss.item()
                global_step += 1
                
                # Update progress bar
                pbar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
                
                # Log to wandb
                if args.use_wandb:
                    wandb.log({
                        "train_loss": loss.item(),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "global_step": global_step
                    })
                
                # Save checkpoint periodically
                if global_step % args.save_steps == 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'global_step': global_step,
                        'epoch': epoch
                    }
                    torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_{global_step}.pt'))
                
                # Stop if we've reached total steps
                if global_step >= args.total_steps:
                    break
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_data, pdb_paths in tqdm(val_loader, desc="Validation"):
                # Move data to device
                tokens = batch_data['tokens'].to(device)
                rna_fm_tokens = batch_data['rna_fm_tokens'].to(device)
                seq = batch_data['seq']
                evo2_fea = batch_data['evo2_fea'].to(device)
                
                # Forward pass
                outputs = model(
                    tokens=tokens,
                    rna_fm_tokens=rna_fm_tokens,
                    seq=seq,
                    evo2_fea=evo2_fea
                )
                
                # Compute loss
                targets = torch.zeros_like(outputs[-1]['cord_tns_pred'][-1])  # Placeholder
                loss = compute_loss(outputs, targets)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {val_loss:.6f}")
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                "val_loss": val_loss,
                "epoch": epoch
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'global_step': global_step,
                'epoch': epoch,
                'val_loss': val_loss
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
        
        # Stop if we've reached total steps
        if global_step >= args.total_steps:
            break
    
    # Save final model
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'global_step': global_step,
        'epoch': epoch,
        'val_loss': val_loss
    }
    torch.save(checkpoint, os.path.join(args.output_dir, 'final_model.pt'))
    
    logger.info("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RhoFold with Evo2 embeddings")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    
    # Model arguments
    parser.add_argument("--ckpt", type=str, default="./pretrained/RhoFold_pretrained.pt", help="Path to pretrained model")
    
    # Training arguments
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1600, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Number of warmup steps")
    parser.add_argument("--total_steps", type=int, default=300000, help="Total number of training steps")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    
    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--run_name", type=str, default="rhofold-evo2", help="Name of the run for W&B")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    train(args) 