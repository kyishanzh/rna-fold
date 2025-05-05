import os
import sys
import argparse
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
from rhofold.utils.alphabet import get_features
from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.utils.loss import compute_fape

# Training configuration
CHECKPOINT_DIR = "checkpoints"
USE_EVO2 = True
BATCH_SIZE = 1
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
CHECKPOINT_EVERY = 2
WARMUP_STEPS = 100

# Set to your specific project/entity here or use environment variables
os.environ["WANDB_API_KEY"] = "71278e965a6e50657c6b254d59ba8fac486c97dd"  # Uncomment and set your API key if needed
os.environ["WANDB_PROJECT"] = "rhofold"  # Uncomment to override default project

# Get wandb configuration from environment or use defaults
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "rhofold")
WANDB_DISABLED = False

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
        
        logging.info(f"Found {len(self.seq_ids)} RNA sequences with MSA data for training")
        
    def __len__(self):
        return len(self.seq_ids)
    
    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]
        
        # Load sequence and MSA features
        input_fas = os.path.join(self.data_dir, f"RNA3D_DATA/seq/{seq_id}.seq")
        input_a3m = os.path.join(self.data_dir, f"RNA3D_DATA/rMSA/{seq_id}.a3m")
        data_dict = get_features(input_fas, input_a3m)
        
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

def train(args):
    # Initialize wandb
    if args.use_wandb:
        # Try to login with API key from environment if available
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
            }
        )
        logging.info(f"Initialized wandb with project={WANDB_PROJECT}")
    
    # Create checkpoint dir if it doesn't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Set device
    device = get_device(args.device)
    
    # Initialize model
    model = RhoFold(rhofold_config).to(device)
    print(f"Number of params: {sum(p.numel() for p in model.parameters())}")
    
    # If starting from a checkpoint, load it
    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Loaded checkpoint from {args.checkpoint}, starting from epoch {start_epoch}")
    
    # Create dataset and dataloader
    dataset = RNADataset(args.data_dir, use_evo2=USE_EVO2)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            # Move data to device
            seq_id = batch['seq_id'][0]
            tokens = batch['tokens'][0].to(device)
            rna_fm_tokens = batch['rna_fm_tokens'][0].to(device)
            assert tokens.shape[-1] == rna_fm_tokens.shape[-1]
            seq = batch['seq'][0]
            pdb_path = batch['pdb_path'][0]
            
            # Get chain ID from seq_id if available
            chain_id = None
            
            # Handle evo2 features
            evo2_fea = None
            if USE_EVO2 and batch['evo2_fea'] is not None:
                evo2_fea = batch['evo2_fea'][0].to(device).to(torch.float32)
            
            # Forward pass
            outputs = model(tokens=tokens, rna_fm_tokens=rna_fm_tokens, seq=seq, evo2_fea=evo2_fea)
            
            # Take the last output from recycles
            output = outputs[-1]
            
            # Compute FAPE loss
            loss = compute_fape_loss(output, pdb_path, chain_id=chain_id)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            
            # Log batch metrics
            if args.use_wandb:
                wandb.log({
                    "batch_loss": loss.item(),
                    "pLDDT": output["plddt"][1].item(),
                    "seq_id": seq_id,
                    "seq_length": len(seq)
                })
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        
        # Log epoch metrics
        print(f"Epoch {epoch+1} - Loss: {avg_epoch_loss:.4f}")
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_epoch_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        # Update learning rate scheduler
        scheduler.step(avg_epoch_loss)
        
        # Save checkpoint
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"rhofold_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_checkpoint_path = os.path.join(CHECKPOINT_DIR, "rhofold_final.pt")
    torch.save({
        'epoch': NUM_EPOCHS,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': avg_epoch_loss,
    }, final_checkpoint_path)
    
    if args.use_wandb:
        wandb.finish()
    print(f"Training completed. Final model saved to {final_checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description="Train RhoFold with FAPE loss")
    parser.add_argument("--data_dir", type=str, default="/home/user/rna-fold/data", 
                        help="Directory containing RNA data")
    parser.add_argument("--device", type=str, default="cpu", 
                        help="Device to run training on (cuda or cpu)")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--use_wandb", action="store_true", default=WANDB_DISABLED,
                        help="Whether to use Weights & Biases for logging")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    train(args)

if __name__ == "__main__":
    main() 