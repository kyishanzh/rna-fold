# Import required libraries and modules
from Dataset import *
from Network import *
from Functions import *
from tqdm import tqdm
from sklearn.model_selection import KFold
from ranger import Ranger  # Specialized optimizer
import argparse
from sklearn.metrics import mean_squared_error
from accelerate import Accelerator  # For distributed training/inference
import time
import json

# Record start time for performance tracking
start_time = time.time()

# Set up command line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default="configs/pairwise.yaml")
args = parser.parse_args()

# Load configuration from YAML file
config = load_config_from_yaml(args.config_path)

# Initialize accelerator for mixed precision training
accelerator = Accelerator(mixed_precision='fp16')

# Set up GPU and directory configuration
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
os.system('mkdir predictions')  # Create directory for predictions
os.system('mkdir plots')        # Create directory for plots
os.system('mkdir subs')         # Create directory for submissions

# Load and preprocess test data
data = pl.read_csv(f"{config.input_dir}/test_sequences.csv")
# Calculate sequence lengths and sort by length
lengths = data['sequence'].apply(len).to_list()
data = data.with_columns(pl.Series('sequence_length', lengths))
data = data.sort('sequence_length', descending=True)
print(data['sequence_length'])

# Extract test sequences and IDs
test_ids = data['sequence_id'].to_list()
sequences = data['sequence'].to_list()
# Generate attention mask based on sequence distances
attention_mask = torch.tensor(get_distance_mask(max(lengths))).float()

# Create data dictionary for dataset
data_dict = {
    'sequences': sequences,
    'sequence_ids': test_ids,
    "attention_mask": attention_mask
}
assert len(test_ids) == len(data)  # Verify data integrity

# Create test dataset and dataloader
val_dataset = TestRNAdataset(np.arange(len(data)), data_dict, k=config.k)
val_loader = DataLoader(
    val_dataset,
    batch_size=config.test_batch_size,
    shuffle=False,
    collate_fn=Custom_Collate_Obj_test(),
    num_workers=min(config.batch_size, 32)
)

print(accelerator.distributed_type)

# Load models
models = []
for i in range(1):
    model = RibonanzaNet(config)
    model.eval()  # Set model to evaluation mode
    model.load_state_dict(torch.load(f"models/model{i}.pt", map_location='cpu'))
    models.append(model)

# Prepare model and dataloader for distributed inference
model, val_loader = accelerator.prepare(model, val_loader)

# Initialize progress bar and prediction list
tbar = tqdm(val_loader)
val_loss = 0
preds = []
model.eval()

# Main inference loop
for idx, batch in enumerate(tbar):
    src = batch['sequence']
    masks = batch['masks']
    bs = len(src)
    
    # Create flipped version of input for test-time augmentation
    src_flipped = src.clone()
    length = batch['length']
    
    # Flip sequences along length dimension
    for batch_idx in range(len(src)):
        src_flipped[batch_idx, :length[batch_idx]] = src_flipped[batch_idx, :length[batch_idx]].flip(0)
    
    # Generate predictions
    with torch.no_grad():
        with accelerator.autocast():
            output = []
            for model in models:
                # Get prediction for normal sequence
                output.append(model(src, masks))
                
                # If using flip augmentation, get prediction for flipped sequence
                if config.use_flip_aug:
                    flipped_output = model(src_flipped, masks)
                    # Flip predictions back to original orientation
                    for batch_idx in range(len(flipped_output)):
                        flipped_output[batch_idx, :length[batch_idx]] = flipped_output[batch_idx, :length[batch_idx]].flip(0)
                    output.append(flipped_output)
                    
            # Average predictions from all models and augmentations
            output = torch.stack(output).mean(0)
    
    # Gather predictions from all processes
    output = accelerator.pad_across_processes(output, 1)
    all_output = accelerator.gather(output).cpu().numpy()
    preds.append(all_output)

# Save predictions if this is the main process
if accelerator.is_local_main_process:
    import pickle
    
    # Create dictionary mapping sequence IDs to predictions
    preds_dict = {}
    for i, id in tqdm(enumerate(test_ids)):
        batch_number = i // (config.test_batch_size * accelerator.num_processes)
        in_batch_index = i % (config.test_batch_size * accelerator.num_processes)
        preds_dict[id] = preds[batch_number][in_batch_index]
    
    # Save predictions to file
    with open("preds.p", 'wb+') as f:
        pickle.dump(preds_dict, f)
    
    # Record and save execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    with open("inference_stats.json", 'w') as file:
        json.dump({'Total_execution_time': elapsed_time}, file, indent=4)