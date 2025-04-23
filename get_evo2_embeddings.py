import requests
import numpy as np
from Bio import SeqIO
import io
import base64
import zipfile
import json
import os
import gc  # For garbage collection
from tqdm.auto import tqdm
import pickle
import umap
from collections import defaultdict
import constants

URL = os.getenv("URL", "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward")
layer_name = 'blocks.28.mlp.l3'

def get_embedding(sequence):
    r = requests.post(
        url=URL,
        headers={"Authorization": f"Bearer {constants.NVIDIA_API}"},
        json={"sequence": sequence, "output_layers": [layer_name]},
    )
    content_type = r.headers.get("Content-Type", "")
    if "application/json" in content_type:
        response_json = r.json()
        b64_npz = response_json["data"]
    elif "application/zip" in content_type:
        zip_data = io.BytesIO(r.content)
        with zipfile.ZipFile(zip_data, "r") as zf:
            response_filename = zf.namelist()[0]
            with zf.open(response_filename) as f:
                response_text = f.read().decode("utf-8")
        response_json = json.loads(response_text)
        b64_npz = response_json["data"]
    else:
        print("Response content:", r.content)
        raise ValueError(f"Unexpected content type: {content_type}")
    
    npz_bytes = io.BytesIO(base64.b64decode(b64_npz))
    npz_file = np.load(npz_bytes)
    key = npz_file.files[0]
    embedding = npz_file[key]  # shape: (1, seq_len, 8192)
    return embedding.squeeze(0)  # shape: (seq_len, 8192)

def get_embeddings_for_sequences(sequences, target_ids, batch_size=32):
    """Get Evo2 embeddings for RNA sequences using NVIDIA's API
    
    Args:
        sequences (list): List of RNA sequences
        target_ids (list): List of corresponding target IDs
        batch_size (int): Number of sequences to process at once
        
    Returns:
        dict: Mapping of target_id to embedding
    """
    embeddings_dict = {}
    
    # process sequences in batches
    for target_id, sequence in tqdm(list(zip(target_ids, sequences))):
        try:
            # get embedding for sequence
            embedding = get_embedding(sequence)
            embeddings_dict[target_id] = embedding
        except Exception as e:
            print(f"Error processing sequence {target_id}: {str(e)}")
            continue
            
    return embeddings_dict

def process_rna3d_data(batch_size=1000):
    """
    Process RNA3D data files, get embeddings, and save as pickle in batches.
    
    Reads all sequence files in data/RNA3D_DATA/seq directory,
    extracts target IDs and sequences, gets embeddings for them,
    and saves the results as pickle files in batches to prevent memory issues.
    Only combines all batches at the very end to minimize memory usage.
    
    Args:
        batch_size (int): Number of sequences to process in each batch
    """
    # Directory containing sequence files
    seq_dir = "data/RNA3D_DATA/seq"
    output_dir = "data/RNA3D_DATA/embeddings_batches"
    final_output_file = "data/RNA3D_DATA/nvidia_evo2_embeddings.pkl"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Lists to store sequences and target IDs
    all_sequences = []
    all_target_ids = []
    
    # Read all files in the directory
    print("Reading sequence files...")
    for filename in tqdm(os.listdir(seq_dir)):
        file_path = os.path.join(seq_dir, filename)
        
        with open(file_path, "r", encoding="latin-1") as f:
            target_id = filename.split(".")[0]
            lines = f.readlines()
            sequence = lines[1].strip() if len(lines) > 1 else ""

            if sequence:
                all_target_ids.append(target_id)
                all_sequences.append(sequence)
    
    # Process in batches of batch_size
    total_sequences = len(all_sequences)
    print(f"Found {total_sequences} sequences to process")
    
    # Track which batch files we create
    batch_files = []
    
    # Process in batches
    for batch_idx in range(0, total_sequences, batch_size):
        batch_end = min(batch_idx + batch_size, total_sequences)
        batch_sequences = all_sequences[batch_idx:batch_end]
        batch_target_ids = all_target_ids[batch_idx:batch_end]
        
        batch_name = f"batch_{batch_idx//batch_size + 1}_of_{(total_sequences + batch_size - 1)//batch_size}"
        batch_output_file = os.path.join(output_dir, f"embeddings_{batch_name}.pkl")
        batch_files.append(batch_output_file)
        
        print(f"\nProcessing {batch_name}: sequences {batch_idx+1} to {batch_end} of {total_sequences}")
        
        # Skip if this batch already exists
        if os.path.exists(batch_output_file):
            print(f"Batch file {batch_output_file} already exists, skipping...")
            continue
        
        # Get embeddings for this batch
        batch_embeddings = get_embeddings_for_sequences(batch_sequences, batch_target_ids)
        
        # Save this batch to its own file
        with open(batch_output_file, 'wb') as f:
            pickle.dump(batch_embeddings, f)
        
        print(f"Saved {len(batch_embeddings)} embeddings to {batch_output_file}")
        
        # Clear batch data and force garbage collection to free memory
        del batch_sequences
        del batch_target_ids
        del batch_embeddings
        gc.collect()
    
    # Clear large lists from memory before combining batches
    del all_sequences
    del all_target_ids
    gc.collect()
    
    # Now combine all batch files into the final output file
    print("\nCombining all batch files into final output...")
    all_embeddings = {}
    
    for batch_file in tqdm(batch_files):
        if os.path.exists(batch_file):
            try:
                with open(batch_file, 'rb') as f:
                    batch_embeddings = pickle.load(f)
                    all_embeddings.update(batch_embeddings)
                    # Immediately delete to free memory
                    del batch_embeddings
                    gc.collect()
            except Exception as e:
                print(f"Error loading {batch_file}: {str(e)}")
    
    # Save complete embeddings dictionary to final output file
    with open(final_output_file, 'wb') as f:
        pickle.dump(all_embeddings, f)
    
    print(f"\nCompleted processing. Saved embeddings for {len(all_embeddings)} sequences to {final_output_file}")
    
    # Final garbage collection
    del all_embeddings
    gc.collect()

if __name__ == "__main__":
    process_rna3d_data(batch_size=1000)
