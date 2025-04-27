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
from collections import defaultdict
import constants
import torch  # Add PyTorch import
import asyncio
import aiohttp
import time

URL = os.getenv("URL", "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward")
layer_name = 'blocks.28.mlp.l3'

# Rate limiter semaphore - limits to 1 request every 3 seconds
class RateLimiter:
    def __init__(self, rate_limit=3.0):
        self.rate_limit = rate_limit  # seconds between requests
        self.last_request_time = 0
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        async with self.lock:
            now = time.time()
            time_since_last = now - self.last_request_time
            
            if time_since_last < self.rate_limit:
                wait_time = self.rate_limit - time_since_last
                await asyncio.sleep(wait_time)
                
            self.last_request_time = time.time()

rate_limiter = RateLimiter(rate_limit=3.0)

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

async def get_embedding_async(sequence):
    # Respect rate limit
    await rate_limiter.acquire()
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url=URL,
            headers={"Authorization": f"Bearer {constants.NVIDIA_API}"},
            json={"sequence": sequence, "output_layers": [layer_name]},
        ) as r:
            content_type = r.headers.get("Content-Type", "")
            if "application/json" in content_type:
                response_json = await r.json()
                b64_npz = response_json["data"]
            elif "application/zip" in content_type:
                zip_data = io.BytesIO(await r.read())
                with zipfile.ZipFile(zip_data, "r") as zf:
                    response_filename = zf.namelist()[0]
                    with zf.open(response_filename) as f:
                        response_text = f.read().decode("utf-8")
                response_json = json.loads(response_text)
                b64_npz = response_json["data"]
            else:
                content = await r.read()
                print("Response content:", content)
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

async def get_embeddings_for_sequences_async(sequences, target_ids, max_concurrent=5):
    """Get Evo2 embeddings for RNA sequences using NVIDIA's API asynchronously
    
    Args:
        sequences (list): List of RNA sequences
        target_ids (list): List of corresponding target IDs
        max_concurrent (int): Maximum number of concurrent requests
        
    Returns:
        dict: Mapping of target_id to embedding
    """
    embeddings_dict = {}
    
    # Use a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_sequence(target_id, sequence):
        async with semaphore:
            try:
                embedding = await get_embedding_async(sequence)
                return target_id, embedding
            except Exception as e:
                print(f"Error processing sequence {target_id}: {str(e)}")
                return target_id, None
    
    # Create tasks for all sequences
    tasks = [process_sequence(target_id, sequence) 
             for target_id, sequence in zip(target_ids, sequences)]
    
    # Initialize two progress bars
    total_tasks = len(tasks)
    overall_pbar = tqdm(total=total_tasks, desc="Processing", position=0)
    success_pbar = tqdm(total=total_tasks, desc="Successful", position=1)
    
    # Process in batches and update progress bars
    for future in asyncio.as_completed(tasks):
        target_id, embedding = await future
        overall_pbar.update(1)
        
        if embedding is not None:
            embeddings_dict[target_id] = embedding
            success_pbar.update(1)
    
    # Close progress bars
    overall_pbar.close()
    success_pbar.close()
    
    return embeddings_dict

def seq_id_to_embedding(seq_id):
    """
    Get Evo2 embeddings for a given sequence ID from a .pt file
    """
    embeddings_dir = "data/RNA3D_DATA/evo2_embeddings"
    file_path = os.path.join(embeddings_dir, f"{seq_id}.pt")
    
    if os.path.exists(file_path):
        return torch.load(file_path)
    else:
        raise FileNotFoundError(f"Embedding file for {seq_id} not found at {file_path}")
    

async def process_rna3d_data_async():
    """
    Process RNA3D data files, get embeddings, and save as individual .pt files.
    Uses asynchronous processing with rate limiting.
    """
    # Directory containing sequence files
    seq_dir = "data/RNA3D_DATA/seq"
    output_dir = "data/RNA3D_DATA/evo2_embeddings"
    
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
                # Skip if this file already exists
                output_file = os.path.join(output_dir, f"{target_id}.pt")
                if not os.path.exists(output_file):
                    all_target_ids.append(target_id)
                    all_sequences.append(sequence)
    
    total_sequences = len(all_sequences)
    print(f"Found {total_sequences} sequences to process")
    
    if total_sequences == 0:
        print("No new sequences to process")
        return
    
    # Process sequences with async function
    max_concurrent = 10  # Adjust based on API limits
    
    # Use a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_and_save_sequence(target_id, sequence):
        output_file = os.path.join(output_dir, f"{target_id}.pt")
        
        async with semaphore:
            try:
                # Get embedding for sequence
                embedding = await get_embedding_async(sequence)
                
                # Save immediately to file to avoid storing in RAM
                tensor = torch.tensor(embedding)
                torch.save(tensor, output_file)
                
                # Free memory
                del tensor
                del embedding
                gc.collect()
                
                return target_id, True  # Success
            except Exception as e:
                print(f"Error processing sequence {target_id}: {str(e)}")
                return target_id, False  # Failure
    
    # Create tasks for all sequences
    tasks = [process_and_save_sequence(target_id, sequence) 
             for target_id, sequence in zip(all_target_ids, all_sequences)]
    
    # Initialize two progress bars
    total_tasks = len(tasks)
    overall_pbar = tqdm(total=total_tasks, desc="Processing", position=0)
    success_pbar = tqdm(total=total_tasks, desc="Successful", position=1)
    
    # Process tasks and update progress bars
    successful_count = 0
    for future in asyncio.as_completed(tasks):
        target_id, success = await future
        overall_pbar.update(1)
        
        if success:
            successful_count += 1
            success_pbar.update(1)
    
    # Close progress bars
    overall_pbar.close()
    success_pbar.close()
    
    print(f"Completed processing: {successful_count}/{total_tasks} successful")

def process_rna3d_data():
    """
    Process RNA3D data files, get embeddings, and save as individual .pt files.
    
    Reads all sequence files in data/RNA3D_DATA/seq directory,
    extracts target IDs and sequences, gets embeddings for them,
    and saves each embedding as a separate .pt file in data/RNA3D_DATA/evo2_embeddings.
    """
    # Directory containing sequence files
    seq_dir = "data/RNA3D_DATA/seq"
    output_dir = "data/RNA3D_DATA/evo2_embeddings"
    
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
    
    total_sequences = len(all_sequences)
    print(f"Found {total_sequences} sequences to process")
    
    # Process all sequences at once
    for idx, (target_id, sequence) in enumerate(tqdm(zip(all_target_ids, all_sequences), total=total_sequences)):
        output_file = os.path.join(output_dir, f"{target_id}.pt")
        
        # Skip if this file already exists
        if os.path.exists(output_file):
            print(f"File {output_file} already exists, skipping...")
            continue
            
        try:
            # Get embedding for this sequence
            embedding = get_embedding(sequence)
            
            # Convert numpy array to PyTorch tensor
            tensor = torch.tensor(embedding)
            
            # Save tensor to file
            torch.save(tensor, output_file)
                
        except Exception as e:
            print(f"Error processing sequence {target_id}: {str(e)}")
            continue

if __name__ == "__main__":
    # Run the async version
    asyncio.run(process_rna3d_data_async())
