#!/bin/bash

# Script to run the RhoFold training with torchrun

# Clean up any stray semaphores first
echo "Cleaning up any stray semaphores..."
./cleanup_semaphores.sh

# Default values
NUM_GPUS=8
DATA_DIR=""  # Empty by default
USE_WANDB=true  # Enable wandb by default
CHECKPOINT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --use_wandb)
      USE_WANDB=true
      shift
      ;;
    --no_wandb)
      USE_WANDB=false
      shift
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Build the command
CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS train_rhofold.py"

# Add data_dir only if it's not empty
if [ ! -z "$DATA_DIR" ]; then
  CMD="$CMD --data_dir $DATA_DIR"
fi

# Add optional arguments
if [ "$USE_WANDB" = true ]; then
  CMD="$CMD --use_wandb"
fi

if [ ! -z "$CHECKPOINT" ]; then
  CMD="$CMD --checkpoint $CHECKPOINT"
fi

# Execute the command
echo "Running command: $CMD"
eval $CMD

# Clean up any semaphores that might have been left behind
echo "Cleaning up semaphores after training..."
./cleanup_semaphores.sh

echo "Training completed"