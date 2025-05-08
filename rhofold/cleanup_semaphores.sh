#!/bin/bash

# Find and remove semaphores created by python multiprocessing
# This will find semaphores owned by your user and remove them

echo "Cleaning up semaphores for user $USER..."

# Find semaphores created by current user
SEMAPHORES=$(ipcs -s | grep $USER | awk '{print $2}')

if [ -z "$SEMAPHORES" ]; then
    echo "No semaphores found for user $USER"
    exit 0
fi

# Remove each semaphore
for SEM in $SEMAPHORES; do
    echo "Removing semaphore $SEM"
    ipcrm -s $SEM
done

echo "Semaphore cleanup complete"
