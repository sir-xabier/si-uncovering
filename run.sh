#!/bin/bash

source .venv/bin/activate
echo "Using Python from: $(which python3)"

# Directory containing the datasets
DATASET_DIR="datasets/blobs"
OUTPUT_DIR="results"

# List of algorithms to run
ALGORITHMS=("kmeans")

# Define specific k values, including 2, 4, 8, 16, and 32
K_VALUES=(2 4 8 16 32 3 10 20 25 50)

# Iterate over all .npy files in the dataset directory
for dataset in "$DATASET_DIR"/*.npy; do
    dataset_name=$(basename "$dataset" .npy)  

    # Iterate over selected k values
    for k in "${K_VALUES[@]}"; do
        # Run experiments for each algorithm in parallel
        for algorithm in "${ALGORITHMS[@]}"; do
            # Run the Python script for each combination
            tsp ./venv/bin/python3 src/experiments/experiment.py \
                -algorithm "$algorithm" \
                -dataset "$dataset_name" \
                -data_path "$dataset" \
                -output_dir "$OUTPUT_DIR" \
                -n_clusters "$k" \
                -alpha $alpha \
                --random_state 31416
        done
    done
done