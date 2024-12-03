#!/bin/bash

source .venv/bin/activate
echo "Using Python from: $(which python3)"

# Directory containing the datasets
DATASET_DIR="datasets/real"
OUTPUT_DIR="results"

# List of algorithms to run
ALGORITHMS=("kmeans" "gmm" "fcm" "hdbscan")

# Iterate over all .npy files in the dataset directory
for dataset in $DATASET_DIR/*.npy; do
    dataset_name=$(basename "$dataset" .npy)  # Get dataset name without extension

    # Run experiments for each algorithm in parallel
    for algorithm in "${ALGORITHMS[@]}"; do
        # Run the Python script for each combination of dataset and algorithm in parallel
        # The python script should accept the dataset, algorithm, and other parameters as arguments
        # Using GNU Parallel for parallel execution
        tsp ./.venv/bin/python3 src/experiment.py \
            -algorithm "$algorithm" \
            -dataset "$dataset_name" \
            -data_path "$dataset" \
            -output_dir "$OUTPUT_DIR" \
            --n_clusters 3 \
            --random_state 31416 \
            --kmeans_max_iter 300 \
            --gmm_covariance_type "full" \
            --hdbscan_min_cluster_size 5 \
            --fcm_m 2.0
    done
done
