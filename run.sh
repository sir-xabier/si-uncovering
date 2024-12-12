#!/bin/bash

source .venv/bin/activate
echo "Using Python from: $(which python3)"

# Directory containing the datasets
DATASET_DIR="datasets/synthetic"
OUTPUT_DIR="results"

# List of algorithms to run
ALGORITHMS=("kmeans")

# Iterate over all .npy files in the dataset directory
for dataset in $DATASET_DIR/*.npy; do
    dataset_name=$(basename "$dataset" .npy)  # Get dataset name without extension

    # Iterate over k (number of clusters) from 1 to 25
    for k in {1..25}; do
        # Run experiments for each algorithm in parallel
        for algorithm in "${ALGORITHMS[@]}"; do
            # Run the Python script for each combination of dataset, algorithm, and k in parallel
            tsp ./.venv/bin/python3 src/experiment.py \
                -algorithm "$algorithm" \
                -dataset "$dataset_name" \
                -data_path "$dataset" \
                -output_dir "$OUTPUT_DIR" \
                -n_clusters "$k" \
                --random_state 31416 \
                --kmeans_max_iter 300 \
                --gmm_covariance_type "full" \
                --hdbscan_min_cluster_size 5 \
                --fcm_m 2.0
        done
    done
done
