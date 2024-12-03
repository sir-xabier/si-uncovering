import argparse
import os
import numpy as np
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
import hdbscan
from fcmeans import FCM
from joblib import Parallel, delayed
import json
from utils import sugeno_inspired_global_uncovering_index, compute_centroids

# Define clustering algorithms
def get_clustering_algorithm(name, n_clusters, seed):
    if name == "kmeans":
        return KMeans(n_clusters=n_clusters, random_state=seed)
    elif name == "spectral":
        return SpectralClustering(n_clusters=n_clusters, random_state=seed)
    elif name == "gmm":
        return GaussianMixture(n_components=n_clusters, random_state=seed)
    elif name == "hdbscan":
        return hdbscan.HDBSCAN(min_cluster_size=5)
    elif name == "fcm":
        return FCM(n_clusters=n_clusters)
    else:
        raise ValueError(f"Unknown algorithm: {name}")

# Function to process a single experiment
def process_experiment(algorithm, dataset_name, data, labels, output_dir):
    n_clusters = len(np.unique(labels))
    model = get_clustering_algorithm(algorithm, n_clusters)
    predictions = model.fit_predict(data)

    # Compute cluster centroids for Sugeno-inspired index
    centroids = compute_centroids(data, predictions, n_clusters)

    # Compute metrics
    rand_score = adjusted_rand_score(labels, predictions)
    acc = accuracy_score(labels, predictions)
    sigui = sugeno_inspired_global_uncovering_index(data, centroids, predictions)

    # Save results
    result = {
        "dataset": dataset_name,
        "algorithm": algorithm,
        "rand_score": rand_score,
        "accuracy": acc,
        "sigui": sigui,
    }
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}_{algorithm}.txt")
    with open(output_path, "w") as f:
        f.write(json.dumps(result, indent=4))

# Main function
def main():
    parser = argparse.ArgumentParser(description="Clustering Experiment Runner")
    parser.add_argument("--algorithm", type=str, required=True, help="Clustering algorithm to use")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset file (npy format)")
    parser.add_argument("--labels_path", type=str, required=True, help="Path to labels file (npy format)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    args = parser.parse_args()
    
    # Load data
    data = np.load(args.data_path)
    labels = np.load(args.labels_path)
    
    # Run experiment
    process_experiment(args.algorithm, args.dataset, data, labels, args.output_dir)

if __name__ == "__main__":
    main()