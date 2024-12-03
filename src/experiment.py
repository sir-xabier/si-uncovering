import argparse
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import adjusted_rand_score, accuracy_score, pairwise_distances

from sklearn.cluster import KMeans, SpectralClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from fcmeans import FCM
import json

from utils import sugeno_inspired_global_uncovering_index, compute_centroids 
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def relabel_predictions(labels, predictions):
    all_classes = np.union1d(np.unique(labels), np.unique(predictions))

    cm = confusion_matrix(labels, predictions, labels=all_classes)

    row_ind, col_ind = linear_sum_assignment(-cm)  
    
    label_mapping = {col: row for row, col in zip(row_ind, col_ind)}
    relabeled_predictions = np.array([label_mapping[pred] for pred in predictions])
    
    return relabeled_predictions

    
def compute_metrics(labels, predictions, average='macro'):
    predictions = relabel_predictions(labels, predictions)
    
    precision = precision_score(labels, predictions, average=average, zero_division=0)
    recall = recall_score(labels, predictions, average=average, zero_division=0)
    f1 = f1_score(labels, predictions, average=average, zero_division=0)
    acc = accuracy_score(labels, predictions)
    
    return acc, f1, precision, recall

def get_clustering_algorithm(name, **kwargs):
    n_clusters = kwargs["n_clusters"]
    seed = kwargs["random_state"]
    
    kwargs = {key[len(name)+1:]: value for key, value in kwargs.items() if key.startswith(f"{name}_")}

    if name == "kmeans":
        return KMeans(n_clusters= n_clusters, random_state=seed, **kwargs)
    elif name == "spectral":
        return SpectralClustering(n_clusters= n_clusters, random_state=seed,**kwargs)
    elif name == "gmm":
        return GaussianMixture(n_components= n_clusters, random_state=seed, **kwargs)
    elif name == "hdbscan":
        return HDBSCAN(**kwargs)
    elif name == "fcm":
        return FCM(n_clusters= n_clusters, random_state=seed, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {name}")

def process_experiment(algorithm, dataset_name, data, labels, output_dir, **kwargs):
    n_clusters = len(np.unique(labels))
    kwargs["n_clusters"] = n_clusters

    model = get_clustering_algorithm(algorithm, **kwargs)
    if algorithm != "fcm":  
        predictions = model.fit_predict(data)
    else:
        model.fit(data)
        predictions = model.predict(data)
     
    # Compute cluster centroids for Sugeno-inspired index
    centroids, n_clusters = compute_centroids(data, predictions, n_clusters)
    sigui = sugeno_inspired_global_uncovering_index(data, centroids, predictions)

    for i, label in enumerate(predictions):
        if label == -1:  # Noise point
            # Compute distances to all centroids
            distances = pairwise_distances(data[i].reshape(1, -1), centroids).flatten()
            # Assign to the nearest cluster
            predictions[i] = np.argmin(distances)
    
    # Use LabelEncoder to map valid cluster labels to sequential integers
    label_encoder = LabelEncoder()

    predictions = label_encoder.fit_transform(predictions)
    labels = label_encoder.fit_transform(labels)

    predictions = relabel_predictions(labels, predictions)
    
    # Compute metrics
    rand_score = adjusted_rand_score(labels, predictions)
    acc, f1, precision, recall = compute_metrics(labels, predictions)
 
    # Save results
    result = {
        "dataset": dataset_name,
        "algorithm": algorithm,
        "rand_score": rand_score,
        "accuracy": acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "sigui": sigui,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}_{algorithm}.txt")
    with open(output_path, "w") as f:
        f.write(json.dumps(result, indent=4))
        
# Main function
def main():
    parser = argparse.ArgumentParser(description="Clustering Experiment Runner")
    parser.add_argument("-algorithm", type=str, required=True, help="Clustering algorithm to use")
    parser.add_argument("-dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("-data_path", type=str, required=True, help="Path to dataset file (npy format)")
    parser.add_argument("-output_dir", type=str, default="results", help="Directory to save results")
    
    parser.add_argument("--n_clusters", type=int, default=1, help="Number of clusters")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")

    # For KMeans
    parser.add_argument("--kmeans_max_iter", type=int, default=300, help="Maximum number of iterations for KMeans")

    # For GMM
    parser.add_argument("--gmm_covariance_type", type=str, default="full", choices=["full", "tied", "diag", "spherical"], help="Covariance type for GMM")

    # For HDBSCAN
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=5, help="Minimum cluster size for HDBSCAN")

    # For FCM
    parser.add_argument("--fcm_m", type=float, default=150, help="Fuzziness parameter for FCM")

    args = parser.parse_args()

    # Prepare data
    data = np.load(args.data_path)
    X, y = data[:, :-1], data[:, -1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = y.astype(np.int64)
 
    kwargs = {key: value for key, value in vars(args).items() if key not in ["algorithm", "dataset", "data_path", "output_dir"]}
    process_experiment(args.algorithm, args.dataset, X, y, args.output_dir, **kwargs)

if __name__ == "__main__":
    main()