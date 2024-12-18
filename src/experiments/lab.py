from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

#from sklearn_extra.cluster import KMedoids
from sklearn.datasets import make_circles, make_moons, make_blobs
import seaborn as sns
from utils import *

def plot_clustering_results_with_spaces(datasets, clustering_spaces, save_path):
    """
    Generate scatter plots for clustering results based on parameter spaces, and save them.
    
    Parameters:
    - datasets: dict
        Dictionary with dataset names as keys and tuples (X, y) as values, where X is the data and y is labels (optional).
    - clustering_spaces: dict
        Dictionary with algorithm names as keys and lists of parameter configurations as values.
    - save_path: str
        Path to save the generated plots.
    """
    
    # Define the palette and markers
    sns.set(style="whitegrid")  # Use a clean grid style
    palette = sns.color_palette("husl", 10)  # Color palette for clusters

    for dataset_name, (X, _) in datasets.items():
        X = StandardScaler().fit_transform(X)
        
        for algo_name, param_space in clustering_spaces.items():
            for param_idx, params in enumerate(param_space):
                plt.figure(figsize=(12, 8))
                
                # Instantiate and fit the model with given parameters
                if algo_name == "DBSCAN":
                    model = DBSCAN(eps=params.get("eps", 0.5), 
                                   min_samples=params.get("min_samples", 5), 
                                   metric=params.get("metric", "euclidean"))
                elif algo_name == "KMeans":
                    model = KMeans(n_clusters=params.get("k", 2), 
                                   init=params.get("init", "k-means++"), 
                                   n_init=params.get("n_init", 10), 
                                   max_iter=params.get("max_iter", 300))
                elif algo_name == "Agglomerative":
                    model = AgglomerativeClustering(n_clusters=params.get("k", 2), 
                                                    linkage=params.get("linkage", "ward"), 
                                                    metric=params.get("metric", "euclidean"))
                elif algo_name == "GMM":
                    model = GaussianMixture(n_components=params.get("n_components", 2), 
                                            covariance_type=params.get("covariance_type", "full"), 
                                            init_params=params.get("init_params", "kmeans"))
                else:
                    continue  # Skip unsupported algorithms
                
                # Fit and get labels
                model.fit(X)
                labels = model.labels_ if hasattr(model, "labels_") else model.predict(X)

                # Get cluster centers if available
                if hasattr(model, "cluster_centers_"):
                    centers = model.cluster_centers_
                elif algo_name == "GMM":
                    centers = model.means_
                else:
                    centers = np.array([X[labels == i].mean(axis=0) for i in range(max(labels) + 1) if (labels == i).any()])
                
                # Calculate Sugeno-inspired Global Uncovering Index
                sigui, partial_siui, partial_entropies = sugeno_inspired_global_uncovering_index(X, centers, labels, get_info=True)
                
                # Scatter plot for each cluster
                for cluster_idx in range(len(centers)):
                    cluster_data = X[labels == cluster_idx]
                    plt.scatter(
                        cluster_data[:, 0], cluster_data[:, 1],
                        color=palette[cluster_idx % len(palette)],
                        alpha=0.8
                    )
                
                # Plot centroids and annotate
                for center_idx, center in enumerate(centers):
                    plt.scatter(center[0], center[1], color="black", marker="x", s=100)  # Centroid marker
                    plt.text(center[0], center[1], 
                             f"SIUI: {partial_siui[center_idx]:.2f}\nEntropy: {partial_entropies[center_idx]:.2f}",
                             fontsize=10, color="black", ha="center", bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'))
                
                # Final plot adjustments
                plt.title(f"{algo_name} Clustering for {dataset_name}\nParams: {params}, SIGUI={sigui:.2f}", fontsize=16)
                plt.xlabel("Feature 1", fontsize=14)
                plt.ylabel("Feature 2", fontsize=14)
                plt.tight_layout()
                
                # Save plot to the specified directory
                param_str = "_".join(f"{key}_{val}" for key, val in params.items())
                figure_path = f"{save_path}/{dataset_name.replace(' ', '_')}_{algo_name}_{param_idx}_{param_str}_scatter.png"
                plt.savefig(figure_path, dpi=300)
                plt.close()

def plot_elbow_with_siui(datasets, search_spaces, save_path):
    """
    Generate and save elbow plots with Sugeno-inspired Global Uncovering Index (SIUI)
    for clustering algorithms applied to datasets using specified search spaces.
    
    Parameters:
    - datasets: dict
        Dictionary with dataset names as keys and tuples (X, y) as values, where X is the data and y is labels (optional).
    - search_spaces: dict
        Dictionary with algorithm names as keys and lists of parameter configurations as values.
    - save_path: str
        Path to save the generated plots.
    """
    # Define the palette and markers
    palette = sns.color_palette("husl", len(search_spaces))
    markers = ['o', 's', 'D', 'P', '^', 'X']
    
    sns.set(style="whitegrid")  # Use a clean grid style
    
    for dataset_name, (X, _) in datasets.items():
        plt.figure(figsize=(12, 8))
        X = StandardScaler().fit_transform(X)
        
        for idx, (algo_name, configs) in enumerate(search_spaces.items()):
            results = []  # To store (config, siui_score) tuples
            
            for config in configs:
                # Instantiate the clustering model based on the configuration
                if algo_name == "KMeans":
                    model = KMeans(
                        n_clusters=config["k"],
                        init=config["init"],
                        n_init=config["n_init"],
                        max_iter=config["max_iter"],
                        random_state=0
                    ).fit(X)
                elif algo_name == "Agglomerative":
                    model = AgglomerativeClustering(
                        n_clusters=config["k"],
                        linkage=config["linkage"],
                        metric=config["metric"]
                    ).fit(X)
                elif algo_name == "GMM":
                    model = GaussianMixture(
                        n_components=config["n_components"],
                        covariance_type=config["covariance_type"],
                        init_params=config["init_params"],
                        random_state=0
                    ).fit(X)
                elif algo_name == "DBSCAN":
                    model = DBSCAN(
                        eps=config["eps"],
                        min_samples=config["min_samples"],
                        metric=config["metric"]
                    ).fit(X)
                else:
                    raise ValueError(f"Unsupported algorithm: {algo_name}")
                
                # Retrieve labels
                labels = model.labels_ if hasattr(model, "labels_") else model.predict(X)
                
                # Get cluster centers for algorithms that provide them
                if hasattr(model, "cluster_centers_"):
                    centers = model.cluster_centers_
                elif algo_name == "GMM":
                    centers = model.means_
                else:
                    centers = np.array([
                        X[labels == i].mean(axis=0) 
                        for i in np.unique(labels) if i != -1  # Exclude noise points (-1) for DBSCAN
                    ])
                
                # Calculate the Sugeno-inspired Global Uncovering Index
                siui = sugeno_inspired_global_uncovering_index(X, centers, labels)
                results.append((config, siui))
            
            # Sort results by SIUI in descending order
            results = sorted(results, key=lambda x: x[1], reverse=True)
            sorted_configs, siui_scores = zip(*results)
            
            # Plot using Seaborn
            sns.lineplot(
                x=list(range(len(sorted_configs))), y=siui_scores,
                label=f"{algo_name}",
                marker=markers[idx % len(markers)],  # Cycle through markers
                color=palette[idx % len(palette)],
                linewidth=2
            )
        
        # Final plot adjustments
        plt.title(f"Elbow Plot for {dataset_name}", fontsize=16)
        plt.xlabel("Configuration Index (Sorted by SIUI)", fontsize=14)
        plt.ylabel("Sugeno-inspired Global Uncovering Index", fontsize=14)
        plt.legend(title="Algorithms", fontsize=12)
        plt.tight_layout()
        
        # Save plot to the specified directory
        figure_path = f"{save_path}/elbow_{dataset_name.replace(' ', '_')}.png"
        plt.savefig(figure_path, dpi=300)
        plt.close()

search_spaces = {
    "KMeans": [
        {"k": k, "n_init": n_init, "init": init, "max_iter": max_iter}
        for k, n_init, init, max_iter in product(
            range(1, 30),  # k: 2 a 15
            [10],  # n_init: repeticiones
            ["k-means++"],  # inicialización
            [100]  # max_iter
        )
    ],
    "Agglomerative": [
        {"k": k, "linkage": linkage, "metric": metric}
        for k, linkage, metric in product(
            range(2, 10),  # k: 2 a 12
            ["ward", "complete", "average", "single"],  # vínculo
            ["euclidean"]  # métrica
        )
    ],
    "GMM": [
        {"n_components": n, "init_params": init, "covariance_type": cov}
        for n, init, cov in product(
            range(2, 10),  # n_components: 2 a 15
            ["kmeans"],  # inicialización
            ["full", "tied", "diag", "spherical"]  # covarianza
        )
    ],
    "DBSCAN": [
        {"eps": eps, "min_samples": min_samples, "metric": metric}
        for eps, min_samples, metric in product(
            [round(x * 0.1, 1) for x in range(2, 16)],  # eps: 0.2 a 1.5
            [5, 10, 50, 100],  # min_samples
            ["euclidean"]  # métrica
        )
    ],
}

# Mezclar y tomar los primeros 30
for key, space in search_spaces.items():
    np.random.shuffle(space)  # Mezclar aleatoriamente
    search_spaces[key] = space[:30]  # Tomar los primeros 30

n_samples = 500
# Datasets
datasets = {
    "Circles": make_circles(n_samples=n_samples, noise=0.05, factor=0.5),
    "Moons": make_moons(n_samples=n_samples, noise=0.05),
    "Blobs": make_blobs(n_samples=n_samples, centers=2, cluster_std=0.5, random_state=0),
    "Blobs2": make_blobs(n_samples=n_samples, centers=5, cluster_std=2.0, random_state=0),
    "Blobs3": make_blobs(n_samples=n_samples, centers=10, cluster_std=0.1, random_state=0),

}

plot_elbow_with_siui(datasets, search_spaces, "./figures")
#plot_clustering_results(datasets, clustering_algorithms, "./figures")
plot_clustering_results_with_spaces(datasets, search_spaces, "./figures")

import os

os.getcwd()+ "/datasets"