import time
import warnings
import os
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from utils import *

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 500
seed = 30
noisy_circles = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
rng = np.random.RandomState(seed)
no_structure = rng.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(9 * 2 + 3, 13))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
)

plt.figure(figsize=(20, 15))  # Increase figure size for better spacing
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Add space between subplots

plot_num = 1

default_base = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 3,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
    "allow_single_cluster": True,
    "hdbscan_min_cluster_size": 15,
    "hdbscan_min_samples": 3,
    "random_state": 42,
}

datasets = [
    (
        noisy_circles,
        {
            "damping": 0.77,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.08,
        },
    ),
    (
        noisy_moons,
        {
            "damping": 0.75,
            "preference": -220,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.1,
        },
    ),
    (
        varied,
        {
            "eps": 0.18,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.01,
            "min_cluster_size": 0.2,
        },
    ),
    (
        aniso,
        {
            "eps": 0.15,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.1,
            "min_cluster_size": 0.2,
        },
    ),
    (blobs, {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2}),
    (no_structure, {}),
]
 

# Apply Seaborn styles
sns.set(style="whitegrid", context="talk")
palette = sns.color_palette("husl", n_colors=6)  # Choose a vibrant palette
output_dir = "figures"

# Iterate over datasets and generate one figure per dataset
for i_dataset, (dataset, algo_params) in enumerate(datasets):
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset
    X = StandardScaler().fit_transform(X)

    clustering_algorithms = (
        ("MiniBatch\nKMeans", cluster.MiniBatchKMeans(
            n_clusters=params["n_clusters"], random_state=params["random_state"])),
        ("Affinity\nPropagation", cluster.AffinityPropagation(
            damping=params["damping"], preference=params["preference"], random_state=params["random_state"])),
        ("MeanShift", cluster.MeanShift(
            bandwidth=cluster.estimate_bandwidth(X, quantile=params["quantile"]), bin_seeding=True)),
        ("Spectral\nClustering", cluster.SpectralClustering(
            n_clusters=params["n_clusters"], eigen_solver="arpack",
            affinity="nearest_neighbors", random_state=params["random_state"])),
        ("Ward", cluster.AgglomerativeClustering(
            n_clusters=params["n_clusters"], linkage="ward")),
        ("Agglomerative\nClustering", cluster.AgglomerativeClustering(
            linkage="average", metric="cityblock", n_clusters=params["n_clusters"])),
        ("DBSCAN", cluster.DBSCAN(eps=params["eps"])),
        ("OPTICS", cluster.OPTICS(
            min_samples=params["min_samples"], xi=params["xi"], min_cluster_size=params["min_cluster_size"])),
        ("BIRCH", cluster.Birch(n_clusters=params["n_clusters"])),
        ("Gaussian\nMixture", mixture.GaussianMixture(
            n_components=params["n_clusters"], covariance_type="full", random_state=params["random_state"]))
    )

    # Create a figure for the current dataset
    plt.figure(figsize=(20, 15))
    plt.suptitle(f"Clustering Results for Dataset {i_dataset + 1}", fontsize=20, fontweight="bold")
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    for i_algo, (name, algorithm) in enumerate(clustering_algorithms):
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            algorithm.fit(X)
        t1 = time.time()

        if hasattr(algorithm, "labels_"):
            labels = algorithm.labels_.astype(int)
        else:
            labels = algorithm.predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 0:
            centroids = compute_centroids(X, labels, n_clusters)
            global_siui, partial_siui, entropies = sugeno_inspired_global_uncovering_index(
                X, centroids, labels, get_info=True)
        else:
            centroids = []
            global_siui = 0
            partial_siui = []
            entropies = []

        plt.subplot(3, 4, i_algo + 1)
        plt.title(f"{name}\nSIUI: {global_siui:.3f}", size=12, fontweight="bold")

        # Use Seaborn palette colors
        colors = np.array(
            list(islice(cycle(sns.color_palette("husl", int(max(labels) + 1))), 
                        int(max(labels) + 1)))
        )
        colors = np.append(colors, [(0, 0, 0)])  # Add black for outliers

        # Scatter plot for data points
        plt.scatter(X[:, 0], X[:, 1], s=14, c=colors[labels], alpha=0.7)

        # Highlight centroids
        if n_clusters > 0:
            for i, centroid in enumerate(centroids):
                if not np.isnan(centroid).any():
                    plt.text(centroid[0], centroid[1], 
                             f"SIUI: {partial_siui[i]:.2f}\nEntropy: {entropies[i]:.2f}",
                             fontsize=10, color="black", ha="center", bbox=dict(facecolor='white', alpha=0.6, edgecolor='grey'))

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(()), plt.yticks(())
        plt.text(0.99, 0.01, ("%.2fs" % (t1 - t0)).lstrip("0"),
                 transform=plt.gca().transAxes, size=12, ha="right", fontweight="bold")  # Larger execution time text

    # Save the figure for the current dataset
    output_path = os.path.join(output_dir, f"dataset_{i_dataset + 1}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
# Show the plot
plt.tight_layout()

plt.savefig("./figures/overview.png")