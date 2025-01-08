import os
import json
import warnings
from typing import Generator, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn import datasets

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import itertools

# Suppress warnings
warnings.filterwarnings("ignore")

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for handling NumPy arrays.
    """
    def default(self, obj: Union[np.ndarray, object]) -> Union[list, object]:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def generate_scenario(
    n_blobs: int = 10,
    k_low: int = 1,
    k_high: int = 1,
    p_low: int = 2,
    p_high: int = 2,
    s_low: float = 1.0,
    s_high: float = 1.0,
    n_samples: int = 500,
    initial_seed: int = 0,
    get_class: bool = True
) -> Tuple[List[str], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Generate a synthetic dataset with varying scenarios.

    :param n_blobs: Number of blobs to generate.
    :param k_low: Minimum number of clusters.
    :param k_high: Maximum number of clusters.
    :param p_low: Minimum number of dimensions.
    :param p_high: Maximum number of dimensions.
    :param s_low: Minimum scaling factor.
    :param s_high: Maximum scaling factor.
    :param n_samples: Number of samples per blob.
    :param initial_seed: Random seed for reproducibility.
    :param get_class: Whether to include class labels.
    :return: A tuple containing blob names and generated data.
    """
    data = []
    class_counts = []   
    names = []

    rng = np.random.default_rng(seed=initial_seed)

    if s_high == 0.5:
        scaling_factors = [0.3,0.32,0.34,0.36,0.38,0.4,0.425,0.45,0.475,0.5]
    elif s_low == 1.0 and s_high == 1.0:
        scaling_factors = [1.0] * n_blobs
    else:
        scaling_factors = np.linspace(s_low, s_high, n_blobs)

    for i, scale in enumerate(scaling_factors):
        n_clusters = rng.integers(k_low, k_high + 1)
        n_features = rng.integers(p_low, p_high + 1)
        
        centers = np.zeros(shape = (n_clusters, n_features))
        for k in range(n_clusters):
            center=rng.integers(1,n_clusters,endpoint=True,size=(1,n_features))
            if k== 0:
                centers[k,:] = center
            else:
                igual=True
                while igual:
                    if np.any(np.all(centers==np.repeat(center,n_clusters,axis=0),axis=1)):
                        center=rng.integers(1,n_clusters,endpoint=True,size=(1,n_features))
                    else:
                        centers[k,:]=center
                        igual=False
        
        centers=centers-0.5
        
        min_dist = np.amin(distance.cdist(centers,centers) + np.identity(n_clusters) * n_clusters * np.sqrt(n_features))
        
        # Create blob
        blobs = datasets.make_blobs(
            n_samples=n_samples,
            centers=centers,
            cluster_std=min_dist * scale,
            random_state=initial_seed + i
        )
        data.append(blobs if get_class else blobs[0])
        names.append(
            f"blobs-P{n_features}-K{n_clusters}-N{n_samples}-dt{scale:.2f}-S{i}"
        )
        class_counts.append(n_clusters)

    return (names, data) if get_class else (data, names, class_counts)


def generate_blobs(
    n_blobs: int = 10,
    k_low: int = 1,
    k_high: int = 10,
    n_features: int = 2,
    n_samples: int = 500,
    initial_seed: int = 1,
    get_class: bool = False,
    interval: int = 1
) -> Tuple[List[str], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Generate synthetic blobs for clustering.

    :param n_blobs: Number of blobs to generate.
    :param k_low: Minimum number of clusters.
    :param k_high: Maximum number of clusters.
    :param n_features: Number of dimensions/features.
    :param n_samples: Number of samples per blob.
    :param initial_seed: Random seed for reproducibility.
    :param get_class: Whether to include class labels.
    :param interval: Step interval for the number of clusters.
    :return: A tuple containing blob names and generated data.
    """
    data = []
    class_counts = []
    names = []

    for n_clusters in range(k_low, k_high + 1, interval):
        for blob_id in range(n_blobs):
            blobs = datasets.make_blobs(
                n_samples=n_samples,
                n_features=n_features,
                centers=n_clusters,
                random_state=initial_seed + blob_id
            )
            data.append(blobs if get_class else blobs[0])
            names.append(
                f"blobs-P{n_features}-K{n_clusters}-N{n_samples}-S{blob_id + 1}"
            )
            class_counts.append(n_clusters)

    return (names, data) if get_class else (data, names, class_counts)



def save_dataset(path, filename, X, y):
    """
    Save dataset as a NumPy .npy file.

    :param path: Directory path to save the file.
    :param filename: Name of the file (without extension).
    :param X: Feature matrix.
    :param y: Target vector.
    """
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{filename}.npy")
    np.save(file_path, np.concatenate((X, y), axis=1))

def generate_synthetic_datasets(path, n_samples, random_state):
    """
    Generate and save synthetic datasets.

    :param path: Directory path to save the datasets.
    :param n_samples: Number of samples per dataset.
    :param random_state: Seed for reproducibility.
    """

    # Circles dataset
    X, y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    save_dataset(path, "circles", X, y.reshape(-1, 1))
    
    # Moons dataset
    X, y = datasets.make_moons(n_samples=n_samples, noise=0.05)
    save_dataset(path, "moons", X, y.reshape(-1, 1))
    
    # Random no-structure dataset
    X = np.random.rand(n_samples, 2)
    save_dataset(path, "no_structure", X, np.zeros((n_samples, 1)))
    
    # Anisotropic blobs
    transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    X = np.dot(X, transformation)
    save_dataset(path, "aniso", X, y.reshape(-1, 1))
    
    # Varied blob sizes
    X, y = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )
    save_dataset(path, "varied", X, y.reshape(-1, 1))

def generate_real_datasets(path):
    """
    Load and save real-world datasets.

    :param path: Directory path to save the datasets.
    """ 
    
    datasets_to_load = {
        "iris": datasets.load_iris,
        "digits": datasets.load_digits,
        "wine": datasets.load_wine,
        "breast_cancer": datasets.load_breast_cancer
    }
    
    for name, loader in datasets_to_load.items():
        X, y = loader(return_X_y=True)
        save_dataset(path, name, X, y.reshape(-1, 1))
        
    """ I comment this because I no longer have the files in .dat format and I have them all in .npy format.
    real_path = "./datasets/real/"
    for file in os.listdir(real_path):
        generator = get_dataframe_from_dat(real_path  + file)
        data = list(generator)
        data = np.array(data)
        X = data[:, :-1]
        y = data[:, -1]
        unique_labels, y = np.unique(y, return_inverse=True)
        y += 1
        y = y.reshape(-1, 1)z
        np.save(path + file.split(".")[0] + ".npy", np.concatenate((X, y), axis=1))
    """
    
def generate_scenario_datasets(path, n_blobs, initial_seed, scenarios_file):
    """
    Generate and save datasets based on scenario configurations.

    :param path: Directory path to save the datasets.
    :param n_blobs: Number of blobs per scenario.
    :param initial_seed: Seed for reproducibility.
    :param scenarios_file: Path to the CSV file with scenario configurations.
    """
    scenarios = pd.read_csv(scenarios_file)
    
    for j,row in enumerate(scenarios.iterrows()):
        scenario_data = generate_scenario(
            n_blobs=n_blobs,
            k_low=row[1]['kl'], k_high=row[1]['ku'],
            p_low=row[1]['pl'], p_high=row[1]['pu'],
            s_low=row[1]['sl'], s_high=row[1]['su'],
            n_samples=row[1]['n'],
            initial_seed=initial_seed + j * n_blobs
        )
        
        for i, key in enumerate(scenario_data[0]):
            X, y = scenario_data[1][i]
            save_dataset(path, key, X, y.reshape(-1, 1))
         

def generate_data(
    path, n_samples=500, n_blobs=10, initial_seed=500, random_state=131416, scenarios_file="out_files/scenarios.csv"
):
    """
    Generate and save synthetic, real-world, and scenario-based test datasets.

    :param path: Directory path to save all datasets.
    :param n_samples: Number of samples for synthetic datasets.
    :param n_blobs: Number of blobs for scenario-based datasets.
    :param initial_seed: Seed for scenario generation.
    :param random_state: Seed for synthetic datasets.
    :param scenarios_file: Path to the CSV file with scenario configurations.
    """ 
    
    # Generate synthetic datasets
    generate_synthetic_datasets(path + "control/", n_samples, random_state)
    
    # Generate real-world datasets
    generate_real_datasets(path + "control/")
    
    # Generate scenario-based datasets
    generate_scenario_datasets(path + "blobs/", n_blobs, initial_seed, scenarios_file)
    
 
def ensure_dirs_exist(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            
    
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    initial_seed=200
    n_blobs=20
    k_max=37
    max_pred=35
    kh=15
    suffix=str(n_blobs)+"blobs"+str(kh)+"K"+str(k_max)+"S"+str(initial_seed)
        
    # Generate combinations for user-specific requirements
    k_values_custom = [2, 4, 8, 16, 32]
    p_values_custom = [2, 5, 10]
    n_values_custom = [64, 128, 1024]
    d_intervals_custom = [(0.1, 1.0)]

    scenarios_custom = []
    for k, p, n, (sl, su) in itertools.product(k_values_custom, p_values_custom, n_values_custom, d_intervals_custom):
        scenario_name = f"K{k}_P{p}_N{n}_D{sl}-{su}"
        scenarios_custom.append({
            "Scenario": scenario_name,
            "kl": k,
            "ku": k,
            "pl": p,
            "pu": p,
            "n": n,
            "sl": sl,
            "su": su
        })

    # Create a DataFrame for custom scenarios
    df_custom = pd.DataFrame(scenarios_custom)
    
    # Ensure directories exist
    ensure_dirs_exist([
        "./datasets",
        "./results",
        "./out_files",
        "./figures"
    ])
    
    
    # Save the custom scenarios to a new file and display a preview
    file_path_custom = "./out_files/scenarios.csv"
    df_custom.to_csv(file_path_custom, index=False)
    df_custom.head(), file_path_custom

    generate_data(path="./datasets/")