import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

def read_results(directory, output_csv=None):
    # If CSV file exists, load and return it
    if output_csv and os.path.exists(output_csv):
        print(f"Loading existing results from {output_csv}...")
        df = pd.read_csv(output_csv)
        return df

    # Initialize a list to store all results
    all_results = []

    # Iterate over all .txt files in the specified directory with tqdm progress bar
    for filename in tqdm(os.listdir(directory), desc="Reading files"):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            
            # Read the JSON data from the file
            with open(filepath, "r") as f:
                data = json.load(f)
                
                # Extract partitions info
                partitions = data.get("partitions", {})
                X = partitions.get("X", [])
                y = partitions.get("y", [])
                k = partitions.get("k", [])
                
                # Compute k (number of unique clusters), n (number of samples), d (dimensionality)
                n = len(y)
                d = len(X[0]) if X else 0
                k_true = len(set(y)) if y else 0
               
                # Add these values to the result
                data["n_samples"] = n
                data["dimensionality"] = d
                data["n_clusters"] = k_true
                data["k"] = k
                data.pop('partitions')
                all_results.append(data)
                
    # Create a DataFrame from the collected results
    df = pd.DataFrame(all_results)

    # Save to CSV if output_csv is specified
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)  # Ensure the output directory exists
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    return df

def compute_metrics(df):
    # Basic summary statistics
    summary = df.describe(include="all")
    
    # Group by dataset and algorithm and compute mean
    grouped = df.groupby(["dataset", "algorithm"]).mean()
    
    return summary, grouped

def compute_correlation(df):
    # Select only numeric columns (exclude non-numeric columns like 'dataset', 'algorithm')
    numeric_df = df.select_dtypes(include=[float, int])

    # Correlation between 'sigui' and other metrics (general)
    correlation_matrix = numeric_df.corr(method='spearman')
    
    # Correlation between 'sigui' and other metrics
    sigui_correlation = correlation_matrix['sigui'].drop('sigui')  # Drop 'sigui' to avoid self-correlation
    
    return correlation_matrix, sigui_correlation

def compute_correlation_per_dataset(df):
    # Initialize a dictionary to store correlation per dataset
    correlation_per_dataset = {}
    df = df.drop(columns = ["n_clusters", "algorithm", "n_samples", "dimensionality", "k"])
    # Compute correlation for each dataset individually
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        numeric_df = dataset_df.select_dtypes(include=[float, int])
        
        # Skip datasets with no variability in quality scores
        if numeric_df.nunique().min() <= 1:  # Check if all columns have at least 2 unique values
            continue
        
        # Compute Spearman rank correlation
        spearman_corr = numeric_df.corr(method='spearman')
        sigui_correlation = spearman_corr['sigui'].drop('sigui')  # Drop 'sigui' to avoid self-correlation
        
        correlation_per_dataset[dataset] = sigui_correlation

    # Convert dictionary to a DataFrame
    return pd.DataFrame.from_dict(correlation_per_dataset)

def compute_correlation_per_k(df):
    # Initialize a dictionary to store correlation per dataset
    correlation_per_dataset = {}
    df = df.drop(columns = ["n_clusters", "algorithm", "n_samples", "dimensionality", "dataset"])
    # Compute correlation for each dataset individually
    for dataset in df['k'].unique():
        dataset_df = df[df['k'] == dataset]
        numeric_df = dataset_df.select_dtypes(include=[float, int])
        
        # Compute Spearman rank correlation
        spearman_corr = numeric_df.corr(method='spearman')
        sigui_correlation = spearman_corr['sigui'].drop('sigui')  # Drop 'sigui' to avoid self-correlation
        
        correlation_per_dataset[dataset] = sigui_correlation

    # Convert dictionary to a DataFrame
    return pd.DataFrame.from_dict(correlation_per_dataset)

# Example function to visualize partitions
def visualize_partitions(results_df, dataset_name):
    # Filter rows for the given dataset
    dataset_results = results_df[results_df["dataset"] == dataset_name]
    
    # Set up the figure
    n_algorithms = len(dataset_results)
    fig, axes = plt.subplots(1, n_algorithms, figsize=(5 * n_algorithms, 5), sharey=True)
    if n_algorithms == 1:
        axes = [axes]  # Ensure axes is iterable for single algorithm case
    
    # Loop through algorithms and create plots
    for ax, (_, row) in zip(axes, dataset_results.iterrows()):
        # Extract partitions
        partitions = row["partitions"]
        X = np.array(partitions["X"])
        predictions = np.array(partitions["predictions"])
        
        # Scatter plot for the data colored by predictions
        scatter = ax.scatter(X[:, 0], X[:, 1], c=predictions, cmap="tab10", s=10, alpha=0.7)
        ax.set_title(f"{row['algorithm']}", fontsize=14)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        
        # Annotate with metrics
        metrics = f"Rand Score: {row['rand_score']:.2f}\n" \
                  f"Accuracy: {row['accuracy']:.2f}\n" \
                  f"F1 Score: {row['f1_score']:.2f}\n" \
                  f"Precision: {row['precision']:.2f}\n" \
                  f"Recall: {row['recall']:.2f}\n" \
                  f"Sigui: {row['sigui']:.2f}"
        ax.text(0.05, 0.95, metrics, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.5, facecolor="white"))

    plt.tight_layout()
    plt.savefig(f"./figures/{dataset_name}")


# Directory containing result files
results_directory = "./results"

# Read and process results
results_df = read_results(results_directory)
results_df[results_df["dataset"]=="blobs-P2-K2-N10000-dt0.1-S0"]


# Compute overall metrics and correlations
correlation_matrix, _ = compute_correlation(results_df)

# Plot general correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, vmin=-1, vmax=1)
plt.title("General Correlation Matrix")
plt.tight_layout()
plt.savefig("./figures/corr_matrix.png")

# Compute correlation per dataset
correlation_per_dataset = compute_correlation_per_dataset(results_df)
correlation_per_dataset = correlation_per_dataset.dropna()
correlation_per_dataset= correlation_per_dataset.T.sort_values(by=["rand_score"])

# Plot all dataset correlations in a single heatmap
plt.figure(figsize=(10, 8)) 
sns.heatmap(correlation_per_dataset, annot=False, cmap="coolwarm", cbar=True, vmin=-1, vmax=1)
plt.title("Correlation of 'sigui' with Other Metrics for All Datasets")
plt.tight_layout()

# Save the figure
plt.savefig("./figures/corr_all_datasets.png")

# Compute correlation per dataset
correlation_per_k = compute_correlation_per_k(results_df)
correlation_per_k  = correlation_per_k.dropna()
correlation_per_k = correlation_per_k .T.sort_values(by=["rand_score"])

# Plot all dataset correlations in a single heatmap
plt.figure(figsize=(10, 8)) 
sns.heatmap(correlation_per_k, annot=False, cmap="coolwarm", cbar=True, vmin=-1, vmax=1)
plt.title("Correlation of 'sigui' with Other Metrics for All k")
plt.tight_layout()

# Save the figure
plt.savefig("./figures/corr_all_k.png")