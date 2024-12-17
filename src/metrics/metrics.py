import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

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
# Function to compute correlations per x (e.g., per dataset, per k, etc.)
def compute_correlation_per_x(df, x="dataset"):
    correlation_per_x = {}

    # List of columns to drop (keeping only the relevant ones for correlation)
    columns = ["n_clusters", "algorithm", "n_samples", "dimensionality", "k", "dt", "dataset"]
    columns_to_drop = [col for col in columns if col != x]
    
    # Drop irrelevant columns
    df_filtered = df.drop(columns=columns_to_drop)
    
    # Compute correlation for each unique value in the specified column (x)
    for x_ in df_filtered[x].unique():
        # Filter DataFrame by the current value
        df_filtered_value = df_filtered[df_filtered[x] == x_]
        
        # Only select numerical columns (int and float types)
        df_filtered_value = df_filtered_value.select_dtypes(include=[float, int])
    
        # Compute Spearman rank correlation
        spearman_corr = df_filtered_value.corr(method='spearman')
        
        # Extract correlation with 'sigui' (assuming 'sigui' exists in the DataFrame)
        if 'sigui' in spearman_corr.columns:
            sigui_correlation = spearman_corr['sigui'].drop('sigui')  # Drop 'sigui' to avoid self-correlation
            correlation_per_x[x_] = sigui_correlation

    return pd.DataFrame.from_dict(correlation_per_x, orient='index')


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
results_df = read_results(results_directory, "out_files/merged.csv")

# Extracting 'dt' values using regex
results_df['dt'] = results_df['dataset'].apply(lambda x: float(re.search(r'dt(\d+\.\d+)', x).group(1)))


# Compute overall metrics and correlations
correlation_matrix, _ = compute_correlation(results_df)

# Plot general correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, vmin=-1, vmax=1)
plt.title("General Correlation Matrix")
plt.tight_layout()
plt.savefig("./figures/corr_matrix.png")

# List of columns to iterate over for correlation
columns_to_compute =  ["n_clusters", "algorithm", "n_samples", "dimensionality", "k", "dt", "dataset"]  # You can add more columns as needed

for column_name in columns_to_compute:
    # Compute the correlation per x (e.g., per dataset, per k, etc.)
    correlation_per_x = compute_correlation_per_x(results_df, column_name)
    
    # Clean the data by dropping NaN values and sorting
    correlation_per_x = correlation_per_x.dropna()
    correlation_per_x = correlation_per_x.T.sort_values(by=["rand_score"])

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_per_x, annot=False, cmap="coolwarm", cbar=True, vmin=-1, vmax=1)
    
    # Set title dynamically based on the column
    plt.title(f"Correlation of 'sigui' with Other Metrics for All {column_name.capitalize()}")
    plt.tight_layout()

    # Save the heatmap figure
    plt.savefig(f"./figures/corr_all_{column_name}.png")
    plt.close()  # Close the plot to free memory for the next one