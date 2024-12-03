import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_results(directory):
    # Initialize a list to store all results
    all_results = []

    # Iterate over all .txt files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            
            # Read the JSON data from the file
            with open(filepath, "r") as f:
                data = json.load(f)
                all_results.append(data)
    
    # Create a DataFrame from the collected results
    df = pd.DataFrame(all_results)
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
    correlation_matrix = numeric_df.corr()
    
    # Correlation between 'sigui' and other metrics
    sigui_correlation = correlation_matrix['sigui'].drop('sigui')  # Drop 'sigui' to avoid self-correlation
    
    return correlation_matrix, sigui_correlation

def compute_correlation_per_dataset(df):
    # Initialize a dictionary to store correlation per dataset
    correlation_per_dataset = {}

    # Compute correlation for each dataset individually
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        numeric_df = dataset_df.select_dtypes(include=[float, int])
        correlation_matrix = numeric_df.corr()
        sigui_correlation = correlation_matrix['sigui'].drop('sigui')  # Drop 'sigui' to avoid self-correlation
        correlation_per_dataset[dataset] = sigui_correlation

    return pd.DataFrame.from_dict(correlation_per_dataset)

# Directory containing result files
results_directory = "./results"

# Read and process results
results_df = read_results(results_directory)

# Compute overall metrics and correlations
summary_stats, grouped_stats = compute_metrics(results_df)
correlation_matrix, sigui_correlation = compute_correlation(results_df)
 
# Plot general correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, vmin=-1, vmax=1)
plt.title("General Correlation Matrix")
plt.tight_layout()
plt.savefig("./figures/corr_matrix")

# Compute correlation per dataset
correlation_per_dataset = compute_correlation_per_dataset(results_df)

# Plot all dataset correlations in a single heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_per_dataset, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, vmin=-1, vmax=1)
plt.title("Correlation of 'sigui' with Other Metrics for All Datasets")
plt.tight_layout()

# Save the figure
plt.savefig("./figures/corr_all_datasets.png")
 