import numpy as np 

def uncovering(x, c):
    """
    Computes the uncovering index for a single point `x` with respect to cluster `c`.
    
    Parameters:
    - x: A single data point (1D array-like).
    - c: A cluster center (1D array-like).
    
    Returns:
    - Uncovering value (float).
    """
    D = x.shape[-1]  # Dimensionality of `x`
    distance = np.linalg.norm(x - c)  # Euclidean distance
    return 1 - np.exp(2 * np.log(10) * (-distance / (5 * np.sqrt(D))))
     
def F(X, e_value):
    
    max_X = np.max(X)
    min_X = np.min(X)
    
    return max_X - (1- e_value) * (max_X - min_X)
    
def entropy(X):
    
    hist,_ =np.histogram(X, bins=10,  density = True, range=(0, 1))
    hist = hist[hist > 0]
    hist /= hist.sum()
    
    if len(hist) > 0:
        max_entropy = np.log(len(hist))
    else:
        max_entropy = 0  # Handle cases where histogram is empty or invalid

    entropy = -np.sum(hist * np.log(hist)) / (max_entropy if max_entropy > 0 else 1.)
    return entropy 
 
def sugeno_inspired_uncovering_index(X, c):
    """
    Computes the uncovering index for a single point `x` with respect to cluster `c`.
    
    Parameters:
    - X: A single data point ().
    - c: A cluster center (1D array-like).
    
    Returns:
    - Uncovering value (float).
    """
    uncovering_values = [uncovering(i, c) for i in X]
    result = []
    e = entropy(uncovering_values) 

    for i in range(len(uncovering_values)):
        # Create a new list excluding the current element
        remaining_values = uncovering_values[:i] + uncovering_values[i+1:]
        result.append(np.minimum(uncovering_values[i], F(remaining_values, e)))

    return np.max(result), e
 
def sugeno_inspired_global_uncovering_index(X, C, labels, get_info= False):
    """
    Computes the Sugeno-inspired global uncovering index for the entire dataset.
    
    Parameters:
    - X: Dataset as a 2D array (shape: [n_samples, n_features]).
    - C: Cluster centers as a 2D array (shape: [n_clusters, n_features]).
    - labels: Cluster labels for each data point in `X` (1D array-like, shape: [n_samples]).
    
    Returns:
    - Global uncovering index (float).
    """
    
    partial_siui = []
    partial_entropies = [] 
    n = X.shape[0]  # Total number of data points
    
    labels = np.array(labels)  # Ensure labels is a NumPy array
    
    for c_i, c in enumerate(C):
        # Extract points assigned to cluster `c_i`
        X_c = X[labels == c_i]
        
        if X_c.shape[0] <= 1:  # Handle empty clusters
            partial_entropies.append(0.0)
            partial_siui.append(0.0)
        else:    
            # Compute partial SIUI for the current cluster
            partial_value, entropy_value = sugeno_inspired_uncovering_index(X_c, c) 
            partial_siui.append(partial_value * (X_c.shape[0] / n))
            partial_entropies.append(entropy_value)
    if len(partial_entropies) < C.shape[0]:
        pass  
    # Sum partial SIUI values to get the global uncovering index
    if get_info:
        return np.sum(partial_siui), partial_siui, partial_entropies
    else:
        return np.sum(partial_siui)
 
def compute_centroids(X, labels, n_clusters):
    centroids = []
    for cluster_id in range(n_clusters):
        cluster_points = X[labels == cluster_id]
        if len(cluster_points) > 0:
            centroids.append(cluster_points.mean(axis=0))
        else:
            n_clusters-=1
    return np.array(centroids), n_clusters

 