import numpy as np 

from typing import Generator, List, Tuple, Union

from sklearn.metrics import silhouette_score as sc
from sklearn.metrics import calinski_harabasz_score as chc
from sklearn.metrics import davies_bouldin_score as dbc

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

def generate_decreasing_sequence(n, a):
    S = (1 - a) / (1 - a**n)  # Factor de normalización
    sequence = [S * a**(i) for i in range(n)]
    
    return np.array(sequence)

def global_covering_index(X, labels):
    partial_mci = []
    n = X.shape[0]  # Total number of data points
    
    labels = np.array(labels)  # Ensure labels is a NumPy array
    unique_y = np.unique(labels)
    
    for c_i in unique_y:
        # Extract points assigned to cluster `c_i`
        X_c = X[labels == c_i]
        c = X_c.mean(axis=0)
        
        if X_c.shape[0] <= 1:  # Handle empty clusters
            partial_mci.append(0.0)
        else:    
            # Compute partial SIUI for the current cluster
            mci = np.mean([uncovering(j, c) for j in X_c])
            partial_mci.append(mci)
    return np.mean(partial_mci)

def sugeno_inspired_uncovering_index(X, c, alpha):
    """
    Computes the uncovering index for a set of points `X` with respect to cluster `c`.
    
    Parameters:
    - X: List or array of data points.
    - c: Cluster center (1D array-like).

    Returns:
    - Maximum uncovering value.
    """
    uncovering_values = [uncovering(i, c) for i in X] 
    n = len(uncovering_values)
    uncovering_values = np.sort(uncovering_values)[::-1].tolist()

    w = generate_decreasing_sequence(n-1, alpha) 
    result = []
    sum_ = np.sum(uncovering_values) 
    for i in range(n):
        remaining_values = uncovering_values[:i] + uncovering_values[i+1:]
        remaining_values = np.array(remaining_values)
        result.append(np.minimum(uncovering_values[i], np.sum(w * remaining_values)))
    return np.max(result)

def sugeno_inspired_global_uncovering_index(X, labels, alpha, get_info=False):
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
    n = X.shape[0]  # Total number of data points
    
    labels = np.array(labels)  # Ensure labels is a NumPy array
    unique_y = np.unique(labels)
    
    for c_i in unique_y:
        # Extract points assigned to cluster `c_i`
        X_c = X[labels == c_i]
        c = X_c.mean(axis=0)
        
        if X_c.shape[0] <= 1:  # Handle empty clusters
            partial_siui.append(0.0)
        else:    
            # Compute partial SIUI for the current cluster
            partial_value = sugeno_inspired_uncovering_index(X_c, c, alpha) 
            partial_siui.append(partial_value * (X_c.shape[0] / n))

    if len(partial_siui) < unique_y.shape[0]:
        pass  
    # Sum partial SIUI values to get the global uncovering index
    if get_info:
        return np.sum(partial_siui), partial_siui
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

def silhouette_score(X, y):
    """
    Modified silhouette score to return None if there is only one cluster in y.
    """
    if np.amax(y) == 0:
        return None
    else:
        return sc(X, y)


def calinski_harabasz_score(X, y):
    """
    Modified CH score to return None if there is only one cluster in y.
    """
    if np.amax(y) == 0:
        return None
    else:
        return chc(X, y)


def davies_bouldin_score(X, y):
    """
    Modified DB score to return None if there is only one cluster in y.
    """
    if np.amax(y) == 0:
        return None
    else:
        return dbc(X, y)


def SSE(X, y, centroids):
    sse = 0.0
    for i, centroid in enumerate(centroids):
        idx = np.where(y == i)[0]
        sse += np.sum((X[idx] - centroid)**2)
    return sse



# Taken from https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
def bic_fixed(X, y, sse):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    
    #number of clusters
    m = len(np.unique(y))
    # size of the clusters
    n = np.bincount(np.array(y,dtype='int64'))
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sse


    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)

def xie_beni_ts(y, centroids, sse):
    
    K = len(np.unique(y))

    intraclass_similarity= sse
    cluster_dispersion    = 0.0
    min_dispersion        = np.inf
    if K==1:
        return None

    for k1 in range(K):
        for k2 in range(K):
            if k1 != k2:
                aux= np.sum((centroids[k2] - centroids[k1])**2)
                cluster_dispersion   += aux
    
                if aux < min_dispersion:
                    min_dispersion= aux
        
    return (intraclass_similarity + (1/(K * (K-1))) * cluster_dispersion) / (1/K + min_dispersion)


 