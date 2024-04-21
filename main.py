import pandas as pd
import numpy as np
from scipy.stats import zscore

dataset = pd.read_csv("imdb_top_2000_movies.csv")
X = dataset[['IMDB Rating']].values
def remove_outliers(X, outlier_indices):
    return np.delete(X, outlier_indices, axis=0)

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Function to initialize centroids randomly
def initialize_centroids(X, k):
    centroids_indices = np.random.choice(range(len(X)), size=k, replace=False)
    centroids = X[centroids_indices]
    return centroids

def assign_to_clusters(X, centroids):
    clusters = []
    for point in X:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return clusters
# Function to update centroids based on the mean of the points in each cluster
def update_centroids(X, clusters, k):
    centroids = np.zeros((k, X.shape[1]))
    for cluster_id in range(k):
        cluster_points = X[np.array(clusters) == cluster_id]
        if len(cluster_points) > 0:
            centroids[cluster_id] = np.mean(cluster_points, axis=0)
    return centroids

# Function to detect outliers using Z-score
def detect_outliers_zscore(X, threshold):
    z_scores = zscore(X)
    outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
    outliers = X[outlier_indices]
    return outliers, outlier_indices


# Function to perform k-means clustering
def kmeans_with_zscore(X, k, threshold, max_iterations=100):
    outliers, outlier_indices = detect_outliers_zscore(X, threshold)
    if len(outliers) > 0:
        X = remove_outliers(X, outlier_indices) 
    #print(X)
    
    centroids = initialize_centroids(X, k)
    
    for _ in range(max_iterations):
        clusters = assign_to_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if np.array_equal(new_centroids, centroids):
            break
        centroids = new_centroids
    return clusters, centroids, outliers, outlier_indices

# User input for percentage of records to analyze
percentage = float(50)

num_records = int(len(X) * (percentage / 100))
X = X[:num_records]

# User input for number of clusters
k = int(3)

# User input for Z-score threshold
threshold = float(3)

# Perform k-means clustering with Z-score outlier detection
clusters, centroids, outliers, outlier_indices = kmeans_with_zscore(X, k, threshold)

# Print clusters
for cluster_id in range(k):
    print(f"Cluster {cluster_id + 1}:")
    cluster_indices = np.where(np.array(clusters) == cluster_id)[0]
    for idx in cluster_indices:
        print(dataset.iloc[idx]['Movie Name'], dataset.iloc[idx]['IMDB Rating'])

# Print outlier records
print("Outlier records:")
for idx in outlier_indices:
    print(dataset.iloc[idx]['Movie Name'], dataset.iloc[idx]['IMDB Rating'])