import pandas as pd
import numpy as np
from scipy.stats import zscore

def euclidean_dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def remove_outliers(data, outlier_indices):
    return np.delete(data, outlier_indices, axis=0)


def initialize_centroids(data, k):
    centroids_indices = np.random.choice(range(len(data)), size=k, replace=False)
    centroids = data[centroids_indices]
    return centroids

#assign each point to the nearest centroid
def AssignToClusters(data, centroids):
    clusters = []
    for point in data:
        distances = [euclidean_dist(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return clusters

#update centroids based on the mean of the points in each cluster
def update_centroids(data, clusters, k):
    # matrix to store updated positions of the centroids
    centroids = np.zeros((k, data.shape[1]))
    for cluster_id in range(k):
        cluster_points = data[np.array(clusters) == cluster_id]
        if len(cluster_points) > 0:
            centroids[cluster_id] = np.mean(cluster_points, axis=0)
    
    return centroids


#detect outliers using Z-score
def detect_outliers_zscore(data, threshold):
    z_scores = zscore(data)
    outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
    outliers = data[outlier_indices]
    return outliers, outlier_indices

# Function to perform k-means clustering
def kmeans(data, k, threshold, max_iterations=100):
    outliers, outlier_indices = detect_outliers_zscore(data, threshold)
    if len(outliers) > 0:
        data = remove_outliers(data, outlier_indices) 
    centroids = initialize_centroids(data, k)
    print(outliers)
    print(data)
    
    for _ in range(max_iterations):
        clusters = AssignToClusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        #checks for convergence
        if np.array_equal(new_centroids, centroids):
            break
        centroids = new_centroids
    return clusters, centroids, outliers, outlier_indices

