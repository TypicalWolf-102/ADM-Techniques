# clustering.py - Clustering algorithms implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score  # Kept for GMM evaluation

def perform_kmeans(X, max_clusters=10, output_folder=None):
    """Perform K-means clustering and determine the optimal number of clusters using the Elbow method.
    
    Args:
        X (numpy.ndarray): The data to cluster
        max_clusters (int, optional): Maximum number of clusters to try. Defaults to 10.
        output_folder (str, optional): Folder to save visualizations. Defaults to None.
        
    Returns:
        tuple: (kmeans, cluster_labels, optimal_k) - trained model, cluster labels, optimal cluster count
    """
    print("\n--- Performing K-means Clustering (Elbow Method) ---")
    
    # Calculate inertia for different number of clusters
    inertia = []
    
    for n_clusters in range(1, max_clusters + 1):
        # Create and fit KMeans model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        
        # Store inertia (sum of squared distances to closest centroid)
        inertia.append(kmeans.inertia_)
        
        print(f"Clusters: {n_clusters}, Inertia: {kmeans.inertia_:.3f}")
    
    # Visualize Elbow method
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), inertia, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True)
    plt.tight_layout()
    if output_folder:
        plt.savefig(f"{output_folder}/kmeans_elbow_method.png")
    else:
        plt.savefig('kmeans_elbow_method.png')
    plt.close()
    
    # Find optimal K using the elbow method
    # Calculate the rate of decrease in inertia
    inertia_diff = np.diff(inertia)
    inertia_diff_rate = np.diff(inertia_diff)
    
    # The elbow is where the rate of decrease sharply changes
    # We add 2 because:
    # 1. np.diff reduces array length by 1 each time
    # 2. We start from 1 cluster, so index 0 = 1 cluster
    optimal_k = np.argmax(inertia_diff_rate) + 2
    
    print(f"Optimal number of clusters based on Elbow method: {optimal_k}")
    
    # Train the final K-means model with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    return kmeans, cluster_labels, optimal_k

def perform_gmm(X, max_components=10, output_folder=None):
    """Perform Gaussian Mixture Model clustering and determine the optimal number of components using BIC.
    
    Args:
        X (numpy.ndarray): The data to cluster
        max_components (int, optional): Maximum number of components to try. Defaults to 10.
        output_folder (str, optional): Folder to save visualizations. Defaults to None.
        
    Returns:
        tuple: (gmm, cluster_labels, optimal_n) - trained model, cluster labels, optimal component count
    """
    print("\n--- Performing Gaussian Mixture Model (BIC Method) ---")
    
    # Calculate BIC for different number of components
    bic_scores = []
    
    for n_components in range(1, max_components + 1):
        # Create and fit GMM model
        gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=5)
        gmm.fit(X)
        
        # Store BIC scores
        bic_score = gmm.bic(X)
        bic_scores.append(bic_score)
        
        print(f"Components: {n_components}, BIC: {bic_score:.3f}")
    
    # Visualize BIC scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_components + 1), bic_scores, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('BIC Score')
    plt.title('BIC Score for GMM Component Selection')
    plt.grid(True)
    plt.tight_layout()
    if output_folder:
        plt.savefig(f"{output_folder}/gmm_bic_scores.png")
    else:
        plt.savefig('gmm_bic_scores.png')
    plt.close()
    
    # Find optimal number of components based on lowest BIC
    optimal_n = np.argmin(bic_scores) + 1  # +1 because we start from 1 component
    
    print(f"Optimal number of components based on BIC: {optimal_n}")
    
    # Train the final GMM model with the optimal number of components
    gmm = GaussianMixture(n_components=optimal_n, random_state=42, n_init=5)
    gmm.fit(X)
    cluster_labels = gmm.predict(X)
    
    return gmm, cluster_labels, optimal_n

def visualize_clusters(X_pca, kmeans_labels, gmm_labels, y, output_folder=None):
    """Visualize clustering results using PCA for dimensionality reduction.
    
    Args:
        X_pca (numpy.ndarray): PCA-transformed data
        kmeans_labels (numpy.ndarray): K-means cluster labels
        gmm_labels (numpy.ndarray): GMM cluster labels
        y (numpy.ndarray or pandas.Series): True class labels
        output_folder (str, optional): Folder to save visualizations. Defaults to None.
    """
    plt.figure(figsize=(18, 6))
    
    # Convert target to numeric if it's not already
    if not np.issubdtype(y.dtype, np.number):
        y_numeric = y.map({'yes': 1, 'no': 0})
    else:
        y_numeric = y
    
    # Plot original classes
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_numeric, cmap='viridis', alpha=0.5, s=10)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('True Classes')
    plt.colorbar(scatter, label='Class')
    
    # Plot K-means clusters
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.5, s=10)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('K-means Clusters')
    plt.colorbar(scatter, label='Cluster')
    
    # Plot GMM clusters
    plt.subplot(1, 3, 3)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='tab10', alpha=0.5, s=10)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('GMM Clusters')
    plt.colorbar(scatter, label='Cluster')
    
    plt.tight_layout()
    if output_folder:
        plt.savefig(f"{output_folder}/cluster_visualization.png")
    else:
        plt.savefig('cluster_visualization.png')
    plt.close()

def evaluate_clusters(kmeans_labels, gmm_labels, y, output_folder=None):
    """Evaluate how well clusters align with true classes.
    
    Args:
        kmeans_labels (numpy.ndarray): K-means cluster labels
        gmm_labels (numpy.ndarray): GMM cluster labels
        y (numpy.ndarray or pandas.Series): True class labels
        output_folder (str, optional): Folder to save visualizations. Defaults to None.
    """
    # Convert target to numeric if it's not already
    if not np.issubdtype(y.dtype, np.number):
        y_numeric = np.array(y.map({'yes': 1, 'no': 0}))
    else:
        y_numeric = np.array(y)
    
    # Create a function to calculate cluster purity
    def purity_score(clusters, classes):
        contingency_matrix = pd.crosstab(clusters, classes)
        # Purity is the sum of the maximum value in each cluster divided by the total number of samples
        return np.sum(np.max(contingency_matrix, axis=1)) / np.sum(contingency_matrix.values)
    
    # Calculate purity for K-means
    kmeans_purity = purity_score(kmeans_labels, y_numeric)
    
    # Calculate purity for GMM
    gmm_purity = purity_score(gmm_labels, y_numeric)
    
    print("\n--- Cluster Evaluation ---")
    print(f"K-means cluster purity: {kmeans_purity:.4f}")
    print(f"GMM cluster purity: {gmm_purity:.4f}")
    
    # Create contingency tables
    kmeans_contingency = pd.crosstab(
        kmeans_labels, 
        y_numeric,
        rownames=['K-means Cluster'],
        colnames=['True Class']
    )
    
    gmm_contingency = pd.crosstab(
        gmm_labels, 
        y_numeric,
        rownames=['GMM Cluster'],
        colnames=['True Class']
    )
    
    print("\nK-means Contingency Table:")
    print(kmeans_contingency)
    
    print("\nGMM Contingency Table:")
    print(gmm_contingency)
    
    # Visualize contingency tables
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(kmeans_contingency, annot=True, fmt='d', cmap='Blues')
    plt.title('K-means Clusters vs. True Classes')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(gmm_contingency, annot=True, fmt='d', cmap='Blues')
    plt.title('GMM Clusters vs. True Classes')
    
    plt.tight_layout()
    if output_folder:
        plt.savefig(f"{output_folder}/cluster_evaluation.png")
    else:
        plt.savefig('cluster_evaluation.png')
    plt.close()