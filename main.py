# main.py - Main entry point for the banking data analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import from separate modules
from utils import load_data, explore_data, get_next_test_folder
from preprocessing import preprocess_data
from pca import perform_pca, visualize_pca
from clustering import perform_kmeans, perform_gmm, visualize_clusters, evaluate_clusters
from neural_network import build_neural_network, train_and_evaluate_model, compare_models
from gbm import train_and_evaluate_gbm, tune_gbm_hyperparameters, compare_gbm_with_other_models
from config import DATA_CONFIG, PCA_CONFIG, CLUSTERING_CONFIG, NN_CONFIG, GBM_CONFIG

def main():
    """Main function to run the complete analysis pipeline."""
    # Create a new test folder for this run
    output_folder = get_next_test_folder()
    print(f"Saving all outputs to folder: {output_folder}")
    
    # Load data
    file_path = 'processed_bank_data.csv'
    data = load_data(file_path)
    
    if data is None:
        return
    
    # Explore the data
    explore_data(data, output_folder=output_folder)
    
    # Preprocess data
    print("\n--- Preprocessing Data ---")
    X_scaled, y, scaler = preprocess_data(data)
    print(f"X shape after preprocessing: {X_scaled.shape}")
    print(f"y shape after preprocessing: {y.shape}")
    print(f"Class balance: {np.bincount(y)}")
    
    # Perform PCA
    print("\n--- Performing PCA ---")
    X_pca, pca, explained_variance = perform_pca(X_scaled)
    print(f"PCA explained variance: {explained_variance}")
    print(f"PCA transformed shape: {X_pca.shape}")
    
    # Visualize PCA results
    print("\n--- Visualizing PCA Results ---")
    visualize_pca(pca, explained_variance, output_folder=output_folder)
    
    # Find the number of components that explain 95% of variance
    cumulative_variance = np.cumsum(explained_variance)
    n_components_95 = np.argmax(cumulative_variance >= PCA_CONFIG['variance_threshold']) + 1
    print(f"Number of components explaining {PCA_CONFIG['variance_threshold']*100}% of variance: {n_components_95}")
    
    # Get PCA with optimal number of components
    X_pca_optimal, _, _ = perform_pca(X_scaled, n_components=n_components_95)
    
    # Perform K-means clustering on the scaled data
    kmeans, kmeans_labels, optimal_k = perform_kmeans(X_scaled, max_clusters=10, output_folder=output_folder)
    
    # Perform GMM on the scaled data
    gmm, gmm_labels, optimal_n = perform_gmm(X_scaled, max_components=10, output_folder=output_folder)
    
    # Visualize clustering results
    visualize_clusters(X_pca, kmeans_labels, gmm_labels, y, output_folder=output_folder)
    
    # Evaluate clusters against true classes
    evaluate_clusters(kmeans_labels, gmm_labels, y, output_folder=output_folder)
    
    # Add cluster assignments as features
    X_with_clusters = np.column_stack((X_scaled, kmeans_labels.reshape(-1, 1), gmm_labels.reshape(-1, 1)))
    print(f"Shape with cluster features: {X_with_clusters.shape}")
    
    # Compare neural network models with different features
    print("\n--- Comparing Neural Network Models ---")
    nn_results = compare_models(X_scaled, X_pca_optimal, y, X_with_clusters, output_folder=output_folder)
    
    # Train and evaluate GBM models with different features
    print("\n--- Training and Evaluating GBM Models ---")
    gbm_results = {}
    
    # GBM with original features
    _, gbm_acc_original = train_and_evaluate_gbm(
        X_scaled, y, 
        test_size=GBM_CONFIG['test_size'], 
        feature_type="original", 
        output_folder=output_folder
    )
    gbm_results['original'] = gbm_acc_original
    
    # GBM with PCA features
    _, gbm_acc_pca = train_and_evaluate_gbm(
        X_pca_optimal, y, 
        test_size=GBM_CONFIG['test_size'], 
        feature_type="pca", 
        output_folder=output_folder
    )
    gbm_results['pca'] = gbm_acc_pca
    
    # GBM with cluster-enhanced features
    _, gbm_acc_clusters = train_and_evaluate_gbm(
        X_with_clusters, y, 
        test_size=GBM_CONFIG['test_size'], 
        feature_type="cluster_enhanced", 
        output_folder=output_folder
    )
    gbm_results['cluster_enhanced'] = gbm_acc_clusters
    
    # Optional: Tune GBM hyperparameters (commented out to avoid long runtime)
    # print("\n--- Tuning GBM Hyperparameters ---")
    # best_gbm, best_params = tune_gbm_hyperparameters(
    #     X_scaled, y, 
    #     param_grid=GBM_CONFIG['grid_search']['param_grid'],
    #     cv=GBM_CONFIG['grid_search']['cv'],
    #     feature_type="original",
    #     output_folder=output_folder
    # )
    
    # Compare all models (Neural Network and GBM)
    print("\n--- Comparing All Models ---")
    # Create a dictionary with all model results
    all_model_results = {
        'Neural Network': nn_results,
        'GBM': gbm_results
    }
    feature_types = ['original', 'pca', 'cluster_enhanced']
    
    # Compare GBM with Neural Network
    compare_gbm_with_other_models(all_model_results, feature_types, output_folder=output_folder)

    # Summary of all approaches
    print("\n=== Summary of All Approaches ===")
    print("1. PCA Analysis:")
    print(f"   - Number of components explaining {PCA_CONFIG['variance_threshold']*100}% of variance: {n_components_95}")
    print(f"   - Explained variance ratios: {explained_variance}")
    
    print("\n2. Clustering Analysis:")
    print(f"   - Optimal K-means clusters: {optimal_k}")
    print(f"   - Optimal GMM components: {optimal_n}")
    
    print("\n3. Neural Network Performance:")
    for feature_type, accuracy in nn_results.items():
        print(f"   - Neural Network with {feature_type} features: {accuracy:.4f}")
    
    print("\n4. GBM Performance:")
    for feature_type, accuracy in gbm_results.items():
        print(f"   - GBM with {feature_type} features: {accuracy:.4f}")
    
    print(f"\nAll plots and results saved to: {output_folder}")

if __name__ == "__main__":
    main()