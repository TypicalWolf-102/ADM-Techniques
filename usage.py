#!/usr/bin/env python3
# run_analysis.py - Example script to run different analyses

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Import from our package
from utils import load_data, explore_data, get_next_test_folder
from preprocessing import preprocess_data
from pca_analysis import perform_pca, visualize_pca
from clustering import perform_kmeans, perform_gmm, visualize_clusters, evaluate_clusters
from neural_network import train_and_evaluate_model, compare_models
from linear_regression import run_linear_regression_analysis
from model_comparison import compare_models_with_linear
from config import DATA_CONFIG, PCA_CONFIG, CLUSTERING_CONFIG, NN_CONFIG

def run_pca_analysis(data, output_folder):
    """Run PCA analysis on the data."""
    print("\n=== Running PCA Analysis ===")
    
    # Preprocess data
    X_scaled, y, scaler = preprocess_data(data)
    
    # Perform PCA
    X_pca, pca, explained_variance = perform_pca(X_scaled)
    
    # Visualize PCA results
    visualize_pca(pca, explained_variance, output_folder=output_folder)
    
    # Find optimal number of components
    cumulative_variance = np.cumsum(explained_variance)
    n_components_95 = np.argmax(cumulative_variance >= PCA_CONFIG['variance_threshold']) + 1
    print(f"Number of components explaining {PCA_CONFIG['variance_threshold']*100}% of variance: {n_components_95}")
    
    # Get PCA with optimal number of components
    X_pca_optimal, _, _ = perform_pca(X_scaled, n_components=n_components_95)
    
    return X_scaled, X_pca_optimal, y

def run_clustering_analysis(X_scaled, X_pca, y, output_folder):
    """Run clustering analysis on the data."""
    print("\n=== Running Clustering Analysis ===")
    
    # Perform K-means clustering
    kmeans, kmeans_labels, optimal_k = perform_kmeans(
        X_scaled, 
        max_clusters=CLUSTERING_CONFIG['kmeans']['max_clusters'],
        output_folder=output_folder
    )
    
    # Perform GMM clustering
    gmm, gmm_labels, optimal_n = perform_gmm(
        X_scaled, 
        max_components=CLUSTERING_CONFIG['gmm']['max_components'],
        output_folder=output_folder
    )
    
    # Visualize clustering results
    visualize_clusters(X_pca, kmeans_labels, gmm_labels, y, output_folder=output_folder)
    
    # Evaluate clusters against true classes
    evaluate_clusters(kmeans_labels, gmm_labels, y, output_folder=output_folder)
    
    return kmeans_labels, gmm_labels

def run_neural_network_analysis(X_scaled, X_pca, y, kmeans_labels=None, gmm_labels=None, output_folder=None):
    """Run neural network analysis with different feature sets."""
    print("\n=== Running Neural Network Analysis ===")
    
    # Prepare the cluster-enhanced features if clustering was performed
    X_with_clusters = None
    if kmeans_labels is not None and gmm_labels is not None:
        print("--- Using Clustering as Additional Features ---")
        X_with_clusters = np.column_stack((X_scaled, kmeans_labels.reshape(-1, 1), gmm_labels.reshape(-1, 1)))
        print(f"Shape with cluster features: {X_with_clusters.shape}")
    
    # Compare all feature types
    compare_models(X_scaled, X_pca, y, X_with_clusters, output_folder=output_folder)

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Run bank data analysis with different methods')
    parser.add_argument('--pca', action='store_true', help='Run PCA analysis')
    parser.add_argument('--clustering', action='store_true', help='Run clustering analysis')
    parser.add_argument('--nn', action='store_true', help='Run neural network analysis')
    parser.add_argument('--linear', action='store_true', help='Run linear regression analysis')
    parser.add_argument('--compare', action='store_true', help='Run model comparison (NN vs. Linear)')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    
    args = parser.parse_args()
    
    # If no specific analyses are selected, run all
    if not (args.pca or args.clustering or args.nn or args.linear or args.compare):
        args.all = True
    
    # Create a new test folder for this run
    output_folder = get_next_test_folder()
    print(f"Saving all outputs to folder: {output_folder}")
    
    # Load data
    data = load_data(DATA_CONFIG['file_path'])
    if data is None:
        return
    
    # Explore the data
    explore_data(data, output_folder=output_folder)
    
    X_scaled = X_pca = y = None
    kmeans_labels = gmm_labels = None
    
    # Run requested analyses
    if args.pca or args.all:
        X_scaled, X_pca, y = run_pca_analysis(data, output_folder)
    
    if args.clustering or args.all:
        if X_scaled is None or X_pca is None or y is None:
            # If PCA wasn't run but clustering is requested
            X_scaled, y, _ = preprocess_data(data)
            X_pca, _, _ = perform_pca(X_scaled)
        
        kmeans_labels, gmm_labels = run_clustering_analysis(X_scaled, X_pca, y, output_folder)
    
    if args.linear or args.all:
        print("\n=== Running Linear Regression Analysis ===")
        if X_scaled is None or X_pca is None or y is None:
            # If PCA wasn't run but linear regression is requested
            X_scaled, y, _ = preprocess_data(data)
            _, _, explained_variance = perform_pca(X_scaled)
            cumulative_variance = np.cumsum(explained_variance)
            n_components_95 = np.argmax(cumulative_variance >= PCA_CONFIG['variance_threshold']) + 1
            X_pca, _, _ = perform_pca(X_scaled, n_components=n_components_95)
        
        # Run linear regression with or without cluster features
        run_linear_regression_analysis(X_scaled, X_pca, y, kmeans_labels, gmm_labels, output_folder=output_folder)
    
    if args.nn or args.all:
        if X_scaled is None or X_pca is None or y is None:
            # If PCA wasn't run but NN is requested
            X_scaled, y, _ = preprocess_data(data)
            _, _, explained_variance = perform_pca(X_scaled)
            cumulative_variance = np.cumsum(explained_variance)
            n_components_95 = np.argmax(cumulative_variance >= PCA_CONFIG['variance_threshold']) + 1
            X_pca, _, _ = perform_pca(X_scaled, n_components=n_components_95)
        
        run_neural_network_analysis(X_scaled, X_pca, y, kmeans_labels, gmm_labels, output_folder=output_folder)
    
    if args.compare or args.all:
        print("\n=== Running Model Comparison (Neural Network vs. Logistic Regression) ===")
        if X_scaled is None or X_pca is None or y is None:
            # If PCA wasn't run but comparison is requested
            X_scaled, y, _ = preprocess_data(data)
            _, _, explained_variance = perform_pca(X_scaled)
            cumulative_variance = np.cumsum(explained_variance)
            n_components_95 = np.argmax(cumulative_variance >= PCA_CONFIG['variance_threshold']) + 1
            X_pca, _, _ = perform_pca(X_scaled, n_components=n_components_95)
        
        # Prepare the cluster-enhanced features if clustering was performed
        X_with_clusters = None
        if kmeans_labels is not None and gmm_labels is not None:
            X_with_clusters = np.column_stack((X_scaled, kmeans_labels.reshape(-1, 1), gmm_labels.reshape(-1, 1)))
        
        # Run the comparison
        compare_models_with_linear(X_scaled, X_pca, y, X_with_clusters, output_folder=output_folder)
    
    print(f"\n=== Analysis Complete - All results saved to {output_folder} ===")

if __name__ == "__main__":
    main()