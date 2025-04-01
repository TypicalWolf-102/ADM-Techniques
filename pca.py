# pca_analysis.py - Principal Component Analysis functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def perform_pca(X_scaled, n_components=None):
    """Perform PCA and return the transformed data.
    
    Args:
        X_scaled (numpy.ndarray): Standardized features
        n_components (int, optional): Number of components to keep. Defaults to None (keep all).
        
    Returns:
        tuple: (X_pca, pca, explained_variance) - transformed data, PCA object, and explained variance
    """
    # Initialize PCA
    pca = PCA(n_components=n_components)
    
    # Fit and transform the data
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    
    return X_pca, pca, explained_variance

def visualize_pca(pca, explained_variance, output_folder=None):
    """Visualize PCA results and explained variance.
    
    Args:
        pca (sklearn.decomposition.PCA): Fitted PCA object
        explained_variance (numpy.ndarray): Explained variance ratios
        output_folder (str, optional): Folder to save visualizations. Defaults to None.
    """
    # Plot explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.8)
    plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', color='red')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Component')
    if output_folder:
        plt.savefig(f"{output_folder}/pca_explained_variance.png")
    else:
        plt.savefig('pca_explained_variance.png')
    plt.close()
    
    # Plot feature importance
    if hasattr(pca, 'components_'):
        plt.figure(figsize=(12, 8))
        
        # Get original feature names
        # Assuming we have 4 features: age, duration, euribor3m, nr.employed
        feature_names = ['age', 'duration', 'euribor3m', 'nr.employed']
        
        # Create components DataFrame with actual feature names
        components = pd.DataFrame(
            pca.components_,
            columns=feature_names
        )
        
        plt.imshow(components, cmap='viridis')
        plt.colorbar()
        plt.xticks(range(len(feature_names)), feature_names, rotation=90)
        plt.yticks(range(len(components.index)), [f'PC{i+1}' for i in range(len(components.index))])
        plt.title('PCA Components Heatmap')
        if output_folder:
            plt.savefig(f"{output_folder}/pca_components_heatmap.png")
        else:
            plt.savefig('pca_components_heatmap.png')
        plt.close()

def pca_feature_importance(pca, feature_names, output_folder=None):
    """Calculate and visualize feature importance in PCA components.
    
    Args:
        pca (sklearn.decomposition.PCA): Fitted PCA object
        feature_names (list): List of feature names
        output_folder (str, optional): Folder to save visualizations. Defaults to None.
        
    Returns:
        pandas.DataFrame: DataFrame with feature importance for each component
    """
    # Get absolute value of component loadings
    loadings = pd.DataFrame(
        np.abs(pca.components_),
        columns=feature_names
    )
    
    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
    })
    
    # Calculate importance for each principal component
    for i in range(pca.n_components_):
        importance_df[f'PC{i+1}'] = loadings.iloc[i]
    
    # Calculate overall importance across all components
    importance_df['Overall'] = importance_df.iloc[:, 1:].mean(axis=1)
    
    # Sort by overall importance
    importance_df = importance_df.sort_values('Overall', ascending=False)
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    importance_df.set_index('Feature')['Overall'].plot(kind='bar')
    plt.title('Overall Feature Importance in PCA')
    plt.ylabel('Average Absolute Loading')
    plt.tight_layout()
    if output_folder:
        plt.savefig(f"{output_folder}/pca_feature_importance.png")
    else:
        plt.savefig('pca_feature_importance.png')
    plt.close()
    
    return importance_df