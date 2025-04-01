# config.py - Configuration settings for the analysis

# Data configuration
DATA_CONFIG = {
    'file_path': 'processed_bank_data.csv',
    'target_column': 'y',  # Column name for the target variable
    'random_state': 42,    # Random seed for reproducibility
}

# Feature names (used for PCA visualization)
FEATURE_NAMES = ['age', 'duration', 'euribor3m', 'nr.employed']

# PCA configuration
PCA_CONFIG = {
    'max_components': None,  # None means keep all components
    'variance_threshold': 0.95,  # Threshold for explained variance
}

# Clustering configuration
CLUSTERING_CONFIG = {
    'kmeans': {
        'max_clusters': 10,
        'n_init': 10,
        'random_state': 42,
        'method': 'elbow',  # Only using the elbow method for determining optimal K
    },
    'gmm': {
        'max_components': 10,
        'n_init': 5,
        'random_state': 42,
        'method': 'bic',    # Only using BIC for determining optimal components
    }
}

# Linear Regression configuration
LINEAR_REG_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'regularization': {
        'ridge_alpha': 1.0,
        'lasso_alpha': 0.1,
        'elasticnet_alpha': 0.1,
        'elasticnet_l1_ratio': 0.5
    },
    'cross_validation': {
        'cv_folds': 5,
        'scoring': 'r2'
    }
}

# Neural Network configuration
NN_CONFIG = {
    'hidden_layers': [64, 32, 16],  # Neurons in each hidden layer
    'dropout_rates': [0.3, 0.2, 0.1],  # Dropout rates for each layer
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 100,
    'validation_split': 0.2,
    'test_size': 0.2,
    'early_stopping_patience': 10,
}

# GBM configuration
GBM_CONFIG = {
    'default_params': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'subsample': 0.8,
        'random_state': 42
    },
    'grid_search': {
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 1.0]
        },
        'cv': 5,
        'n_jobs': -1
    },
    'test_size': 0.2
}

# Visualization configuration
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 100,
    'save_path': './',  # Path to save visualizations
    'cmap': 'viridis',  # Default colormap
}

# Output configuration
OUTPUT_CONFIG = {
    'save_models': True,
    'model_path': './models/',
    'results_path': './results/',
}