# __init__.py - Package initialization file

# Import key functions from each module
from .utils import load_data, explore_data
from .preprocessing import preprocess_data, handle_categorical_features, handle_missing_values
from .pca_analysis import perform_pca, visualize_pca, pca_feature_importance
from .clustering import perform_kmeans, perform_gmm, visualize_clusters, evaluate_clusters
from .neural_network import (
    build_neural_network, 
    train_and_evaluate_model, 
    compare_models,
    build_custom_neural_network
)
from .gbm import (
    build_gbm_model,
    train_and_evaluate_gbm,
    tune_gbm_hyperparameters,
    compare_gbm_with_other_models,
    save_gbm_model,
    load_gbm_model
)

# Version information
__version__ = '1.0.0'