# gbm.py - Gradient Boosting Machine models implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

def build_gbm_model(random_state=42):
    """Build a default Gradient Boosting Machine model.
    
    Args:
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        
    Returns:
        sklearn.ensemble.GradientBoostingClassifier: Default GBM model
    """
    # Create a GBM model with default parameters
    gbm = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=random_state
    )
    
    return gbm

def train_and_evaluate_gbm(X, y, test_size=0.2, feature_type="original", output_folder=None):
    """Train and evaluate a Gradient Boosting Machine model.
    
    Args:
        X (numpy.ndarray): Features
        y (numpy.ndarray or pandas.Series): Target variable
        test_size (float, optional): Proportion of data for testing. Defaults to 0.2.
        feature_type (str, optional): Type of features being used. Defaults to "original".
        output_folder (str, optional): Folder to save results. Defaults to None.
        
    Returns:
        tuple: (model, accuracy) - trained model and test accuracy
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=42, stratify=y)
    
    # Build the model
    gbm = build_gbm_model()
    
    # Train the model
    print(f"\nTraining GBM model with {feature_type} features...")
    gbm.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = gbm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"GBM Accuracy ({feature_type} features): {accuracy:.4f}")
    
    # Print classification report
    print(f"\nGBM Classification Report ({feature_type} features):")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'GBM Confusion Matrix - {feature_type.capitalize()} Features')
    
    if output_folder:
        plt.savefig(f"{output_folder}/gbm_confusion_matrix_{feature_type}.png")
    else:
        plt.savefig(f'gbm_confusion_matrix_{feature_type}.png')
    plt.close()
    
    # Feature importance
    if feature_type != 'pca':  # Feature importance makes more sense for original features
        plot_feature_importance(gbm, X, feature_type, output_folder)
    
    # ROC curve
    y_pred_prob = gbm.predict_proba(X_test)[:,1]
    plot_roc_curve(y_test, y_pred_prob, feature_type, output_folder)
    
    return gbm, accuracy

def tune_gbm_hyperparameters(X, y, param_grid=None, cv=5, feature_type="original", output_folder=None):
    """Tune hyperparameters for GBM using GridSearchCV.
    
    Args:
        X (numpy.ndarray): Features
        y (numpy.ndarray or pandas.Series): Target variable
        param_grid (dict, optional): Parameter grid for search. If None, uses default grid.
        cv (int, optional): Number of cross-validation folds. Defaults to 5.
        feature_type (str, optional): Type of features being used. Defaults to "original".
        output_folder (str, optional): Folder to save results. Defaults to None.
        
    Returns:
        tuple: (best_model, best_params) - best model and parameters
    """
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 1.0]
        }
    
    # Base model
    gbm = GradientBoostingClassifier(random_state=42)
    
    # Grid search with cross-validation
    print(f"\nTuning GBM hyperparameters with {feature_type} features...")
    grid_search = GridSearchCV(
        estimator=gbm,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X, y)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"Best parameters found: {best_params}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    # Visualize parameter importance
    plot_parameter_importance(grid_search, feature_type, output_folder)
    
    return best_model, best_params

def plot_feature_importance(gbm, X, feature_type="original", output_folder=None):
    """Plot feature importance from the GBM model.
    
    Args:
        gbm (sklearn.ensemble.GradientBoostingClassifier): Trained GBM model
        X (numpy.ndarray): Features used for training
        feature_type (str, optional): Type of features being used. Defaults to "original".
        output_folder (str, optional): Folder to save results. Defaults to None.
    """
    # Get feature importance
    feature_importance = gbm.feature_importances_
    
    # If X is a DataFrame, use column names; otherwise, create generic feature names
    if hasattr(X, 'columns'):
        feature_names = X.columns
    else:
        if feature_type == 'cluster_enhanced':
            # For cluster-enhanced features, we need to account for the additional cluster columns
            base_features = ['age', 'duration', 'euribor3m', 'nr.employed']
            feature_names = base_features + ['kmeans_cluster', 'gmm_cluster']
        else:
            # Get basic feature names from config or create generic ones
            feature_names = ['age', 'duration', 'euribor3m', 'nr.employed']
        
        # If the number of features doesn't match, create generic names
        if len(feature_names) != len(feature_importance):
            feature_names = [f'Feature {i+1}' for i in range(len(feature_importance))]
    
    # Create DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'GBM Feature Importance - {feature_type.capitalize()} Features')
    plt.tight_layout()
    
    if output_folder:
        plt.savefig(f"{output_folder}/gbm_feature_importance_{feature_type}.png")
    else:
        plt.savefig(f'gbm_feature_importance_{feature_type}.png')
    plt.close()

def plot_roc_curve(y_true, y_pred_prob, feature_type="original", output_folder=None):
    """Plot ROC curve for the model.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred_prob (numpy.ndarray): Predicted probabilities
        feature_type (str, optional): Type of features being used. Defaults to "original".
        output_folder (str, optional): Folder to save results. Defaults to None.
    """
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'GBM ROC Curve - {feature_type.capitalize()} Features')
    plt.legend(loc="lower right")
    
    if output_folder:
        plt.savefig(f"{output_folder}/gbm_roc_curve_{feature_type}.png")
    else:
        plt.savefig(f'gbm_roc_curve_{feature_type}.png')
    plt.close()

def plot_parameter_importance(grid_search, feature_type="original", output_folder=None):
    """Visualize parameter importance from grid search results.
    
    Args:
        grid_search (sklearn.model_selection.GridSearchCV): Completed grid search
        feature_type (str, optional): Type of features being used. Defaults to "original".
        output_folder (str, optional): Folder to save results. Defaults to None.
    """
    # Get results from grid search
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Extract the parameter columns
    param_cols = [col for col in results.columns if col.startswith('param_')]
    
    # Plot each parameter's effect on the mean test score
    plt.figure(figsize=(15, 10))
    
    for i, param in enumerate(param_cols):
        param_name = param.replace('param_', '')
        plt.subplot(2, 3, i+1)
        
        # Convert parameter values to strings for categorical plotting
        param_values = results[param].astype(str)
        
        # Group by parameter value and calculate mean score
        grouped = results.groupby(param)[['mean_test_score']].mean()
        
        # Plot
        sns.barplot(x=grouped.index.astype(str), y='mean_test_score', data=grouped.reset_index())
        plt.title(f'Effect of {param_name}')
        plt.ylabel('Mean CV Score')
        plt.xlabel(param_name)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if output_folder:
        plt.savefig(f"{output_folder}/gbm_parameter_effects_{feature_type}.png")
    else:
        plt.savefig(f'gbm_parameter_effects_{feature_type}.png')
    plt.close()

def compare_gbm_with_other_models(model_results, feature_types, output_folder=None):
    """Compare GBM performance with other models.
    
    Args:
        model_results (dict): Dictionary with model performances
        feature_types (list): List of feature types
        output_folder (str, optional): Folder to save results. Defaults to None.
    """
    # Prepare data for comparison
    models = list(model_results.keys())
    
    # Create a figure for each feature type
    for feature_type in feature_types:
        plt.figure(figsize=(12, 6))
        
        # Extract accuracies for this feature type
        accuracies = [model_results[model].get(feature_type, None) for model in models]
        
        # Filter out None values
        valid_models = [model for i, model in enumerate(models) if accuracies[i] is not None]
        valid_accuracies = [acc for acc in accuracies if acc is not None]
        
        # Create bar chart
        bars = plt.bar(valid_models, valid_accuracies, color=['blue', 'green', 'purple', 'orange'])
        plt.ylim(0.5, 1.0)  # Adjust as needed
        plt.xlabel('Model Type')
        plt.ylabel('Accuracy')
        plt.title(f'Model Performance Comparison - {feature_type.capitalize()} Features')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add accuracy values on top of each bar
        for i, (model, accuracy) in enumerate(zip(valid_models, valid_accuracies)):
            plt.text(i, accuracy + 0.01, f'{accuracy:.4f}', ha='center')
        
        plt.tight_layout()
        
        if output_folder:
            plt.savefig(f"{output_folder}/model_comparison_{feature_type}.png")
        else:
            plt.savefig(f'model_comparison_{feature_type}.png')
        plt.close()

def save_gbm_model(model, filename):
    """Save the trained GBM model to disk.
    
    Args:
        model (sklearn.ensemble.GradientBoostingClassifier): Trained GBM model
        filename (str): Filename to save the model
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the model
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_gbm_model(filename):
    """Load a trained GBM model from disk.
    
    Args:
        filename (str): Filename of the saved model
        
    Returns:
        sklearn.ensemble.GradientBoostingClassifier: Loaded GBM model
    """
    try:
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None