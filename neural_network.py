# neural_network.py - Neural network model implementations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def build_neural_network(input_dim):
    """Build a simple neural network model.
    
    Args:
        input_dim (int): Dimension of the input features
        
    Returns:
        tensorflow.keras.models.Sequential: Compiled neural network model
    """
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_and_evaluate_model(X, y, test_size=0.2, feature_type="original", output_folder=None):
    """Split the data, train the model, and evaluate its performance.
    
    Args:
        X (numpy.ndarray): Features
        y (numpy.ndarray or pandas.Series): Target variable
        test_size (float, optional): Proportion of data to use for testing. Defaults to 0.2.
        feature_type (str, optional): Type of features being used for visualization naming. 
                                    Defaults to "original".
        output_folder (str, optional): Folder to save visualizations. Defaults to None.
        
    Returns:
        tuple: (model, history, accuracy) - trained model, training history, test accuracy
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Build the model
    model = build_neural_network(X_train.shape[1])
    
    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Print classification report
    print(f"\nClassification Report ({feature_type} features):")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {feature_type.capitalize()} Features')
    
    if output_folder:
        plt.savefig(f"{output_folder}/confusion_matrix_{feature_type}.png")
    else:
        plt.savefig(f'confusion_matrix_{feature_type}.png')
    plt.close()
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model Accuracy - {feature_type.capitalize()} Features')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss - {feature_type.capitalize()} Features')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    if output_folder:
        plt.savefig(f"{output_folder}/training_history_{feature_type}.png")
    else:
        plt.savefig(f'training_history_{feature_type}.png')
    plt.close()
    
    return model, history, accuracy_score(y_test, y_pred)

def compare_models(X_scaled, X_pca, y, X_with_clusters=None, output_folder=None):
    """Compare models trained on different feature sets.
    
    Args:
        X_scaled (numpy.ndarray): Standardized original features
        X_pca (numpy.ndarray): PCA-transformed features
        y (numpy.ndarray or pandas.Series): Target variable
        X_with_clusters (numpy.ndarray, optional): Features enhanced with cluster assignments
        output_folder (str, optional): Folder to save visualizations. Defaults to None.
        
    Returns:
        dict: Dictionary of accuracy scores for each feature type
    """
    results = {}
    
    print("\n--- Training model with original features ---")
    _, _, acc_original = train_and_evaluate_model(X_scaled, y, feature_type="original", output_folder=output_folder)
    results['original'] = acc_original
    
    print("\n--- Training model with PCA features ---")
    _, _, acc_pca = train_and_evaluate_model(X_pca, y, feature_type="pca", output_folder=output_folder)
    results['pca'] = acc_pca
    
    if X_with_clusters is not None:
        print("\n--- Training model with cluster-enhanced features ---")
        _, _, acc_with_clusters = train_and_evaluate_model(X_with_clusters, y, feature_type="cluster_enhanced", output_folder=output_folder)
        results['cluster_enhanced'] = acc_with_clusters
    
    # Print comparison summary
    print("\n=== Model Performance Comparison ===")
    for feature_type, accuracy in results.items():
        print(f"Neural Network with {feature_type} features: {accuracy:.4f}")
    
    # Visualize comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results.keys(), results.values(), color=['blue', 'green', 'purple'])
    plt.ylim(0.5, 1.0)  # Adjust as needed based on your accuracy range
    plt.xlabel('Feature Type')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison Across Feature Types')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add accuracy values on top of each bar
    for i, (_, accuracy) in enumerate(results.items()):
        plt.text(i, accuracy + 0.01, f'{accuracy:.4f}', ha='center')
    
    plt.tight_layout()
    if output_folder:
        plt.savefig(f"{output_folder}/model_comparison.png")
    else:
        plt.savefig('model_comparison.png')
    plt.close()
    
    return results

def build_custom_neural_network(input_dim, hidden_layers=[64, 32], dropout_rates=[0.3, 0.2]):
    """Build a customizable neural network with variable architecture.
    
    Args:
        input_dim (int): Input dimension
        hidden_layers (list, optional): List of neurons in each hidden layer. Defaults to [64, 32].
        dropout_rates (list, optional): List of dropout rates after each hidden layer. Defaults to [0.3, 0.2].
        
    Returns:
        tensorflow.keras.models.Sequential: Compiled neural network model
    """
    model = Sequential()
    
    # Add input layer
    model.add(Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
    model.add(Dropout(dropout_rates[0]))
    
    # Add hidden layers
    for i in range(1, len(hidden_layers)):
        model.add(Dense(hidden_layers[i], activation='relu'))
        if i < len(dropout_rates):
            model.add(Dropout(dropout_rates[i]))
    
    # Add output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model