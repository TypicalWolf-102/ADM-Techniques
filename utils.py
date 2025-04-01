# utils.py - Utility functions for data loading and exploration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_next_test_folder():
    """Create a new numbered test folder.
    
    Returns:
        str: Path to the new test folder
    """
    i = 1
    while True:
        folder_name = f"test_{i}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            return folder_name
        i += 1

def load_data(file_path):
    """Load and prepare the bank marketing dataset.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded and cleaned dataframe, or None if error
    """
    try:
        # Read CSV with additional options to handle potential issues
        df = pd.read_csv(file_path, skipinitialspace=True)
        
        # Clean up column names - strip whitespace
        df.columns = df.columns.str.strip()
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def explore_data(df, output_folder=None):
    """Explore the dataset to understand its characteristics.
    
    Args:
        df (pandas.DataFrame): The dataset to explore
        output_folder (str, optional): Folder to save visualizations. Defaults to None.
    """
    print("\n--- Data Exploration ---")
    print(f"Dataset Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    print("\nClass Distribution:")
    if 'y' in df.columns:
        print(df['y'].value_counts(normalize=True) * 100)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing Values:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values found.")
    
    # Create basic visualizations if the DataFrame is not too large
    if df.shape[0] <= 10000:  # Only for reasonably sized datasets
        # Correlation heatmap for numerical features
        plt.figure(figsize=(10, 8))
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            if output_folder:
                plt.savefig(f"{output_folder}/feature_correlation.png")
            else:
                plt.savefig('feature_correlation.png')
            plt.close()
            
        # Distribution of target variable if it exists
        if 'y' in df.columns:
            plt.figure(figsize=(8, 6))
            sns.countplot(x='y', data=df)
            plt.title('Distribution of Target Variable')
            plt.tight_layout()
            if output_folder:
                plt.savefig(f"{output_folder}/target_distribution.png")
            else:
                plt.savefig('target_distribution.png')
            plt.close()