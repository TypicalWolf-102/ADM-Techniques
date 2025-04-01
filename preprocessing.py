# preprocessing.py - Data preprocessing functions

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """Preprocess the data for analysis and modeling.
    
    Args:
        df (pandas.DataFrame): The raw dataframe to preprocess
        
    Returns:
        tuple: (X_scaled, y, scaler) - scaled features, target variable, and scaler object
    """
    # Make a copy of the dataframe
    data = df.copy()
    
    # Fix column name issue - the target column has a space at the end
    # Rename the column to remove the trailing space
    if 'y ' in data.columns:
        data = data.rename(columns={'y ': 'y'})
    
    # Convert target to binary numeric
    data['y'] = data['y'].map({'yes': 1, 'no': 0})
    
    # Split features and target
    X = data.drop('y', axis=1)
    y = data['y']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def handle_categorical_features(df, categorical_cols):
    """Handle categorical features with one-hot encoding.
    
    Args:
        df (pandas.DataFrame): The dataframe to process
        categorical_cols (list): List of categorical column names
        
    Returns:
        pandas.DataFrame: DataFrame with one-hot encoded categorical features
    """
    # This function is prepared for future extension if categorical features are added
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)

def handle_missing_values(df, numeric_strategy='mean', categorical_strategy='most_frequent'):
    """Handle missing values in the dataframe.
    
    Args:
        df (pandas.DataFrame): The dataframe with missing values
        numeric_strategy (str): Strategy for numeric columns ('mean', 'median', 'mode')
        categorical_strategy (str): Strategy for categorical columns ('most_frequent', 'constant')
        
    Returns:
        pandas.DataFrame: DataFrame with filled missing values
    """
    # This function is prepared for future extension if missing values need to be handled
    data = df.copy()
    
    # Handle numeric features
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].isnull().sum() > 0:
            if numeric_strategy == 'mean':
                data[col].fillna(data[col].mean(), inplace=True)
            elif numeric_strategy == 'median':
                data[col].fillna(data[col].median(), inplace=True)
            elif numeric_strategy == 'mode':
                data[col].fillna(data[col].mode()[0], inplace=True)
    
    # Handle categorical features
    cat_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if data[col].isnull().sum() > 0:
            if categorical_strategy == 'most_frequent':
                data[col].fillna(data[col].mode()[0], inplace=True)
            elif categorical_strategy == 'constant':
                data[col].fillna('unknown', inplace=True)
    
    return data
