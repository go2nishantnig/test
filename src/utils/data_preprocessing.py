"""
Data preprocessing utilities for fraud detection
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os


class FraudDataPreprocessor:
    """Preprocessor for fraud detection data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def generate_synthetic_data(self, n_samples=10000, fraud_ratio=0.02):
        """
        Generate synthetic fraud transaction data for demonstration
        
        Args:
            n_samples: Number of samples to generate
            fraud_ratio: Ratio of fraudulent transactions
            
        Returns:
            DataFrame with transaction features and labels
        """
        np.random.seed(42)
        
        n_fraud = int(n_samples * fraud_ratio)
        n_normal = n_samples - n_fraud
        
        # Generate normal transactions
        normal_data = {
            'amount': np.random.exponential(scale=100, size=n_normal),
            'time': np.random.uniform(0, 86400, size=n_normal),
            'distance_from_home': np.random.exponential(scale=10, size=n_normal),
            'distance_from_last_transaction': np.random.exponential(scale=5, size=n_normal),
            'ratio_to_median_purchase_price': np.random.normal(1, 0.3, size=n_normal),
            'repeat_retailer': np.random.choice([0, 1], size=n_normal, p=[0.3, 0.7]),
            'used_chip': np.random.choice([0, 1], size=n_normal, p=[0.1, 0.9]),
            'used_pin_number': np.random.choice([0, 1], size=n_normal, p=[0.2, 0.8]),
            'online_order': np.random.choice([0, 1], size=n_normal, p=[0.4, 0.6]),
        }
        
        # Generate fraudulent transactions (with different patterns)
        fraud_data = {
            'amount': np.random.exponential(scale=500, size=n_fraud),
            'time': np.random.uniform(0, 86400, size=n_fraud),
            'distance_from_home': np.random.exponential(scale=50, size=n_fraud),
            'distance_from_last_transaction': np.random.exponential(scale=30, size=n_fraud),
            'ratio_to_median_purchase_price': np.random.normal(3, 1, size=n_fraud),
            'repeat_retailer': np.random.choice([0, 1], size=n_fraud, p=[0.8, 0.2]),
            'used_chip': np.random.choice([0, 1], size=n_fraud, p=[0.6, 0.4]),
            'used_pin_number': np.random.choice([0, 1], size=n_fraud, p=[0.7, 0.3]),
            'online_order': np.random.choice([0, 1], size=n_fraud, p=[0.2, 0.8]),
        }
        
        # Additional synthetic features
        for i in range(9, 30):
            normal_data[f'feature_{i}'] = np.random.normal(0, 1, size=n_normal)
            fraud_data[f'feature_{i}'] = np.random.normal(0.5, 1.5, size=n_fraud)
        
        # Create DataFrames
        normal_df = pd.DataFrame(normal_data)
        normal_df['is_fraud'] = 0
        
        fraud_df = pd.DataFrame(fraud_data)
        fraud_df['is_fraud'] = 1
        
        # Combine and shuffle
        df = pd.concat([normal_df, fraud_df], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def preprocess_data(self, data, fit=True):
        """
        Preprocess the data
        
        Args:
            data: DataFrame with features and labels
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Tuple of (X, y) where X is scaled features and y is labels
        """
        # Separate features and labels
        if 'is_fraud' in data.columns:
            y = data['is_fraud'].values
            X = data.drop('is_fraud', axis=1)
        else:
            y = None
            X = data
        
        # Only set feature names when fitting (training)
        if fit:
            self.feature_names = X.columns.tolist()
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Reshape for transformer input (batch_size, sequence_length=1, num_features)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        return X_scaled, y
    
    def save_scaler(self, filepath):
        """Save the scaler to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath):
        """Load the scaler from disk"""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Scaler loaded from {filepath}")
    
    def prepare_train_test_data(self, test_size=0.2, random_state=42):
        """
        Generate and prepare train/test data
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Generate synthetic data
        data = self.generate_synthetic_data()
        
        # Split into train and test
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state, stratify=data['is_fraud']
        )
        
        # Preprocess
        X_train, y_train = self.preprocess_data(train_data, fit=True)
        X_test, y_test = self.preprocess_data(test_data, fit=False)
        
        return X_train, X_test, y_train, y_test
