import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os


class FraudDataPreprocessor:

    def __init__(self):
        """Initialize the preprocessor with default scalers and encoders."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def generate_synthetic_data(self, n_samples=10000, fraud_ratio=0.02):
        np.random.seed(42)
        
        n_fraud = int(n_samples * fraud_ratio)
        n_normal = n_samples - n_fraud
        
        # Transaction types
        transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
        
        # Generate normal transactions
        normal_data = {
            'step': np.random.randint(1, 744, size=n_normal),  # Hour of simulation
            'type': np.random.choice(transaction_types, size=n_normal, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
            'amount': np.random.exponential(scale=100, size=n_normal),
            'nameOrig': [f'C{np.random.randint(1000000, 9999999)}' for _ in range(n_normal)],
            'oldbalanceOrg': np.abs(np.random.normal(50000, 30000, size=n_normal)),
            'newbalanceOrig': np.abs(np.random.normal(49000, 30000, size=n_normal)),
            'nameDest': [f'M{np.random.randint(1000000, 9999999)}' if np.random.random() > 0.5 
                        else f'C{np.random.randint(1000000, 9999999)}' for _ in range(n_normal)],
            'oldbalanceDest': np.abs(np.random.normal(100000, 50000, size=n_normal)),
            'newbalanceDest': np.abs(np.random.normal(101000, 50000, size=n_normal)),
            'isFlaggedFraud': np.zeros(n_normal, dtype=int),
        }
        
        # Generate fraudulent transactions (typically TRANSFER or CASH_OUT with larger amounts)
        fraud_data = {
            'step': np.random.randint(1, 744, size=n_fraud),
            'type': np.random.choice(['TRANSFER', 'CASH_OUT'], size=n_fraud, p=[0.5, 0.5]),
            'amount': np.random.exponential(scale=50000, size=n_fraud),  # Higher amounts
            'nameOrig': [f'C{np.random.randint(1000000, 9999999)}' for _ in range(n_fraud)],
            'oldbalanceOrg': np.abs(np.random.normal(100000, 50000, size=n_fraud)),
            'newbalanceOrig': np.abs(np.random.normal(10000, 10000, size=n_fraud)),  # Much lower after fraud
            'nameDest': [f'C{np.random.randint(1000000, 9999999)}' for _ in range(n_fraud)],
            'oldbalanceDest': np.abs(np.random.normal(50000, 30000, size=n_fraud)),
            'newbalanceDest': np.abs(np.random.normal(150000, 50000, size=n_fraud)),  # Higher after fraud
            'isFlaggedFraud': np.random.choice([0, 1], size=n_fraud, p=[0.9, 0.1]),  # Rarely flagged
        }
        
        # Create DataFrames
        normal_df = pd.DataFrame(normal_data)
        normal_df['isFraud'] = 0
        
        fraud_df = pd.DataFrame(fraud_data)
        fraud_df['isFraud'] = 1
        
        # Combine and shuffle
        df = pd.concat([normal_df, fraud_df], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def load_from_csv(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Load the CSV
        df = pd.read_csv(csv_path)
        
        # Standardize the fraud label column name
        # Handle both 'isFraud' and 'is_fraud' naming conventions
        if 'is_fraud' in df.columns and 'isFraud' not in df.columns:
            df = df.rename(columns={'is_fraud': 'isFraud'})
        
        # Verify required columns exist
        required_cols = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
                        'newbalanceOrig', 'nameDest', 'oldbalanceDest', 
                        'newbalanceDest', 'isFraud']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing expected columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
        
        return df
    
    def preprocess_data(self, data, fit=True):
        # Copy data to avoid modifying original
        df = data.copy()
        
        # Separate features and labels
        if 'isFraud' in df.columns:
            y = df['isFraud'].values
            df = df.drop('isFraud', axis=1)
        else:
            y = None
        
        # Encode categorical features
        categorical_cols = ['type']
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Drop ID columns (nameOrig, nameDest) - not useful for prediction
        id_cols = ['nameOrig', 'nameDest']
        for col in id_cols:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        # Only set feature names when fitting (training)
        if fit:
            self.feature_names = df.columns.tolist()
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(df)
        else:
            X_scaled = self.scaler.transform(df)
        
        # Reshape for transformer input (batch_size, sequence_length=1, num_features)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        return X_scaled, y
    
    def save_scaler(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_dict = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Scaler and encoders saved to {filepath}")
    
    def load_scaler(self, filepath):
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        self.scaler = save_dict.get('scaler', self.scaler)
        self.label_encoders = save_dict.get('label_encoders', {})
        self.feature_names = save_dict.get('feature_names', None)
        print(f"Scaler and encoders loaded from {filepath}")
    
    def prepare_train_test_data(self, test_size=0.2, random_state=42):
        # Generate synthetic data
        data = self.generate_synthetic_data()
        
        # Split into train and test
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state, stratify=data['isFraud']
        )
        
        # Preprocess
        X_train, y_train = self.preprocess_data(train_data, fit=True)
        X_test, y_test = self.preprocess_data(test_data, fit=False)
        
        return X_train, X_test, y_train, y_test
