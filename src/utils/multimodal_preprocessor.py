import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.tabular_preprocessor import FraudDataPreprocessor
from src.utils.image_preprocessor import QRCodePreprocessor


class MultimodalDataPreprocessor:

    def __init__(self, image_size=(128, 128)):
        self.fraud_preprocessor = FraudDataPreprocessor()
        self.qr_preprocessor = QRCodePreprocessor(image_size=image_size)
        
    def generate_synthetic_multimodal_data(self, n_samples=5000, fraud_ratio=0.1):
        # Generate tabular data
        tabular_data = self.fraud_preprocessor.generate_synthetic_data(
            n_samples=n_samples, fraud_ratio=fraud_ratio
        )
        
        # Vectorized image generation
        # Determine which samples get malicious QR codes based on fraud status
        fraud_labels = tabular_data['isFraud'].values
        malicious_probs = np.where(fraud_labels == 1, 0.8, 0.1)
        is_malicious = np.random.random(n_samples) < malicious_probs
        
        # Generate all images (batch processing for efficiency)
        images = np.array([
            self.qr_preprocessor._generate_qr_pattern(malicious=mal)
            for mal in is_malicious
        ], dtype=np.float32) / 255.0
        
        return {
            'tabular': tabular_data,
            'images': images,
            'labels': fraud_labels
        }
    
    def preprocess_multimodal_data(self, tabular_data, images, fit=True):
        X_tabular, y = self.fraud_preprocessor.preprocess_data(tabular_data, fit=fit)
        
        return {
            'tabular': X_tabular,
            'images': images,
            'labels': y
        }
    
    def prepare_train_test_data(self, test_size=0.2, random_state=42, n_samples=5000):
        # Generate synthetic multimodal data
        data = self.generate_synthetic_multimodal_data(n_samples=n_samples)
        
        # Create indices for splitting
        indices = np.arange(len(data['labels']))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=data['labels']
        )
        
        # Split tabular data
        train_tabular = data['tabular'].iloc[train_indices].reset_index(drop=True)
        test_tabular = data['tabular'].iloc[test_indices].reset_index(drop=True)
        
        # Preprocess tabular data
        X_train_tabular, y_train = self.fraud_preprocessor.preprocess_data(train_tabular, fit=True)
        X_test_tabular, y_test = self.fraud_preprocessor.preprocess_data(test_tabular, fit=False)
        
        # Split and get images
        X_train_images = data['images'][train_indices]
        X_test_images = data['images'][test_indices]
        
        return {
            'X_train_tabular': X_train_tabular,
            'X_train_images': X_train_images,
            'y_train': y_train,
            'X_test_tabular': X_test_tabular,
            'X_test_images': X_test_images,
            'y_test': y_test
        }
    
    def save_preprocessors(self, filepath):
        self.fraud_preprocessor.save_scaler(filepath)
    
    def load_preprocessors(self, filepath):
        self.fraud_preprocessor.load_scaler(filepath)
