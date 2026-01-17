"""
Multimodal data preprocessor for combined tabular and image data.

This module provides the MultimodalDataPreprocessor class that combines
tabular transaction data with QR code images for multimodal fraud detection.
"""
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.tabular_preprocessor import FraudDataPreprocessor
from src.utils.image_preprocessor import QRCodePreprocessor
from config.config import DATA_DIR, QRCODE_DATASET_PATH


class MultimodalDataPreprocessor:
    """
    Combined Preprocessor for Multimodal Fraud Detection
    
    This class combines tabular transaction data preprocessing with QR code
    image preprocessing to create multimodal inputs for the fraud detection model.
    
    The multimodal approach simulates a scenario where each transaction has
    an associated QR code that was scanned during the transaction.
    
    Attributes:
        fraud_preprocessor: FraudDataPreprocessor for tabular data
        qr_preprocessor: QRCodePreprocessor for image data
    """
    
    def __init__(self, image_size=(128, 128)):
        """
        Initialize the multimodal preprocessor.
        
        Args:
            image_size: Target size (height, width) for QR code images
        """
        self.fraud_preprocessor = FraudDataPreprocessor()
        self.qr_preprocessor = QRCodePreprocessor(image_size=image_size)
        
    def generate_synthetic_multimodal_data(self, n_samples=5000, fraud_ratio=0.1):
        """
        Generate synthetic multimodal data combining tabular and image data
        
        This simulates a scenario where each transaction also has an associated 
        QR code image that was scanned.
        
        For fraudulent transactions, there's a higher probability (80%) that the
        associated QR code is malicious. For normal transactions, there's a lower
        probability (10%) of having a malicious QR code.
        
        Args:
            n_samples: Number of samples to generate
            fraud_ratio: Ratio of fraudulent transactions
            
        Returns:
            Dictionary with 'tabular', 'images', and 'labels' keys
        """
        print(f"\n{'='*70}")
        print(f"MULTIMODAL DATA GENERATION")
        print(f"{'='*70}")
        print(f"CSV Data Directory: {DATA_DIR}")
        print(f"Image Data Directory: {QRCODE_DATASET_PATH}")
        print(f"{'='*70}")
        
        # Generate tabular data
        tabular_data = self.fraud_preprocessor.generate_synthetic_data(
            n_samples=n_samples, fraud_ratio=fraud_ratio
        )
        
        # Display image generation info
        print(f"\n[Image Processing] Generating QR code images...")
        print(f"[Image Processing] Total images to generate: {n_samples}")
        
        # Vectorized image generation
        # Determine which samples get malicious QR codes based on fraud status
        fraud_labels = tabular_data['isFraud'].values
        malicious_probs = np.where(fraud_labels == 1, 0.8, 0.1)
        is_malicious = np.random.random(n_samples) < malicious_probs
        
        # Count benign vs malicious
        n_malicious = np.sum(is_malicious)
        n_benign = n_samples - n_malicious
        print(f"[Image Processing] - Benign images: {n_benign}")
        print(f"[Image Processing] - Malicious images: {n_malicious}")
        
        # Generate all images (batch processing for efficiency)
        images = np.array([
            self.qr_preprocessor._generate_qr_pattern(malicious=mal)
            for mal in is_malicious
        ], dtype=np.float32) / 255.0
        
        print(f"[Image Processing] ✓ Successfully generated {n_samples} images")
        print(f"[Image Processing] Image shape: {images.shape}")
        
        print(f"\n{'='*70}")
        print(f"DATA GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total samples: {n_samples}")
        print(f"CSV records: {len(tabular_data)}")
        print(f"Images: {len(images)}")
        print(f"{'='*70}\n")
        
        return {
            'tabular': tabular_data,
            'images': images,
            'labels': fraud_labels
        }
    
    def preprocess_multimodal_data(self, tabular_data, images, fit=True):
        """
        Preprocess multimodal data
        
        Processes tabular data through the fraud preprocessor while
        keeping images as-is (already normalized).
        
        Args:
            tabular_data: DataFrame with tabular features
            images: Array of images (normalized)
            fit: Whether to fit the preprocessors
            
        Returns:
            Dictionary with preprocessed 'tabular', 'images', and 'labels'
        """
        X_tabular, y = self.fraud_preprocessor.preprocess_data(tabular_data, fit=fit)
        
        return {
            'tabular': X_tabular,
            'images': images,
            'labels': y
        }
    
    def prepare_train_test_data(self, test_size=0.2, random_state=42, n_samples=5000):
        """
        Generate and prepare multimodal train/test data
        
        Creates stratified train/test splits maintaining the fraud ratio
        in both sets.
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary with train/test splits for both modalities:
            - X_train_tabular: Training tabular features
            - X_train_images: Training images
            - y_train: Training labels
            - X_test_tabular: Test tabular features
            - X_test_images: Test images
            - y_test: Test labels
        """
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
        
        print(f"\n[Data Preprocessing] Processing CSV data for training...")
        print(f"[Data Preprocessing] Training CSV records: {len(train_tabular)}")
        print(f"[Data Preprocessing] Testing CSV records: {len(test_tabular)}")
        
        # Preprocess tabular data
        X_train_tabular, y_train = self.fraud_preprocessor.preprocess_data(train_tabular, fit=True)
        X_test_tabular, y_test = self.fraud_preprocessor.preprocess_data(test_tabular, fit=False)
        
        # Split and get images
        X_train_images = data['images'][train_indices]
        X_test_images = data['images'][test_indices]
        
        print(f"[Data Preprocessing] ✓ Training images prepared: {len(X_train_images)}")
        print(f"[Data Preprocessing] ✓ Testing images prepared: {len(X_test_images)}")
        
        return {
            'X_train_tabular': X_train_tabular,
            'X_train_images': X_train_images,
            'y_train': y_train,
            'X_test_tabular': X_test_tabular,
            'X_test_images': X_test_images,
            'y_test': y_test
        }
    
    def save_preprocessors(self, filepath):
        """
        Save all preprocessors to disk
        
        Args:
            filepath: Path to save the preprocessor state
        """
        self.fraud_preprocessor.save_scaler(filepath)
    
    def load_preprocessors(self, filepath):
        """
        Load all preprocessors from disk
        
        Args:
            filepath: Path to load the preprocessor state from
        """
        self.fraud_preprocessor.load_scaler(filepath)
