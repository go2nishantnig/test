"""
Multimodal data preprocessor for combined tabular and image data.

This module provides the MultimodalDataPreprocessor class that combines
tabular transaction data with QR code images for multimodal fraud detection.
"""
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

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
    
    def load_all_csv_data(self, csv_dir=None):
        """
        Load all CSV files from the CSV directory.
        
        Args:
            csv_dir: Directory containing CSV files (default: DATA_DIR from config)
            
        Returns:
            DataFrame with all CSV data combined
        """
        if csv_dir is None:
            csv_dir = DATA_DIR
        
        print(f"\n{'='*70}")
        print(f"LOADING ACTUAL CSV DATA")
        print(f"{'='*70}")
        print(f"CSV Data Directory: {csv_dir}")
        
        if not os.path.exists(csv_dir):
            print(f"[CSV Data Processing] ✗ CSV directory not found: {csv_dir}")
            raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
        
        # Find all CSV files in the directory
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"[CSV Data Processing] ✗ No CSV files found in: {csv_dir}")
            raise FileNotFoundError(f"No CSV files found in: {csv_dir}")
        
        print(f"[CSV Data Processing] Found {len(csv_files)} CSV file(s)")
        for csv_file in csv_files:
            print(f"  - {csv_file}")
        
        # Load and combine all CSV files
        all_dataframes = []
        total_records = 0
        
        for csv_file in csv_files:
            csv_path = os.path.join(csv_dir, csv_file)
            df = self.fraud_preprocessor.load_from_csv(csv_path)
            all_dataframes.append(df)
            total_records += len(df)
            print(f"[CSV Data Processing]   Loaded {len(df)} records from {csv_file}")
        
        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        print(f"\n[CSV Data Processing] ✓ Total CSV records loaded: {total_records}")
        print(f"[CSV Data Processing] Combined DataFrame shape: {combined_df.shape}")
        
        if 'isFraud' in combined_df.columns:
            fraud_count = combined_df['isFraud'].sum()
            fraud_ratio = fraud_count / len(combined_df)
            print(f"[CSV Data Processing] Fraud records: {fraud_count} ({fraud_ratio:.2%})")
            print(f"[CSV Data Processing] Normal records: {len(combined_df) - fraud_count} ({1-fraud_ratio:.2%})")
        
        print(f"{'='*70}\n")
        
        return combined_df
    
    def load_all_actual_data(self, csv_dir=None, image_dir=None):
        """
        Load all actual CSV and image data from directories.
        
        This method loads all CSV files from csv_dir and all images from image_dir,
        then pairs them for multimodal training. If there's a mismatch in counts,
        it will replicate the smaller dataset to match the larger one.
        
        Args:
            csv_dir: Directory containing CSV files (default: DATA_DIR from config)
            image_dir: Directory containing image data (default: QRCODE_DATASET_PATH from config)
            
        Returns:
            Dictionary with 'tabular', 'images', and 'labels' keys
        """
        if csv_dir is None:
            csv_dir = DATA_DIR
        if image_dir is None:
            image_dir = QRCODE_DATASET_PATH
        
        print(f"\n{'='*70}")
        print(f"LOADING ALL ACTUAL DATA FOR MULTIMODAL TRAINING")
        print(f"{'='*70}")
        
        # Load all CSV data
        tabular_data = self.load_all_csv_data(csv_dir)
        n_csv = len(tabular_data)
        
        # Load all image data
        print(f"Image Data Directory: {image_dir}")
        images, image_labels = self.qr_preprocessor.load_images_from_directory(image_dir)
        n_images = len(images)
        
        print(f"\n{'='*70}")
        print(f"DATA LOADED - HANDLING SIZE MISMATCH")
        print(f"{'='*70}")
        print(f"CSV records: {n_csv}")
        print(f"Images available: {n_images}")
        
        # Handle size mismatch by replicating images to match CSV count
        if n_csv > n_images:
            print(f"\n[Data Pairing] CSV records ({n_csv}) > Images ({n_images})")
            print(f"[Data Pairing] Strategy: Replicate images with augmentation to match CSV count")
            
            # Calculate how many times we need to replicate
            replications_needed = (n_csv + n_images - 1) // n_images  # Ceiling division
            
            # Replicate images
            replicated_images = []
            replicated_labels = []
            
            for i in range(replications_needed):
                # Add some noise/augmentation to replicated images for variety
                if i == 0:
                    # First copy: use original images
                    replicated_images.append(images)
                    replicated_labels.append(image_labels)
                else:
                    # Subsequent copies: add slight noise for variety
                    noise = np.random.normal(0, 0.02, images.shape)
                    augmented_images = np.clip(images + noise, 0, 1)
                    replicated_images.append(augmented_images)
                    replicated_labels.append(image_labels)
            
            images = np.concatenate(replicated_images, axis=0)[:n_csv]
            image_labels = np.concatenate(replicated_labels, axis=0)[:n_csv]
            
            print(f"[Data Pairing] ✓ Images replicated to {len(images)} samples")
            
        elif n_images > n_csv:
            print(f"\n[Data Pairing] Images ({n_images}) > CSV records ({n_csv})")
            print(f"[Data Pairing] Strategy: Use first {n_csv} images to match CSV count")
            
            # Use only the first n_csv images
            images = images[:n_csv]
            image_labels = image_labels[:n_csv]
            
            print(f"[Data Pairing] ✓ Using first {len(images)} images")
        else:
            print(f"\n[Data Pairing] ✓ CSV records and images are already matched ({n_csv} samples)")
        
        # Shuffle the pairing to avoid any ordering bias
        print(f"\n[Data Pairing] Shuffling data for random pairing...")
        indices = np.random.permutation(len(images))
        images = images[indices]
        
        print(f"\n{'='*70}")
        print(f"MULTIMODAL DATA LOADING COMPLETE")
        print(f"{'='*70}")
        print(f"Total samples: {n_csv}")
        print(f"CSV records: {len(tabular_data)}")
        print(f"Images: {len(images)}")
        print(f"{'='*70}\n")
        
        return {
            'tabular': tabular_data,
            'images': images,
            'labels': tabular_data['isFraud'].values
        }
        
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
        print(f"  Example CSV file: {DATA_DIR}/PS_20174392719_1491204439457_log.csv")
        print(f"\nImage Data Directory: {QRCODE_DATASET_PATH}")
        print(f"  Benign images: {QRCODE_DATASET_PATH}/benign/benign/")
        print(f"  Malicious images: {QRCODE_DATASET_PATH}/malicious/malicious/")
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
    
    def prepare_train_test_data(self, test_size=0.2, random_state=42, n_samples=None, use_actual_data=True):
        """
        Prepare multimodal train/test data from actual files or synthetic generation
        
        Creates stratified train/test splits maintaining the fraud ratio
        in both sets.
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            n_samples: Number of samples to generate (only used if use_actual_data=False)
            use_actual_data: If True, loads actual CSV and image data from directories.
                           If False, generates synthetic data with n_samples.
            
        Returns:
            Dictionary with train/test splits for both modalities:
            - X_train_tabular: Training tabular features
            - X_train_images: Training images
            - y_train: Training labels
            - X_test_tabular: Test tabular features
            - X_test_images: Test images
            - y_test: Test labels
        """
        # Load actual data or generate synthetic data
        if use_actual_data:
            print(f"\n{'='*70}")
            print(f"USING ACTUAL DATA FROM FILES")
            print(f"{'='*70}\n")
            data = self.load_all_actual_data()
        else:
            print(f"\n{'='*70}")
            print(f"USING SYNTHETIC DATA GENERATION")
            print(f"{'='*70}\n")
            if n_samples is None:
                n_samples = 5000
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
