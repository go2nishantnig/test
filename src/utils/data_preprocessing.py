"""
Data preprocessing utilities for multimodal fraud detection

Handles two data modalities:
1. Tabular data: Online Payments Fraud Detection features
2. Image data: QR Code images (benign vs malicious)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os


class FraudDataPreprocessor:
    """Preprocessor for tabular fraud detection data (Online Payments Fraud Dataset)"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def generate_synthetic_data(self, n_samples=10000, fraud_ratio=0.02):
        """
        Generate synthetic fraud transaction data matching Online Payments Fraud Detection format
        
        Dataset columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
                        nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
        
        Args:
            n_samples: Number of samples to generate
            fraud_ratio: Ratio of fraudulent transactions
            
        Returns:
            DataFrame with transaction features and labels
        """
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
    
    def preprocess_data(self, data, fit=True):
        """
        Preprocess the tabular data
        
        Args:
            data: DataFrame with features and labels
            fit: Whether to fit the scaler and encoders (True for training data)
            
        Returns:
            Tuple of (X, y) where X is scaled features and y is labels
        """
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
        """Save the scaler and label encoders to disk"""
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
        """Load the scaler and label encoders from disk"""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        self.scaler = save_dict.get('scaler', self.scaler)
        self.label_encoders = save_dict.get('label_encoders', {})
        self.feature_names = save_dict.get('feature_names', None)
        print(f"Scaler and encoders loaded from {filepath}")
    
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
            data, test_size=test_size, random_state=random_state, stratify=data['isFraud']
        )
        
        # Preprocess
        X_train, y_train = self.preprocess_data(train_data, fit=True)
        X_test, y_test = self.preprocess_data(test_data, fit=False)
        
        return X_train, X_test, y_train, y_test


class QRCodePreprocessor:
    """Preprocessor for QR Code image data (Benign vs Malicious)"""
    
    def __init__(self, image_size=(128, 128)):
        self.image_size = image_size
        
    def generate_synthetic_qr_images(self, n_samples=1000, malicious_ratio=0.3):
        """
        Generate synthetic QR code-like images for demonstration
        
        In production, this should load actual QR code images from the dataset.
        
        Args:
            n_samples: Number of samples to generate
            malicious_ratio: Ratio of malicious QR codes
            
        Returns:
            Tuple of (images, labels) where images are normalized and labels are 0/1
        """
        np.random.seed(42)
        
        n_malicious = int(n_samples * malicious_ratio)
        n_benign = n_samples - n_malicious
        
        # Generate benign QR code-like images (more structured patterns)
        benign_images = []
        for _ in range(n_benign):
            img = self._generate_qr_pattern(malicious=False)
            benign_images.append(img)
        
        # Generate malicious QR code-like images (slightly different patterns)
        malicious_images = []
        for _ in range(n_malicious):
            img = self._generate_qr_pattern(malicious=True)
            malicious_images.append(img)
        
        # Combine
        images = np.array(benign_images + malicious_images)
        labels = np.array([0] * n_benign + [1] * n_malicious)
        
        # Shuffle
        indices = np.random.permutation(n_samples)
        images = images[indices]
        labels = labels[indices]
        
        # Normalize to [0, 1]
        images = images.astype(np.float32) / 255.0
        
        return images, labels
    
    def _generate_qr_pattern(self, malicious=False):
        """Generate a synthetic QR code-like pattern"""
        h, w = self.image_size
        
        # Create base image (white background)
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Add QR code-like patterns
        block_size = 8
        
        # Add position detection patterns (corner squares)
        self._add_position_pattern(img, 0, 0, block_size * 3)
        self._add_position_pattern(img, 0, w - block_size * 3, block_size * 3)
        self._add_position_pattern(img, h - block_size * 3, 0, block_size * 3)
        
        # Add random data modules
        for i in range(block_size * 3, h - block_size * 3, block_size):
            for j in range(block_size * 3, w - block_size * 3, block_size):
                if np.random.random() > 0.5:
                    img[i:i+block_size, j:j+block_size] = 0  # Black block
        
        # For malicious QR codes, add some noise/distortion
        if malicious:
            noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
            img = np.clip(img.astype(np.int32) + noise - 25, 0, 255).astype(np.uint8)
            
            # Add subtle color tint
            tint = np.random.choice([0, 1, 2])
            img[:, :, tint] = np.clip(img[:, :, tint].astype(np.int32) + 30, 0, 255).astype(np.uint8)
        
        return img
    
    def _add_position_pattern(self, img, row, col, size):
        """Add a position detection pattern (finder pattern)"""
        # Outer black square
        img[row:row+size, col:col+size] = 0
        # Middle white square
        inner_start = size // 7
        inner_size = size - 2 * inner_start
        img[row+inner_start:row+inner_start+inner_size, 
            col+inner_start:col+inner_start+inner_size] = 255
        # Inner black square
        center_start = size // 3
        center_size = size // 3
        img[row+center_start:row+center_start+center_size,
            col+center_start:col+center_start+center_size] = 0
    
    def load_images_from_directory(self, directory, target_size=None):
        """
        Load QR code images from a directory structure:
        directory/
            benign/
                image1.png
                image2.png
            malicious/
                image1.png
                image2.png
        
        Args:
            directory: Path to the dataset directory
            target_size: Target size for resizing (default: self.image_size)
            
        Returns:
            Tuple of (images, labels)
        """
        if target_size is None:
            target_size = self.image_size
            
        images = []
        labels = []
        
        # This is a placeholder - actual implementation would use PIL/cv2
        # For now, return synthetic data if directory doesn't exist
        if not os.path.exists(directory):
            print(f"Directory {directory} not found. Using synthetic data.")
            return self.generate_synthetic_qr_images()
        
        # Load benign images
        benign_dir = os.path.join(directory, 'benign')
        if os.path.exists(benign_dir):
            for filename in os.listdir(benign_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Load and preprocess image
                    # Placeholder: actual loading would use PIL
                    img = self._load_and_resize_image(
                        os.path.join(benign_dir, filename), target_size
                    )
                    if img is not None:
                        images.append(img)
                        labels.append(0)
        
        # Load malicious images
        malicious_dir = os.path.join(directory, 'malicious')
        if os.path.exists(malicious_dir):
            for filename in os.listdir(malicious_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = self._load_and_resize_image(
                        os.path.join(malicious_dir, filename), target_size
                    )
                    if img is not None:
                        images.append(img)
                        labels.append(1)
        
        if len(images) == 0:
            print("No images found. Using synthetic data.")
            return self.generate_synthetic_qr_images()
        
        return np.array(images), np.array(labels)
    
    def _load_and_resize_image(self, filepath, target_size):
        """Load and resize an image
        
        Note: This is a placeholder. In production, use PIL or OpenCV:
            from PIL import Image
            img = Image.open(filepath).resize(target_size).convert('RGB')
            return np.array(img) / 255.0
        """
        # Placeholder - actual implementation requires PIL/cv2
        # Return None to signal that synthetic data should be used instead
        return None
    
    def prepare_train_test_data(self, test_size=0.2, random_state=42):
        """
        Generate and prepare train/test data for QR codes
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Generate synthetic data
        images, labels = self.generate_synthetic_qr_images()
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        return X_train, X_test, y_train, y_test


class MultimodalDataPreprocessor:
    """Combined preprocessor for multimodal fraud detection"""
    
    def __init__(self, image_size=(128, 128)):
        self.fraud_preprocessor = FraudDataPreprocessor()
        self.qr_preprocessor = QRCodePreprocessor(image_size=image_size)
        
    def generate_synthetic_multimodal_data(self, n_samples=5000, fraud_ratio=0.1):
        """
        Generate synthetic multimodal data combining tabular and image data
        
        This simulates a scenario where each transaction also has an associated 
        QR code image that was scanned.
        
        Args:
            n_samples: Number of samples to generate
            fraud_ratio: Ratio of fraudulent transactions
            
        Returns:
            Dictionary with 'tabular', 'images', and 'labels' keys
        """
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
        """
        Preprocess multimodal data
        
        Args:
            tabular_data: DataFrame with tabular features
            images: Array of images (normalized)
            fit: Whether to fit the preprocessors
            
        Returns:
            Dictionary with preprocessed 'tabular' and 'images'
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
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary with train/test splits for both modalities
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
        """Save all preprocessors to disk"""
        self.fraud_preprocessor.save_scaler(filepath)
    
    def load_preprocessors(self, filepath):
        """Load all preprocessors from disk"""
        self.fraud_preprocessor.load_scaler(filepath)
