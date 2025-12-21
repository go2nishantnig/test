"""
Image data preprocessor for QR code analysis.

This module provides the QRCodePreprocessor class for processing
QR code images from the Benign and Malicious QR Codes dataset.
"""
import numpy as np
from sklearn.model_selection import train_test_split
import os

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL/Pillow not installed. Image loading from files will not work.")
    print("Install with: pip install Pillow")


class QRCodePreprocessor:
    """
    Preprocessor for QR Code Image Data
    
    Handles preprocessing of QR code images for fraud detection.
    Supports both synthetic data generation for testing and loading
    real images from the Benign and Malicious QR Codes dataset.
    
    Attributes:
        image_size: Target size (height, width) for image resizing
    """
    
    def __init__(self, image_size=(128, 128)):
        """
        Initialize the QR code preprocessor.
        
        Args:
            image_size: Target size (height, width) for images
        """
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
        """
        Generate a synthetic QR code-like pattern
        
        Creates a realistic QR code pattern with position detection patterns
        and random data modules.
        
        Args:
            malicious: If True, adds noise and color tint to simulate malicious QR codes
            
        Returns:
            Image array of shape (height, width, 3)
        """
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
        """
        Add a position detection pattern (finder pattern)
        
        Creates the characteristic QR code finder pattern with nested squares.
        
        Args:
            img: Image array to modify
            row: Top row position
            col: Left column position
            size: Size of the pattern
        """
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
        """
        Load and resize an image
        
        Args:
            filepath: Path to the image file
            target_size: Target size (height, width)
            
        Returns:
            Image array (normalized to [0, 1]) or None if loading fails
        """
        if not PIL_AVAILABLE:
            return None
            
        try:
            # Load image and convert to RGB
            img = Image.open(filepath).convert('RGB')
            
            # Resize to target size (PIL expects (width, height))
            img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)
            
            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            return img_array
        except Exception as e:
            print(f"Error loading image {filepath}: {e}")
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
