"""
Configuration file for multimodal fraud detection transformer model

This model supports two modalities:
1. Tabular data: Online Payments Fraud Detection features
2. Image data: QR Code images (benign vs malicious)
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Ensure directories exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Tabular data configuration (Online Payments Fraud Detection)
TABULAR_CONFIG = {
    'num_features': 8,  # After preprocessing: step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFlaggedFraud
    'feature_names': [
        'step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud'
    ],
    'd_model': 64,  # Embedding dimension for tabular features
}

# Image data configuration (QR Codes)
IMAGE_CONFIG = {
    'image_size': (128, 128),  # QR code image size
    'channels': 3,  # RGB channels
    'patch_size': 16,  # Size of image patches for Vision Transformer
    'num_patches': 64,  # (128/16) * (128/16) = 64 patches
    'd_model': 64,  # Embedding dimension for image patches
}

# Multimodal Transformer Model hyperparameters
MODEL_CONFIG = {
    # Tabular branch
    'tabular_num_features': TABULAR_CONFIG['num_features'],
    'tabular_d_model': TABULAR_CONFIG['d_model'],
    
    # Image branch (Vision Transformer style)
    'image_size': IMAGE_CONFIG['image_size'],
    'image_channels': IMAGE_CONFIG['channels'],
    'patch_size': IMAGE_CONFIG['patch_size'],
    'image_d_model': IMAGE_CONFIG['d_model'],
    
    # Shared transformer settings
    'd_model': 64,  # Unified dimension for cross-modal fusion
    'num_heads': 4,  # Number of attention heads
    'num_layers': 2,  # Number of transformer blocks per modality
    'cross_modal_layers': 2,  # Number of cross-modal attention layers
    'dff': 128,  # Dimension of feed-forward network
    'dropout_rate': 0.1,
    
    # For backward compatibility
    'num_features': TABULAR_CONFIG['num_features'],
    'max_sequence_length': 1,
}

# Training parameters
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'early_stopping_patience': 5,
}

# Model saving
MODEL_NAME = 'multimodal_fraud_detection_transformer'
MODEL_VERSION = 'v2.0'

# Dataset paths (for real data loading)
FRAUD_DATASET_PATH = os.path.join(DATA_DIR, 'online_payments_fraud.csv')
QRCODE_DATASET_PATH = os.path.join(DATA_DIR, 'qr_codes')
