"""
Configuration file for AWS EC2 G5 XLarge instance setup
This configuration adapts the model for use on EC2 with Jupyter notebook
"""
import os

# EC2-specific paths
EC2_USER_HOME = '/home/ec2-user'
EC2_QRDATA_DIR = os.path.join(EC2_USER_HOME, 'qrdata')
EC2_CSV_DATA_DIR = os.path.join(EC2_USER_HOME, 'csv-data')
EC2_MODEL_DIR = os.path.join(EC2_USER_HOME, 'model')
EC2_LOG_DIR = os.path.join(EC2_MODEL_DIR, 'logs')

# Ensure directories exist (will be created by setup script)
EC2_DIRS = {
    'qrdata': EC2_QRDATA_DIR,
    'csv_data': EC2_CSV_DATA_DIR,
    'model': EC2_MODEL_DIR,
    'logs': EC2_LOG_DIR,
}

# Override base configuration for EC2
MODEL_SAVE_DIR = EC2_MODEL_DIR
DATA_DIR = EC2_CSV_DATA_DIR
LOG_DIR = EC2_LOG_DIR

# Tabular data configuration (Online Payments Fraud Detection)
# Note: nameOrig and nameDest are dropped during preprocessing as they are ID columns
TABULAR_CONFIG = {
    'num_features': 8,  # After preprocessing: step, type (encoded), amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFlaggedFraud
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

# Dataset paths for EC2
FRAUD_DATASET_PATH = os.path.join(EC2_CSV_DATA_DIR, 'online_payments_fraud.csv')
QRCODE_DATASET_PATH = EC2_QRDATA_DIR

# GPU Configuration for G5 XLarge (NVIDIA A10G)
GPU_CONFIG = {
    'memory_growth': True,  # Allow memory growth instead of allocating all GPU memory
    'mixed_precision': True,  # Use mixed precision for faster training
}
