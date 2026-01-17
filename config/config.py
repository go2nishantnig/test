"""
Configuration file for multimodal fraud detection transformer model

This model supports two modalities:
1. Tabular data: Online Payments Fraud Detection features
2. Image data: QR Code images (benign vs malicious)

Environment-aware configuration:
- Automatically detects EC2 environment (/home/ec2-user exists)
- Can be forced with USE_EC2_CONFIG environment variable
- Falls back to local paths for development

IMPORTANT: To switch between environments, change DATA_BASE_PATH below:
- For AWS EC2: DATA_BASE_PATH = '/home/ec2-user'
- For GitHub Codespaces: DATA_BASE_PATH = '/workspaces/test/data'
"""
import os

# ============================================================================
# CONFIGURABLE BASE PATH - CHANGE THIS TO SWITCH ENVIRONMENTS
# ============================================================================
# For AWS EC2, use: DATA_BASE_PATH = '/home/ec2-user'
# For GitHub Codespaces, use: DATA_BASE_PATH = '/workspaces/test/data'
DATA_BASE_PATH = '/home/ec2-user'
# ============================================================================

# Detect environment: Check if we're using a cloud/remote data directory structure
# This includes both EC2 and Codespaces when DATA_BASE_PATH is set to a remote path
IS_EC2 = (
    DATA_BASE_PATH != os.path.dirname(os.path.dirname(os.path.abspath(__file__))) or
    os.path.exists('/home/ec2-user') or 
    os.environ.get('USE_EC2_CONFIG') == '1'
)

# Base paths - adapt based on environment
if IS_EC2:
    # Remote/cloud environment paths (EC2, Codespaces, etc.)
    # Uses DATA_BASE_PATH as the root for all data directories
    EC2_USER_HOME = DATA_BASE_PATH
    BASE_DIR = os.path.join(EC2_USER_HOME, 'test')  # Repository location
    MODEL_SAVE_DIR = os.path.join(EC2_USER_HOME, 'model')
    DATA_DIR = os.path.join(EC2_USER_HOME, 'csvdata')
    LOG_DIR = os.path.join(MODEL_SAVE_DIR, 'logs')
    QRCODE_DATASET_PATH = os.path.join(EC2_USER_HOME, 'qrimages', 'QR codes')
    
    # Directory mapping for convenience
    EC2_DIRS = {
        'qrdata': QRCODE_DATASET_PATH,
        'csv_data': DATA_DIR,
        'model': MODEL_SAVE_DIR,
        'logs': LOG_DIR,
    }
else:
    # Local development paths (relative to repository)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    QRCODE_DATASET_PATH = os.path.join(DATA_DIR, 'qr_codes')

# Ensure directories exist (best effort - may fail on EC2 if not yet set up)
try:
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
except PermissionError:
    # Directory creation may fail on EC2 before setup script runs
    # This is expected - the setup script will create them with proper permissions
    pass
except FileNotFoundError:
    # Parent directories may not exist yet on fresh EC2 instances
    # The setup script will create the full directory tree
    pass

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

# Dataset paths (for real data loading)
FRAUD_DATASET_PATH = os.path.join(DATA_DIR, 'online_payments_fraud.csv')

# GPU Configuration (for EC2 G5 XLarge with NVIDIA A10G or other GPU systems)
GPU_CONFIG = {
    'memory_growth': True,  # Allow memory growth instead of allocating all GPU memory
    'mixed_precision': True,  # Use mixed precision for faster training
}

# Backward compatibility aliases for EC2-specific code
EC2_QRDATA_DIR = QRCODE_DATASET_PATH
EC2_CSV_DATA_DIR = DATA_DIR
EC2_MODEL_DIR = MODEL_SAVE_DIR
EC2_LOG_DIR = LOG_DIR
