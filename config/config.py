"""
Configuration file for fraud detection transformer model
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

# Model hyperparameters
MODEL_CONFIG = {
    'num_features': 30,  # Number of input features
    'd_model': 64,  # Dimension of the transformer model
    'num_heads': 4,  # Number of attention heads
    'num_layers': 2,  # Number of transformer blocks
    'dff': 128,  # Dimension of feed-forward network
    'dropout_rate': 0.1,
    'max_sequence_length': 1,  # For single transaction classification
}

# Training parameters
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
}

# Model saving
MODEL_NAME = 'fraud_detection_transformer'
MODEL_VERSION = 'v1.0'
