# config.py - Configuration Documentation

## Overview
This configuration file defines all the parameters and settings for the multimodal fraud detection transformer model. It supports two data modalities: tabular transaction data and QR code images.

## Key Components

### Directory Setup
```python
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
```
**Purpose**: Establishes the base directory structure for the project, ensuring all models, data, and logs are organized in consistent locations.

### Tabular Data Configuration
```python
TABULAR_CONFIG = {
    'num_features': 8,
    'feature_names': [
        'step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud'
    ],
    'd_model': 64,
}
```
**Purpose**: Configures the tabular data branch of the model. Defines 8 transaction features from the Online Payments Fraud Detection dataset and sets the embedding dimension to 64.

### Image Data Configuration
```python
IMAGE_CONFIG = {
    'image_size': (128, 128),
    'channels': 3,
    'patch_size': 16,
    'num_patches': 64,
    'd_model': 64,
}
```
**Purpose**: Configures the Vision Transformer (ViT) style image processing. Images are divided into 16x16 patches, resulting in 64 patches per image (8x8 grid), each embedded into a 64-dimensional space.

### Model Architecture Configuration
```python
MODEL_CONFIG = {
    'tabular_num_features': 8,
    'tabular_d_model': 64,
    'image_d_model': 64,
    'd_model': 64,
    'num_heads': 4,
    'num_layers': 2,
    'cross_modal_layers': 2,
    'dff': 128,
    'dropout_rate': 0.1,
}
```
**Purpose**: Defines the transformer architecture:
- **d_model (64)**: Unified dimension for all representations
- **num_heads (4)**: Number of attention heads in multi-head attention
- **num_layers (2)**: Number of transformer encoder blocks per modality
- **cross_modal_layers (2)**: Number of cross-modal fusion layers
- **dff (128)**: Dimension of feed-forward network (2x d_model)
- **dropout_rate (0.1)**: Dropout probability for regularization

### Training Configuration
```python
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'early_stopping_patience': 5,
}
```
**Purpose**: Sets training hyperparameters including batch size, number of epochs, learning rate, validation split ratio, and early stopping patience to prevent overfitting.

### Model Persistence
```python
MODEL_NAME = 'multimodal_fraud_detection_transformer'
MODEL_VERSION = 'v2.0'
FRAUD_DATASET_PATH = os.path.join(DATA_DIR, 'online_payments_fraud.csv')
QRCODE_DATASET_PATH = os.path.join(DATA_DIR, 'qr_codes')
```
**Purpose**: Defines model naming convention and dataset paths for loading real data when available.

## Usage
Import configurations in other modules:
```python
from config.config import MODEL_CONFIG, TRAINING_CONFIG, MODEL_SAVE_DIR
```
