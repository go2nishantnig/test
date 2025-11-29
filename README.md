# Multimodal Fraud Detection Transformer Model

A TensorFlow-based multimodal transformer model for detecting fraudulent transactions. This project implements a complete pipeline for training, evaluating, and deploying a transformer-based fraud detection system that combines:

1. **Tabular Data**: Transaction features from [Online Payments Fraud Detection Dataset](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset)
2. **Image Data**: QR code images from [Benign and Malicious QR Codes Dataset](https://www.kaggle.com/datasets/samahsadiq/benign-and-malicious-qr-codes)

## Features

- **Multimodal Transformer Architecture**: Combines tabular and image data using cross-modal attention
- **Vision Transformer (ViT) for Images**: Patch-based image encoding for QR code analysis
- **Tabular Transformer**: Self-attention mechanism for transaction feature analysis
- **Cross-Modal Attention Fusion**: Allows each modality to attend to the other
- **Automated Model Saving**: Trained models are automatically saved to a dedicated directory
- **Data Preprocessing**: Built-in utilities for both tabular and image data
- **Synthetic Data Generation**: Demo dataset generators for testing and development
- **Backward Compatible**: Supports tabular-only mode for legacy use cases

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Multimodal Transformer                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐              ┌─────────────────┐              │
│  │  Tabular Input  │              │   Image Input   │              │
│  │  (Transaction)  │              │   (QR Code)     │              │
│  └────────┬────────┘              └────────┬────────┘              │
│           │                                │                        │
│           ▼                                ▼                        │
│  ┌─────────────────┐              ┌─────────────────┐              │
│  │ Feature Embed   │              │ Patch Embedding │              │
│  └────────┬────────┘              └────────┬────────┘              │
│           │                                │                        │
│           ▼                                ▼                        │
│  ┌─────────────────┐              ┌─────────────────┐              │
│  │  Transformer    │              │  Transformer    │              │
│  │  Encoder        │              │  Encoder (ViT)  │              │
│  └────────┬────────┘              └────────┬────────┘              │
│           │                                │                        │
│           └──────────┬─────────────────────┘                       │
│                      ▼                                              │
│           ┌─────────────────────┐                                   │
│           │  Cross-Modal        │                                   │
│           │  Attention Fusion   │                                   │
│           └──────────┬──────────┘                                   │
│                      │                                              │
│                      ▼                                              │
│           ┌─────────────────────┐                                   │
│           │  Classification     │                                   │
│           │  Head               │                                   │
│           └──────────┬──────────┘                                   │
│                      │                                              │
│                      ▼                                              │
│           ┌─────────────────────┐                                   │
│           │  Fraud Prediction   │                                   │
│           │  (0: Normal, 1: Fraud)                                  │
│           └─────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
.
├── config/                      # Configuration files
│   ├── __init__.py
│   └── config.py               # Model and training configuration
├── data/                       # Data directory
├── models/                     # Model storage
│   └── saved_models/          # Trained models saved here
├── logs/                       # Training logs
├── notebooks/                  # Jupyter notebooks for experiments
├── src/                        # Source code
│   ├── __init__.py
│   ├── models/                # Model architectures
│   │   ├── __init__.py
│   │   └── transformer_model.py
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py
│   │   └── data_preprocessing.py
│   ├── train.py              # Training script
│   └── predict.py            # Inference script
└── requirements.txt           # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd test
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Multimodal Model

Train the multimodal fraud detection transformer model (default):

```bash
python src/train.py --mode multimodal
```

Or train the tabular-only model:

```bash
python src/train.py --mode tabular
```

This will:
- Generate synthetic training data (or load real data if available)
- Build the transformer model
- Train the model with early stopping and learning rate reduction
- Save the trained model to `models/saved_models/`
- Save the data preprocessors
- Generate TensorBoard logs

### Making Predictions

Run predictions on new data:

```bash
# Multimodal prediction
python src/predict.py --mode multimodal

# Tabular-only prediction
python src/predict.py --mode tabular
```

### Quick Start

```bash
# Multimodal demo
python quick_start.py --mode multimodal

# Tabular-only demo
python quick_start.py --mode tabular
```

### Using the Model in Your Code

**Multimodal Prediction:**
```python
from src.predict import MultimodalFraudDetectionPredictor

# Initialize predictor
predictor = MultimodalFraudDetectionPredictor()

# Predict for a transaction with QR code
result = predictor.predict_single_transaction(
    tabular_features={
        'step': 1,
        'type': 'TRANSFER',
        'amount': 50000.0,
        'nameOrig': 'C1234567890',
        'oldbalanceOrg': 100000.0,
        'newbalanceOrig': 50000.0,
        'nameDest': 'C9876543210',
        'oldbalanceDest': 0.0,
        'newbalanceDest': 50000.0,
        'isFlaggedFraud': 0,
    },
    image=qr_code_image  # Normalized numpy array
)
print(f"Is Fraud: {result['is_fraud']}")
print(f"Fraud Probability: {result['fraud_probability']:.2%}")
```

**Tabular-only Prediction:**
```python
from src.predict import FraudDetectionPredictor

# Initialize predictor
predictor = FraudDetectionPredictor()

# Predict for a single transaction
transaction = {
    'step': 1,
    'type': 'PAYMENT',
    'amount': 250.0,
    'nameOrig': 'C1234567890',
    'oldbalanceOrg': 10000.0,
    'newbalanceOrig': 9750.0,
    'nameDest': 'M9876543210',
    'oldbalanceDest': 5000.0,
    'newbalanceDest': 5250.0,
    'isFlaggedFraud': 0,
}

result = predictor.predict_single_transaction(transaction)
print(f"Is Fraud: {result['is_fraud']}")
print(f"Fraud Probability: {result['fraud_probability']:.2%}")
```

## Model Architecture

### Multimodal Architecture

1. **Tabular Branch** (Transaction Features):
   - Input Layer: 10 transaction features
   - Feature Embedding: Projects to 64-dimensional space
   - Positional Encoding
   - 2 Transformer Blocks with 4 attention heads

2. **Image Branch** (QR Codes):
   - Input: 128x128 RGB images
   - Patch Embedding: 16x16 patches → 64 patches
   - Positional Encoding
   - 2 Transformer Blocks with 4 attention heads

3. **Cross-Modal Fusion**:
   - 2 Cross-Modal Attention Layers
   - Bidirectional attention between modalities

4. **Classification Head**:
   - Global Average Pooling
   - Dense layers with dropout
   - Sigmoid output for binary classification

## Configuration

Modify `config/config.py` to adjust:

- **Model parameters**: `d_model`, `num_heads`, `num_layers`, etc.
- **Training parameters**: `batch_size`, `epochs`, `learning_rate`
- **Image settings**: `image_size`, `patch_size`
- **Directory paths**: `MODEL_SAVE_DIR`, `DATA_DIR`, `LOG_DIR`

Default configuration:
```python
MODEL_CONFIG = {
    # Tabular branch
    'tabular_num_features': 10,
    'tabular_d_model': 64,
    
    # Image branch
    'image_size': (128, 128),
    'image_channels': 3,
    'patch_size': 16,
    
    # Shared settings
    'd_model': 64,
    'num_heads': 4,
    'num_layers': 2,
    'cross_modal_layers': 2,
    'dff': 128,
    'dropout_rate': 0.1,
}

TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'early_stopping_patience': 5,
}
```

## Datasets

This model is designed to work with:

1. **Online Payments Fraud Detection Dataset**
   - Source: [Kaggle](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset)
   - Features: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

2. **Benign and Malicious QR Codes Dataset**
   - Source: [Kaggle](https://www.kaggle.com/datasets/samahsadiq/benign-and-malicious-qr-codes)
   - Contains benign and malicious QR code images

## Monitoring Training

View training metrics with TensorBoard:

```bash
tensorboard --logdir logs/
```

Then open your browser to `http://localhost:6006`

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Scikit-learn >= 1.3.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- Jupyter >= 1.0.0

## License

This project is provided as-is for educational and development purposes.