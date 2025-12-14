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
├── config/                          # Configuration files
│   ├── __init__.py
│   └── config.py                   # Model and training configuration
├── data/                           # Data directory
├── diagrams/                        # PlantUML architecture diagrams
│   ├── *.puml                      # PlantUML source files
│   ├── README.md                   # Diagrams documentation
│   └── generate_diagrams.sh        # Script to generate diagram images
├── models/                         # Model storage
│   └── saved_models/              # Trained models saved here
├── logs/                           # Training logs
├── notebooks/                      # Jupyter notebooks for experiments
├── src/                            # Source code
│   ├── __init__.py
│   ├── models/                    # Model architectures (modular)
│   │   ├── __init__.py            # Package exports
│   │   ├── attention.py           # Attention mechanisms
│   │   ├── layers.py              # Core layers (FFN, Residual)
│   │   ├── blocks.py              # Transformer blocks
│   │   ├── embeddings.py          # Embedding layers
│   │   ├── transformer.py         # Model builders
│   │   └── transformer_model.py   # Legacy (backward compatible)
│   ├── utils/                     # Utility functions (modular)
│   │   ├── __init__.py            # Package exports
│   │   ├── tabular_preprocessor.py     # Tabular data preprocessing
│   │   ├── image_preprocessor.py       # QR code preprocessing
│   │   ├── multimodal_preprocessor.py  # Combined preprocessing
│   │   └── data_preprocessing.py       # Legacy (backward compatible)
│   ├── train.py                  # Training script
│   └── predict.py                # Inference script
├── quick_start.py                 # Quick start demo
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── SUMMARY.md                     # Project summary
```

## Architecture Diagrams

Comprehensive PlantUML diagrams are available in the [`diagrams/`](diagrams/) folder, documenting:

- **System Architecture**: High-level overview of components and data flow
- **Component Structure**: Package and module organization
- **Class Diagrams**: Detailed class hierarchies for models and utilities
- **Sequence Diagrams**: Training and prediction workflows
- **Data Flow**: End-to-end data transformation pipeline
- **Attention Mechanisms**: Detailed view of self-attention and cross-modal attention
- **Deployment Architecture**: Production deployment options

See [`diagrams/README.md`](diagrams/README.md) for details on viewing and generating these diagrams.

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

### Package Structure

The codebase is organized into modular packages for better understanding and maintainability:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         src/models/ Package                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────┐    ┌─────────────────────┐                    │
│  │    attention.py     │    │      layers.py      │                    │
│  ├─────────────────────┤    ├─────────────────────┤                    │
│  │ MultiHeadSelf       │    │ FeedForward         │                    │
│  │   Attention         │    │ ResidualConnection  │                    │
│  │ MaskedMultiHead     │    └─────────────────────┘                    │
│  │   Attention         │                                                │
│  │ CrossModalAttention │                                                │
│  └─────────────────────┘                                                │
│              │                         │                                │
│              └────────────┬────────────┘                                │
│                           ▼                                             │
│              ┌─────────────────────┐                                    │
│              │      blocks.py      │                                    │
│              ├─────────────────────┤                                    │
│              │ TransformerBlock    │                                    │
│              │ TransformerDecoder  │                                    │
│              │   Block             │                                    │
│              │ CrossModalTrans     │                                    │
│              │   formerBlock       │                                    │
│              └─────────────────────┘                                    │
│                           │                                             │
│              ┌────────────┼────────────┐                               │
│              ▼            ▼            ▼                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │  embeddings.py  │  │  transformer.py  │  │ transformer_    │        │
│  ├─────────────────┤  ├─────────────────┤  │   model.py      │        │
│  │ PatchEmbedding  │  │ Multimodal      │  ├─────────────────┤        │
│  └─────────────────┘  │   FraudDetect   │  │ (Legacy module  │        │
│                       │   ionTransformer │  │  for backward   │        │
│                       │ FraudDetection  │  │  compatibility) │        │
│                       │   Transformer   │  └─────────────────┘        │
│                       └─────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         src/utils/ Package                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────┐    ┌─────────────────────────┐            │
│  │ tabular_preprocessor.py │    │  image_preprocessor.py  │            │
│  ├─────────────────────────┤    ├─────────────────────────┤            │
│  │ FraudDataPreprocessor   │    │ QRCodePreprocessor      │            │
│  │ - generate_synthetic    │    │ - generate_synthetic    │            │
│  │ - preprocess_data       │    │ - generate_qr_pattern   │            │
│  │ - save/load_scaler      │    │ - load_images           │            │
│  └─────────────────────────┘    └─────────────────────────┘            │
│              │                             │                            │
│              └──────────┬──────────────────┘                           │
│                         ▼                                               │
│            ┌─────────────────────────┐                                  │
│            │multimodal_preprocessor.py│                                  │
│            ├─────────────────────────┤                                  │
│            │ MultimodalDataPreprocessor                                 │
│            │ - generate_multimodal    │                                  │
│            │ - preprocess_multimodal  │                                  │
│            │ - prepare_train_test     │                                  │
│            └─────────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Transformer Components

This implementation includes all standard transformer components:

| Component | Description |
|-----------|-------------|
| **MultiHeadSelfAttention** | Standard multi-head self-attention for encoder |
| **MaskedMultiHeadAttention** | Masked multi-head attention for decoder (causal masking) |
| **CrossModalAttention** | Cross multi-head attention for cross-modal fusion |
| **FeedForward** | Position-wise feed-forward network |
| **ResidualConnection** | Residual connection with layer normalization (Add & Norm) |
| **TransformerBlock** | Complete encoder block |
| **TransformerDecoderBlock** | Complete decoder block with masked attention |
| **CrossModalTransformerBlock** | Cross-modal transformer block for multimodal fusion |

### Multimodal Architecture

1. **Tabular Branch** (Transaction Features):
   - Input Layer: 10 transaction features
   - Feature Embedding: Projects to 64-dimensional space
   - Positional Encoding
   - 2 Transformer Blocks with 4 attention heads
   - Each block contains: MultiHeadSelfAttention + Residual + FeedForward + Residual

2. **Image Branch** (QR Codes):
   - Input: 128x128 RGB images
   - Patch Embedding: 16x16 patches → 64 patches
   - Positional Encoding
   - 2 Transformer Blocks with 4 attention heads
   - Each block contains: MultiHeadSelfAttention + Residual + FeedForward + Residual

3. **Cross-Modal Fusion**:
   - 2 Cross-Modal Attention Layers (CrossModalTransformerBlock)
   - Bidirectional CrossModalAttention between modalities
   - Tabular attends to image, Image attends to tabular
   - FeedForward networks + Residual connections for each modality

4. **Classification Head**:
   - Global Average Pooling
   - Dense layers with dropout
   - Sigmoid output for binary classification

### Detailed Class Flow Diagrams

#### Attention Mechanism Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MultiHeadSelfAttention Flow                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input: x (batch_size, seq_len, d_model)                              │
│                                                                         │
│          ┌───────────┬───────────┬───────────┐                         │
│          │    Wq     │    Wk     │    Wv     │                         │
│          │  (Dense)  │  (Dense)  │  (Dense)  │                         │
│          └─────┬─────┴─────┬─────┴─────┬─────┘                         │
│                │           │           │                                │
│                ▼           ▼           ▼                                │
│          ┌─────────┐ ┌─────────┐ ┌─────────┐                           │
│          │ Q       │ │ K       │ │ V       │                           │
│          └────┬────┘ └────┬────┘ └────┬────┘                           │
│               │           │           │                                 │
│               └─────┬─────┴───────────┘                                │
│                     │                                                   │
│               ┌─────▼─────┐                                            │
│               │ Split to  │                                            │
│               │ num_heads │                                            │
│               └─────┬─────┘                                            │
│                     │                                                   │
│               ┌─────▼─────────────────────────────┐                    │
│               │ Scaled Dot-Product Attention      │                    │
│               │ Attention(Q,K,V) = softmax(QK^T   │                    │
│               │                    /√dk) · V      │                    │
│               └─────────────┬─────────────────────┘                    │
│                             │                                           │
│               ┌─────────────▼─────────────┐                            │
│               │ Concatenate heads         │                            │
│               └─────────────┬─────────────┘                            │
│                             │                                           │
│               ┌─────────────▼─────────────┐                            │
│               │ Output Projection (Dense) │                            │
│               └─────────────┬─────────────┘                            │
│                             │                                           │
│   Output: (batch_size, seq_len, d_model)                               │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Transformer Block Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TransformerBlock Flow                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input: x (batch_size, seq_len, d_model)                              │
│          │                                                              │
│          ├───────────────────────────────────────────┐                 │
│          │                                           │                 │
│          ▼                                           │ (Residual)      │
│   ┌──────────────────────────────┐                  │                 │
│   │   MultiHeadSelfAttention     │                  │                 │
│   └──────────────┬───────────────┘                  │                 │
│                  │                                   │                 │
│                  ▼                                   │                 │
│   ┌──────────────────────────────┐                  │                 │
│   │        Dropout               │                  │                 │
│   └──────────────┬───────────────┘                  │                 │
│                  │                                   │                 │
│                  └───────────┬───────────────────────┘                 │
│                              │                                          │
│                              ▼                                          │
│   ┌──────────────────────────────────────────────────┐                 │
│   │              Add & LayerNorm                      │                 │
│   └──────────────────────┬───────────────────────────┘                 │
│                          │                                              │
│                          ├───────────────────────────────┐             │
│                          │                               │             │
│                          ▼                               │ (Residual)  │
│   ┌──────────────────────────────┐                      │             │
│   │       FeedForward            │                      │             │
│   │  (Dense → ReLU → Dense)      │                      │             │
│   └──────────────┬───────────────┘                      │             │
│                  │                                       │             │
│                  ▼                                       │             │
│   ┌──────────────────────────────┐                      │             │
│   │        Dropout               │                      │             │
│   └──────────────┬───────────────┘                      │             │
│                  │                                       │             │
│                  └─────────────┬─────────────────────────┘             │
│                                │                                        │
│                                ▼                                        │
│   ┌──────────────────────────────────────────────────┐                 │
│   │              Add & LayerNorm                      │                 │
│   └──────────────────────┬───────────────────────────┘                 │
│                          │                                              │
│   Output: (batch_size, seq_len, d_model)                               │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Cross-Modal Transformer Block Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│              CrossModalTransformerBlock Flow                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Tabular Input                        Image Input                      │
│        │                                    │                           │
│        ├──────────────────────┐            ├──────────────────────┐    │
│        │                      │            │                      │    │
│        ▼                      │ (Res)      ▼                      │ (Res)
│   ┌──────────────┐           │       ┌──────────────┐           │    │
│   │ CrossModal   ├───────────│───────┤ CrossModal   │           │    │
│   │ Attention    │ Q←Tab     │       │ Attention    │ Q←Img     │    │
│   │ (Tab→Img)    │ K,V←Img   │       │ (Img→Tab)    │ K,V←Tab   │    │
│   └──────┬───────┘           │       └──────┬───────┘           │    │
│          │                   │              │                   │    │
│          ▼                   │              ▼                   │    │
│   ┌──────────────┐           │       ┌──────────────┐           │    │
│   │   Dropout    │           │       │   Dropout    │           │    │
│   └──────┬───────┘           │       └──────┬───────┘           │    │
│          │                   │              │                   │    │
│          └───────┬───────────┘              └───────┬───────────┘    │
│                  ▼                                  ▼                 │
│   ┌────────────────────┐            ┌────────────────────┐           │
│   │   Add & LayerNorm  │            │   Add & LayerNorm  │           │
│   └─────────┬──────────┘            └─────────┬──────────┘           │
│             │                                  │                      │
│             ├───────────────┐                  ├───────────────┐     │
│             │               │(Res)             │               │(Res)│
│             ▼               │                  ▼               │     │
│   ┌──────────────┐          │       ┌──────────────┐          │     │
│   │ FeedForward  │          │       │ FeedForward  │          │     │
│   │  (Tabular)   │          │       │  (Image)     │          │     │
│   └──────┬───────┘          │       └──────┬───────┘          │     │
│          │                  │              │                  │     │
│          ▼                  │              ▼                  │     │
│   ┌──────────────┐          │       ┌──────────────┐          │     │
│   │   Dropout    │          │       │   Dropout    │          │     │
│   └──────┬───────┘          │       └──────┬───────┘          │     │
│          │                  │              │                  │     │
│          └────────┬─────────┘              └────────┬─────────┘     │
│                   ▼                                 ▼                │
│   ┌────────────────────┐            ┌────────────────────┐          │
│   │   Add & LayerNorm  │            │   Add & LayerNorm  │          │
│   └─────────┬──────────┘            └─────────┬──────────┘          │
│             │                                  │                     │
│   Tabular Output                       Image Output                  │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Data Preprocessing Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Data Preprocessing Pipeline                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Raw Transaction Data              Raw QR Code Images                  │
│   (CSV/DataFrame)                   (PNG/JPG files)                     │
│        │                                    │                           │
│        ▼                                    ▼                           │
│   ┌──────────────────────┐          ┌──────────────────────┐           │
│   │ FraudDataPreprocessor│          │  QRCodePreprocessor  │           │
│   ├──────────────────────┤          ├──────────────────────┤           │
│   │ 1. Encode categories │          │ 1. Load/resize images│           │
│   │ 2. Drop ID columns   │          │ 2. Normalize [0,1]   │           │
│   │ 3. Scale features    │          │ 3. Extract patches   │           │
│   │ 4. Reshape for model │          └──────────┬───────────┘           │
│   └──────────┬───────────┘                     │                        │
│              │                                  │                        │
│              └──────────────┬───────────────────┘                       │
│                             │                                            │
│                             ▼                                            │
│              ┌──────────────────────────────┐                           │
│              │ MultimodalDataPreprocessor   │                           │
│              ├──────────────────────────────┤                           │
│              │ 1. Combine modalities        │                           │
│              │ 2. Align samples             │                           │
│              │ 3. Create train/test splits  │                           │
│              └──────────────┬───────────────┘                           │
│                             │                                            │
│              ┌──────────────┴───────────────┐                           │
│              │                              │                            │
│              ▼                              ▼                            │
│   ┌─────────────────────┐      ┌─────────────────────┐                  │
│   │ Training Data       │      │ Test Data           │                  │
│   │ - X_train_tabular   │      │ - X_test_tabular    │                  │
│   │ - X_train_images    │      │ - X_test_images     │                  │
│   │ - y_train           │      │ - y_test            │                  │
│   └─────────────────────┘      └─────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
```

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