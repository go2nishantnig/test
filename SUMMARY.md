# Project Summary: Multimodal Fraud Detection Transformer Model

## Overview
This project implements a multimodal transformer-based neural network model for detecting fraudulent transactions using TensorFlow. The model combines two data modalities:

1. **Tabular Data**: Transaction features from [Online Payments Fraud Detection Dataset](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset)
2. **Image Data**: QR code images from [Benign and Malicious QR Codes Dataset](https://www.kaggle.com/datasets/samahsadiq/benign-and-malicious-qr-codes)

The architecture uses cross-modal attention to fuse information from both modalities for improved fraud detection.

## Transformer Components

This implementation includes all standard transformer components:

### Attention Mechanisms

| Component | Description | Use Case |
|-----------|-------------|----------|
| **MultiHeadSelfAttention** | Standard multi-head self-attention | Encoder self-attention |
| **MaskedMultiHeadAttention** | Multi-head attention with causal masking | Decoder self-attention (autoregressive) |
| **CrossModalAttention** | Cross multi-head attention | Cross-modal fusion, encoder-decoder attention |

### Core Components

| Component | Description |
|-----------|-------------|
| **FeedForward** | Position-wise feed-forward network (FFN): two linear transformations with ReLU activation |
| **ResidualConnection** | Residual connection with layer normalization (Add & Norm) |

### Transformer Blocks

| Component | Description | Contains |
|-----------|-------------|----------|
| **TransformerBlock** | Standard encoder block | MultiHeadSelfAttention + Residual + FFN + Residual |
| **TransformerDecoderBlock** | Standard decoder block | MaskedMultiHeadAttention + Residual + CrossModalAttention + Residual + FFN + Residual |
| **CrossModalTransformerBlock** | Cross-modal fusion block | Bidirectional CrossModalAttention + Residuals + FFN + Residuals |

## Architecture Details

### Multimodal Transformer Components

#### 1. Tabular Branch (Transaction Features)
- **Input**: 10 transaction features (step, type, amount, etc.)
- **Feature Embedding**: Projects to 64-dimensional space
- **Transformer Blocks** (x2):
  - Multi-head self-attention (4 heads)
  - Feed-forward network (128 dimensions)
  - Layer normalization
  - Residual connections
  - Dropout (0.1 rate)

#### 2. Image Branch (QR Codes - Vision Transformer)
- **Input**: 128x128 RGB images
- **Patch Embedding**: 16x16 patches → 64 patches total
- **Positional Encoding**: Learnable position embeddings
- **Transformer Blocks** (x2):
  - Multi-head self-attention (4 heads)
  - Feed-forward network (128 dimensions)
  - Layer normalization
  - Residual connections
  - Dropout (0.1 rate)

#### 3. Cross-Modal Fusion
- **Cross-Modal Attention Layers** (x2):
  - Tabular attends to image features (Cross Multi-Head Attention)
  - Image attends to tabular features (Cross Multi-Head Attention)
  - Bidirectional information exchange
  - Feed-forward networks for each modality
  - Residual connections throughout

#### 4. Classification Head
- **Global Average Pooling**: Aggregates sequence information
- **Concatenation**: Merges both modality representations
- **Dense Layers**: 128 → 64 with dropout
- **Output**: Sigmoid activation for binary classification

### Model Performance
- **Metrics**: Accuracy, Precision, Recall, AUC
- **Training**: Early stopping, learning rate reduction, model checkpointing
- **Validation**: 20% validation split with stratification

## Directory Structure

```
.
├── config/                          # Configuration files
│   ├── __init__.py
│   └── config.py                   # Model and training parameters
├── data/                           # Data directory (empty initially)
├── models/                         # Model storage
│   ├── logs/                      # TensorBoard logs (gitignored)
│   └── saved_models/              # Trained models (gitignored)
│       ├── multimodal_fraud_detection_transformer_v2.0_best.keras
│       ├── multimodal_fraud_detection_transformer_v2.0_final.keras
│       └── multimodal_fraud_detection_transformer_v2.0_preprocessor.pkl
├── notebooks/                      # Jupyter notebooks
│   └── fraud_detection_demo.ipynb
├── src/                            # Source code
│   ├── models/                    # Model architectures (MODULAR)
│   │   ├── __init__.py            # Package exports
│   │   ├── attention.py           # Attention mechanisms
│   │   ├── layers.py              # Core layers (FeedForward, Residual)
│   │   ├── blocks.py              # Transformer blocks
│   │   ├── embeddings.py          # Embedding layers (PatchEmbedding)
│   │   ├── transformer.py         # Model builders
│   │   └── transformer_model.py   # Legacy (backward compatible)
│   ├── utils/                     # Utility functions (MODULAR)
│   │   ├── __init__.py            # Package exports
│   │   ├── tabular_preprocessor.py     # FraudDataPreprocessor
│   │   ├── image_preprocessor.py       # QRCodePreprocessor
│   │   ├── multimodal_preprocessor.py  # MultimodalDataPreprocessor
│   │   └── data_preprocessing.py       # Legacy (backward compatible)
│   ├── __init__.py
│   ├── train.py                  # Training script
│   └── predict.py                # Inference script
├── .gitignore                     # Git ignore rules
├── README.md                      # Main documentation
├── requirements.txt               # Python dependencies
├── quick_start.py                # Quick start example
└── SUMMARY.md                    # This file
```

## Package Architecture

### Model Package (src/models/)

The model package is organized into separate modules for clarity:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        src/models/ Package                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        attention.py                                │  │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐     │  │
│  │  │MultiHeadSelf    │ │MaskedMultiHead  │ │CrossModal       │     │  │
│  │  │Attention        │ │Attention        │ │Attention        │     │  │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────┘     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                     │
│                                    ▼                                     │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                          layers.py                                 │  │
│  │         ┌─────────────────┐    ┌─────────────────┐                │  │
│  │         │  FeedForward    │    │ResidualConnection│                │  │
│  │         └─────────────────┘    └─────────────────┘                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                     │
│                                    ▼                                     │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                          blocks.py                                 │  │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐     │  │
│  │  │TransformerBlock │ │TransformerDeco  │ │CrossModalTrans  │     │  │
│  │  │                 │ │derBlock         │ │formerBlock      │     │  │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────┘     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                     │
│              ┌─────────────────────┼─────────────────────┐              │
│              ▼                     ▼                     ▼              │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐       │
│  │ embeddings.py   │   │ transformer.py  │   │transformer_     │       │
│  │                 │   │                 │   │model.py (Legacy)│       │
│  │ PatchEmbedding  │   │ Multimodal      │   │                 │       │
│  │                 │   │ Transformer     │   │ All classes in  │       │
│  │                 │   │                 │   │ single file     │       │
│  │                 │   │ Fraud Detection │   │ (backward       │       │
│  │                 │   │ Transformer     │   │ compatible)     │       │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Utils Package (src/utils/)

The utils package provides modular data preprocessing:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         src/utils/ Package                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────┐    ┌────────────────────────────┐       │
│  │  tabular_preprocessor.py   │    │   image_preprocessor.py    │       │
│  │  ──────────────────────────│    │  ──────────────────────────│       │
│  │  FraudDataPreprocessor     │    │  QRCodePreprocessor        │       │
│  │  ├─ generate_synthetic_data│    │  ├─ generate_qr_images     │       │
│  │  ├─ preprocess_data        │    │  ├─ load_from_directory    │       │
│  │  ├─ save_scaler            │    │  ├─ _generate_qr_pattern   │       │
│  │  ├─ load_scaler            │    │  └─ prepare_train_test_data│       │
│  │  └─ prepare_train_test_data│    │                            │       │
│  └────────────────────────────┘    └────────────────────────────┘       │
│                   │                              │                       │
│                   └──────────────┬───────────────┘                      │
│                                  ▼                                       │
│              ┌────────────────────────────────────────┐                 │
│              │      multimodal_preprocessor.py        │                 │
│              │  ──────────────────────────────────────│                 │
│              │  MultimodalDataPreprocessor            │                 │
│              │  ├─ generate_synthetic_multimodal_data │                 │
│              │  ├─ preprocess_multimodal_data         │                 │
│              │  ├─ prepare_train_test_data            │                 │
│              │  ├─ save_preprocessors                 │                 │
│              │  └─ load_preprocessors                 │                 │
│              └────────────────────────────────────────┘                 │
│                                                                          │
│              ┌────────────────────────────────────────┐                 │
│              │    data_preprocessing.py (Legacy)      │                 │
│              │  ──────────────────────────────────────│                 │
│              │  All classes in single file            │                 │
│              │  (For backward compatibility)          │                 │
│              └────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Multimodal Transformer Implementation
- Tabular data encoder with self-attention
- Vision Transformer (ViT) for QR code images
- Cross-modal attention for information fusion
- Properly registered Keras serializable layers
- Configurable architecture
- **Modular code structure** for easy understanding

### 2. Data Processing
- **Tabular Data**:
  - Feature encoding (categorical → numerical)
  - Feature scaling and normalization
  - Handles transaction features matching Kaggle dataset
- **Image Data**:
  - Patch extraction for Vision Transformer
  - Image normalization
  - Synthetic QR code generation for testing

## Class Flow Diagrams

### Complete Model Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Complete Fraud Detection Pipeline                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Raw Data                                                                   │
│   ────────                                                                   │
│   ┌─────────────────┐              ┌─────────────────┐                      │
│   │ Transaction CSV │              │  QR Code Images │                      │
│   └────────┬────────┘              └────────┬────────┘                      │
│            │                                │                                │
│            ▼                                ▼                                │
│   ┌─────────────────┐              ┌─────────────────┐                      │
│   │FraudData        │              │QRCode           │                      │
│   │Preprocessor     │              │Preprocessor     │                      │
│   └────────┬────────┘              └────────┬────────┘                      │
│            │                                │                                │
│            └────────────┬───────────────────┘                               │
│                         ▼                                                    │
│            ┌────────────────────────┐                                        │
│            │MultimodalData          │                                        │
│            │Preprocessor            │                                        │
│            └────────────┬───────────┘                                        │
│                         │                                                    │
│   Preprocessed Data     │                                                    │
│   ─────────────────     │                                                    │
│   ┌─────────────────┐   │   ┌─────────────────┐                             │
│   │ Tabular Features│◄──┴──►│  Image Patches  │                             │
│   │ (1, 8) scaled   │       │ (128,128,3)     │                             │
│   └────────┬────────┘       └────────┬────────┘                             │
│            │                         │                                       │
│            ▼                         ▼                                       │
│   ┌──────────────────────────────────────────────────────┐                  │
│   │         MultimodalFraudDetectionTransformer          │                  │
│   │  ┌────────────────┐    ┌────────────────┐           │                  │
│   │  │ Tabular        │    │ Image          │           │                  │
│   │  │ Encoder        │    │ Encoder (ViT)  │           │                  │
│   │  │ ┌────────────┐ │    │ ┌────────────┐ │           │                  │
│   │  │ │Transformer │ │    │ │PatchEmbed  │ │           │                  │
│   │  │ │Block x2    │ │    │ │Transformer │ │           │                  │
│   │  │ └────────────┘ │    │ │Block x2    │ │           │                  │
│   │  └───────┬────────┘    └──────┬───────┘│           │                  │
│   │          │                    │         │           │                  │
│   │          └─────────┬──────────┘         │           │                  │
│   │                    ▼                    │           │                  │
│   │          ┌──────────────────┐           │           │                  │
│   │          │CrossModal        │           │           │                  │
│   │          │TransformerBlock  │           │           │                  │
│   │          │x2                │           │           │                  │
│   │          └────────┬─────────┘           │           │                  │
│   │                   │                     │           │                  │
│   │                   ▼                     │           │                  │
│   │          ┌──────────────────┐           │           │                  │
│   │          │Classification    │           │           │                  │
│   │          │Head (Dense)      │           │           │                  │
│   │          └────────┬─────────┘           │           │                  │
│   └───────────────────┼─────────────────────┘           │                  │
│                       │                                  │                  │
│   Output              ▼                                                     │
│   ──────   ┌─────────────────┐                                              │
│            │ Fraud Probability│                                              │
│            │ 0.0 - 1.0        │                                              │
│            └─────────────────┘                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### TransformerBlock Internal Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TransformerBlock Internal Flow                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Input: x ──────────────────────────────────────────────────────┐          │
│          │                                                        │ (Skip)   │
│          ▼                                                        │          │
│   ┌──────────────────────────────────────────────────────────────┐│          │
│   │                  MultiHeadSelfAttention                       ││          │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐                       ││          │
│   │  │   Wq    │  │   Wk    │  │   Wv    │                       ││          │
│   │  └────┬────┘  └────┬────┘  └────┬────┘                       ││          │
│   │       │            │            │                             ││          │
│   │       ▼            ▼            ▼                             ││          │
│   │  ┌──────────────────────────────────────────────────────┐    ││          │
│   │  │ Q·K^T / √d_k  →  softmax  →  × V  →  concat heads   │    ││          │
│   │  └────────────────────────────────────────────────────────┘    ││          │
│   │                            │                                   ││          │
│   │                            ▼                                   ││          │
│   │                     ┌────────────┐                             ││          │
│   │                     │  Wo (Dense)│                             ││          │
│   │                     └──────┬─────┘                             ││          │
│   └────────────────────────────┼───────────────────────────────────┘│          │
│                                │                                    │          │
│                                ▼                                    │          │
│                         ┌───────────┐                               │          │
│                         │  Dropout  │                               │          │
│                         └─────┬─────┘                               │          │
│                               │◄────────────────────────────────────┘          │
│                               ▼                                                 │
│                         ┌───────────┐                                           │
│                         │    Add    │                                           │
│                         └─────┬─────┘                                           │
│                               ▼                                                 │
│                         ┌───────────┐                                           │
│                         │ LayerNorm │                                           │
│                         └─────┬─────┘                                           │
│                               │                                                 │
│                               ├─────────────────────────────────────┐ (Skip)   │
│                               ▼                                     │          │
│   ┌─────────────────────────────────────────────────────────────┐   │          │
│   │                      FeedForward                             │   │          │
│   │  ┌────────────────┐  ┌────────────┐  ┌────────────────┐    │   │          │
│   │  │ Dense(d → dff) │→ │   ReLU     │→ │ Dense(dff → d) │    │   │          │
│   │  └────────────────┘  └────────────┘  └────────────────┘    │   │          │
│   └───────────────────────────┬─────────────────────────────────┘   │          │
│                               ▼                                     │          │
│                         ┌───────────┐                               │          │
│                         │  Dropout  │                               │          │
│                         └─────┬─────┘                               │          │
│                               │◄────────────────────────────────────┘          │
│                               ▼                                                 │
│                         ┌───────────┐                                           │
│                         │    Add    │                                           │
│                         └─────┬─────┘                                           │
│                               ▼                                                 │
│                         ┌───────────┐                                           │
│                         │ LayerNorm │                                           │
│                         └─────┬─────┘                                           │
│                               │                                                 │
│   Output: y ◄─────────────────┘                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Cross-Modal Attention Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CrossModalTransformerBlock Flow                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│        Tabular                                  Image                        │
│   Features (T)                            Features (I)                       │
│          │                                       │                           │
│          │          ┌───────────────────────────┐│                           │
│          │          │   CrossModalAttention     ││                           │
│          ├─────────►│   Q: from T               ││                           │
│          │          │   K,V: from I             │◄────────────────────────┤ │
│          │          │   T attends to I          ││                        │ │
│          │          └───────────┬───────────────┘│                        │ │
│          │                      │                │                        │ │
│          │◄─────────Add & Norm──┘                │                        │ │
│          │                                       │                        │ │
│          │          ┌───────────────────────────┐│                        │ │
│          │          │       FeedForward         ││                        │ │
│          │          │       (for T)             ││                        │ │
│          │          └───────────┬───────────────┘│                        │ │
│          │                      │                │                        │ │
│          │◄─────────Add & Norm──┘                │                        │ │
│          │                                       │                        │ │
│          │          ┌───────────────────────────┐│                        │ │
│          │          │   CrossModalAttention     ││                        │ │
│          ├─────────►│   Q: from I               ││                        │ │
│          │          │   K,V: from T             │├────────────────────────┘ │
│          │          │   I attends to T          ││                          │
│          │          └───────────┬───────────────┘│                          │
│          │                      │                │                          │
│          │                      └────────────────┼──Add & Norm──►│          │
│          │                                       │               │          │
│          │          ┌───────────────────────────┐│               │          │
│          │          │       FeedForward         ││               │          │
│          │          │       (for I)             ││               │          │
│          │          └───────────┬───────────────┘│               │          │
│          │                      │                │               │          │
│          │                      └────────────────┼──Add & Norm──►│          │
│          │                                       │               │          │
│          ▼                                       ▼               ▼          │
│   Updated T                               Updated I                         │
│   (Attended to I)                        (Attended to T)                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Training Pipeline
- Multimodal batch training
- Automated model checkpointing
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- TensorBoard logging for monitoring
- Multiple save formats (Keras)

### 4. Inference Pipeline
- Multimodal predictor class
- Tabular-only predictor (backward compatible)
- Single transaction prediction
- Batch prediction support
- Automatic model format detection

### 5. Model Persistence
All models are automatically saved to `models/saved_models/` directory:
- **Best checkpoint**: Saved during training based on validation loss
- **Final model**: Saved at the end of training
- **Preprocessor**: Saved for consistent preprocessing during inference
- **Logs**: TensorBoard logs for training visualization

## Usage Examples

### Training
```bash
# Train multimodal model (default)
python src/train.py --mode multimodal

# Train tabular-only model
python src/train.py --mode tabular
```

### Inference
```bash
# Multimodal inference demo
python src/predict.py --mode multimodal

# Tabular-only inference demo
python src/predict.py --mode tabular
```

### Quick Start
```bash
# Multimodal quick start
python quick_start.py --mode multimodal

# Tabular-only quick start
python quick_start.py --mode tabular
```

### Using in Code
```python
from src.predict import MultimodalFraudDetectionPredictor

predictor = MultimodalFraudDetectionPredictor()
result = predictor.predict_single_transaction(
    tabular_features={'step': 1, 'type': 'TRANSFER', 'amount': 50000.0, ...},
    image=qr_code_image  # Normalized numpy array
)
print(f"Fraud Probability: {result['fraud_probability']:.2%}")
```

## Dependencies
- TensorFlow >= 2.13.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Scikit-learn >= 1.3.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- Jupyter >= 1.0.0

## Datasets

### Online Payments Fraud Detection Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset)
- **Features**:
  - `step`: Hour of simulation
  - `type`: Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN)
  - `amount`: Transaction amount
  - `nameOrig`: Customer originating the transaction
  - `oldbalanceOrg`: Initial balance before transaction
  - `newbalanceOrig`: Balance after transaction
  - `nameDest`: Recipient of the transaction
  - `oldbalanceDest`: Initial recipient balance
  - `newbalanceDest`: Recipient balance after transaction
  - `isFraud`: Target variable (1 = fraud, 0 = not fraud)
  - `isFlaggedFraud`: Business rule flagged as fraud

### Benign and Malicious QR Codes Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/samahsadiq/benign-and-malicious-qr-codes)
- **Structure**:
  - `benign/`: Directory containing benign QR code images
  - `malicious/`: Directory containing malicious QR code images
- **Image Size**: Resized to 128x128 for model input

## Model Characteristics

### Input Features
1. **Tabular** (10 features after preprocessing):
   - step, type (encoded), amount, oldbalanceOrg, newbalanceOrig
   - oldbalanceDest, newbalanceDest, isFlaggedFraud

2. **Image** (128x128x3):
   - RGB QR code images normalized to [0, 1]

### Model Parameters
- **Model dimension**: 64
- **Attention heads**: 4
- **Transformer layers**: 2 per modality
- **Cross-modal layers**: 2
- **Feed-forward dimension**: 128
- **Dropout rate**: 0.1
- **Patch size**: 16 (for images)

### Training Parameters
- **Batch size**: 32
- **Epochs**: 10 (with early stopping at 5 patience)
- **Learning rate**: 0.001 (with reduction on plateau)
- **Validation split**: 20%
- **Optimizer**: Adam

## Future Enhancements
1. Load and train on real Kaggle datasets
2. Add more sophisticated feature engineering
3. Implement attention visualization for explainability
4. Add API endpoint for production deployment
5. Implement online learning capabilities
6. Add model monitoring and drift detection
7. Support for additional image modalities (ID scans, receipts)

## Notes
- Models are saved in Keras format for flexibility
- Custom layers are properly registered for serialization
- All paths are configurable via `config/config.py`
- The system supports both CPU and GPU training
- TensorBoard logs are available for detailed training analysis
- Backward compatible with tabular-only mode
