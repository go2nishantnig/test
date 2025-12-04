# End-to-End Flow Documentation
# Multimodal Fraud Detection Transformer

This document provides a **complete end-to-end flow** of the multimodal fraud detection system, showing exactly when and how each documentation file and component is used throughout the pipeline.

---

## Table of Contents
1. [Quick Navigation](#quick-navigation)
2. [Complete System Flow](#complete-system-flow)
3. [Setup and Configuration Phase](#phase-1-setup-and-configuration)
4. [Data Preparation Phase](#phase-2-data-preparation)
5. [Model Architecture Phase](#phase-3-model-architecture)
6. [Training Phase](#phase-4-training)
7. [Inference Phase](#phase-5-inference)
8. [Quick Start Demo](#phase-6-quick-start-demo)

---

## Quick Navigation

### Documentation Files Reference
| File | Purpose | Used In Phase |
|------|---------|---------------|
| **README.md** | Project overview, installation, usage | Initial setup |
| **SUMMARY.md** | Technical summary, architecture details | Architecture understanding |
| **quick_start.md** | Quick start script documentation | Demo and testing |
| **config/config.md** | Configuration parameters | Setup, Training, Inference |
| **src/train.md** | Training pipeline documentation | Training |
| **src/predict.md** | Inference pipeline documentation | Inference |
| **src/models/attention.md** | Attention mechanisms | Architecture |
| **src/models/layers.md** | Core transformer layers | Architecture |
| **src/models/blocks.md** | Transformer blocks | Architecture |
| **src/models/embeddings.md** | Patch embedding for images | Architecture |
| **src/models/transformer.md** | Model builders | Architecture |
| **src/models/transformer_model.md** | Complete model implementation | Architecture |
| **src/utils/tabular_preprocessor.md** | Transaction data preprocessing | Data preparation |
| **src/utils/image_preprocessor.md** | QR code image preprocessing | Data preparation |
| **src/utils/multimodal_preprocessor.md** | Multimodal data coordination | Data preparation |
| **src/utils/data_preprocessing.md** | Legacy preprocessing (backward compatible) | Data preparation |

---

## Complete System Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      MULTIMODAL FRAUD DETECTION SYSTEM                    │
│                              End-to-End Pipeline                          │
└──────────────────────────────────────────────────────────────────────────┘

                              START HERE
                                  ↓
        ┌─────────────────────────────────────────────┐
        │  PHASE 1: Setup & Configuration             │
        │  📄 README.md, config/config.md             │
        └─────────────────────────────────────────────┘
                                  ↓
        ┌─────────────────────────────────────────────┐
        │  PHASE 2: Data Preparation                  │
        │  📄 tabular_preprocessor.md                 │
        │  📄 image_preprocessor.md                   │
        │  📄 multimodal_preprocessor.md              │
        └─────────────────────────────────────────────┘
                                  ↓
        ┌─────────────────────────────────────────────┐
        │  PHASE 3: Model Architecture                │
        │  📄 attention.md, layers.md, blocks.md      │
        │  📄 embeddings.md, transformer.md           │
        │  📄 SUMMARY.md (architecture diagrams)      │
        └─────────────────────────────────────────────┘
                                  ↓
        ┌─────────────────────────────────────────────┐
        │  PHASE 4: Training                          │
        │  📄 train.md, config.md                     │
        └─────────────────────────────────────────────┘
                                  ↓
        ┌─────────────────────────────────────────────┐
        │  PHASE 5: Inference                         │
        │  📄 predict.md                              │
        └─────────────────────────────────────────────┘
                                  ↓
        ┌─────────────────────────────────────────────┐
        │  PHASE 6: Quick Start Demo                  │
        │  📄 quick_start.md                          │
        └─────────────────────────────────────────────┘
                                  ↓
                              COMPLETE
```

---

## PHASE 1: Setup and Configuration

### Entry Point
**Start here:** `README.md`

### Flow
```
README.md (Project Overview)
    ↓
    ├─ Installation instructions
    ├─ Directory structure
    ├─ Feature overview
    └─ Quick start guide
    
    → THEN READ: config/config.md
    
config/config.md (Configuration Parameters)
    ↓
    ├─ MODEL_CONFIG (architecture settings)
    ├─ TRAINING_CONFIG (hyperparameters)
    ├─ Directory paths
    └─ Dataset paths
```

### Key Files Referenced
1. **README.md** - Complete project documentation
   - **When to use:** First time setup, understanding project structure
   - **What it provides:** Installation steps, architecture overview, usage examples
   
2. **config/config.md** - Configuration documentation
   - **When to use:** Before training or customizing model
   - **What it provides:** All configurable parameters, their meanings, and defaults

### Configuration Setup
```python
# Import configuration (as documented in config/config.md)
from config.config import MODEL_CONFIG, TRAINING_CONFIG, MODEL_SAVE_DIR

# Key configurations:
MODEL_CONFIG = {
    'd_model': 64,           # Embedding dimension
    'num_heads': 4,          # Attention heads
    'num_layers': 2,         # Transformer layers per modality
    'cross_modal_layers': 2, # Cross-modal fusion layers
    'dff': 128,             # Feed-forward dimension
    'dropout_rate': 0.1     # Dropout rate
}
```

**📄 Documentation:** See `config/config.md` for detailed parameter explanations

---

## PHASE 2: Data Preparation

### Overview
This phase prepares both tabular transaction data and QR code images for the model.

### Flow Diagram
```
Data Preparation Phase
        ↓
┌───────────────────────────────┐
│ Tabular Data                  │ 📄 tabular_preprocessor.md
│ (Transaction Features)        │
├───────────────────────────────┤
│ • Generate synthetic data     │
│ • Encode categorical features │
│ • Scale numeric features      │
│ • Reshape for transformer     │
└───────────────────────────────┘
        ↓
┌───────────────────────────────┐
│ Image Data                    │ 📄 image_preprocessor.md
│ (QR Code Images)              │
├───────────────────────────────┤
│ • Generate synthetic QR codes │
│ • Add malicious patterns      │
│ • Normalize to [0, 1]         │
│ • Resize to (128, 128, 3)     │
└───────────────────────────────┘
        ↓
┌───────────────────────────────┐
│ Multimodal Integration        │ 📄 multimodal_preprocessor.md
│ (Combine Both Modalities)     │
├───────────────────────────────┤
│ • Correlate QR with fraud     │
│ • Align samples               │
│ • Train/test split            │
│ • Stratified sampling         │
└───────────────────────────────┘
        ↓
    Ready for Training
```

### Step-by-Step Data Flow

#### Step 1: Tabular Data Preprocessing
**📄 Reference:** `src/utils/tabular_preprocessor.md`

```python
from src.utils.tabular_preprocessor import FraudDataPreprocessor

# Initialize preprocessor
preprocessor = FraudDataPreprocessor()

# Generate synthetic transaction data
data = preprocessor.generate_synthetic_data(n_samples=5000, fraud_ratio=0.1)

# Preprocess: encode, scale, reshape
X_tabular, y = preprocessor.preprocess_data(data, fit=True)
```

**What happens:**
1. Transaction types encoded (PAYMENT→0, TRANSFER→1, etc.)
2. Numeric features scaled to mean=0, std=1
3. ID columns dropped
4. Reshaped to (n_samples, 1, 8) for transformer input

**📄 For details:** See `tabular_preprocessor.md` sections:
- "Transaction Features" - Feature descriptions
- "Synthetic Data Generation" - Normal vs fraud patterns
- "Preprocessing Pipeline" - Step-by-step transformations

#### Step 2: Image Data Preprocessing
**📄 Reference:** `src/utils/image_preprocessor.md`

```python
from src.utils.image_preprocessor import QRCodePreprocessor

# Initialize preprocessor
qr_preprocessor = QRCodePreprocessor(image_size=(128, 128))

# Generate QR code images
images, labels = qr_preprocessor.generate_synthetic_qr_images(
    n_samples=5000,
    malicious_ratio=0.3
)
```

**What happens:**
1. Creates 128×128 QR code patterns
2. Adds position detection patterns (3 corners)
3. Fills data area with random blocks
4. For malicious: adds noise + color tint
5. Normalizes to [0, 1] range

**📄 For details:** See `image_preprocessor.md` sections:
- "QR Code Structure" - QR code components
- "Position Detection Patterns" - Visual patterns
- "Malicious QR Code Characteristics" - How fraud is simulated

#### Step 3: Multimodal Integration
**📄 Reference:** `src/utils/multimodal_preprocessor.md`

```python
from src.utils.multimodal_preprocessor import MultimodalDataPreprocessor

# Initialize multimodal preprocessor
preprocessor = MultimodalDataPreprocessor(image_size=(128, 128))

# Generate correlated multimodal data
data = preprocessor.prepare_train_test_data(
    test_size=0.2,
    random_state=42,
    n_samples=5000
)
```

**What happens:**
1. Generates tabular transaction data
2. For each transaction, determines QR maliciousness:
   - If fraud: 80% chance malicious QR
   - If normal: 10% chance malicious QR
3. Generates correlated QR images
4. Stratified train/test split (maintains fraud ratio)
5. Preprocesses both modalities with fit/transform pattern

**Output:**
```python
{
    'X_train_tabular': (4000, 1, 8),      # Transaction features
    'X_train_images': (4000, 128, 128, 3), # QR code images
    'y_train': (4000,),                    # Labels
    'X_test_tabular': (1000, 1, 8),
    'X_test_images': (1000, 128, 128, 3),
    'y_test': (1000,)
}
```

**📄 For details:** See `multimodal_preprocessor.md` sections:
- "Correlated Generation Strategy" - How modalities are linked
- "Correlation Logic" - Probability distributions
- "Synchronized Splitting" - Maintaining alignment

---

## PHASE 3: Model Architecture

### Overview
This phase builds the multimodal transformer architecture combining tabular and image encoders with cross-modal fusion.

### Architecture Documentation Flow
```
Understanding the Architecture
        ↓
┌────────────────────────────────┐
│ SUMMARY.md                     │ ← START HERE for overview
│ • Complete architecture diagram│
│ • Component relationships      │
│ • Flow diagrams                │
└────────────────────────────────┘
        ↓
┌────────────────────────────────┐
│ Attention Mechanisms           │ 📄 attention.md
│ • MultiHeadSelfAttention       │
│ • MaskedMultiHeadAttention     │
│ • CrossModalAttention          │
└────────────────────────────────┘
        ↓
┌────────────────────────────────┐
│ Core Layers                    │ 📄 layers.md
│ • FeedForward                  │
│ • ResidualConnection           │
└────────────────────────────────┘
        ↓
┌────────────────────────────────┐
│ Transformer Blocks             │ 📄 blocks.md
│ • TransformerBlock (encoder)   │
│ • TransformerDecoderBlock      │
│ • CrossModalTransformerBlock   │
└────────────────────────────────┘
        ↓
┌────────────────────────────────┐
│ Embeddings                     │ 📄 embeddings.md
│ • PatchEmbedding (for images)  │
└────────────────────────────────┘
        ↓
┌────────────────────────────────┐
│ Model Builders                 │ 📄 transformer.md
│ • MultimodalFraudDetectionTransformer│
│ • FraudDetectionTransformer    │
└────────────────────────────────┘
```

### Component-by-Component Build

#### Level 1: Attention Mechanisms
**📄 Reference:** `src/models/attention.md`

```python
from src.models.attention import (
    MultiHeadSelfAttention,
    MaskedMultiHeadAttention,
    CrossModalAttention
)

# Multi-head self-attention for encoders
self_attention = MultiHeadSelfAttention(d_model=64, num_heads=4)

# Cross-modal attention for fusion
cross_attention = CrossModalAttention(d_model=64, num_heads=4)
```

**When to read `attention.md`:**
- Understanding how attention works
- Learning about Q, K, V projections
- Understanding scaled dot-product attention
- Learning cross-modal information exchange

**Key sections:**
- "Attention Formula" - Mathematical foundation
- "MultiHeadSelfAttention" - Self-attention implementation
- "CrossModalAttention" - Cross-modal fusion mechanism
- "Attention Visualization" - Examples of attention patterns

#### Level 2: Core Layers
**📄 Reference:** `src/models/layers.md`

```python
from src.models.layers import FeedForward, ResidualConnection

# Feed-forward network
ffn = FeedForward(d_model=64, dff=128, dropout_rate=0.1)

# Residual connection with layer normalization
residual = ResidualConnection(d_model=64, dropout_rate=0.1)
```

**When to read `layers.md`:**
- Understanding feed-forward networks
- Learning about residual connections
- Understanding layer normalization
- Learning why these are needed

**Key sections:**
- "FeedForward" - Position-wise FFN details
- "ResidualConnection" - Add & Norm implementation
- "Why Residual Connections?" - Benefits explained
- "Layer Normalization" - How it stabilizes training

#### Level 3: Transformer Blocks
**📄 Reference:** `src/models/blocks.md`

```python
from src.models.blocks import (
    TransformerBlock,
    TransformerDecoderBlock,
    CrossModalTransformerBlock
)

# Encoder block
encoder_block = TransformerBlock(
    d_model=64,
    num_heads=4,
    dff=128,
    dropout_rate=0.1
)

# Cross-modal fusion block
cross_modal_block = CrossModalTransformerBlock(
    d_model=64,
    num_heads=4,
    dff=128,
    dropout_rate=0.1
)
```

**When to read `blocks.md`:**
- Understanding complete transformer blocks
- Learning how components are assembled
- Understanding encoder vs decoder blocks
- Learning bidirectional cross-modal fusion

**Key sections:**
- "TransformerBlock" - Standard encoder block flow
- "CrossModalTransformerBlock" - Multimodal fusion
- "Why Bidirectional?" - Explanation of two-way attention
- "Stacking Blocks" - Building deeper networks

#### Level 4: Embeddings
**📄 Reference:** `src/models/embeddings.md`

```python
from src.models.embeddings import PatchEmbedding

# Convert images to patch sequences
patch_embedding = PatchEmbedding(
    image_size=(128, 128),
    patch_size=16,
    d_model=64
)
```

**When to read `embeddings.md`:**
- Understanding Vision Transformer (ViT)
- Learning patch extraction
- Understanding position embeddings
- Learning image-to-sequence conversion

**Key sections:**
- "Concept" - Vision Transformer overview
- "Patch Extraction" - How images become sequences
- "Position Embeddings" - Adding spatial information
- "Why This Works" - Advantages for QR codes

#### Level 5: Complete Model
**📄 Reference:** `src/models/transformer.md` and `SUMMARY.md`

```python
from src.models.transformer import MultimodalFraudDetectionTransformer

# Build complete multimodal model
model_builder = MultimodalFraudDetectionTransformer(MODEL_CONFIG)
model = model_builder.build_model()
```

**Architecture Flow:**
```
Tabular Input (1, 8)                Image Input (128, 128, 3)
        ↓                                      ↓
   Dense(64)                            PatchEmbedding
        ↓                                      ↓
Position Embedding                      Position Embedding
        ↓                                      ↓
TransformerBlock ×2                     TransformerBlock ×2
        ↓                                      ↓
        └──────────────────┬───────────────────┘
                           ↓
              CrossModalTransformerBlock ×2
                    (Bidirectional Fusion)
                           ↓
                ┌──────────┴──────────┐
                ↓                     ↓
        GlobalAvgPool          GlobalAvgPool
                ↓                     ↓
                └──────────┬──────────┘
                           ↓
                     Concatenate
                           ↓
                    Dense(128) + ReLU
                           ↓
                      Dropout(0.1)
                           ↓
                     Dense(64) + ReLU
                           ↓
                      Dropout(0.1)
                           ↓
                    Dense(1) + Sigmoid
                           ↓
                  Fraud Probability [0, 1]
```

**When to read:**
- `transformer.md` - Model builder implementation
- `SUMMARY.md` - Complete architecture diagrams and flow charts

---

## PHASE 4: Training

### Entry Point
**📄 Reference:** `src/train.md`

### Training Flow
```
Training Phase
        ↓
┌─────────────────────────────────────┐
│ 1. Load Configuration               │ 📄 config/config.md
│    • MODEL_CONFIG                   │
│    • TRAINING_CONFIG                │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ 2. Prepare Data                     │ 📄 multimodal_preprocessor.md
│    • Generate synthetic data        │
│    • Preprocess both modalities     │
│    • Train/test split               │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ 3. Build Model                      │ 📄 All architecture docs
│    • Create multimodal transformer  │
│    • Compile with optimizer         │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ 4. Setup Callbacks                  │ 📄 train.md
│    • ModelCheckpoint (save best)    │
│    • EarlyStopping (prevent overfit)│
│    • ReduceLROnPlateau (lr schedule)│
│    • TensorBoard (logging)          │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ 5. Train Model                      │
│    • Fit on training data           │
│    • Validate on test data          │
│    • Monitor metrics                │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ 6. Save Artifacts                   │
│    • Best model checkpoint          │
│    • Final model (multiple formats) │
│    • Preprocessor state             │
│    • Training logs                  │
└─────────────────────────────────────┘
        ↓
    Training Complete
```

### Command-Line Usage
```bash
# Train multimodal model (default)
python src/train.py --mode multimodal

# Train tabular-only model (backward compatible)
python src/train.py --mode tabular
```

### Detailed Training Steps

#### Step 1: Initialize Training
**📄 Reference:** `src/train.md` → "train_multimodal_model()"

```python
from src.models.transformer import MultimodalFraudDetectionTransformer
from src.utils.multimodal_preprocessor import MultimodalDataPreprocessor
from config.config import MODEL_CONFIG, TRAINING_CONFIG

# Load configuration (see config/config.md)
print("Configuration:")
print(f"  d_model: {MODEL_CONFIG['d_model']}")
print(f"  num_heads: {MODEL_CONFIG['num_heads']}")
print(f"  batch_size: {TRAINING_CONFIG['batch_size']}")
print(f"  epochs: {TRAINING_CONFIG['epochs']}")
```

#### Step 2: Prepare Training Data
**📄 Reference:** `src/train.md` → "Data Preparation"

```python
# Initialize preprocessor (see multimodal_preprocessor.md)
preprocessor = MultimodalDataPreprocessor(
    image_size=MODEL_CONFIG.get('image_size', (128, 128))
)

# Generate and prepare data
data = preprocessor.prepare_train_test_data(
    test_size=TRAINING_CONFIG['validation_split'],  # 0.2
    n_samples=5000
)

print(f"Training samples: {len(data['y_train'])}")
print(f"Test samples: {len(data['y_test'])}")
print(f"Fraud ratio: {data['y_train'].mean():.2%}")
```

#### Step 3: Build and Compile Model
**📄 Reference:** `src/train.md` → "Model Building"

```python
# Build model (architecture from attention.md, layers.md, blocks.md)
multimodal_model = MultimodalFraudDetectionTransformer(MODEL_CONFIG)
model = multimodal_model.build_model()

# Compile model
model = multimodal_model.compile_model(
    learning_rate=TRAINING_CONFIG['learning_rate']
)

model.summary()
```

#### Step 4: Configure Callbacks
**📄 Reference:** `src/train.md` → "Training Callbacks"

```python
import tensorflow.keras as keras

# Save best model during training
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='models/saved_models/best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Stop if no improvement
early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate on plateau
reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# TensorBoard logging
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='logs/',
    histogram_freq=1
)
```

#### Step 5: Train
**📄 Reference:** `src/train.md` → "Training Loop"

```python
# Train model with dual inputs
history = model.fit(
    [data['X_train_tabular'], data['X_train_images']],
    data['y_train'],
    batch_size=TRAINING_CONFIG['batch_size'],
    epochs=TRAINING_CONFIG['epochs'],
    validation_data=(
        [data['X_test_tabular'], data['X_test_images']],
        data['y_test']
    ),
    callbacks=[
        checkpoint_callback,
        early_stopping_callback,
        reduce_lr_callback,
        tensorboard_callback
    ],
    verbose=1
)
```

#### Step 6: Evaluate and Save
**📄 Reference:** `src/train.md` → "Model Saving"

```python
# Evaluate on test set
test_loss, test_acc = model.evaluate(
    [data['X_test_tabular'], data['X_test_images']],
    data['y_test'],
    verbose=1
)

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Save final model in multiple formats
model.save('models/saved_models/final_model.keras')
model.save('models/saved_models/final_model.h5')

# Save preprocessor
preprocessor.save_preprocessors('models/saved_models/preprocessor.pkl')
```

### Monitoring Training
**📄 Reference:** `README.md` → "Monitoring Training"

```bash
# View training progress in TensorBoard
tensorboard --logdir logs/

# Open browser to http://localhost:6006
```

---

## PHASE 5: Inference

### Entry Point
**📄 Reference:** `src/predict.md`

### Inference Flow
```
Inference Phase
        ↓
┌─────────────────────────────────────┐
│ 1. Load Trained Model               │
│    • Load .keras or .h5 model       │
│    • Load preprocessor state        │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ 2. Prepare New Data                 │
│    • Format transaction features    │
│    • Load/prepare QR code image     │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ 3. Preprocess Inputs                │ 📄 multimodal_preprocessor.md
│    • Encode transaction type        │
│    • Scale numeric features         │
│    • Normalize image to [0, 1]      │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ 4. Run Prediction                   │
│    • Forward pass through model     │
│    • Get fraud probability          │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ 5. Return Results                   │
│    • Fraud probability [0, 1]       │
│    • Binary prediction (threshold)  │
│    • Confidence score               │
└─────────────────────────────────────┘
```

### Command-Line Usage
```bash
# Run multimodal inference demo
python src/predict.py --mode multimodal

# Run tabular-only inference demo
python src/predict.py --mode tabular
```

### Detailed Inference Steps

#### Step 1: Initialize Predictor
**📄 Reference:** `src/predict.md` → "MultimodalFraudDetectionPredictor"

```python
from src.predict import MultimodalFraudDetectionPredictor

# Initialize predictor (loads model and preprocessor)
predictor = MultimodalFraudDetectionPredictor()

# Model loaded from: models/saved_models/final_model.keras
# Preprocessor loaded from: models/saved_models/preprocessor.pkl
```

**What happens:**
1. Loads trained Keras model
2. Loads fitted preprocessor (scaler, encoders)
3. Ready for inference

#### Step 2: Prepare Transaction Data
**📄 Reference:** `src/predict.md` → "Single Transaction Prediction"

```python
# New transaction to predict
transaction = {
    'step': 100,
    'type': 'TRANSFER',
    'amount': 50000.0,
    'nameOrig': 'C1234567890',
    'oldbalanceOrg': 100000.0,
    'newbalanceOrig': 50000.0,
    'nameDest': 'C9876543210',
    'oldbalanceDest': 10000.0,
    'newbalanceDest': 60000.0,
    'isFlaggedFraud': 0
}
```

#### Step 3: Prepare QR Code Image
**📄 Reference:** `src/predict.md` and `image_preprocessor.md`

```python
import numpy as np

# Load QR code image (normalized to [0, 1])
# In practice, load from file and normalize
qr_image = np.random.rand(128, 128, 3)  # Placeholder

# Or generate synthetic for testing
from src.utils.image_preprocessor import QRCodePreprocessor
qr_gen = QRCodePreprocessor()
qr_image = qr_gen._generate_qr_pattern(malicious=False) / 255.0
```

#### Step 4: Make Prediction
**📄 Reference:** `src/predict.md` → "Prediction Method"

```python
# Single transaction prediction
result = predictor.predict_single_transaction(
    tabular_features=transaction,
    image=qr_image,
    threshold=0.5
)

print(f"Is Fraud: {result['is_fraud']}")
print(f"Fraud Probability: {result['fraud_probability']:.4f}")
print(f"Prediction: {result['prediction']}")
```

**Output:**
```
Is Fraud: True
Fraud Probability: 0.8523
Prediction: 1
```

#### Step 5: Batch Prediction
**📄 Reference:** `src/predict.md` → "Batch Prediction"

```python
import pandas as pd

# Multiple transactions
transactions_df = pd.DataFrame([
    {'step': 1, 'type': 'PAYMENT', 'amount': 100.0, ...},
    {'step': 2, 'type': 'TRANSFER', 'amount': 50000.0, ...},
    {'step': 3, 'type': 'CASH_OUT', 'amount': 25000.0, ...}
])

# Multiple QR images (n_samples, 128, 128, 3)
qr_images = np.random.rand(3, 128, 128, 3)

# Batch prediction
results = predictor.predict(transactions_df, qr_images, threshold=0.5)

print(f"Predictions: {results['predictions']}")
print(f"Probabilities: {results['probabilities']}")
print(f"Is Fraud: {results['is_fraud']}")
```

### Tabular-Only Inference (Backward Compatible)
**📄 Reference:** `src/predict.md` → "FraudDetectionPredictor"

```python
from src.predict import FraudDetectionPredictor

# Tabular-only predictor
predictor = FraudDetectionPredictor()

# Predict with transaction only (no QR code)
result = predictor.predict_single_transaction(transaction)

print(f"Fraud Probability: {result['fraud_probability']:.4f}")
```

---

## PHASE 6: Quick Start Demo

### Entry Point
**📄 Reference:** `quick_start.md`

### Demo Flow
```
Quick Start Demo
        ↓
┌─────────────────────────────────────┐
│ Choose Demo Mode                    │
│ • multimodal (default)              │
│ • tabular (legacy)                  │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ Multimodal Demo                     │ 📄 quick_start.md
│ • Load multimodal predictor         │
│ • Generate sample transactions      │
│ • Generate sample QR codes          │
│ • Run predictions                   │
│ • Display results                   │
└─────────────────────────────────────┘
        ↓
        OR
        ↓
┌─────────────────────────────────────┐
│ Tabular Demo                        │ 📄 quick_start.md
│ • Load tabular predictor            │
│ • Create sample transactions        │
│ • Run predictions                   │
│ • Display results                   │
└─────────────────────────────────────┘
```

### Command-Line Usage
```bash
# Run multimodal demo (default)
python quick_start.py

# Or explicitly
python quick_start.py --mode multimodal

# Run tabular-only demo
python quick_start.py --mode tabular
```

### Demo Examples

#### Multimodal Demo
**📄 Reference:** `quick_start.md` → "demo_multimodal()"

```python
from src.predict import MultimodalFraudDetectionPredictor
from src.utils.multimodal_preprocessor import MultimodalDataPreprocessor

# Initialize
predictor = MultimodalFraudDetectionPredictor()
preprocessor = MultimodalDataPreprocessor()

# Example 1: Normal transaction with benign QR
normal_data = preprocessor.generate_synthetic_multimodal_data(
    n_samples=1, 
    fraud_ratio=0.0
)
processed = preprocessor.preprocess_multimodal_data(
    normal_data['tabular'], 
    normal_data['images'], 
    fit=True
)
result = predictor.predict(processed['tabular'], processed['images'])

print("Normal Transaction + Benign QR:")
print(f"  Fraud Probability: {result['probabilities'][0]:.4f}")
print(f"  Predicted: {'FRAUD' if result['is_fraud'][0] else 'NORMAL'}")

# Example 2: Fraudulent transaction with malicious QR
fraud_data = preprocessor.generate_synthetic_multimodal_data(
    n_samples=1, 
    fraud_ratio=1.0
)
processed = preprocessor.preprocess_multimodal_data(
    fraud_data['tabular'], 
    fraud_data['images'], 
    fit=False
)
result = predictor.predict(processed['tabular'], processed['images'])

print("Fraudulent Transaction + Malicious QR:")
print(f"  Fraud Probability: {result['probabilities'][0]:.4f}")
print(f"  Predicted: {'FRAUD' if result['is_fraud'][0] else 'NORMAL'}")
```

#### Tabular Demo
**📄 Reference:** `quick_start.md` → "demo_tabular()"

```python
from src.predict import FraudDetectionPredictor

predictor = FraudDetectionPredictor()

# Example 1: Normal transaction
normal_transaction = {
    'step': 1,
    'type': 'PAYMENT',
    'amount': 50.00,
    'nameOrig': 'C1234567890',
    'oldbalanceOrg': 50000.0,
    'newbalanceOrig': 49950.0,
    'nameDest': 'M1234567890',
    'oldbalanceDest': 5000.0,
    'newbalanceDest': 5050.0,
    'isFlaggedFraud': 0
}
result = predictor.predict_single_transaction(normal_transaction)
print(f"Normal Transaction: {result['fraud_probability']:.4f}")

# Example 2: Suspicious transaction
suspicious_transaction = {
    'step': 100,
    'type': 'TRANSFER',
    'amount': 50000.0,
    'nameOrig': 'C1234567890',
    'oldbalanceOrg': 100000.0,
    'newbalanceOrig': 50000.0,
    'nameDest': 'C9876543210',
    'oldbalanceDest': 10000.0,
    'newbalanceDest': 60000.0,
    'isFlaggedFraud': 0
}
result = predictor.predict_single_transaction(suspicious_transaction)
print(f"Suspicious Transaction: {result['fraud_probability']:.4f}")
```

---

## Complete Documentation Map

### By Learning Path

#### Path 1: Quick Start (30 minutes)
1. **README.md** - Project overview and installation
2. **quick_start.md** - Run demo immediately
3. **predict.md** - Understand inference

#### Path 2: Development (2-3 hours)
1. **README.md** - Full project documentation
2. **config/config.md** - Understand configuration
3. **SUMMARY.md** - Architecture overview
4. **train.md** - Training pipeline
5. **predict.md** - Inference pipeline

#### Path 3: Deep Dive (Full day)
1. **README.md** - Foundation
2. **SUMMARY.md** - Architecture overview
3. **Data Processing:**
   - `tabular_preprocessor.md`
   - `image_preprocessor.md`
   - `multimodal_preprocessor.md`
4. **Architecture Components:**
   - `attention.md` - Attention mechanisms
   - `layers.md` - Core layers
   - `blocks.md` - Transformer blocks
   - `embeddings.md` - Patch embeddings
   - `transformer.md` - Model builders
5. **Pipeline:**
   - `train.md` - Training
   - `predict.md` - Inference
6. **Testing:**
   - `quick_start.md` - Demo

### By Task

#### Want to: Train a Model
**Read in order:**
1. `README.md` → Installation and setup
2. `config/config.md` → Configure hyperparameters
3. `multimodal_preprocessor.md` → Understand data preparation
4. `train.md` → Training process
5. `SUMMARY.md` → Architecture overview (optional)

**Run:**
```bash
python src/train.py --mode multimodal
```

#### Want to: Make Predictions
**Read in order:**
1. `README.md` → Project overview
2. `predict.md` → Inference process
3. `quick_start.md` → Example usage

**Run:**
```bash
python src/predict.py --mode multimodal
# or
python quick_start.py
```

#### Want to: Understand the Architecture
**Read in order:**
1. `SUMMARY.md` → High-level architecture
2. `attention.md` → How attention works
3. `layers.md` → Basic building blocks
4. `blocks.md` → How blocks are assembled
5. `embeddings.md` → Image-to-sequence conversion
6. `transformer.md` → Complete model

#### Want to: Modify Data Processing
**Read in order:**
1. `tabular_preprocessor.md` → Transaction processing
2. `image_preprocessor.md` → QR code processing
3. `multimodal_preprocessor.md` → Integration

#### Want to: Customize the Model
**Read in order:**
1. `config/config.md` → Configuration options
2. `SUMMARY.md` → Architecture overview
3. `attention.md`, `layers.md`, `blocks.md` → Components
4. `transformer.md` → Model builder

---

## Cross-Reference Index

### Configuration Parameters
- **Defined in:** `config/config.md`
- **Used in:** `train.md`, `predict.md`, `transformer.md`
- **Examples in:** `README.md`, `SUMMARY.md`

### Data Preprocessing
- **Tabular:** `tabular_preprocessor.md`
  - Used by: `multimodal_preprocessor.md`, `train.md`, `predict.md`
- **Image:** `image_preprocessor.md`
  - Used by: `multimodal_preprocessor.md`, `train.md`, `predict.md`
- **Multimodal:** `multimodal_preprocessor.md`
  - Used by: `train.md`, `predict.md`, `quick_start.md`

### Model Components
- **Attention:** `attention.md`
  - Used by: `blocks.md`, `transformer.md`
  - Referenced in: `SUMMARY.md`
- **Layers:** `layers.md`
  - Used by: `blocks.md`, `transformer.md`
  - Referenced in: `SUMMARY.md`
- **Blocks:** `blocks.md`
  - Used by: `transformer.md`
  - Referenced in: `SUMMARY.md`, `train.md`
- **Embeddings:** `embeddings.md`
  - Used by: `transformer.md`
  - Referenced in: `SUMMARY.md`

### Training Pipeline
- **Main doc:** `train.md`
- **References:** `config/config.md`, `multimodal_preprocessor.md`, `transformer.md`
- **Used in:** `README.md`, `SUMMARY.md`, `quick_start.md`

### Inference Pipeline
- **Main doc:** `predict.md`
- **References:** `config/config.md`, `multimodal_preprocessor.md`
- **Used in:** `README.md`, `SUMMARY.md`, `quick_start.md`

---

## Summary: When to Use Each File

| Documentation File | When to Use It |
|-------------------|----------------|
| **README.md** | First time setup, project overview, installation |
| **SUMMARY.md** | Understanding overall architecture, technical summary |
| **quick_start.md** | Running demos, quick testing, example usage |
| **config/config.md** | Configuring model, adjusting hyperparameters |
| **train.md** | Training models, understanding training pipeline |
| **predict.md** | Making predictions, understanding inference |
| **attention.md** | Understanding attention mechanisms, Q/K/V |
| **layers.md** | Understanding FFN, residual connections |
| **blocks.md** | Understanding transformer blocks, encoder/decoder |
| **embeddings.md** | Understanding patch embeddings, ViT |
| **transformer.md** | Understanding complete model architecture |
| **tabular_preprocessor.md** | Transaction data processing |
| **image_preprocessor.md** | QR code image processing |
| **multimodal_preprocessor.md** | Multimodal data integration |

---

## Quick Commands Reference

### Setup
```bash
# Clone repository
git clone <repository-url>
cd test

# Install dependencies (see README.md)
pip install -r requirements.txt
```

### Training
```bash
# Multimodal training (see train.md)
python src/train.py --mode multimodal

# Tabular-only training
python src/train.py --mode tabular
```

### Inference
```bash
# Multimodal inference demo (see predict.md)
python src/predict.py --mode multimodal

# Tabular-only inference demo
python src/predict.py --mode tabular
```

### Quick Start
```bash
# Run demo (see quick_start.md)
python quick_start.py

# Specify mode
python quick_start.py --mode multimodal
python quick_start.py --mode tabular
```

### Monitoring
```bash
# View training with TensorBoard (see README.md)
tensorboard --logdir logs/
```

---

## Conclusion

This document provides a complete map of the multimodal fraud detection system, showing exactly when and how each documentation file is used throughout the pipeline. Follow the phases sequentially for a complete understanding, or jump to specific sections based on your needs.

**For more details on any component, refer to the specific documentation file mentioned in each section.**

---

**Last Updated:** 2025-12-04
**Documentation Version:** 2.0
**Model Version:** v2.0
