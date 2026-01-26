# Multimodal Fraud Detection Transformer - Architecture Documentation

This document provides comprehensive architectural documentation for the Multimodal Fraud Detection Transformer system.

## Architecture Diagram

The complete architectural flow is documented in `architecture.puml` using PlantUML notation. This diagram shows:

- All classes with exact names from the codebase
- Complete class attributes and methods
- Relationships between components
- Data flow through the system
- Notes explaining key architectural patterns

## Viewing the PlantUML Diagram

### Option 1: Online PlantUML Viewer
1. Copy the contents of `architecture.puml`
2. Visit [PlantUML Online Server](http://www.plantuml.com/plantuml/uml/)
3. Paste the content and view the rendered diagram

### Option 2: VS Code Extension
1. Install the "PlantUML" extension in VS Code
2. Open `architecture.puml`
3. Press `Alt+D` to preview the diagram

### Option 3: Command Line (with PlantUML installed)
```bash
# Install PlantUML (requires Java)
# On Ubuntu/Debian:
sudo apt-get install plantuml

# Generate PNG image
plantuml architecture.puml

# Generate SVG image (better for documentation)
plantuml -tsvg architecture.puml
```

### Option 4: Docker
```bash
# Generate diagram using Docker (no local installation needed)
docker run --rm -v $(pwd):/data plantuml/plantuml architecture.puml
```

## Key Architectural Components

### 1. Entry Points
- **train.py**: Main training script with GPU configuration and model training
- **predict.py**: Inference script for making predictions

### 2. Data Preprocessing (src/utils/)
- **FraudDataPreprocessor**: Handles tabular transaction data preprocessing
- **QRCodePreprocessor**: Handles QR code image preprocessing
- **MultimodalDataPreprocessor**: Combines both data modalities

### 3. Core Transformer Components (src/models/)

#### Attention Mechanisms (attention.py)
- **MultiHeadSelfAttention**: Standard self-attention for encoder
- **MaskedMultiHeadAttention**: Causal masked attention for decoder
- **CrossModalAttention**: Cross-attention between different modalities

#### Transformer Blocks (blocks.py)
- **TransformerBlock**: Standard encoder block with self-attention + FFN
- **TransformerDecoderBlock**: Decoder block with masked attention + cross-attention + FFN
- **CrossModalTransformerBlock**: Bidirectional cross-modal fusion block

#### Core Layers (layers.py)
- **FeedForward**: Position-wise feed-forward network
- **ResidualConnection**: Residual connection with layer normalization

#### Embeddings (embeddings.py)
- **PatchEmbedding**: Vision Transformer style patch embedding for images

#### Main Models (transformer_model.py)
- **MultimodalFraudDetectionTransformer**: Combined tabular + image model
- **FraudDetectionTransformer**: Tabular-only model (backward compatible)

### 4. Prediction Classes (predict.py)
- **MultimodalFraudDetectionPredictor**: Predictor for multimodal model
- **FraudDetectionPredictor**: Predictor for tabular-only model

## Architecture Highlights

### Multimodal Architecture Flow
1. **Tabular Path**: Transaction features → Dense projection → Position encoding → TransformerBlocks
2. **Image Path**: QR codes → PatchEmbedding → TransformerBlocks
3. **Fusion**: Both modalities → CrossModalTransformerBlock (bidirectional attention)
4. **Output**: Global pooling → Concatenate → Dense layers → Sigmoid (fraud prediction)

### Tabular-Only Architecture Flow (Backward Compatible)
1. Transaction features → Dense projection → Position encoding
2. Stack of TransformerBlocks
3. Global pooling → Dense layers → Sigmoid (fraud prediction)

### Cross-Modal Attention
The `CrossModalTransformerBlock` implements bidirectional cross-attention:
- Tabular features attend to image features (learning visual fraud patterns)
- Image features attend to tabular features (contextualizing visual patterns)

This enables rich multimodal fusion where both modalities inform each other.

### Vision Transformer Approach
Images are processed using the ViT (Vision Transformer) approach:
1. Divide image into non-overlapping patches
2. Flatten each patch to a vector
3. Project to model dimension
4. Add learnable position embeddings
5. Process through transformer blocks

## Training Modes

### 1. Multimodal Mode (--mode multimodal)
Uses both tabular transaction data and QR code images for fraud detection.

```bash
python src/train.py --mode multimodal
```

### 2. Tabular-Only Mode (--mode tabular)
Uses only tabular transaction data (backward compatible with original design).

```bash
python src/train.py --mode tabular
```

## GPU Configuration
The system includes robust GPU configuration with automatic CPU fallback:
- `configure_gpu()`: Sets up GPU with memory growth
- `force_cpu_execution()`: Forces CPU execution if GPU fails
- `build_model_with_fallback()`: Automatically retries on CPU if GPU initialization fails

## Class Relationships

### Composition
- `MultimodalDataPreprocessor` contains `FraudDataPreprocessor` and `QRCodePreprocessor`
- `TransformerBlock` contains `MultiHeadSelfAttention` and `FeedForward`
- `CrossModalTransformerBlock` contains two `CrossModalAttention` instances and two `FeedForward` instances

### Usage
- Models use various transformer blocks during construction
- Predictors use preprocessors for data transformation
- Training functions use models and preprocessors

## Data Flow Summary

```
Raw Data (CSV + Images)
    ↓
Preprocessors (normalization, encoding, resizing)
    ↓
Model Input (tensors)
    ↓
Encoders (tabular + image transformers)
    ↓
Cross-Modal Fusion (attention-based)
    ↓
Classification Head (dense layers)
    ↓
Prediction (fraud probability)
```

## File Structure
```
src/
├── models/
│   ├── transformer_model.py    # Main models
│   ├── attention.py             # Attention mechanisms
│   ├── blocks.py                # Transformer blocks
│   ├── layers.py                # Core layers
│   └── embeddings.py            # Embedding layers
├── utils/
│   ├── data_preprocessing.py    # Tabular preprocessor
│   ├── image_preprocessor.py    # Image preprocessor
│   ├── multimodal_preprocessor.py  # Combined preprocessor
│   └── plotting.py              # Visualization utilities
├── train.py                     # Training script
└── predict.py                   # Inference script
config/
└── config.py                    # Configuration settings
```

## Notes

All class names, method names, and architectural components in the PlantUML diagram are **exact** matches to the actual code implementation. This ensures the documentation stays synchronized with the codebase.

The architecture supports:
- ✅ Multimodal learning (tabular + image)
- ✅ Tabular-only learning (backward compatible)
- ✅ GPU/CPU execution with automatic fallback
- ✅ Flexible configuration
- ✅ Production-ready prediction interfaces
- ✅ Comprehensive preprocessing pipelines
