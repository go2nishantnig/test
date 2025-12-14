# PlantUML Architecture Diagrams

This directory contains comprehensive PlantUML diagrams that document the architecture, design, and implementation of the Multimodal Fraud Detection Transformer system.

## Diagram Overview

### 1. System Architecture (`01_system_architecture.puml`)
**Purpose:** High-level overview of the entire system

**Key Components:**
- External data sources (Online Payments Fraud Dataset, QR Code Images)
- Data layer (Tabular and Image data)
- Processing layer (Preprocessors)
- Model layer (Encoders, Fusion, Classifier)
- Training and Inference pipelines

**Use Case:** Understanding the overall system design and component interactions

### 2. Component Diagram (`02_component_diagram.puml`)
**Purpose:** Package and module structure

**Key Components:**
- `config/` package with configuration
- `src/models/` package hierarchy
- `src/utils/` package hierarchy
- Main scripts (train.py, predict.py, quick_start.py)
- Storage components (data, models, logs)

**Use Case:** Understanding code organization and dependencies

### 3. Models Package Class Diagram (`03_models_class_diagram.puml`)
**Purpose:** Detailed class structure of the models package

**Key Classes:**
- **Attention mechanisms:** `MultiHeadSelfAttention`, `MaskedMultiHeadAttention`, `CrossModalAttention`
- **Layer components:** `FeedForward`, `ResidualConnection`
- **Transformer blocks:** `TransformerBlock`, `TransformerDecoderBlock`, `CrossModalTransformerBlock`
- **Embeddings:** `PatchEmbedding`
- **Model builders:** `MultimodalFraudDetectionTransformer`, `FraudDetectionTransformer`

**Use Case:** Understanding model architecture implementation

### 4. Utils Package Class Diagram (`04_utils_class_diagram.puml`)
**Purpose:** Data preprocessing class hierarchy

**Key Classes:**
- `FraudDataPreprocessor` - Handles tabular transaction data
- `QRCodePreprocessor` - Handles QR code image data
- `MultimodalDataPreprocessor` - Combines both modalities

**Use Case:** Understanding data preprocessing pipeline

### 5. Training Sequence (`05_training_sequence.puml`)
**Purpose:** Step-by-step training flow

**Key Interactions:**
1. Load configuration
2. Create preprocessors
3. Generate/load synthetic data
4. Preprocess data (tabular and images)
5. Build multimodal model
6. Compile and train model
7. Save trained model and preprocessor

**Use Case:** Understanding the training process

### 6. Prediction Sequence (`06_prediction_sequence.puml`)
**Purpose:** Step-by-step inference flow

**Key Interactions:**
1. Load saved model and preprocessor
2. Receive transaction data and QR code image
3. Preprocess inputs
4. Forward pass through model
5. Generate fraud probability
6. Return prediction result

**Use Case:** Understanding how predictions are made

### 7. Data Flow Diagram (`07_data_flow.puml`)
**Purpose:** End-to-end data transformation pipeline

**Key Transformations:**
- Raw CSV → Encoded/Scaled → Reshaped tabular data
- Raw images → Resized/Normalized → Patch embeddings
- Feature encoding through transformer blocks
- Cross-modal fusion
- Classification output

**Use Case:** Understanding data transformations and dimensions

### 8. Architecture Detail (`08_architecture_detail.puml`)
**Purpose:** Detailed layer-by-layer model architecture

**Key Details:**
- Tabular encoder: Dense → Positional → 2× Transformer Blocks
- Image encoder: Patch Embedding → Positional → 2× Transformer Blocks
- Cross-modal fusion: 2× CrossModalTransformerBlock
- Classification head: Pooling → Dense layers → Sigmoid

**Use Case:** Understanding exact model structure and layer connections

### 9. Attention Mechanism Detail (`09_attention_detail.puml`)
**Purpose:** Detailed view of attention mechanisms

**Key Components:**
- **Multi-Head Self-Attention:** Q, K, V from same input, split to multiple heads
- **Cross-Modal Attention:** Q from one modality, K, V from another
- **Masked Attention:** Causal masking for sequence generation
- Attention formula: `Attention(Q,K,V) = softmax(QK^T/√dk) × V`

**Use Case:** Understanding attention computation in detail

### 10. Deployment Diagram (`10_deployment.puml`)
**Purpose:** Production deployment architecture

**Key Infrastructure:**
- Client applications (Web, Mobile, API)
- API Gateway with load balancing
- Application server cluster
- Model serving infrastructure
- Training server (GPU-enabled)
- Data storage (Databases, S3)
- Monitoring and logging (TensorBoard, Grafana, ELK)

**Use Case:** Understanding production deployment options

## How to Use These Diagrams

### PlantUML Version Compatibility

These diagrams are designed to work with PlantUML version 1.2020.02 and newer. They have been tested and successfully generate images. Some older PlantUML versions may show syntax warnings during validation, but the diagrams will still render correctly.

For best results, use:
- **PlantUML 1.2021.0 or newer** (recommended)
- **Online PlantUML Server** (always up-to-date)
- **VS Code PlantUML Extension** (uses latest version)

### Viewing PlantUML Diagrams

You have several options to view these diagrams:

1. **Online PlantUML Editor:**
   - Visit [PlantUML Online Server](http://www.plantuml.com/plantuml/uml/)
   - Copy and paste the diagram code
   - View the rendered diagram

2. **VS Code Extension:**
   ```bash
   # Install the PlantUML extension
   code --install-extension jebbs.plantuml
   ```
   - Open any `.puml` file
   - Press `Alt+D` to preview

3. **Command Line (with PlantUML installed):**
   ```bash
   # Install PlantUML (requires Java)
   # On Ubuntu/Debian:
   sudo apt-get install plantuml
   
   # Generate PNG images
   plantuml diagrams/*.puml
   
   # Generate SVG images
   plantuml -tsvg diagrams/*.puml
   ```

4. **IntelliJ IDEA / PyCharm:**
   - Install PlantUML integration plugin
   - Right-click on `.puml` file → "Show PlantUML Diagram"

5. **GitHub:**
   - Some GitHub viewers support PlantUML rendering
   - Or use GitHub Actions to auto-generate images

### Generating Image Files

To generate all diagrams as PNG images:

```bash
cd diagrams
plantuml *.puml
```

To generate as SVG (scalable):

```bash
cd diagrams
plantuml -tsvg *.puml
```

## Diagram Reading Guide

### Color Coding

- **Light Blue:** Standard components, classes, processing steps
- **Light Green:** Model-related components
- **Light Coral:** Data processing components
- **Light Yellow:** External data sources, attention mechanisms
- **Pink:** Fraud/negative outcomes
- **Light Green:** Normal/positive outcomes

### Notation

- **Solid arrows (→):** Direct dependencies or data flow
- **Dashed arrows (..>):** Usage relationships
- **Composition (◆):** "has-a" relationship
- **Inheritance (▷):** "is-a" relationship
- **Notes:** Additional context and explanations

## Architecture Highlights

### Multimodal Approach

The system processes two types of data:
1. **Tabular Data:** Transaction features (amount, balances, type, etc.)
2. **Image Data:** QR code images (benign vs malicious)

### Transformer-Based

Uses state-of-the-art transformer architecture:
- Self-attention for each modality
- Cross-modal attention for fusion
- Vision Transformer (ViT) for images
- Positional encodings

### Key Innovations

1. **Cross-Modal Fusion:** Bidirectional attention between modalities
2. **Patch-Based Image Processing:** ViT approach for QR codes
3. **Modular Design:** Easy to extend with additional modalities
4. **Backward Compatible:** Supports tabular-only mode

## Documentation Mapping

These diagrams complement the following documentation:

- `README.md` - High-level overview and usage
- `SUMMARY.md` - Project summary
- `END_TO_END_FLOW.md` - End-to-end workflow
- `quick_start.md` - Quick start guide
- `src/models/*.md` - Model component documentation
- `src/utils/*.md` - Utility component documentation

## Maintenance

When updating the codebase:

1. Review affected diagrams
2. Update relevant `.puml` files
3. Regenerate images if needed
4. Update this README if adding new diagrams

## Questions?

For questions about:
- **Architecture:** Refer to diagrams 01, 02, 08
- **Code structure:** Refer to diagrams 02, 03, 04
- **Training process:** Refer to diagram 05
- **Inference process:** Refer to diagram 06
- **Data flow:** Refer to diagram 07
- **Deployment:** Refer to diagram 10
- **Attention details:** Refer to diagram 09

## Additional Resources

- [PlantUML Language Reference](https://plantuml.com/guide)
- [PlantUML Component Diagram](https://plantuml.com/component-diagram)
- [PlantUML Class Diagram](https://plantuml.com/class-diagram)
- [PlantUML Sequence Diagram](https://plantuml.com/sequence-diagram)
- [PlantUML Deployment Diagram](https://plantuml.com/deployment-diagram)
- [PlantUML Activity Diagram](https://plantuml.com/activity-diagram-beta)
