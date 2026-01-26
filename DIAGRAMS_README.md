# Architecture and Sequence Diagrams

This directory contains PlantUML diagrams for the Multimodal Fraud Detection Transformer system.

## Available Diagrams

### 1. Architecture Diagram (`architecture.puml`)
**Component Diagram** showing the complete system architecture including:
- Entry points (train.py, predict.py)
- Configuration modules
- Training and prediction modules
- Data preprocessing components
- Core transformer layers
- Attention mechanisms
- Transformer blocks
- Embedding layers
- Main transformer models
- Utility functions

**Generated Image:** `architecture_diagram.png`

### 2. Training Flow Sequence Diagram (`training_sequence.puml`)
**Sequence Diagram** showing the training workflow for both multimodal and tabular modes:

#### Multimodal Training Flow:
1. User executes `python train.py --mode multimodal`
2. GPU configuration
3. Multimodal data preparation:
   - Initialize MultimodalDataPreprocessor
   - Load CSV files from `data/csvdata/`
   - Load QR code images from `data/qrimages/`
   - Prepare train/test split
4. Model building with GPU/CPU fallback:
   - Build tabular encoder (TransformerBlocks)
   - Build image encoder (PatchEmbedding + TransformerBlocks)
   - Build cross-modal fusion (CrossModalTransformerBlock)
   - Compile model
5. Model training:
   - Train on multimodal data
   - Save model (*.keras) and preprocessor (*.pkl)
   - Generate training report

#### Tabular Training Flow:
1. User executes `python train.py --mode tabular`
2. GPU configuration
3. Tabular data preparation:
   - Initialize FraudDataPreprocessor
   - Generate or load synthetic data
   - Preprocess and scale data
4. Model building with GPU/CPU fallback:
   - Build encoder blocks
   - Compile model
5. Model training:
   - Train on tabular data
   - Save model (*.keras) and scaler (*.pkl)
   - Generate training report

### 3. Prediction Flow Sequence Diagram (`prediction_sequence.puml`)
**Sequence Diagram** showing the inference workflow for both multimodal and tabular modes:

#### Multimodal Prediction Flow:
1. User executes `python predict.py --mode multimodal`
2. GPU configuration
3. Load multimodal model:
   - Load trained MultimodalFraudDetectionTransformer
   - Initialize and load preprocessors (fraud + QR code)
4. Generate or provide test data
5. Preprocess data (tabular + images)
6. Make predictions:
   - Model predicts fraud probabilities
   - Apply threshold (default: 0.5)
7. Display results and metrics
8. Generate inference report with visualizations

#### Tabular Prediction Flow:
1. User executes `python predict.py --mode tabular`
2. GPU configuration
3. Load tabular model:
   - Load trained FraudDetectionTransformer
   - Load scaler and preprocessor
4. Generate or provide test data
5. Preprocess data (tabular only)
6. Make predictions:
   - Model predicts fraud probabilities
   - Apply threshold (default: 0.5)
7. Display results and metrics
8. Generate inference report with visualizations

#### Single Transaction Prediction:
- Process individual transactions via `predict_single_transaction()`
- Returns: prediction, probability, is_fraud flag

## Class Details

### Prediction Module Classes

#### `MultimodalFraudDetectionPredictor`
Main predictor class for multimodal fraud detection.

**Attributes:**
- `model`: Loaded MultimodalFraudDetectionTransformer
- `preprocessor`: MultimodalDataPreprocessor instance

**Methods:**
- `__init__(model_path, preprocessor_path)`: Initialize predictor with saved model and preprocessor
- `predict(tabular_data, images, threshold)`: Predict fraud for batch of multimodal data
- `predict_single_transaction(tabular_features, image, threshold)`: Predict fraud for single transaction

#### `FraudDetectionPredictor`
Predictor class for tabular-only fraud detection (backward compatible).

**Attributes:**
- `model`: Loaded FraudDetectionTransformer
- `preprocessor`: FraudDataPreprocessor instance

**Methods:**
- `__init__(model_path, scaler_path)`: Initialize predictor with saved model and scaler
- `predict(data, threshold)`: Predict fraud for batch of tabular data
- `predict_single_transaction(transaction_features, threshold)`: Predict fraud for single transaction

## Generating Diagrams

To generate PNG images from these PlantUML files, you can use:

### Option 1: Online PlantUML Editor
Visit [PlantUML Online Editor](http://www.plantuml.com/plantuml/) and paste the content of any `.puml` file.

### Option 2: Command Line (requires PlantUML and Java)
```bash
# Install PlantUML
# On Ubuntu/Debian:
sudo apt-get install plantuml

# On macOS:
brew install plantuml

# Generate diagrams
plantuml architecture.puml
plantuml training_sequence.puml
plantuml prediction_sequence.puml
```

### Option 3: Using Docker
```bash
docker run -v $(pwd):/data plantuml/plantuml:latest \
  /data/architecture.puml \
  /data/training_sequence.puml \
  /data/prediction_sequence.puml
```

## Syntax Verification

All PlantUML files have been verified for correct syntax:
- ✅ Proper `@startuml` and `@enduml` tags
- ✅ No typos in component/class declarations
- ✅ Valid PlantUML syntax for sequence and component diagrams
- ✅ Proper relationship declarations

## Notes

### Architecture Diagram Features
- Shows component relationships with different line types:
  - `-->` Solid arrows for direct dependencies
  - `..>` Dotted lines for usage dependencies
  - `*--` Composition relationships
- Includes detailed notes explaining key architectural patterns

### Sequence Diagram Features
- Shows temporal ordering of operations
- Includes alt/else blocks for different execution paths
- Annotations with important implementation details
- Clear separation of initialization, processing, and output phases

### Training Module Fallback Mechanism
The `build_model_with_fallback()` function implements automatic CPU fallback:
1. Attempts to build model with current GPU configuration
2. If GPU initialization fails, forces CPU execution
3. Retries model building on CPU
4. Raises exception if both attempts fail

This ensures training can proceed even in environments with GPU issues.
