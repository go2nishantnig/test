# Fraud Detection Transformer Model

A TensorFlow-based transformer model for detecting fraudulent transactions. This project implements a complete pipeline for training, evaluating, and deploying a transformer-based fraud detection system.

## Features

- **Transformer Architecture**: Custom multi-head self-attention mechanism for transaction analysis
- **Automated Model Saving**: Trained models are automatically saved to a dedicated directory
- **Data Preprocessing**: Built-in utilities for data scaling and preparation
- **Synthetic Data Generation**: Demo dataset generator for testing and development
- **Model Persistence**: Saves models, scalers, and training logs
- **Inference Pipeline**: Easy-to-use prediction interface for new transactions

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

### Training the Model

Train the fraud detection transformer model:

```bash
python src/train.py
```

This will:
- Generate synthetic training data
- Build the transformer model
- Train the model with early stopping and learning rate reduction
- Save the trained model to `models/saved_models/`
- Save the data scaler for preprocessing
- Generate TensorBoard logs

**Output location**: All trained models and artifacts are saved to `models/saved_models/` directory:
- `fraud_detection_transformer_v1.0_best.h5` - Best checkpoint during training
- `fraud_detection_transformer_v1.0_final.keras` - Final model in Keras format
- `fraud_detection_transformer_v1.0_final.h5` - Final model in H5 format (for compatibility)
- `fraud_detection_transformer_v1.0_savedmodel/` - SavedModel format (for deployment)
- `fraud_detection_transformer_v1.0_scaler.pkl` - Data preprocessing scaler

Training logs are saved to `models/logs/` for TensorBoard visualization.

### Making Predictions

Run predictions on new transactions:

```bash
python src/predict.py
```

This demonstrates:
- Loading the trained model
- Making predictions on sample transactions
- Displaying fraud probabilities and classifications

### Using the Model in Your Code

```python
from src.predict import FraudDetectionPredictor

# Initialize predictor
predictor = FraudDetectionPredictor()

# Predict for a single transaction
transaction = {
    'amount': 250.0,
    'time': 14530,
    'distance_from_home': 15.2,
    'distance_from_last_transaction': 8.5,
    'ratio_to_median_purchase_price': 2.1,
    'repeat_retailer': 0,
    'used_chip': 1,
    'used_pin_number': 1,
    'online_order': 1,
    # ... other features
}

result = predictor.predict_single_transaction(transaction)
print(f"Is Fraud: {result['is_fraud']}")
print(f"Fraud Probability: {result['fraud_probability']:.2%}")
```

## Model Architecture

The fraud detection system uses a transformer-based architecture:

1. **Input Layer**: Accepts transaction features (30 features by default)
2. **Embedding Layer**: Projects inputs to model dimension (d_model=64)
3. **Positional Encoding**: Adds positional information
4. **Transformer Blocks**: 2 layers of multi-head self-attention
   - 4 attention heads per layer
   - Feed-forward dimension: 128
   - Dropout: 0.1
5. **Global Average Pooling**: Aggregates sequence information
6. **Dense Layers**: Classification head with dropout
7. **Output Layer**: Sigmoid activation for binary classification

## Configuration

Modify `config/config.py` to adjust:

- **Model parameters**: `d_model`, `num_heads`, `num_layers`, etc.
- **Training parameters**: `batch_size`, `epochs`, `learning_rate`
- **Directory paths**: `MODEL_SAVE_DIR`, `DATA_DIR`, `LOG_DIR`

Default configuration:
```python
MODEL_CONFIG = {
    'num_features': 30,
    'd_model': 64,
    'num_heads': 4,
    'num_layers': 2,
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

## Model Saving

The trained model is automatically saved to the virtual machine's file system at the location specified in `config/config.py`:

```python
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
```

This ensures all trained models are stored persistently and can be loaded for inference later.

## Monitoring Training

View training metrics with TensorBoard:

```bash
tensorboard --logdir logs/
```

Then open your browser to `http://localhost:6006`

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## License

This project is provided as-is for educational and development purposes.