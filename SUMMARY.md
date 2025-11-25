# Project Summary: Fraud Detection Transformer Model

## Overview
This project implements a transformer-based neural network model for detecting fraudulent transactions using TensorFlow. The model uses multi-head self-attention mechanisms to analyze transaction patterns and classify them as fraudulent or legitimate.

## Architecture Details

### Transformer Model Components
1. **Input Layer**: Accepts 30 transaction features
2. **Embedding Layer**: Projects features to 64-dimensional space
3. **Positional Encoding**: Adds position information to embeddings
4. **Transformer Blocks** (x2):
   - Multi-head self-attention (4 heads)
   - Feed-forward network (128 dimensions)
   - Layer normalization
   - Residual connections
   - Dropout (0.1 rate)
5. **Global Average Pooling**: Aggregates sequence information
6. **Classification Head**: Dense layers with dropout
7. **Output**: Sigmoid activation for binary classification

### Model Performance
- **Metrics**: Accuracy, Precision, Recall, AUC
- **Training**: Early stopping, learning rate reduction, model checkpointing
- **Validation**: 20% validation split with stratification

## Directory Structure

```
.
├── config/                    # Configuration files
│   ├── __init__.py
│   └── config.py             # Model and training parameters
├── data/                     # Data directory (empty initially)
├── models/                   # Model storage
│   ├── logs/                # TensorBoard logs (gitignored)
│   └── saved_models/        # Trained models (gitignored)
│       ├── fraud_detection_transformer_v1.0_best.h5
│       ├── fraud_detection_transformer_v1.0_final.keras
│       ├── fraud_detection_transformer_v1.0_final.h5
│       ├── fraud_detection_transformer_v1.0_savedmodel/
│       └── fraud_detection_transformer_v1.0_scaler.pkl
├── notebooks/                # Jupyter notebooks
│   └── fraud_detection_demo.ipynb
├── src/                      # Source code
│   ├── models/              # Model architectures
│   │   ├── __init__.py
│   │   └── transformer_model.py
│   ├── utils/               # Utility functions
│   │   ├── __init__.py
│   │   └── data_preprocessing.py
│   ├── __init__.py
│   ├── train.py            # Training script
│   └── predict.py          # Inference script
├── .gitignore               # Git ignore rules
├── README.md                # Main documentation
├── requirements.txt         # Python dependencies
├── quick_start.py          # Quick start example
└── SUMMARY.md              # This file
```

## Key Features

### 1. Custom Transformer Implementation
- Multi-head self-attention mechanism
- Properly registered Keras serializable layers
- Residual connections and layer normalization
- Configurable architecture

### 2. Data Processing
- Synthetic data generation for testing
- Feature scaling and normalization
- Train/test split with stratification
- Scaler persistence for inference

### 3. Training Pipeline
- Automated model checkpointing
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- TensorBoard logging for monitoring
- Multiple save formats (Keras, H5, SavedModel)

### 4. Inference Pipeline
- Easy-to-use predictor class
- Single transaction prediction
- Batch prediction support
- Automatic model format detection

### 5. Model Persistence
All models are automatically saved to `models/saved_models/` directory in the virtual machine:
- **Best checkpoint**: Saved during training based on validation loss
- **Final model**: Saved at the end of training in multiple formats
- **Scaler**: Saved for consistent preprocessing during inference
- **Logs**: TensorBoard logs for training visualization

## Usage Examples

### Training
```bash
python src/train.py
```

### Inference
```bash
python src/predict.py
```

### Quick Start
```bash
python quick_start.py
```

### Using in Code
```python
from src.predict import FraudDetectionPredictor

predictor = FraudDetectionPredictor()
result = predictor.predict_single_transaction({
    'amount': 250.0,
    'time': 14530,
    # ... other features
})
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

## Model Characteristics

### Input Features (30 total)
- `amount`: Transaction amount
- `time`: Time of transaction
- `distance_from_home`: Distance from home address
- `distance_from_last_transaction`: Distance from previous transaction
- `ratio_to_median_purchase_price`: Ratio to median purchase
- `repeat_retailer`: Binary flag for repeat retailer
- `used_chip`: Binary flag for chip usage
- `used_pin_number`: Binary flag for PIN usage
- `online_order`: Binary flag for online order
- `feature_9` to `feature_29`: Additional synthetic features

### Model Parameters
- **Model dimension**: 64
- **Attention heads**: 4
- **Transformer layers**: 2
- **Feed-forward dimension**: 128
- **Dropout rate**: 0.1

### Training Parameters
- **Batch size**: 32
- **Epochs**: 10 (with early stopping at 5 patience)
- **Learning rate**: 0.001 (with reduction on plateau)
- **Validation split**: 20%
- **Optimizer**: Adam

## Performance Notes

### Test Results (Sample Run)
- Successfully identifies normal transactions with low fraud probability (<0.01%)
- Successfully identifies suspicious transactions with high fraud probability (>99%)
- Model handles both edge cases appropriately

### Model Outputs
- **Binary prediction**: Fraud (1) or Not Fraud (0)
- **Probability score**: Continuous value between 0 and 1
- **Threshold**: Default 0.5 (adjustable)

## Future Enhancements
1. Integrate real transaction datasets
2. Add more sophisticated feature engineering
3. Implement explainability (attention visualization)
4. Add API endpoint for production deployment
5. Implement online learning capabilities
6. Add model monitoring and drift detection

## Notes
- Models are saved in multiple formats for flexibility
- Custom layers are properly registered for serialization
- All paths are configurable via `config/config.py`
- The system supports both CPU and GPU training
- TensorBoard logs are available for detailed training analysis

## Testing
The implementation has been tested with:
- ✓ Model training completes successfully
- ✓ Models save to correct directory in VM
- ✓ Model loading works correctly
- ✓ Prediction pipeline functions properly
- ✓ Both normal and fraudulent transactions are classified correctly
