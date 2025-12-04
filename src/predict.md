# predict.py - Model Inference Script Documentation

## Overview
This script provides inference capabilities for the trained fraud detection models. It supports both multimodal predictions (tabular + image) and tabular-only predictions, with classes for batch and single-transaction prediction.

## Key Classes

### `MultimodalFraudDetectionPredictor`
**Purpose**: Predictor class for multimodal fraud detection using both transaction data and QR code images.

#### Initialization
```python
def __init__(self, model_path=None, preprocessor_path=None):
    model_path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_{MODEL_VERSION}_final.keras')
    preprocessor_path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_{MODEL_VERSION}_preprocessor.pkl')
    
    self.model = keras.models.load_model(model_path)
    self.preprocessor = MultimodalDataPreprocessor(image_size=MODEL_CONFIG.get('image_size', (128, 128)))
    self.preprocessor.load_preprocessors(preprocessor_path)
```
**What it does**: Loads the trained multimodal model and its associated preprocessors from disk. Uses default paths if not specified, automatically locating the latest trained model.

#### Batch Prediction
```python
def predict(self, tabular_data, images, threshold=0.5):
    if isinstance(tabular_data, pd.DataFrame):
        X_tabular, _ = self.preprocessor.fraud_preprocessor.preprocess_data(
            tabular_data, fit=False
        )
    
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
    
    probabilities = self.model.predict([X_tabular, images], verbose=0)
    predictions = (probabilities >= threshold).astype(int)
    
    return {
        'predictions': predictions.flatten(),
        'probabilities': probabilities.flatten(),
        'is_fraud': predictions.flatten() == 1
    }
```
**What it does**: 
1. **Preprocesses** tabular data using saved scaler and encoders
2. **Reshapes** images to ensure proper batch dimensions
3. **Predicts** fraud probabilities using the dual-input model
4. **Thresholds** probabilities at 0.5 (default) to get binary predictions
5. **Returns** predictions, probabilities, and boolean fraud flags

#### Single Transaction Prediction
```python
def predict_single_transaction(self, tabular_features, image, threshold=0.5):
    if isinstance(tabular_features, dict):
        df = pd.DataFrame([tabular_features])
    
    if len(image.shape) == 3:
        images = np.expand_dims(image, axis=0)
    
    result = self.predict(df, images, threshold)
    
    return {
        'is_fraud': bool(result['is_fraud'][0]),
        'fraud_probability': float(result['probabilities'][0]),
        'prediction': int(result['predictions'][0])
    }
```
**What it does**: Convenience method for predicting a single transaction. Accepts transaction features as a dictionary and automatically converts to the required format for the model.

### `FraudDetectionPredictor`
**Purpose**: Predictor class for tabular-only fraud detection (backward compatible).

#### Model Loading Strategy
```python
def __init__(self, model_path=None, scaler_path=None):
    keras_path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_{MODEL_VERSION}_final.keras')
    h5_path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_{MODEL_VERSION}_final.h5')
    
    if os.path.exists(keras_path):
        model_path = keras_path
    elif os.path.exists(h5_path):
        model_path = h5_path
    else:
        model_path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_{MODEL_VERSION}_best.h5')
```
**What it does**: Implements flexible model loading with fallback strategy:
1. Try loading native Keras format (.keras)
2. Fall back to H5 format (.h5) if Keras format not found
3. Fall back to best checkpoint if final model not available

#### Prediction Method
```python
def predict(self, data, threshold=0.5):
    if isinstance(data, pd.DataFrame):
        X, _ = self.preprocessor.preprocess_data(data, fit=False)
    else:
        X = data
    
    probabilities = self.model.predict(X, verbose=0)
    predictions = (probabilities >= threshold).astype(int)
    
    return {
        'predictions': predictions.flatten(),
        'probabilities': probabilities.flatten(),
        'is_fraud': predictions.flatten() == 1
    }
```
**What it does**: Similar to multimodal but accepts only tabular data. Automatically handles DataFrame or pre-processed array inputs.

## Demo Functions

### `demo_multimodal_prediction()`
```python
predictor = MultimodalFraudDetectionPredictor()
data = preprocessor.generate_synthetic_multimodal_data(n_samples=10)
processed = preprocessor.preprocess_multimodal_data(data['tabular'], data['images'], fit=True)
results = predictor.predict(processed['tabular'], processed['images'])
```
**What it does**: 
1. Generates 10 synthetic test samples
2. Preprocesses both modalities
3. Makes batch predictions
4. Displays results with actual vs predicted labels
5. Calculates accuracy, precision, recall, and F1 score

### `demo_tabular_prediction()`
```python
predictor = FraudDetectionPredictor()
test_data = preprocessor.generate_synthetic_data(n_samples=10)
results = predictor.predict(test_data)
```
**What it does**: Similar to multimodal demo but uses only transaction features.

## Usage Examples

### Multimodal Prediction
```python
from src.predict import MultimodalFraudDetectionPredictor

predictor = MultimodalFraudDetectionPredictor()

# Single transaction
transaction = {'step': 1, 'type': 'PAYMENT', 'amount': 100.0, ...}
qr_image = load_qr_image('path/to/qr.png')
result = predictor.predict_single_transaction(transaction, qr_image)
print(f"Fraud: {result['is_fraud']}, Probability: {result['fraud_probability']:.2%}")
```

### Tabular-Only Prediction
```python
from src.predict import FraudDetectionPredictor

predictor = FraudDetectionPredictor()

transaction = {
    'step': 1,
    'type': 'TRANSFER',
    'amount': 50000.0,
    'oldbalanceOrg': 100000.0,
    'newbalanceOrig': 50000.0,
    'oldbalanceDest': 10000.0,
    'newbalanceDest': 60000.0,
    'isFlaggedFraud': 0
}

result = predictor.predict_single_transaction(transaction)
print(f"Fraud probability: {result['fraud_probability']:.2%}")
```

## Running Demos

**Multimodal demo**:
```bash
python src/predict.py --mode multimodal
```

**Tabular demo**:
```bash
python src/predict.py --mode tabular
```

## Performance Metrics
The demo functions calculate:
- **Accuracy**: Overall correct predictions
- **Precision**: Correct fraud predictions / All fraud predictions
- **Recall**: Correct fraud predictions / All actual frauds
- **F1 Score**: Harmonic mean of precision and recall

## Custom Layers Registration
```python
from src.models.transformer_model import (
    MultiHeadSelfAttention,
    TransformerBlock,
    CrossModalAttention,
    CrossModalTransformerBlock,
    PatchEmbedding
)
```
**What it does**: Imports custom Keras layers to ensure they're registered when loading saved models. Essential for model deserialization.
