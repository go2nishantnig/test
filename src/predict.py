"""
Inference script for multimodal fraud detection transformer model

Supports two prediction modes:
1. Multimodal: Uses both tabular and image data
2. Tabular-only: Uses only tabular data (backward compatible)
"""
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import MODEL_SAVE_DIR, MODEL_NAME, MODEL_VERSION, MODEL_CONFIG, GPU_CONFIG
from src.utils.data_preprocessing import (
    FraudDataPreprocessor,
    QRCodePreprocessor
)
from src.utils.multimodal_preprocessor import MultimodalDataPreprocessor
from src.utils.plotting import generate_inference_report
# Import custom layers to ensure they are registered
from src.models.transformer_model import (
    MultiHeadSelfAttention,
    TransformerBlock,
    CrossModalAttention,
    CrossModalTransformerBlock,
    PatchEmbedding
)


def configure_gpu():
    """
    Configure GPU settings for TensorFlow to prevent memory allocation issues.
    
    This function should be called before any TensorFlow operations to:
    - Enable memory growth (allocate GPU memory as needed instead of all at once)
    - Configure mixed precision training if enabled
    - Gracefully handle cases where GPU is not available
    """
    try:
        # Get list of physical GPUs
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            # Configure memory growth for each GPU
            if GPU_CONFIG.get('memory_growth', True):
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    # Memory growth must be set before GPUs have been initialized
                    # This is safe to ignore as it means GPU is already configured
                    pass
    except Exception:
        # Silently continue if GPU configuration fails
        # Prediction will fall back to CPU or use default GPU settings
        pass


class MultimodalFraudDetectionPredictor:
    """Predictor class for multimodal fraud detection"""
    
    def __init__(self, model_path=None, preprocessor_path=None):
        """
        Initialize the multimodal predictor
        
        Args:
            model_path: Path to the saved model (if None, uses default)
            preprocessor_path: Path to the saved preprocessor (if None, uses default)
        """
        # Set default paths if not provided
        if model_path is None:
            model_path = os.path.join(
                MODEL_SAVE_DIR,
                f'{MODEL_NAME}_{MODEL_VERSION}_final.keras'
            )
        
        if preprocessor_path is None:
            preprocessor_path = os.path.join(
                MODEL_SAVE_DIR,
                f'{MODEL_NAME}_{MODEL_VERSION}_preprocessor.pkl'
            )
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        
        # Load preprocessor
        print(f"Loading preprocessor from: {preprocessor_path}")
        self.preprocessor = MultimodalDataPreprocessor(
            image_size=MODEL_CONFIG.get('image_size', (128, 128))
        )
        self.preprocessor.load_preprocessors(preprocessor_path)
        
        print("Model and preprocessor loaded successfully!")
    
    def predict(self, tabular_data, images, threshold=0.5):
        """
        Make predictions on multimodal data
        
        Args:
            tabular_data: DataFrame or array with transaction features
            images: Array of QR code images (normalized to [0,1])
            threshold: Probability threshold for fraud classification
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Preprocess tabular data
        if isinstance(tabular_data, pd.DataFrame):
            X_tabular, _ = self.preprocessor.fraud_preprocessor.preprocess_data(
                tabular_data, fit=False
            )
        else:
            X_tabular = tabular_data
        
        # Ensure images are properly formatted
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        
        # Make predictions
        probabilities = self.model.predict([X_tabular, images], verbose=0)
        predictions = (probabilities >= threshold).astype(int)
        
        return {
            'predictions': predictions.flatten(),
            'probabilities': probabilities.flatten(),
            'is_fraud': predictions.flatten() == 1
        }
    
    def predict_single_transaction(self, tabular_features, image, threshold=0.5):
        """
        Predict fraud for a single transaction with QR code image
        
        Args:
            tabular_features: Dictionary or array with transaction features
            image: QR code image (normalized to [0,1])
            threshold: Probability threshold for fraud classification
            
        Returns:
            Dictionary with prediction result
        """
        # Convert tabular features to DataFrame if needed
        if isinstance(tabular_features, dict):
            df = pd.DataFrame([tabular_features])
        elif isinstance(tabular_features, (list, np.ndarray)):
            df = pd.DataFrame(
                [tabular_features],
                columns=self.preprocessor.fraud_preprocessor.feature_names
            )
        else:
            df = tabular_features
        
        # Ensure image has proper shape
        if len(image.shape) == 3:
            images = np.expand_dims(image, axis=0)
        else:
            images = image
        
        result = self.predict(df, images, threshold)
        
        return {
            'is_fraud': bool(result['is_fraud'][0]),
            'fraud_probability': float(result['probabilities'][0]),
            'prediction': int(result['predictions'][0])
        }


class FraudDetectionPredictor:
    """Predictor class for tabular-only fraud detection (backward compatible)"""
    
    def __init__(self, model_path=None, scaler_path=None):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the saved model (if None, uses default)
            scaler_path: Path to the saved scaler (if None, uses default)
        """
        # Set default paths if not provided
        if model_path is None:
            # Try different model formats
            keras_path = os.path.join(
                MODEL_SAVE_DIR,
                f'{MODEL_NAME}_{MODEL_VERSION}_final.keras'
            )
            h5_path = os.path.join(
                MODEL_SAVE_DIR,
                f'{MODEL_NAME}_{MODEL_VERSION}_final.h5'
            )
            
            if os.path.exists(keras_path):
                model_path = keras_path
            elif os.path.exists(h5_path):
                model_path = h5_path
            else:
                # Fallback to best checkpoint
                model_path = os.path.join(
                    MODEL_SAVE_DIR,
                    f'{MODEL_NAME}_{MODEL_VERSION}_best.h5'
                )
        
        if scaler_path is None:
            scaler_path = os.path.join(
                MODEL_SAVE_DIR,
                f'{MODEL_NAME}_{MODEL_VERSION}_scaler.pkl'
            )
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        
        # Load scaler
        print(f"Loading scaler from: {scaler_path}")
        self.preprocessor = FraudDataPreprocessor()
        self.preprocessor.load_scaler(scaler_path)
        
        print("Model and preprocessor loaded successfully!")
    
    def predict(self, data, threshold=0.5):
        """
        Make predictions on new data
        
        Args:
            data: DataFrame or array with transaction features
            threshold: Probability threshold for fraud classification
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Preprocess data
        if isinstance(data, pd.DataFrame):
            X, _ = self.preprocessor.preprocess_data(data, fit=False)
        else:
            # Assume it's already preprocessed
            X = data
        
        # Make predictions
        probabilities = self.model.predict(X, verbose=0)
        predictions = (probabilities >= threshold).astype(int)
        
        return {
            'predictions': predictions.flatten(),
            'probabilities': probabilities.flatten(),
            'is_fraud': predictions.flatten() == 1
        }
    
    def predict_single_transaction(self, transaction_features, threshold=0.5):
        """
        Predict fraud for a single transaction
        
        Args:
            transaction_features: Dictionary or array with transaction features
            threshold: Probability threshold for fraud classification
            
        Returns:
            Dictionary with prediction result
        """
        # Convert to DataFrame if needed
        if isinstance(transaction_features, dict):
            df = pd.DataFrame([transaction_features])
        elif isinstance(transaction_features, (list, np.ndarray)):
            df = pd.DataFrame([transaction_features], columns=self.preprocessor.feature_names)
        else:
            df = transaction_features
        
        result = self.predict(df, threshold)
        
        return {
            'is_fraud': bool(result['is_fraud'][0]),
            'fraud_probability': float(result['probabilities'][0]),
            'prediction': int(result['predictions'][0])
        }


def demo_multimodal_prediction():
    """Demonstrate multimodal model prediction"""
    
    # Configure GPU before any TensorFlow operations
    configure_gpu()
    
    print("=" * 70)
    print("Multimodal Fraud Detection Transformer - Inference Demo")
    print("=" * 70)
    
    # Check if model exists
    model_path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_{MODEL_VERSION}_final.keras')
    
    if not os.path.exists(model_path):
        print(f"\nError: No trained model found at {model_path}")
        print("Please run 'python src/train.py --mode multimodal' first to train the model.")
        return
    
    # Initialize predictor
    predictor = MultimodalFraudDetectionPredictor()
    
    # Generate sample test data
    print("\n1. Generating sample multimodal test data...")
    preprocessor = MultimodalDataPreprocessor(
        image_size=MODEL_CONFIG.get('image_size', (128, 128))
    )
    data = preprocessor.generate_synthetic_multimodal_data(n_samples=10)
    
    # Preprocess
    processed = preprocessor.preprocess_multimodal_data(
        data['tabular'], data['images'], fit=True
    )
    
    # Make predictions
    print("\n2. Making predictions...")
    results = predictor.predict(
        processed['tabular'],
        processed['images']
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("Prediction Results:")
    print("=" * 70)
    
    display_df = data['tabular'][['step', 'type', 'amount', 'isFraud']].copy()
    display_df['predicted_fraud'] = results['predictions']
    display_df['fraud_probability'] = results['probabilities']
    
    print("\nSample predictions:")
    print(display_df.head(10).to_string())
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(data['labels'], results['predictions'])
    precision = precision_score(data['labels'], results['predictions'], zero_division=0)
    recall = recall_score(data['labels'], results['predictions'], zero_division=0)
    f1 = f1_score(data['labels'], results['predictions'], zero_division=0)
    
    print("\n" + "=" * 70)
    print("Performance Metrics:")
    print("=" * 70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Generate and save inference plots
    generate_inference_report(
        y_true=data['labels'],
        y_pred=results['predictions'],
        y_pred_proba=results['probabilities']
    )
    
    print("\n" + "=" * 70)


def demo_tabular_prediction():
    """Demonstrate tabular-only model prediction"""
    
    # Configure GPU before any TensorFlow operations
    configure_gpu()
    
    print("=" * 70)
    print("Fraud Detection Transformer - Tabular Inference Demo")
    print("=" * 70)
    
    # Check if model exists (try multiple formats)
    keras_path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_{MODEL_VERSION}_final.keras')
    h5_path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_{MODEL_VERSION}_final.h5')
    best_path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_{MODEL_VERSION}_best.h5')
    
    if not (os.path.exists(keras_path) or os.path.exists(h5_path) or os.path.exists(best_path)):
        print(f"\nError: No trained model found in {MODEL_SAVE_DIR}")
        print("Please run 'python src/train.py --mode tabular' first to train the model.")
        return
    
    # Initialize predictor
    predictor = FraudDetectionPredictor()
    
    # Generate sample test data
    print("\n1. Generating sample transactions...")
    preprocessor = FraudDataPreprocessor()
    test_data = preprocessor.generate_synthetic_data(n_samples=10)
    
    # Make predictions
    print("\n2. Making predictions...")
    results = predictor.predict(test_data)
    
    # Display results
    print("\n" + "=" * 70)
    print("Prediction Results:")
    print("=" * 70)
    
    test_data['predicted_fraud'] = results['predictions']
    test_data['fraud_probability'] = results['probabilities']
    
    print("\nSample predictions:")
    print(test_data[['amount', 'type', 'isFraud', 'predicted_fraud', 'fraud_probability']].head(10))
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(test_data['isFraud'], results['predictions'])
    precision = precision_score(test_data['isFraud'], results['predictions'], zero_division=0)
    recall = recall_score(test_data['isFraud'], results['predictions'], zero_division=0)
    f1 = f1_score(test_data['isFraud'], results['predictions'], zero_division=0)
    
    print("\n" + "=" * 70)
    print("Performance Metrics:")
    print("=" * 70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Generate and save inference plots
    generate_inference_report(
        y_true=test_data['isFraud'].values,
        y_pred=results['predictions'],
        y_pred_proba=results['probabilities']
    )
    
    print("\n" + "=" * 70)


def demo_prediction(mode='multimodal'):
    """
    Main demo function with mode selection
    
    Args:
        mode: 'multimodal' for combined tabular+image model,
              'tabular' for tabular-only model
    """
    if mode == 'multimodal':
        demo_multimodal_prediction()
    else:
        demo_tabular_prediction()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run fraud detection inference demo')
    parser.add_argument(
        '--mode',
        type=str,
        default='multimodal',
        choices=['multimodal', 'tabular'],
        help='Inference mode: multimodal (tabular+image) or tabular (tabular only)'
    )
    
    args = parser.parse_args()
    demo_prediction(mode=args.mode)
