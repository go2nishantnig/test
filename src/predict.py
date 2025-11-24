"""
Inference script for fraud detection transformer model
"""
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import MODEL_SAVE_DIR, MODEL_NAME, MODEL_VERSION
from src.utils.data_preprocessing import FraudDataPreprocessor


class FraudDetectionPredictor:
    """Predictor class for fraud detection"""
    
    def __init__(self, model_path=None, scaler_path=None):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the saved model (if None, uses default)
            scaler_path: Path to the saved scaler (if None, uses default)
        """
        # Set default paths if not provided
        if model_path is None:
            model_path = os.path.join(
                MODEL_SAVE_DIR,
                f'{MODEL_NAME}_{MODEL_VERSION}_final'
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


def demo_prediction():
    """Demonstrate model prediction"""
    
    print("=" * 70)
    print("Fraud Detection Transformer - Inference Demo")
    print("=" * 70)
    
    # Check if model exists
    model_path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_{MODEL_VERSION}_final')
    if not os.path.exists(model_path):
        print(f"\nError: Model not found at {model_path}")
        print("Please run train.py first to train the model.")
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
    print(test_data[['amount', 'is_fraud', 'predicted_fraud', 'fraud_probability']].head(10))
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(test_data['is_fraud'], results['predictions'])
    precision = precision_score(test_data['is_fraud'], results['predictions'], zero_division=0)
    recall = recall_score(test_data['is_fraud'], results['predictions'], zero_division=0)
    f1 = f1_score(test_data['is_fraud'], results['predictions'], zero_division=0)
    
    print("\n" + "=" * 70)
    print("Performance Metrics:")
    print("=" * 70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Test single transaction prediction
    print("\n" + "=" * 70)
    print("Single Transaction Prediction Example:")
    print("=" * 70)
    
    single_transaction = test_data.drop(['is_fraud'], axis=1).iloc[0].to_dict()
    result = predictor.predict_single_transaction(single_transaction)
    
    print(f"\nTransaction details:")
    print(f"  Amount: ${single_transaction['amount']:.2f}")
    print(f"\nPrediction:")
    print(f"  Is Fraud: {result['is_fraud']}")
    print(f"  Fraud Probability: {result['fraud_probability']:.2%}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_prediction()
