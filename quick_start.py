#!/usr/bin/env python3
"""
Quick start example for multimodal fraud detection transformer model

This demonstrates the multimodal model that combines:
1. Tabular data: Online Payments Fraud Detection features
2. Image data: QR Code images (benign vs malicious)
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predict import MultimodalFraudDetectionPredictor, FraudDetectionPredictor
from src.utils.data_preprocessing import (
    FraudDataPreprocessor,
    QRCodePreprocessor,
    MultimodalDataPreprocessor
)
from config.config import MODEL_CONFIG


def demo_multimodal():
    """Demonstrate multimodal fraud detection"""
    print("=" * 70)
    print("Multimodal Fraud Detection Transformer - Quick Start")
    print("(Combining Tabular Transaction Data + QR Code Images)")
    print("=" * 70)
    
    # Initialize predictor (loads trained model)
    try:
        predictor = MultimodalFraudDetectionPredictor()
    except Exception as e:
        print(f"\nError: Could not load multimodal model. {e}")
        print("\nPlease train the multimodal model first:")
        print("  python src/train.py --mode multimodal")
        return
    
    # Generate sample data
    print("\n1. Generating sample multimodal data...")
    preprocessor = MultimodalDataPreprocessor(
        image_size=MODEL_CONFIG.get('image_size', (128, 128))
    )
    
    # Example: Normal transaction with benign QR code
    print("\n2. Example Normal Transaction with Benign QR Code:")
    print("-" * 70)
    
    qr_preprocessor = QRCodePreprocessor(image_size=MODEL_CONFIG.get('image_size', (128, 128)))
    
    # Generate sample data for normal case
    sample_data = preprocessor.generate_synthetic_multimodal_data(n_samples=1, fraud_ratio=0.0)
    processed = preprocessor.preprocess_multimodal_data(
        sample_data['tabular'], sample_data['images'], fit=True
    )
    
    result = predictor.predict(processed['tabular'], processed['images'])
    
    print(f"Transaction Type: {sample_data['tabular']['type'].values[0]}")
    print(f"Amount: ${sample_data['tabular']['amount'].values[0]:.2f}")
    print(f"QR Code: Benign (synthetic)")
    print(f"\nPrediction:")
    print(f"  Is Fraud: {result['is_fraud'][0]}")
    print(f"  Fraud Probability: {result['probabilities'][0]:.4%}")
    
    # Example: Suspicious transaction with malicious QR code
    print("\n3. Example Suspicious Transaction with Malicious QR Code:")
    print("-" * 70)
    
    # Generate sample data for fraud case
    sample_data = preprocessor.generate_synthetic_multimodal_data(n_samples=1, fraud_ratio=1.0)
    processed = preprocessor.preprocess_multimodal_data(
        sample_data['tabular'], sample_data['images'], fit=False
    )
    
    result = predictor.predict(processed['tabular'], processed['images'])
    
    print(f"Transaction Type: {sample_data['tabular']['type'].values[0]}")
    print(f"Amount: ${sample_data['tabular']['amount'].values[0]:.2f}")
    print(f"QR Code: Potentially Malicious (synthetic)")
    print(f"\nPrediction:")
    print(f"  Is Fraud: {result['is_fraud'][0]}")
    print(f"  Fraud Probability: {result['probabilities'][0]:.4%}")
    
    print("\n" + "=" * 70)
    print("Multimodal Quick Start completed!")
    print("=" * 70)


def demo_tabular():
    """Demonstrate tabular-only fraud detection (backward compatible)"""
    print("=" * 70)
    print("Fraud Detection Transformer - Quick Start (Tabular Only)")
    print("=" * 70)
    
    # Initialize predictor (loads trained model)
    try:
        predictor = FraudDetectionPredictor()
    except Exception as e:
        print(f"\nError: Could not load model. {e}")
        print("\nPlease train the model first:")
        print("  python src/train.py --mode tabular")
        return
    
    # Example transaction data (Online Payments Fraud Detection format)
    print("\n1. Example Normal Transaction:")
    print("-" * 70)
    normal_transaction = {
        'step': 1,
        'type': 'PAYMENT',
        'amount': 50.00,
        'nameOrig': 'C1234567890',
        'oldbalanceOrg': 50000.0,
        'newbalanceOrig': 49950.0,
        'nameDest': 'M1234567890',
        'oldbalanceDest': 10000.0,
        'newbalanceDest': 10050.0,
        'isFlaggedFraud': 0,
    }
    
    result = predictor.predict_single_transaction(normal_transaction)
    print(f"Type: {normal_transaction['type']}")
    print(f"Amount: ${normal_transaction['amount']:.2f}")
    print(f"Balance Change: ${normal_transaction['oldbalanceOrg'] - normal_transaction['newbalanceOrig']:.2f}")
    print(f"\nPrediction:")
    print(f"  Is Fraud: {result['is_fraud']}")
    print(f"  Fraud Probability: {result['fraud_probability']:.4%}")
    
    # Example suspicious transaction
    print("\n2. Example Suspicious Transaction:")
    print("-" * 70)
    suspicious_transaction = {
        'step': 200,
        'type': 'TRANSFER',
        'amount': 200000.00,  # Large transfer
        'nameOrig': 'C9876543210',
        'oldbalanceOrg': 250000.0,
        'newbalanceOrig': 50000.0,  # Large decrease
        'nameDest': 'C5555555555',
        'oldbalanceDest': 1000.0,
        'newbalanceDest': 201000.0,  # Large increase
        'isFlaggedFraud': 0,
    }
    
    result = predictor.predict_single_transaction(suspicious_transaction)
    print(f"Type: {suspicious_transaction['type']}")
    print(f"Amount: ${suspicious_transaction['amount']:.2f}")
    print(f"Balance Change: ${suspicious_transaction['oldbalanceOrg'] - suspicious_transaction['newbalanceOrig']:.2f}")
    print(f"\nPrediction:")
    print(f"  Is Fraud: {result['is_fraud']}")
    print(f"  Fraud Probability: {result['fraud_probability']:.4%}")
    
    print("\n" + "=" * 70)
    print("Quick start completed!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Modify config/config.py to adjust model parameters")
    print("  - Retrain with your own data")
    print("  - Check notebooks/fraud_detection_demo.ipynb for more examples")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick start demo for fraud detection')
    parser.add_argument(
        '--mode',
        type=str,
        default='multimodal',
        choices=['multimodal', 'tabular'],
        help='Demo mode: multimodal (tabular+image) or tabular (tabular only)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'multimodal':
        demo_multimodal()
    else:
        demo_tabular()


if __name__ == "__main__":
    main()
