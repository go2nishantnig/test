#!/usr/bin/env python3
"""
Quick start example for fraud detection transformer model
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predict import FraudDetectionPredictor
from src.utils.data_preprocessing import FraudDataPreprocessor

def main():
    print("=" * 70)
    print("Fraud Detection Transformer - Quick Start Example")
    print("=" * 70)
    
    # Initialize predictor (loads trained model)
    try:
        predictor = FraudDetectionPredictor()
    except Exception as e:
        print(f"\nError: Could not load model. {e}")
        print("\nPlease train the model first:")
        print("  python src/train.py")
        return
    
    # Example transaction data
    print("\n1. Example Normal Transaction:")
    print("-" * 70)
    normal_transaction = {
        'amount': 50.00,
        'time': 43200,  # noon
        'distance_from_home': 5.0,
        'distance_from_last_transaction': 2.0,
        'ratio_to_median_purchase_price': 1.0,
        'repeat_retailer': 1,
        'used_chip': 1,
        'used_pin_number': 1,
        'online_order': 0,
        # Additional features (would normally be calculated from transaction data)
        **{f'feature_{i}': 0.0 for i in range(9, 30)}
    }
    
    result = predictor.predict_single_transaction(normal_transaction)
    print(f"Amount: ${normal_transaction['amount']:.2f}")
    print(f"Time: {normal_transaction['time']/3600:.1f}h")
    print(f"Distance from home: {normal_transaction['distance_from_home']:.1f} miles")
    print(f"\nPrediction:")
    print(f"  Is Fraud: {result['is_fraud']}")
    print(f"  Fraud Probability: {result['fraud_probability']:.4%}")
    
    # Example suspicious transaction
    print("\n2. Example Suspicious Transaction:")
    print("-" * 70)
    suspicious_transaction = {
        'amount': 1500.00,  # Large amount
        'time': 10800,  # 3 AM
        'distance_from_home': 500.0,  # Far from home
        'distance_from_last_transaction': 400.0,  # Far from last transaction
        'ratio_to_median_purchase_price': 10.0,  # Much larger than usual
        'repeat_retailer': 0,  # New retailer
        'used_chip': 0,  # No chip
        'used_pin_number': 0,  # No PIN
        'online_order': 1,  # Online order
        # Additional features
        **{f'feature_{i}': 0.5 for i in range(9, 30)}
    }
    
    result = predictor.predict_single_transaction(suspicious_transaction)
    print(f"Amount: ${suspicious_transaction['amount']:.2f}")
    print(f"Time: {suspicious_transaction['time']/3600:.1f}h (late night)")
    print(f"Distance from home: {suspicious_transaction['distance_from_home']:.1f} miles")
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


if __name__ == "__main__":
    main()
