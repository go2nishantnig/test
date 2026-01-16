#!/usr/bin/env python3
"""
Quick start script for AWS EC2 G5 XLarge
Demonstrates training the multimodal fraud detection model with EC2 configuration

Usage:
    python ec2_quick_start.py [--mode {multimodal|tabular}]
"""
import os
import sys
import argparse

# Use EC2 configuration
os.environ['USE_EC2_CONFIG'] = '1'

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import EC2 configuration
from config.ec2_config import (
    MODEL_CONFIG,
    TRAINING_CONFIG,
    EC2_QRDATA_DIR,
    EC2_CSV_DATA_DIR,
    EC2_MODEL_DIR,
    MODEL_NAME,
    MODEL_VERSION,
    GPU_CONFIG
)

# Import model and preprocessing utilities
from src.models.transformer_model import (
    MultimodalFraudDetectionTransformer,
    FraudDetectionTransformer
)
from src.utils.data_preprocessing import (
    FraudDataPreprocessor,
    MultimodalDataPreprocessor
)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd


def setup_gpu():
    """Configure GPU for optimal performance on EC2 G5 XLarge"""
    print("\n" + "=" * 70)
    print("GPU Configuration")
    print("=" * 70)
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"\n✓ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        
        # Enable memory growth
        if GPU_CONFIG.get('memory_growth', True):
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ GPU memory growth enabled")
        
        # Enable mixed precision for faster training
        if GPU_CONFIG.get('mixed_precision', True):
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print("✓ Mixed precision (FP16) enabled")
    else:
        print("\n⚠ No GPU detected. Training will use CPU (slower).")
        print("For G5 XLarge, ensure NVIDIA drivers are installed:")
        print("  sudo yum install -y nvidia-driver-latest-dkms")


def verify_directories():
    """Verify EC2 directories exist"""
    print("\n" + "=" * 70)
    print("Directory Verification")
    print("=" * 70)
    
    dirs = {
        'QR Data': EC2_QRDATA_DIR,
        'CSV Data': EC2_CSV_DATA_DIR,
        'Model': EC2_MODEL_DIR,
    }
    
    all_exist = True
    for name, path in dirs.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print("\n⚠ Some directories are missing.")
        print("Creating missing directories...")
        for name, path in dirs.items():
            os.makedirs(path, exist_ok=True)
        print("✓ Directories created")
    
    return all_exist


def train_multimodal():
    """Train multimodal fraud detection model"""
    print("\n" + "=" * 70)
    print("Training Multimodal Fraud Detection Model (EC2)")
    print("=" * 70)
    
    # Setup GPU
    setup_gpu()
    
    # Verify directories
    verify_directories()
    
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Initialize preprocessor
    print("\n1. Preparing multimodal data...")
    preprocessor = MultimodalDataPreprocessor(
        image_size=MODEL_CONFIG.get('image_size', (128, 128))
    )
    
    # Prepare data
    data = preprocessor.prepare_train_test_data(
        test_size=TRAINING_CONFIG['validation_split'],
        n_samples=5000
    )
    
    print(f"   Training samples: {len(data['y_train'])}")
    print(f"   Testing samples: {len(data['y_test'])}")
    print(f"   Fraud ratio (train): {np.mean(data['y_train']):.2%}")
    print(f"   Fraud ratio (test): {np.mean(data['y_test']):.2%}")
    
    # Save preprocessors
    preprocessor_path = os.path.join(
        EC2_MODEL_DIR,
        f'{MODEL_NAME}_{MODEL_VERSION}_preprocessor.pkl'
    )
    preprocessor.save_preprocessors(preprocessor_path)
    print(f"   Preprocessors saved to: {preprocessor_path}")
    
    # Build model
    print("\n2. Building multimodal transformer model...")
    multimodal_model = MultimodalFraudDetectionTransformer(MODEL_CONFIG)
    model = multimodal_model.compile_model(
        learning_rate=TRAINING_CONFIG['learning_rate']
    )
    
    print("\n3. Model Summary:")
    model.summary()
    
    # Setup callbacks
    print("\n4. Setting up training callbacks...")
    
    checkpoint_path = os.path.join(
        EC2_MODEL_DIR,
        f'{MODEL_NAME}_{MODEL_VERSION}_best.keras'
    )
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=TRAINING_CONFIG['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(EC2_MODEL_DIR, 'logs'),
            histogram_freq=1
        )
    ]
    
    print(f"   ✓ Model checkpoint: {checkpoint_path}")
    print(f"   ✓ Early stopping (patience={TRAINING_CONFIG['early_stopping_patience']})")
    print(f"   ✓ Learning rate reduction")
    print(f"   ✓ TensorBoard logging")
    
    # Train
    print("\n5. Training model...")
    print(f"   Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"   Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"   Learning rate: {TRAINING_CONFIG['learning_rate']}")
    
    history = model.fit(
        [data['X_train_tabular'], data['X_train_images']],
        data['y_train'],
        validation_data=(
            [data['X_test_tabular'], data['X_test_images']],
            data['y_test']
        ),
        epochs=TRAINING_CONFIG['epochs'],
        batch_size=TRAINING_CONFIG['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n6. Evaluating model...")
    test_loss, test_accuracy = model.evaluate(
        [data['X_test_tabular'], data['X_test_images']],
        data['y_test'],
        verbose=0
    )
    
    print(f"\nTest Results:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Accuracy: {test_accuracy:.4f}")
    
    # Save final model
    final_path = os.path.join(
        EC2_MODEL_DIR,
        f'{MODEL_NAME}_{MODEL_VERSION}_final.keras'
    )
    model.save(final_path)
    print(f"\n✓ Final model saved to: {final_path}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nModel artifacts saved to: {EC2_MODEL_DIR}")
    print("\nTo view TensorBoard:")
    print(f"  tensorboard --logdir={os.path.join(EC2_MODEL_DIR, 'logs')}")


def train_tabular():
    """Train tabular-only model (backward compatible)"""
    print("\n" + "=" * 70)
    print("Training Tabular-Only Fraud Detection Model (EC2)")
    print("=" * 70)
    
    setup_gpu()
    verify_directories()
    
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Initialize preprocessor
    print("\n1. Preparing tabular data...")
    preprocessor = FraudDataPreprocessor()
    
    # Generate or load data
    X, y = preprocessor.generate_synthetic_data(n_samples=5000, fraud_ratio=0.1)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TRAINING_CONFIG['validation_split'],
        stratify=y,
        random_state=42
    )
    
    print(f"   Training samples: {len(y_train)}")
    print(f"   Testing samples: {len(y_test)}")
    
    # Build model
    print("\n2. Building tabular transformer model...")
    tabular_model = FraudDetectionTransformer(MODEL_CONFIG)
    model = tabular_model.compile_model(
        learning_rate=TRAINING_CONFIG['learning_rate']
    )
    
    print("\n3. Model Summary:")
    model.summary()
    
    # Train
    print("\n4. Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=TRAINING_CONFIG['epochs'],
        batch_size=TRAINING_CONFIG['batch_size'],
        verbose=1
    )
    
    # Evaluate
    print("\n5. Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTest Results:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Accuracy: {test_accuracy:.4f}")
    
    # Save model
    final_path = os.path.join(EC2_MODEL_DIR, 'tabular_fraud_detection.keras')
    model.save(final_path)
    print(f"\n✓ Model saved to: {final_path}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='AWS EC2 Quick Start for Fraud Detection'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['multimodal', 'tabular'],
        default='multimodal',
        help='Training mode: multimodal (default) or tabular-only'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("AWS EC2 G5 XLarge - Fraud Detection Quick Start")
    print("=" * 70)
    print(f"\nMode: {args.mode.upper()}")
    print(f"TensorFlow version: {tf.__version__}")
    
    if args.mode == 'multimodal':
        train_multimodal()
    else:
        train_tabular()


if __name__ == '__main__':
    main()
