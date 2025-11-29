"""
Training script for multimodal fraud detection transformer model

Supports two training modes:
1. Multimodal: Uses both tabular and image data
2. Tabular-only: Uses only tabular data (backward compatible)
"""
import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import MODEL_CONFIG, TRAINING_CONFIG, MODEL_SAVE_DIR, MODEL_NAME, MODEL_VERSION
from src.models.transformer_model import (
    FraudDetectionTransformer,
    MultimodalFraudDetectionTransformer
)
from src.utils.data_preprocessing import (
    FraudDataPreprocessor,
    MultimodalDataPreprocessor
)


def train_multimodal_model():
    """Train the multimodal fraud detection transformer model"""
    
    print("=" * 70)
    print("Multimodal Fraud Detection Transformer Model Training")
    print("(Combining Tabular + Image Data)")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Initialize multimodal data preprocessor
    print("\n1. Preparing multimodal data...")
    preprocessor = MultimodalDataPreprocessor(image_size=MODEL_CONFIG.get('image_size', (128, 128)))
    
    # Generate and preprocess multimodal data
    data = preprocessor.prepare_train_test_data(
        test_size=TRAINING_CONFIG['validation_split'],
        n_samples=5000
    )
    
    print(f"   Training samples: {len(data['y_train'])}")
    print(f"   Testing samples: {len(data['y_test'])}")
    print(f"   Fraud ratio in training: {np.mean(data['y_train']):.2%}")
    print(f"   Fraud ratio in testing: {np.mean(data['y_test']):.2%}")
    print(f"   Tabular shape: {data['X_train_tabular'].shape}")
    print(f"   Image shape: {data['X_train_images'].shape}")
    
    # Save preprocessors
    preprocessor_path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_{MODEL_VERSION}_preprocessor.pkl')
    preprocessor.save_preprocessors(preprocessor_path)
    
    # Build and compile model
    print("\n2. Building multimodal transformer model...")
    multimodal_model = MultimodalFraudDetectionTransformer(MODEL_CONFIG)
    model = multimodal_model.compile_model(learning_rate=TRAINING_CONFIG['learning_rate'])
    
    print("\n3. Model architecture:")
    model.summary()
    
    # Define callbacks
    print("\n4. Setting up training callbacks...")
    
    # Model checkpoint - save best model
    checkpoint_path = os.path.join(
        MODEL_SAVE_DIR,
        f'{MODEL_NAME}_{MODEL_VERSION}_best.keras'
    )
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    # Early stopping
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=TRAINING_CONFIG['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    )
    
    # Learning rate reduction
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # TensorBoard
    log_dir = os.path.join(
        os.path.dirname(MODEL_SAVE_DIR),
        'logs',
        f'{MODEL_NAME}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    
    # Train the model
    print("\n5. Training the multimodal model...")
    print(f"   Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"   Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"   Learning rate: {TRAINING_CONFIG['learning_rate']}")
    
    history = model.fit(
        [data['X_train_tabular'], data['X_train_images']],
        data['y_train'],
        batch_size=TRAINING_CONFIG['batch_size'],
        epochs=TRAINING_CONFIG['epochs'],
        validation_data=(
            [data['X_test_tabular'], data['X_test_images']],
            data['y_test']
        ),
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            reduce_lr_callback,
            tensorboard_callback
        ],
        verbose=1
    )
    
    # Evaluate the model
    print("\n6. Evaluating the model on test set...")
    test_results = model.evaluate(
        [data['X_test_tabular'], data['X_test_images']],
        data['y_test'],
        verbose=0
    )
    
    print("\n" + "=" * 70)
    print("Test Results:")
    print("=" * 70)
    for metric_name, metric_value in zip(model.metrics_names, test_results):
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Save final model in Keras format
    final_model_path = os.path.join(
        MODEL_SAVE_DIR,
        f'{MODEL_NAME}_{MODEL_VERSION}_final.keras'
    )
    print(f"\n7. Saving final model to: {final_model_path}")
    model.save(final_model_path)
    
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)
    print(f"\nModel saved to: {MODEL_SAVE_DIR}")
    print(f"  - Best model: {checkpoint_path}")
    print(f"  - Final model: {final_model_path}")
    print(f"  - Preprocessor: {preprocessor_path}")
    print(f"\nTensorBoard logs: {log_dir}")
    print(f"Run 'tensorboard --logdir {log_dir}' to view training metrics")
    
    return model, history


def train_tabular_model():
    """Train the tabular-only fraud detection transformer model (backward compatible)"""
    
    print("=" * 70)
    print("Fraud Detection Transformer Model Training (Tabular Only)")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Initialize data preprocessor
    print("\n1. Preparing data...")
    preprocessor = FraudDataPreprocessor()
    
    # Generate and preprocess data
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_data(
        test_size=TRAINING_CONFIG['validation_split']
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    print(f"   Fraud ratio in training: {np.mean(y_train):.2%}")
    print(f"   Fraud ratio in testing: {np.mean(y_test):.2%}")
    
    # Save scaler
    scaler_path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_{MODEL_VERSION}_scaler.pkl')
    preprocessor.save_scaler(scaler_path)
    
    # Build and compile model
    print("\n2. Building transformer model...")
    fraud_model = FraudDetectionTransformer(MODEL_CONFIG)
    model = fraud_model.compile_model(learning_rate=TRAINING_CONFIG['learning_rate'])
    
    print("\n3. Model architecture:")
    model.summary()
    
    # Define callbacks
    print("\n4. Setting up training callbacks...")
    
    # Model checkpoint - save best model
    checkpoint_path = os.path.join(
        MODEL_SAVE_DIR,
        f'{MODEL_NAME}_{MODEL_VERSION}_best.h5'
    )
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    # Early stopping
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=TRAINING_CONFIG['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    )
    
    # Learning rate reduction
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # TensorBoard
    log_dir = os.path.join(
        os.path.dirname(MODEL_SAVE_DIR),
        'logs',
        f'{MODEL_NAME}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    
    # Train the model
    print("\n5. Training the model...")
    print(f"   Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"   Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"   Learning rate: {TRAINING_CONFIG['learning_rate']}")
    
    history = model.fit(
        X_train, y_train,
        batch_size=TRAINING_CONFIG['batch_size'],
        epochs=TRAINING_CONFIG['epochs'],
        validation_data=(X_test, y_test),
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            reduce_lr_callback,
            tensorboard_callback
        ],
        verbose=1
    )
    
    # Evaluate the model
    print("\n6. Evaluating the model on test set...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    
    print("\n" + "=" * 70)
    print("Test Results:")
    print("=" * 70)
    for metric_name, metric_value in zip(model.metrics_names, test_results):
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Save final model in Keras format
    final_model_path = os.path.join(
        MODEL_SAVE_DIR,
        f'{MODEL_NAME}_{MODEL_VERSION}_final.keras'
    )
    print(f"\n7. Saving final model to: {final_model_path}")
    model.save(final_model_path)
    
    # Also save in H5 format for compatibility
    final_model_h5_path = os.path.join(
        MODEL_SAVE_DIR,
        f'{MODEL_NAME}_{MODEL_VERSION}_final.h5'
    )
    model.save(final_model_h5_path)
    
    # Export SavedModel format for deployment
    saved_model_dir = os.path.join(
        MODEL_SAVE_DIR,
        f'{MODEL_NAME}_{MODEL_VERSION}_savedmodel'
    )
    model.export(saved_model_dir)
    
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)
    print(f"\nModel saved to: {MODEL_SAVE_DIR}")
    print(f"  - Best model: {checkpoint_path}")
    print(f"  - Final model (Keras): {final_model_path}")
    print(f"  - Final model (H5): {final_model_h5_path}")
    print(f"  - SavedModel: {saved_model_dir}")
    print(f"  - Scaler: {scaler_path}")
    print(f"\nTensorBoard logs: {log_dir}")
    print(f"Run 'tensorboard --logdir {log_dir}' to view training metrics")
    
    return model, history


def train_model(mode='multimodal'):
    """
    Main training function with mode selection
    
    Args:
        mode: 'multimodal' for combined tabular+image model,
              'tabular' for tabular-only model
    """
    if mode == 'multimodal':
        return train_multimodal_model()
    else:
        return train_tabular_model()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train fraud detection transformer model')
    parser.add_argument(
        '--mode',
        type=str,
        default='multimodal',
        choices=['multimodal', 'tabular'],
        help='Training mode: multimodal (tabular+image) or tabular (tabular only)'
    )
    
    args = parser.parse_args()
    train_model(mode=args.mode)
