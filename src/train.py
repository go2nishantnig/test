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

from config.config import MODEL_CONFIG, TRAINING_CONFIG, MODEL_SAVE_DIR, MODEL_NAME, MODEL_VERSION, GPU_CONFIG
from src.models.transformer_model import (
    FraudDetectionTransformer,
    MultimodalFraudDetectionTransformer
)
from src.utils.data_preprocessing import (
    FraudDataPreprocessor,
    MultimodalDataPreprocessor
)


def force_cpu_execution():
    """
    Force TensorFlow to use CPU only by disabling GPU devices.
    
    This is a helper function used when GPU initialization fails.
    
    Raises:
        RuntimeError: If GPU cannot be disabled
    """
    try:
        tf.config.set_visible_devices([], 'GPU')
        print("✓ Successfully disabled GPU, using CPU\n")
    except (RuntimeError, ValueError) as e:
        # RuntimeError: If device configuration is already locked
        # ValueError: If device configuration is invalid
        error_msg = f"Failed to disable GPU: {e}"
        print(f"✗ {error_msg}\n")
        raise RuntimeError(error_msg)
    except Exception as e:
        # Catch any other unexpected errors during GPU disabling
        error_msg = f"Unexpected error while disabling GPU: {e}"
        print(f"✗ {error_msg}\n")
        raise RuntimeError(error_msg)


def build_model_with_fallback(model_builder, model_name="model"):
    """
    Build a model with automatic CPU fallback if GPU initialization fails.
    
    Args:
        model_builder: Callable that builds and returns the model
        model_name: Name of the model for logging purposes
        
    Returns:
        The compiled model
        
    Raises:
        Exception: If model building fails on both GPU and CPU
    """
    try:
        # First attempt: build with current device configuration
        return model_builder()
    except (tf.errors.InternalError, tf.errors.UnknownError, 
            tf.errors.ResourceExhaustedError) as e:
        # Common GPU initialization errors:
        # - InternalError: Internal TensorFlow error (often GPU-related)
        # - UnknownError: Unknown error (can include CUDA errors)
        # - ResourceExhaustedError: Out of GPU memory
        print(f"\n⚠ GPU error during {model_name} building: {e}")
        print("Attempting to force CPU execution and retry...")
        
        # Force CPU execution
        try:
            force_cpu_execution()
        except RuntimeError as force_error:
            print(f"Cannot retry on CPU: {force_error}")
            raise e  # Re-raise original error
        
        # Retry model building on CPU
        try:
            model = model_builder()
            print(f"✓ {model_name.capitalize()} successfully built on CPU")
            return model
        except Exception as retry_error:
            print(f"✗ Model building failed even on CPU: {retry_error}")
            raise retry_error
    except Exception as e:
        # Catch other unexpected errors and attempt CPU fallback
        # This includes system errors like "Floating point exception"
        print(f"\n⚠ Unexpected error during {model_name} building: {e}")
        print("Attempting to force CPU execution and retry...")
        
        # Force CPU execution
        try:
            force_cpu_execution()
        except RuntimeError as force_error:
            print(f"Cannot retry on CPU: {force_error}")
            raise e  # Re-raise original error
        
        # Retry model building on CPU
        try:
            model = model_builder()
            print(f"✓ {model_name.capitalize()} successfully built on CPU")
            return model
        except Exception as retry_error:
            print(f"✗ Model building failed even on CPU: {retry_error}")
            raise retry_error


def configure_gpu():
    """
    Configure GPU settings for TensorFlow to prevent memory allocation issues.
    
    This function should be called before any TensorFlow operations to:
    - Enable memory growth (allocate GPU memory as needed instead of all at once)
    - Configure mixed precision training if enabled
    - Gracefully handle cases where GPU is not available or fails to initialize
    - Force CPU usage if GPU initialization fails
    
    Returns:
        bool: True if GPU is successfully configured, False if falling back to CPU
    """
    try:
        # Get list of physical GPUs
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"\n{'='*70}")
            print(f"GPU Configuration")
            print(f"{'='*70}")
            print(f"Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            
            # Configure memory growth for each GPU
            if GPU_CONFIG.get('memory_growth', True):
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"\n✓ GPU memory growth enabled")
                    print(f"  (GPU memory will be allocated as needed)")
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(f"\n⚠ Warning: Could not set memory growth: {e}")
            
            # Configure mixed precision if enabled
            if GPU_CONFIG.get('mixed_precision', False):
                try:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    print(f"✓ Mixed precision training enabled (float16)")
                except Exception as e:
                    print(f"⚠ Warning: Could not enable mixed precision: {e}")
            
            print(f"{'='*70}\n")
            
            # Test GPU initialization to catch early failures
            try:
                # Try to create a simple tensor on GPU to verify it works
                with tf.device('/GPU:0'):
                    test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    _ = test_tensor * 2
                print("✓ GPU initialization test passed\n")
                return True
            except (tf.errors.InternalError, tf.errors.UnknownError, 
                    tf.errors.ResourceExhaustedError) as gpu_error:
                # Common GPU initialization errors:
                # - InternalError: Internal TensorFlow error (often GPU-related)
                # - UnknownError: Unknown error (can include CUDA errors)
                # - ResourceExhaustedError: Out of GPU memory
                print(f"\n{'='*70}")
                print(f"⚠ GPU Initialization Failed")
                print(f"{'='*70}")
                print(f"Error: {gpu_error}")
                print(f"This may be caused by:")
                print(f"  - Missing or incompatible CUDA libraries")
                print(f"  - Driver version mismatch")
                print(f"  - Corrupted TensorFlow installation")
                print(f"\nForcing CPU execution for stability...")
                print(f"{'='*70}\n")
                
                # Force CPU usage
                try:
                    force_cpu_execution()
                except RuntimeError:
                    # If we can't force CPU, continue anyway
                    pass
                
                return False
            except Exception as gpu_error:
                # Catch any other unexpected errors (e.g., system-level errors)
                # This includes "shared object symbol not found" and similar issues
                print(f"\n{'='*70}")
                print(f"⚠ GPU Initialization Failed (Unexpected Error)")
                print(f"{'='*70}")
                print(f"Error: {gpu_error}")
                print(f"This may be caused by:")
                print(f"  - Missing or incompatible CUDA libraries")
                print(f"  - Driver version mismatch")
                print(f"  - Corrupted TensorFlow installation")
                print(f"  - System-level GPU errors")
                print(f"\nForcing CPU execution for stability...")
                print(f"{'='*70}\n")
                
                # Force CPU usage
                try:
                    force_cpu_execution()
                except RuntimeError:
                    # If we can't force CPU, continue anyway
                    pass
                
                return False
        else:
            print(f"\n{'='*70}")
            print(f"No GPU detected - using CPU")
            print(f"{'='*70}\n")
            return False
            
    except Exception as e:
        print(f"\n⚠ Warning: Error during GPU configuration: {e}")
        print(f"Continuing with default configuration...\n")
        return False


def train_multimodal_model():
    """Train the multimodal fraud detection transformer model"""
    
    print("=" * 70)
    print("Multimodal Fraud Detection Transformer Model Training")
    print("(Combining Tabular + Image Data)")
    print("=" * 70)
    
    # Configure GPU before any TensorFlow operations
    # Returns True if GPU is working, False if using CPU fallback
    gpu_available = configure_gpu()
    
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
    
    def build_multimodal():
        multimodal_model = MultimodalFraudDetectionTransformer(MODEL_CONFIG)
        return multimodal_model.compile_model(learning_rate=TRAINING_CONFIG['learning_rate'])
    
    model = build_model_with_fallback(build_multimodal, "multimodal transformer model")
    
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
    
    # Configure GPU before any TensorFlow operations
    # Returns True if GPU is working, False if using CPU fallback
    gpu_available = configure_gpu()
    
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
    
    def build_tabular():
        fraud_model = FraudDetectionTransformer(MODEL_CONFIG)
        return fraud_model.compile_model(learning_rate=TRAINING_CONFIG['learning_rate'])
    
    model = build_model_with_fallback(build_tabular, "transformer model")
    
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
