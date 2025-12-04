# train.py - Model Training Script Documentation

## Overview
This script trains the fraud detection transformer model in either multimodal mode (combining tabular and image data) or tabular-only mode. It includes complete training pipeline with data generation, model building, training, and evaluation.

## Key Functions

### `train_multimodal_model()`
**Purpose**: Trains the multimodal transformer that processes both transaction data and QR code images.

#### Data Preparation
```python
preprocessor = MultimodalDataPreprocessor(image_size=MODEL_CONFIG.get('image_size', (128, 128)))
data = preprocessor.prepare_train_test_data(
    test_size=TRAINING_CONFIG['validation_split'],
    n_samples=5000
)
```
**What it does**: Initializes the multimodal preprocessor and generates 5,000 synthetic samples split into train (80%) and test (20%) sets. Each sample includes transaction features and a QR code image.

#### Model Building
```python
multimodal_model = MultimodalFraudDetectionTransformer(MODEL_CONFIG)
model = multimodal_model.compile_model(learning_rate=TRAINING_CONFIG['learning_rate'])
```
**What it does**: Creates the multimodal transformer architecture with separate encoders for tabular and image data, cross-modal attention fusion, and a classification head. Compiles with Adam optimizer and binary cross-entropy loss.

#### Training Callbacks
```python
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
```
**What it does**: Saves the best model during training based on validation loss. Prevents overfitting by keeping only the model with lowest validation error.

```python
early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=TRAINING_CONFIG['early_stopping_patience'],
    restore_best_weights=True,
    verbose=1
)
```
**What it does**: Stops training if validation loss doesn't improve for 5 consecutive epochs, restoring weights from the best epoch.

```python
reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)
```
**What it does**: Reduces learning rate by half when validation loss plateaus, helping the model fine-tune and escape local minima.

#### Training Loop
```python
history = model.fit(
    [data['X_train_tabular'], data['X_train_images']],
    data['y_train'],
    batch_size=TRAINING_CONFIG['batch_size'],
    epochs=TRAINING_CONFIG['epochs'],
    validation_data=(
        [data['X_test_tabular'], data['X_test_images']],
        data['y_test']
    ),
    callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr_callback, tensorboard_callback],
    verbose=1
)
```
**What it does**: Trains the model with dual inputs (tabular and images), using callbacks for model checkpointing, early stopping, learning rate scheduling, and TensorBoard logging.

### `train_tabular_model()`
**Purpose**: Trains the tabular-only transformer model (backward compatible with single modality).

#### Key Differences from Multimodal
```python
preprocessor = FraudDataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_data(
    test_size=TRAINING_CONFIG['validation_split']
)
fraud_model = FraudDetectionTransformer(MODEL_CONFIG)
model = fraud_model.compile_model(learning_rate=TRAINING_CONFIG['learning_rate'])
```
**What it does**: Uses only tabular preprocessor and single-input transformer architecture. Simpler than multimodal but less powerful for detecting QR code-related fraud patterns.

#### Model Saving
```python
model.save(final_model_path)  # Keras format
model.save(final_model_h5_path)  # H5 format for compatibility
model.export(saved_model_dir)  # SavedModel format for deployment
```
**What it does**: Saves the trained model in multiple formats:
- **Keras format (.keras)**: Native TensorFlow 2.x format
- **H5 format (.h5)**: Legacy format for backward compatibility
- **SavedModel**: TensorFlow serving format for production deployment

### `train_model(mode='multimodal')`
**Purpose**: Main training function that dispatches to the appropriate mode.

```python
def train_model(mode='multimodal'):
    if mode == 'multimodal':
        return train_multimodal_model()
    else:
        return train_tabular_model()
```
**What it does**: Acts as a router to select training mode based on command-line argument or API call.

## Usage

**Train multimodal model** (default):
```bash
python src/train.py
# or explicitly
python src/train.py --mode multimodal
```

**Train tabular-only model**:
```bash
python src/train.py --mode tabular
```

## Training Output
The script displays:
- Data preparation progress (samples, fraud ratio, shapes)
- Model architecture summary
- Training progress (loss, accuracy, precision, recall, AUC per epoch)
- Test set evaluation metrics
- Saved model locations
- TensorBoard log directory

## TensorBoard Visualization
View training metrics in real-time:
```bash
tensorboard --logdir models/logs
```

## Saved Artifacts
After training, the following files are saved:
- **Best model**: Model with lowest validation loss
- **Final model**: Model after all epochs (multiple formats)
- **Preprocessor**: Scaler and encoders for feature preprocessing
- **Logs**: TensorBoard training history
