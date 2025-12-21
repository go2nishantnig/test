# Complete Pipeline Usage Guide

This guide walks you through using the complete multimodal fraud detection pipeline in Google Colab.

## What This Pipeline Does

The complete pipeline implements exactly the flow described in the problem statement:

```
Online Payment Fraud Dataset (Google Drive)
              ↓
         Tabular Data
              ↓
    FraudDataPreprocessor
              ↓
              ├─────────────────────────┐
              ↓                         ↓
QR Code Images Dataset (Google Drive)  |
              ↓                         |
         Image Data                     |
              ↓                         |
      QRCodePreprocessor               |
              ↓                         |
              └─────────────────────────┤
                                        ↓
                  MultimodalDataPreprocessor
                                        ↓
                  Processed Data Storage (Google Drive)
                                        ↓
                ┌───────────────────────┴───────────────┐
                ↓                                       ↓
    TabularTransformerEncoder              VisionTransformerEncoder
                ↓                                       ↓
                └───────────────┬───────────────────────┘
                                ↓
                         Cross-Modal Fusion
                                ↓
                         Fraud Detection
```

## Step-by-Step Walkthrough

### Step 1: Open the Notebook

1. Go to GitHub: https://github.com/go2nishantnig/test
2. Navigate to: `colab/complete_pipeline_demo.ipynb`
3. Click "Open in Colab" or manually open in Google Colab

### Step 2: Mount Google Drive

**Cell 1: Google Drive Setup**
```python
from google.colab import drive
drive.mount('/content/drive')
```

This creates the following structure in your Google Drive:
```
MyDrive/FraudDetection/
├── datasets/
│   ├── online_payment_fraud/
│   └── qr_codes/
├── processed_data/
└── models/
```

**Why Google Drive?**
- Persistent storage across Colab sessions
- No need to re-upload datasets each time
- Easy sharing and collaboration
- Automatic backup

### Step 3: Install Dependencies

**Cell 2: Install and Clone**
```python
!pip install -q tensorflow numpy pandas matplotlib scikit-learn
!git clone https://github.com/go2nishantnig/test.git /content/test
```

Installs all required packages and clones the repository.

### Step 4: Load Online Payment Fraud Dataset

**Cell 4: Load Tabular Data**

The notebook will:
1. Check if `fraud_transactions.csv` exists in Google Drive
2. If yes: Load existing data
3. If no: Generate synthetic data and save to Drive

**Tabular Data Structure:**
- Transaction records with features like amount, type, balances
- Binary fraud labels (0 = normal, 1 = fraud)
- 5,000 samples with ~10% fraud ratio

### Step 5: Load QR Code Images Dataset

**Cell 5: Load Image Data**

The notebook will:
1. Check if `qr_images.npy` and `qr_labels.npy` exist in Drive
2. If yes: Load existing images
3. If no: Generate synthetic QR codes and save to Drive

**Image Data Structure:**
- QR code images (128×128×3 RGB)
- Binary labels (0 = benign, 1 = malicious)
- 5,000 images with ~30% malicious ratio

### Step 6: FraudDataPreprocessor

**Cell 6: Process Tabular Data**

```python
fraud_preprocessor = FraudDataPreprocessor()
X_tabular, y_tabular = fraud_preprocessor.preprocess_data(fraud_data, fit=True)
```

**What happens:**
1. **Encoding**: Convert transaction type to numeric (PAYMENT→0, TRANSFER→1, etc.)
2. **Scaling**: Normalize numeric features using StandardScaler
3. **Reshaping**: Format for transformer input (n_samples, 1, 8)

**Output:** Preprocessed transaction features ready for TabularTransformerEncoder

### Step 7: QRCodePreprocessor

**Cell 7: Process Image Data**

```python
qr_preprocessor = QRCodePreprocessor(image_size=(128, 128))
X_images = qr_images  # Already normalized
```

**What happens:**
1. **Resize**: Ensure all images are 128×128
2. **Normalize**: Scale pixel values to [0, 1]
3. **Format**: RGB format with 3 channels

**Output:** Preprocessed QR code images ready for VisionTransformerEncoder

### Step 8: MultimodalDataPreprocessor

**Cell 8: Combine Both Modalities**

```python
multimodal_preprocessor = MultimodalDataPreprocessor(image_size=(128, 128))
data = multimodal_preprocessor.prepare_train_test_data(
    test_size=0.2,
    random_state=42,
    n_samples=5000
)
```

**What happens:**
1. **Generate Correlated Data**: 
   - Fraud transactions → 80% chance of malicious QR code
   - Normal transactions → 10% chance of malicious QR code
2. **Align Modalities**: Ensure each transaction has a corresponding QR image
3. **Train/Test Split**: 80% train, 20% test with stratification
4. **Preprocess Both**: Apply FraudDataPreprocessor and QRCodePreprocessor

**Output:**
```python
{
    'X_train_tabular': (4000, 1, 8),
    'X_train_images': (4000, 128, 128, 3),
    'y_train': (4000,),
    'X_test_tabular': (1000, 1, 8),
    'X_test_images': (1000, 128, 128, 3),
    'y_test': (1000,)
}
```

### Step 9: Save Processed Data to Google Drive

**Cell 9: Persistent Storage**

```python
np.save(f'{PROCESSED_DATA_PATH}/X_train_tabular.npy', data['X_train_tabular'])
np.save(f'{PROCESSED_DATA_PATH}/X_train_images.npy', data['X_train_images'])
# ... and more
```

**Saved files:**
- Training data: X_train_tabular.npy, X_train_images.npy, y_train.npy
- Test data: X_test_tabular.npy, X_test_images.npy, y_test.npy
- Preprocessor states: preprocessors.pkl
- Metadata: metadata.json

**Benefits:**
- Load preprocessed data instantly in future sessions
- No need to reprocess every time
- Share processed data with team members

### Step 10: TabularTransformerEncoder

**Cell 10: Process Transaction Features**

```python
model_builder = MultimodalFraudDetectionTransformer(MODEL_CONFIG)
tabular_input = keras.Input(shape=(1, 8), name='tabular_input')
tabular_encoded = model_builder.build_tabular_encoder(tabular_input)
```

**Architecture:**
1. **Input**: (batch, 1, 8) transaction features
2. **Dense Projection**: Project to d_model dimension (64)
3. **Positional Encoding**: Add position information
4. **Transformer Blocks** × 2:
   - Multi-head self-attention (4 heads)
   - Feed-forward network (128 units)
   - Layer normalization
   - Residual connections
5. **Output**: (batch, 1, 64) encoded features

**Visualization:**
- Heatmap of encoded features
- Shows learned representations

### Step 11: VisionTransformerEncoder

**Cell 11: Process QR Code Images**

```python
image_input = keras.Input(shape=(128, 128, 3), name='image_input')
image_encoded = model_builder.build_image_encoder(image_input)
```

**Architecture:**
1. **Input**: (batch, 128, 128, 3) QR code images
2. **Patch Embedding**:
   - Divide image into 16×16 patches
   - Creates 64 patches (8×8 grid)
   - Project each patch to d_model dimension
3. **Positional Encoding**: Add spatial position info
4. **Transformer Blocks** × 2:
   - Multi-head self-attention (4 heads)
   - Feed-forward network (128 units)
   - Layer normalization
   - Residual connections
5. **Output**: (batch, 64, 64) encoded patch features

**Visualizations:**
- Original QR code image
- Patch grid overlay (16×16 patches)
- Encoded feature heatmap

### Step 12: Complete Multimodal Model

**Cell 12: Cross-Modal Fusion**

```python
model = model_builder.build_model()
```

**Complete Architecture:**
1. **Tabular Branch**: TabularTransformerEncoder
2. **Image Branch**: VisionTransformerEncoder
3. **Cross-Modal Fusion**: Bidirectional attention
   - Tabular features attend to image features
   - Image features attend to tabular features
   - Enables information exchange between modalities
4. **Classification Head**:
   - Global average pooling on both branches
   - Concatenate features
   - Dense(128) + ReLU + Dropout
   - Dense(64) + ReLU + Dropout
   - Dense(1) + Sigmoid → Fraud probability

**Model Summary:** Shows complete architecture with parameter counts

### Step 13: Model Inference Demo

**Cell 13: Test Predictions**

```python
predictions = model.predict([sample_tabular, sample_images])
```

**Demonstrates:**
- Forward pass through complete model
- Prediction output format
- Visualization of predictions vs. true labels

**Note:** Model is untrained, so predictions are random. For real predictions, train the model first.

### Step 14: Save Model to Google Drive

**Cell 14: Model Persistence**

```python
model.save(f'{MODEL_SAVE_PATH}/multimodal_model.keras')
```

Saves the complete model architecture to Google Drive for future use.

### Step 15: Pipeline Summary

**Cell 15: Complete Overview**

Shows:
- All completed steps
- Storage statistics
- File locations in Google Drive
- Next steps for training

## Using Your Own Datasets

### Option 1: Upload CSV (Tabular Data)

Upload your fraud dataset to:
```
MyDrive/FraudDetection/datasets/online_payment_fraud/fraud_transactions.csv
```

Required columns:
- `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`
- `nameDest`, `oldbalanceDest`, `newbalanceDest`, `is_fraud`, `isFlaggedFraud`

### Option 2: Upload Images (QR Codes)

Save images as NumPy arrays:
```python
# In a separate notebook or script
import numpy as np
from PIL import Image
import glob

# Load your QR code images
images = []
labels = []

for img_path in glob.glob('path/to/your/qr_codes/*.png'):
    img = Image.open(img_path).resize((128, 128))
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    images.append(img_array)
    labels.append(1 if 'malicious' in img_path else 0)

# Save to Google Drive
np.save('/content/drive/MyDrive/FraudDetection/datasets/qr_codes/qr_images.npy', 
        np.array(images))
np.save('/content/drive/MyDrive/FraudDetection/datasets/qr_codes/qr_labels.npy', 
        np.array(labels))
```

## Training the Model

After running the complete pipeline:

```python
# In a new cell or notebook
from config.config import TRAINING_CONFIG

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=TRAINING_CONFIG['learning_rate']),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
)

# Train
history = model.fit(
    [data['X_train_tabular'], data['X_train_images']],
    data['y_train'],
    batch_size=32,
    epochs=50,
    validation_data=([data['X_test_tabular'], data['X_test_images']], data['y_test']),
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            f'{MODEL_SAVE_PATH}/best_model.keras',
            save_best_only=True,
            monitor='val_loss'
        ),
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
)

# Save trained model
model.save(f'{MODEL_SAVE_PATH}/trained_multimodal_model.keras')
```

## Common Issues

### Issue: "Drive not mounted"
**Solution:** Re-run the Google Drive mount cell

### Issue: "Out of memory"
**Solution:** Reduce batch size or use fewer samples

### Issue: "Module not found"
**Solution:** Re-run the installation cell

### Issue: "File not found in Drive"
**Solution:** The notebook will automatically generate synthetic data

## Next Steps

1. ✅ Complete this pipeline to understand the flow
2. 📊 Upload your own datasets
3. 🏋️ Train the model on your data
4. 📈 Evaluate performance
5. 🚀 Deploy for production inference

## Resources

- [Main README](../README.md) - Complete project documentation
- [END_TO_END_FLOW.md](../END_TO_END_FLOW.md) - Detailed flow documentation
- [GOOGLE_DRIVE_SETUP.md](GOOGLE_DRIVE_SETUP.md) - Drive setup guide
- [Colab README](README.md) - Notebooks overview

---

**Happy experimenting! 🎯**
