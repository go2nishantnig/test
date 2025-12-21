# Google Drive Setup Guide for Colab Notebooks

This guide explains how to use the fraud detection notebooks with Google Drive for dataset storage and persistence.

## Overview

The fraud detection system uses **Google Drive** for:
1. **Dataset Storage**: Online Payment Fraud Dataset and QR Code Images Dataset
2. **Processed Data Storage**: Preprocessed tabular and image data
3. **Model Storage**: Trained model checkpoints and artifacts
4. **Persistence**: All data persists across Colab sessions

## Quick Start

### Step 1: Open Notebook in Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File** → **Open notebook**
3. Select **GitHub** tab
4. Enter: `go2nishantnig/test`
5. Choose one of:
   - `colab/complete_pipeline_demo.ipynb` (Recommended - Full pipeline)
   - `colab/transformer_encoders_demo.ipynb` (Encoder demos)

### Step 2: Mount Google Drive

Run the first code cell to mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

You'll be prompted to:
1. Click the authorization link
2. Sign in to your Google account
3. Copy the authorization code
4. Paste it back in Colab

### Step 3: Run All Cells

Click **Runtime** → **Run all** to execute the entire pipeline.

The notebook will automatically:
- Create directory structure in your Google Drive
- Load or generate datasets
- Process all data through the pipeline
- Save everything to Google Drive

## Directory Structure in Google Drive

After running the notebook, you'll find this structure in your Google Drive:

```
MyDrive/
└── FraudDetection/
    ├── datasets/
    │   ├── online_payment_fraud/
    │   │   └── fraud_transactions.csv
    │   └── qr_codes/
    │       ├── qr_images.npy
    │       └── qr_labels.npy
    ├── processed_data/
    │   ├── X_train_tabular.npy
    │   ├── X_train_images.npy
    │   ├── y_train.npy
    │   ├── X_test_tabular.npy
    │   ├── X_test_images.npy
    │   ├── y_test.npy
    │   ├── preprocessors.pkl
    │   └── metadata.json
    └── models/
        └── multimodal_model.keras
```

## Using Your Own Datasets

### Option 1: Upload to Google Drive (Recommended)

1. **For Tabular Data (CSV format)**:
   ```
   Upload to: MyDrive/FraudDetection/datasets/online_payment_fraud/fraud_transactions.csv
   
   Required columns:
   - step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
     nameDest, oldbalanceDest, newbalanceDest, is_fraud, isFlaggedFraud
   ```

2. **For QR Code Images**:
   ```
   Upload to: MyDrive/FraudDetection/datasets/qr_codes/
   
   Format: Save as NumPy arrays
   - qr_images.npy: shape (n_samples, 128, 128, 3), values [0, 1]
   - qr_labels.npy: shape (n_samples,), values 0 (benign) or 1 (malicious)
   ```

3. The notebook will automatically detect and load your data.

### Option 2: Use Synthetic Data

If you don't have datasets, the notebooks will automatically generate synthetic data:
- **Fraud Transactions**: 5,000 samples with 10% fraud ratio
- **QR Code Images**: 5,000 images with 30% malicious ratio

This synthetic data is saved to Google Drive for future use.

## Loading Previously Processed Data

If you've already run the pipeline once, you can load the processed data directly:

```python
import numpy as np

# Load processed data from Google Drive
PROCESSED_DATA_PATH = '/content/drive/MyDrive/FraudDetection/processed_data'

X_train_tabular = np.load(f'{PROCESSED_DATA_PATH}/X_train_tabular.npy')
X_train_images = np.load(f'{PROCESSED_DATA_PATH}/X_train_images.npy')
y_train = np.load(f'{PROCESSED_DATA_PATH}/y_train.npy')

print(f"Loaded training data:")
print(f"  Tabular: {X_train_tabular.shape}")
print(f"  Images: {X_train_images.shape}")
print(f"  Labels: {y_train.shape}")
```

## Storage Requirements

Approximate storage needed in Google Drive:

| Component | Size | Description |
|-----------|------|-------------|
| Fraud Transactions CSV | ~1-2 MB | 5,000 transaction records |
| QR Code Images | ~50-100 MB | 5,000 images (128×128×3) |
| Processed Data | ~50-100 MB | Preprocessed train/test splits |
| Model Checkpoint | ~5-10 MB | Trained model weights |
| **Total** | **~100-200 MB** | For 5,000 samples |

## Benefits of Google Drive Storage

✅ **Persistence**: Data remains available across Colab sessions
✅ **No Re-processing**: Load preprocessed data instantly
✅ **Version Control**: Keep multiple dataset versions
✅ **Collaboration**: Share datasets via Google Drive
✅ **Backup**: Automatic Google Drive backup
✅ **Accessibility**: Access from any device

## Troubleshooting

### Issue: "Google Drive not mounted"

**Solution**: Run the mount cell again and complete authorization.

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### Issue: "Directory not found"

**Solution**: The notebook creates directories automatically. If you get this error:

```python
import os
DRIVE_BASE = '/content/drive/MyDrive/FraudDetection'
os.makedirs(DRIVE_BASE, exist_ok=True)
os.makedirs(f'{DRIVE_BASE}/datasets/online_payment_fraud', exist_ok=True)
os.makedirs(f'{DRIVE_BASE}/datasets/qr_codes', exist_ok=True)
os.makedirs(f'{DRIVE_BASE}/processed_data', exist_ok=True)
os.makedirs(f'{DRIVE_BASE}/models', exist_ok=True)
```

### Issue: "Out of memory"

**Solution**: For large datasets:
1. Use a smaller batch size
2. Reduce the number of samples
3. Upgrade to Colab Pro for more RAM

### Issue: "Storage quota exceeded"

**Solution**: 
1. Free Google Drive accounts have 15 GB free storage
2. Delete old processed data you no longer need
3. Upgrade to Google One for more storage

## Customizing Paths

To use a different location in Google Drive, modify these variables:

```python
# Custom path example
DRIVE_BASE = '/content/drive/MyDrive/MyCustomFolder'
FRAUD_DATASET_PATH = f'{DRIVE_BASE}/my_fraud_data'
QR_DATASET_PATH = f'{DRIVE_BASE}/my_qr_data'
PROCESSED_DATA_PATH = f'{DRIVE_BASE}/processed'
MODEL_SAVE_PATH = f'{DRIVE_BASE}/trained_models'
```

## Best Practices

1. **First Run**: Let the notebook generate synthetic data to verify everything works
2. **Custom Data**: Upload your datasets after verifying the pipeline works
3. **Backups**: Keep a backup of important datasets outside Google Drive
4. **Cleaning**: Periodically clean old processed data to free space
5. **Metadata**: Always check `metadata.json` to understand your processed data

## Security Note

⚠️ **Important**: The notebooks have full access to your Google Drive when mounted. Always:
- Review notebook code before running
- Use trusted notebooks only
- Don't share your authorization tokens
- Revoke access when done (Google Account → Security → Third-party apps)

## Next Steps

After setting up Google Drive:

1. **Run Complete Pipeline**: `complete_pipeline_demo.ipynb`
2. **Explore Encoders**: `transformer_encoders_demo.ipynb`
3. **Train Model**: Use processed data to train the fraud detection model
4. **Deploy**: Export model for production inference

## Support

For issues or questions:
- Check the [main README](../README.md)
- Review [END_TO_END_FLOW.md](../END_TO_END_FLOW.md)
- Check the notebook outputs for error messages

---

**Happy coding! 🚀**
