# Using Real Kaggle Datasets

This guide explains how to use the real Kaggle datasets with the fraud detection transformer model.

## Overview

The notebooks and code now support **three modes**:
1. **Real Kaggle Datasets** (recommended for production use)
2. **Previously Saved Data** (for faster loading)
3. **Synthetic Data** (automatic fallback for testing)

## Kaggle Datasets

### 1. Online Payments Fraud Detection Dataset

**Source**: [Online Payments Fraud Detection Dataset](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset)

**Description**: Contains transaction data with fraud labels. This dataset includes information about online payment transactions including transaction type, amount, balances, and fraud indicators.

**File Format**: CSV file (typically named `PS_20174392719_1491204439457_log.csv`)

**Expected Columns**:
- `step`: Hour of simulation (1-744)
- `type`: Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN)
- `amount`: Transaction amount
- `nameOrig`: Origin account ID
- `oldbalanceOrg`: Initial balance before transaction
- `newbalanceOrig`: Balance after transaction
- `nameDest`: Destination account ID
- `oldbalanceDest`: Initial recipient balance
- `newbalanceDest`: Recipient balance after transaction
- `isFraud`: Fraud label (0 = normal, 1 = fraud)
- `isFlaggedFraud`: Business rule flagged as fraud

**Setup for Google Colab**:
1. Download the dataset from Kaggle
2. Upload to Google Drive at: `/content/drive/MyDrive/FraudDetection/datasets/online_payment_fraud/PS_20174392719_1491204439457_log.csv`
3. The notebook will automatically detect and load it

**Setup for Local Jupyter**:
1. Download the dataset from Kaggle
2. Place at: `data/PS_20174392719_1491204439457_log.csv` (or update the path in the notebook)

### 2. Benign and Malicious QR Codes Dataset

**Source**: [Benign and Malicious QR Codes Dataset](https://www.kaggle.com/datasets/samahsadiq/benign-and-malicious-qr-codes)

**Description**: Contains QR code images categorized as benign or malicious. These images are used for the visual analysis component of the multimodal fraud detection model.

**File Format**: PNG images organized in folders

**Expected Directory Structure**:
```
qr_codes/
├── benign/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── malicious/
    ├── image1.png
    ├── image2.png
    └── ...
```

**Setup for Google Colab**:
1. Download the dataset from Kaggle
2. Extract the folders
3. Upload to Google Drive at: `/content/drive/MyDrive/FraudDetection/datasets/qr_codes/`
4. Ensure you have both `benign/` and `malicious/` folders with PNG files

**Setup for Local Jupyter**:
1. Download the dataset from Kaggle
2. Extract to: `data/qr_codes/` (or update the path in the notebook)
3. Ensure the folder structure matches the expected layout

## Using the Datasets

### In Google Colab (complete_pipeline_demo.ipynb)

The notebook automatically detects and uses available datasets:

```python
# The notebook checks in this order:
# 1. Real Kaggle CSV at: {FRAUD_DATASET_PATH}/PS_20174392719_1491204439457_log.csv
# 2. Previously processed data at: {FRAUD_DATASET_PATH}/fraud_transactions.csv
# 3. Falls back to synthetic data generation
```

**No code changes needed** - just place the datasets in the correct Google Drive locations!

### In Local Jupyter (fraud_detection_demo.ipynb)

Update the path in the data loading cell:

```python
# Update this path to point to your downloaded Kaggle dataset
kaggle_csv_path = 'data/PS_20174392719_1491204439457_log.csv'

if os.path.exists(kaggle_csv_path):
    data = preprocessor.load_from_csv(kaggle_csv_path)
else:
    # Falls back to synthetic data
    data = preprocessor.generate_synthetic_data(n_samples=5000, fraud_ratio=0.02)
```

### In Python Scripts

You can use the preprocessor classes directly:

```python
from src.utils.tabular_preprocessor import FraudDataPreprocessor
from src.utils.image_preprocessor import QRCodePreprocessor

# Load tabular data from CSV
fraud_preprocessor = FraudDataPreprocessor()
fraud_data = fraud_preprocessor.load_from_csv('path/to/dataset.csv')

# Load QR code images from directory
qr_preprocessor = QRCodePreprocessor(image_size=(128, 128))
qr_images, qr_labels = qr_preprocessor.load_images_from_directory('path/to/qr_codes')
```

## Requirements

The following dependencies are required for loading real datasets:

- `pandas>=2.0.0` - For CSV loading
- `Pillow>=10.0.0` - For image loading
- `numpy>=1.24.0` - For data processing

Install with:
```bash
pip install -r requirements.txt
```

## Column Name Compatibility

The code automatically handles different column naming conventions:

- **Kaggle Dataset**: Uses `isFraud` 
- **Synthetic Data**: Uses `is_fraud`
- **Preprocessing**: Standardizes to `isFraud`

Both naming conventions are supported and converted automatically.

## Data Validation

The preprocessors include validation:

### Tabular Data
- Checks for required columns
- Warns about missing columns
- Handles both `isFraud` and `is_fraud` column names

### Image Data
- Validates folder structure
- Supports PNG, JPG, JPEG formats
- Automatically resizes images to target size
- Normalizes pixel values to [0, 1]

## Fallback Behavior

If real datasets are not found, the system automatically:

1. **Prints a helpful message** with dataset download instructions
2. **Generates synthetic data** for demonstration
3. **Saves synthetic data** to the expected location for next time

This ensures the notebooks always run, even without Kaggle datasets.

## Performance Tips

1. **First Run**: Load from Kaggle CSV/images (slower)
2. **Subsequent Runs**: Uses saved numpy arrays (much faster)
3. **Preprocessing**: Results are cached in Google Drive

Example timings:
- Loading CSV: ~10-30 seconds (depends on file size)
- Loading numpy arrays: ~1-2 seconds
- Loading images: ~1-5 minutes (depends on number of images)
- Loading cached images: ~2-5 seconds

## Troubleshooting

### "PIL/Pillow not installed"
```bash
pip install Pillow
```

### "CSV file not found"
- Check the file path is correct
- Ensure the file was uploaded to Google Drive
- Verify the filename matches exactly

### "No images found"
- Check folder structure: should have `benign/` and `malicious/` subdirectories
- Ensure PNG files are in the folders
- Check file permissions

### "Missing expected columns"
- Verify you downloaded the correct Kaggle dataset
- Check the CSV has all required columns
- Ensure no headers are missing

## Example Workflows

### Workflow 1: Complete Colab Setup

1. Mount Google Drive
2. Download Kaggle datasets
3. Upload to Drive at specified paths
4. Run complete_pipeline_demo.ipynb
5. Data is automatically loaded and processed

### Workflow 2: Local Development

1. Download Kaggle datasets
2. Extract to local `data/` folder
3. Update paths in notebook if needed
4. Run fraud_detection_demo.ipynb
5. Data is loaded from local files

### Workflow 3: Mixed Mode (Tabular Real + Image Synthetic)

The system supports using real data for one modality and synthetic for another:
- Place only the CSV in Google Drive
- QR codes will use synthetic data automatically
- Both work together seamlessly

## Dataset Statistics

### Online Payments Fraud Dataset
- **Samples**: ~6.3M transactions
- **Fraud Rate**: ~0.13% (highly imbalanced)
- **File Size**: ~500 MB
- **Features**: 11 columns

### QR Codes Dataset
- **Samples**: Varies by download
- **Image Size**: Various (resized to 128x128)
- **Format**: RGB PNG images
- **Categories**: 2 (benign, malicious)

## Conclusion

The updated code provides seamless support for both real Kaggle datasets and synthetic data generation. No matter which mode you use, the preprocessing pipeline and model training work identically, making it easy to develop with synthetic data and deploy with real data.
