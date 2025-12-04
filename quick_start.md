# quick_start.py - Quick Start Script Documentation

## Overview
This script provides an easy-to-use demonstration of the fraud detection model with both multimodal and tabular-only prediction modes. It showcases how to load trained models and make predictions on sample transactions.

## Key Functions

### `demo_multimodal()`
```python
def demo_multimodal():
    predictor = MultimodalFraudDetectionPredictor()
    preprocessor = MultimodalDataPreprocessor(
        image_size=MODEL_CONFIG.get('image_size', (128, 128))
    )
```
**Purpose**: Demonstrates multimodal fraud detection combining transaction data with QR code images.

**Workflow**:
1. **Load Model**: Initializes the multimodal predictor with pre-trained weights
2. **Generate Sample Data**: Creates synthetic transaction data with associated QR codes
3. **Normal Example**: Shows prediction on a normal transaction with benign QR code
4. **Fraud Example**: Shows prediction on a suspicious transaction with malicious QR code

**Key Code Snippet - Generating Sample Data**:
```python
sample_data = preprocessor.generate_synthetic_multimodal_data(n_samples=1, fraud_ratio=0.0)
processed = preprocessor.preprocess_multimodal_data(
    sample_data['tabular'], sample_data['images'], fit=True
)
result = predictor.predict(processed['tabular'], processed['images'])
```
**What it does**: Generates a single sample transaction, preprocesses both tabular and image data, and runs the multimodal prediction model.

### `demo_tabular()`
```python
def demo_tabular():
    predictor = FraudDetectionPredictor()
    
    normal_transaction = {
        'step': 1,
        'type': 'PAYMENT',
        'amount': 50.00,
        'oldbalanceOrg': 50000.0,
        'newbalanceOrig': 49950.0,
        # ... more fields
    }
    result = predictor.predict_single_transaction(normal_transaction)
```
**Purpose**: Demonstrates tabular-only fraud detection using transaction features alone (backward compatible mode).

**Key Examples**:
1. **Normal Transaction**: Small PAYMENT transaction with balanced account changes
2. **Suspicious Transaction**: Large TRANSFER with significant balance changes

**Key Code Snippet - Single Transaction Prediction**:
```python
result = predictor.predict_single_transaction(normal_transaction)
print(f"Is Fraud: {result['is_fraud']}")
print(f"Fraud Probability: {result['fraud_probability']:.4%}")
```
**What it does**: Passes a transaction dictionary to the predictor, which handles preprocessing and returns fraud prediction with probability score.

### `main()`
```python
parser = argparse.ArgumentParser(description='Quick start demo for fraud detection')
parser.add_argument(
    '--mode',
    type=str,
    default='multimodal',
    choices=['multimodal', 'tabular'],
    help='Demo mode: multimodal (tabular+image) or tabular (tabular only)'
)
```
**Purpose**: Provides command-line interface to choose between demonstration modes.

## Usage Examples

**Run multimodal demo** (default):
```bash
python quick_start.py
```

**Run tabular-only demo**:
```bash
python quick_start.py --mode tabular
```

## Transaction Features Explained
- **step**: Time step in the simulation (1-744 hours)
- **type**: Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN)
- **amount**: Transaction amount in local currency
- **oldbalanceOrg**: Sender's balance before transaction
- **newbalanceOrig**: Sender's balance after transaction
- **oldbalanceDest**: Recipient's balance before transaction
- **newbalanceDest**: Recipient's balance after transaction
- **isFlaggedFraud**: Whether transaction was flagged by business rules

## Output Format
The script displays:
- Transaction details (type, amount, balance changes)
- Fraud prediction (True/False)
- Fraud probability (0.0 to 1.0)
- For multimodal: QR code status (benign/malicious)
