# multimodal_preprocessor.py - Multimodal Preprocessor Documentation

## Overview
This module combines tabular transaction preprocessing with QR code image preprocessing to create complete multimodal inputs for the fraud detection system. It handles data generation, preprocessing, and train/test splitting for both modalities simultaneously.

## `MultimodalDataPreprocessor` Class

### Purpose
Coordinates preprocessing of both tabular and image data, ensuring proper alignment and correlation between modalities.

### Initialization
```python
class MultimodalDataPreprocessor:
    def __init__(self, image_size=(128, 128)):
        self.fraud_preprocessor = FraudDataPreprocessor()
        self.qr_preprocessor = QRCodePreprocessor(image_size=image_size)
```
**What it does**: Creates instances of both specialized preprocessors with shared configuration.

### Multimodal Data Generation

#### Correlated Generation Strategy
```python
def generate_synthetic_multimodal_data(self, n_samples=5000, fraud_ratio=0.1):
    # Step 1: Generate tabular transaction data
    tabular_data = self.fraud_preprocessor.generate_synthetic_data(
        n_samples=n_samples, fraud_ratio=fraud_ratio
    )
    
    # Step 2: Extract fraud labels
    fraud_labels = tabular_data['isFraud'].values
    
    # Step 3: Determine QR maliciousness based on fraud status
    malicious_probs = np.where(fraud_labels == 1, 0.8, 0.1)
    is_malicious = np.random.random(n_samples) < malicious_probs
    
    # Step 4: Generate correlated QR images
    images = np.array([
        self.qr_preprocessor._generate_qr_pattern(malicious=mal)
        for mal in is_malicious
    ], dtype=np.float32) / 255.0
```

**Key Insight**: QR code maliciousness is **correlated** with transaction fraud.

#### Correlation Logic

**For fraudulent transactions** (isFraud=1):
```python
malicious_probs = 0.8  # 80% chance
```
- 80% of fraudulent transactions have malicious QR codes
- 20% of fraudulent transactions have benign QR codes (evasion)

**For normal transactions** (isFraud=0):
```python
malicious_probs = 0.1  # 10% chance
```
- 10% of normal transactions have malicious QR codes (false positives)
- 90% of normal transactions have benign QR codes

#### Why This Correlation?

**Real-world scenario**:
1. **Fraud + Malicious QR**: Attacker uses tampered QR for payment redirection
2. **Fraud + Benign QR**: Traditional fraud without QR manipulation
3. **Normal + Malicious QR**: Compromised QR code but transaction succeeds
4. **Normal + Benign QR**: Legitimate transaction with legitimate QR

**Probability distribution**:
```
                    Benign QR    Malicious QR
Fraud Transaction      20%           80%
Normal Transaction     90%           10%
```

### Vectorized Image Generation

```python
# Efficient batch processing
images = np.array([
    self.qr_preprocessor._generate_qr_pattern(malicious=mal)
    for mal in is_malicious
], dtype=np.float32) / 255.0
```
**What it does**: Generates all QR images in one operation for efficiency.

**Performance optimization**:
- List comprehension with NumPy array conversion
- Pre-allocated array (faster than appending)
- Batch normalization (/ 255.0 on entire array)

### Multimodal Preprocessing

```python
def preprocess_multimodal_data(self, tabular_data, images, fit=True):
    # Preprocess tabular data
    X_tabular, y = self.fraud_preprocessor.preprocess_data(tabular_data, fit=fit)
    
    # Images already normalized during generation
    return {
        'tabular': X_tabular,
        'images': images,
        'labels': y
    }
```
**What it does**:
- Tabular: Encodes, scales, reshapes
- Images: Already in [0, 1] range (no further processing)
- Returns dictionary for clear data organization

### Train/Test Split Strategy

#### Stratified Splitting
```python
def prepare_train_test_data(self, test_size=0.2, random_state=42, n_samples=5000):
    # Generate multimodal data
    data = self.generate_synthetic_multimodal_data(n_samples=n_samples)
    
    # Create indices for splitting
    indices = np.arange(len(data['labels']))
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=data['labels']  # Maintains fraud ratio
    )
```
**Stratification**: Ensures both train and test sets have same fraud ratio.

**Example**:
```
Original: 5000 samples, 10% fraud (500 fraud, 4500 normal)
Train:    4000 samples, 10% fraud (400 fraud, 3600 normal)
Test:     1000 samples, 10% fraud (100 fraud, 900 normal)
```

#### Synchronized Splitting
```python
# Split tabular data
train_tabular = data['tabular'].iloc[train_indices].reset_index(drop=True)
test_tabular = data['tabular'].iloc[test_indices].reset_index(drop=True)

# Split images using same indices
X_train_images = data['images'][train_indices]
X_test_images = data['images'][test_indices]
```
**Critical**: Both modalities split using **same indices** to maintain correspondence.

**Why important?**
```
Sample 42: Transaction X + QR Image Y
         ↓ Must stay together
Train or Test: Transaction X + QR Image Y
```
Breaking this alignment would corrupt the multimodal learning signal.

#### Fit/Transform Pattern
```python
# Preprocess tabular data
X_train_tabular, y_train = self.fraud_preprocessor.preprocess_data(
    train_tabular, fit=True   # Fit scaler on training data
)
X_test_tabular, y_test = self.fraud_preprocessor.preprocess_data(
    test_tabular, fit=False   # Use training scaler on test data
)
```
**What it does**: Prevents data leakage by fitting only on training data.

**Data leakage example** (WRONG):
```python
# BAD: Fitting on all data before split
preprocessor.preprocess_data(all_data, fit=True)
train, test = split(all_data)
# Test data influenced the scaler!
```

**Correct approach** (OUR CODE):
```python
# GOOD: Fit only on training data
train, test = split(all_data)
preprocessor.preprocess_data(train, fit=True)
preprocessor.preprocess_data(test, fit=False)
```

### Output Structure

```python
return {
    'X_train_tabular': X_train_tabular,  # (4000, 1, 8)
    'X_train_images': X_train_images,    # (4000, 128, 128, 3)
    'y_train': y_train,                  # (4000,)
    'X_test_tabular': X_test_tabular,    # (1000, 1, 8)
    'X_test_images': X_test_images,      # (1000, 128, 128, 3)
    'y_test': y_test                     # (1000,)
}
```

### Persistence

#### Saving
```python
def save_preprocessors(self, filepath):
    self.fraud_preprocessor.save_scaler(filepath)
```
**What it saves**: Only tabular preprocessor (scaler and encoders) since images need no learned preprocessing.

#### Loading
```python
def load_preprocessors(self, filepath):
    self.fraud_preprocessor.load_scaler(filepath)
```
**What it loads**: Restores tabular preprocessing state for consistent inference.

## Data Flow Diagram

```
generate_synthetic_multimodal_data()
        ↓
┌───────────────────────────┐
│ Generate Tabular Data     │ → 5000 transactions
│ (fraud_ratio=0.1)         │
└───────────────────────────┘
        ↓
Extract fraud labels: [0, 1, 0, 1, ...]
        ↓
Compute QR maliciousness probabilities:
  fraud=1 → prob=0.8
  fraud=0 → prob=0.1
        ↓
┌───────────────────────────┐
│ Generate QR Images        │ → 5000 QR codes
│ (correlated with fraud)   │
└───────────────────────────┘
        ↓
prepare_train_test_data()
        ↓
Create indices: [0, 1, 2, ..., 4999]
        ↓
Stratified split → train_idx, test_idx
        ↓
┌─────────────────┬─────────────────┐
│ Split Tabular   │ Split Images    │
│ (same indices)  │ (same indices)  │
└─────────────────┴─────────────────┘
        ↓
┌─────────────────┬─────────────────┐
│ Preprocess      │ Already         │
│ Tabular         │ Normalized      │
│ (fit on train)  │ [0, 1]          │
└─────────────────┴─────────────────┘
        ↓
Return dictionary with all splits
```

## Usage Examples

### Basic Usage
```python
preprocessor = MultimodalDataPreprocessor(image_size=(128, 128))

# Generate and prepare data
data = preprocessor.prepare_train_test_data(
    test_size=0.2,
    random_state=42,
    n_samples=5000
)

# Access training data
X_train_tab = data['X_train_tabular']   # (4000, 1, 8)
X_train_img = data['X_train_images']    # (4000, 128, 128, 3)
y_train = data['y_train']               # (4000,)

# Train model
model.fit([X_train_tab, X_train_img], y_train)
```

### Custom Data Generation
```python
preprocessor = MultimodalDataPreprocessor()

# Generate with custom parameters
data = preprocessor.generate_synthetic_multimodal_data(
    n_samples=10000,   # More samples
    fraud_ratio=0.15   # Higher fraud rate
)

print(f"Tabular shape: {data['tabular'].shape}")
print(f"Images shape: {data['images'].shape}")
print(f"Fraud ratio: {data['labels'].mean():.1%}")

# Count QR code types by fraud status
fraud_mask = data['labels'] == 1
fraud_mal = data['images'][fraud_mask]
normal_mal = data['images'][~fraud_mask]
```

### Inference Preparation
```python
# Training phase
preprocessor = MultimodalDataPreprocessor()
data = preprocessor.prepare_train_test_data()
model.fit([data['X_train_tabular'], data['X_train_images']], data['y_train'])
preprocessor.save_preprocessors('preprocessor.pkl')

# Inference phase
preprocessor_inf = MultimodalDataPreprocessor()
preprocessor_inf.load_preprocessors('preprocessor.pkl')

# New transaction + QR code
new_trans = pd.DataFrame([{...}])
new_qr = load_qr_image('new_qr.png')

processed = preprocessor_inf.preprocess_multimodal_data(
    new_trans, 
    new_qr, 
    fit=False
)
prediction = model.predict([processed['tabular'], processed['images']])
```

## Key Design Principles

### 1. Modality Alignment
Transaction and QR code for same sample must stay together throughout pipeline.

### 2. Correlation Modeling
QR maliciousness is probabilistically linked to transaction fraud, not random.

### 3. No Data Leakage
Preprocessing fitted only on training data, applied to test data.

### 4. Stratified Sampling
Train and test sets maintain same fraud distribution.

### 5. Reproducibility
Random seed (42) ensures consistent splits across runs.

## Statistical Properties

**After preprocessing**:
```
Tabular features:
  - Shape: (n, 1, 8)
  - Mean: ≈ 0 (standardized)
  - Std: ≈ 1 (standardized)

Image features:
  - Shape: (n, 128, 128, 3)
  - Range: [0, 1]
  - Mean: ≈ 0.5 (white background)

Labels:
  - Binary: {0, 1}
  - Fraud ratio: Configurable (default 10%)
  
Correlation:
  - P(Malicious QR | Fraud) = 0.8
  - P(Malicious QR | Normal) = 0.1
```
