# data_preprocessing.py - Data Preprocessing Documentation

## Overview
This module provides comprehensive data preprocessing utilities for the multimodal fraud detection system. It handles both tabular transaction data and QR code images, with support for synthetic data generation for testing.

## Key Classes

### `FraudDataPreprocessor`
**Purpose**: Preprocesses tabular transaction data from the Online Payments Fraud Detection dataset format.

#### Initialization
```python
class FraudDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
```
**What it does**: Initializes empty scaler and encoders that will be fitted during training.

#### Synthetic Data Generation
```python
def generate_synthetic_data(self, n_samples=10000, fraud_ratio=0.02):
    # Generate normal transactions
    normal_data = {
        'step': np.random.randint(1, 744, size=n_normal),
        'type': np.random.choice(transaction_types, size=n_normal, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
        'amount': np.random.exponential(scale=100, size=n_normal),
        'oldbalanceOrg': np.abs(np.random.normal(50000, 30000, size=n_normal)),
        'newbalanceOrig': np.abs(np.random.normal(49000, 30000, size=n_normal)),
        # ...
    }
    
    # Generate fraudulent transactions
    fraud_data = {
        'type': np.random.choice(['TRANSFER', 'CASH_OUT'], size=n_fraud),
        'amount': np.random.exponential(scale=50000, size=n_fraud),  # Higher amounts
        'oldbalanceOrg': np.abs(np.random.normal(100000, 50000, size=n_fraud)),
        'newbalanceOrig': np.abs(np.random.normal(10000, 10000, size=n_fraud)),  # Much lower
        # ...
    }
```
**What it does**: 
- Creates realistic synthetic transaction data
- **Normal transactions**: Smaller amounts, balanced account changes
- **Fraud transactions**: Larger amounts, suspicious balance changes, typically TRANSFER or CASH_OUT types
- Uses exponential distribution for amounts (realistic for financial data)
- Uses normal distribution for account balances

**Fraud Patterns Simulated**:
1. **Large transfers**: Higher average amounts
2. **Balance manipulation**: Large drops in sender balance, large increases in recipient
3. **Transaction types**: Frauds more common in TRANSFER and CASH_OUT
4. **Rarely flagged**: Only 10% of frauds are flagged (realistic)

#### Data Preprocessing
```python
def preprocess_data(self, data, fit=True):
    df = data.copy()
    
    # Separate features and labels
    if 'isFraud' in df.columns:
        y = df['isFraud'].values
        df = df.drop('isFraud', axis=1)
    
    # Encode categorical features
    for col in categorical_cols:
        if fit:
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        else:
            df[col] = self.label_encoders[col].transform(df[col].astype(str))
    
    # Drop ID columns
    df = df.drop(['nameOrig', 'nameDest'], axis=1)
    
    # Scale features
    if fit:
        X_scaled = self.scaler.fit_transform(df)
    else:
        X_scaled = self.scaler.transform(df)
    
    # Reshape for transformer
    X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
```
**What it does**:
1. **Separates labels**: Extracts isFraud column if present
2. **Encodes categoricals**: Converts 'type' to numbers (PAYMENT=0, TRANSFER=1, etc.)
3. **Drops IDs**: Removes nameOrig and nameDest (not predictive)
4. **Scales features**: Standardizes to zero mean and unit variance
5. **Reshapes**: Adds sequence dimension for transformer input

**Feature Scaling Example**:
```
Before scaling:
amount: [50.0, 20000.0, 150.0, ...]
oldbalanceOrg: [50000.0, 100000.0, 25000.0, ...]

After scaling:
amount: [-0.42, 2.15, -0.38, ...]  # Mean=0, Std=1
oldbalanceOrg: [-0.31, 1.52, -0.87, ...]
```

### `QRCodePreprocessor`
**Purpose**: Preprocesses QR code images for fraud detection.

#### QR Pattern Generation
```python
def _generate_qr_pattern(self, malicious=False):
    img = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background
    
    # Add position detection patterns (corner squares)
    self._add_position_pattern(img, 0, 0, block_size * 3)
    self._add_position_pattern(img, 0, w - block_size * 3, block_size * 3)
    self._add_position_pattern(img, h - block_size * 3, 0, block_size * 3)
    
    # Add random data modules
    for i in range(block_size * 3, h - block_size * 3, block_size):
        for j in range(block_size * 3, w - block_size * 3, block_size):
            if np.random.random() > 0.5:
                img[i:i+block_size, j:j+block_size] = 0  # Black block
    
    # For malicious QR codes, add noise/distortion
    if malicious:
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        img = np.clip(img.astype(np.int32) + noise - 25, 0, 255).astype(np.uint8)
        
        # Add color tint
        tint = np.random.choice([0, 1, 2])
        img[:, :, tint] = np.clip(img[:, :, tint] + 30, 0, 255)
```
**What it does**:
- **Benign QR codes**: Clean patterns with position markers and data modules
- **Malicious QR codes**: Adds noise and color distortion to simulate tampered QR codes

**Position Pattern Structure**:
```
Finder Pattern (QR corner markers):
┌─────────┐
│█████████│  ← Outer black square
│█       █│  ← Middle white square
│█  ███  █│  ← Inner black square
│█  ███  █│
│█       █│
│█████████│
└─────────┘
```

#### Image Loading (Placeholder)
```python
def load_images_from_directory(self, directory, target_size=None):
    if not os.path.exists(directory):
        print(f"Directory {directory} not found. Using synthetic data.")
        return self.generate_synthetic_qr_images()
    
    # Load benign images
    benign_dir = os.path.join(directory, 'benign')
    # Load malicious images
    malicious_dir = os.path.join(directory, 'malicious')
```
**What it does**: Provides interface for loading real QR code datasets. Falls back to synthetic data if directory doesn't exist.

### `MultimodalDataPreprocessor`
**Purpose**: Combines tabular and image preprocessing for multimodal fraud detection.

#### Multimodal Data Generation
```python
def generate_synthetic_multimodal_data(self, n_samples=5000, fraud_ratio=0.1):
    # Generate tabular data
    tabular_data = self.fraud_preprocessor.generate_synthetic_data(
        n_samples=n_samples, fraud_ratio=fraud_ratio
    )
    
    # Determine QR code maliciousness based on fraud status
    fraud_labels = tabular_data['isFraud'].values
    malicious_probs = np.where(fraud_labels == 1, 0.8, 0.1)
    is_malicious = np.random.random(n_samples) < malicious_probs
    
    # Generate images
    images = np.array([
        self.qr_preprocessor._generate_qr_pattern(malicious=mal)
        for mal in is_malicious
    ], dtype=np.float32) / 255.0
```
**What it does**:
- Generates paired tabular and image data
- **Correlation**: Fraud transactions have 80% chance of malicious QR code
- **Normal transactions**: Only 10% chance of malicious QR code
- Normalizes images to [0, 1] range

**Correlation Logic**:
```
If transaction is fraud (isFraud=1):
    → 80% chance QR code is malicious
    → 20% chance QR code is benign

If transaction is normal (isFraud=0):
    → 10% chance QR code is malicious (false positive)
    → 90% chance QR code is benign
```

#### Train/Test Split
```python
def prepare_train_test_data(self, test_size=0.2, random_state=42, n_samples=5000):
    data = self.generate_synthetic_multimodal_data(n_samples=n_samples)
    
    # Stratified split (maintains fraud ratio)
    indices = np.arange(len(data['labels']))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=data['labels']
    )
    
    # Split tabular data
    train_tabular = data['tabular'].iloc[train_indices].reset_index(drop=True)
    test_tabular = data['tabular'].iloc[test_indices].reset_index(drop=True)
    
    # Preprocess tabular (fit on train only)
    X_train_tabular, y_train = self.fraud_preprocessor.preprocess_data(train_tabular, fit=True)
    X_test_tabular, y_test = self.fraud_preprocessor.preprocess_data(test_tabular, fit=False)
    
    # Split images
    X_train_images = data['images'][train_indices]
    X_test_images = data['images'][test_indices]
```
**What it does**:
- Creates stratified split (preserves fraud ratio in train/test)
- Fits scaler on training data only (prevents data leakage)
- Returns dictionary with all train/test splits

## Persistence

### Saving Preprocessors
```python
def save_scaler(self, filepath):
    save_dict = {
        'scaler': self.scaler,
        'label_encoders': self.label_encoders,
        'feature_names': self.feature_names
    }
    with open(filepath, 'wb') as f:
        pickle.dump(save_dict, f)
```
**What it does**: Saves fitted scaler and encoders for use during inference.

### Loading Preprocessors
```python
def load_scaler(self, filepath):
    with open(filepath, 'rb') as f:
        save_dict = pickle.load(f)
    self.scaler = save_dict.get('scaler')
    self.label_encoders = save_dict.get('label_encoders')
    self.feature_names = save_dict.get('feature_names')
```
**What it does**: Loads saved preprocessors to ensure consistent transformation.

## Feature Engineering

### Transaction Features (After Preprocessing)
1. **step**: Time step (standardized)
2. **type**: Transaction type (encoded: 0-4)
3. **amount**: Transaction amount (standardized)
4. **oldbalanceOrg**: Sender balance before (standardized)
5. **newbalanceOrig**: Sender balance after (standardized)
6. **oldbalanceDest**: Recipient balance before (standardized)
7. **newbalanceDest**: Recipient balance after (standardized)
8. **isFlaggedFraud**: Business rule flag (0 or 1)

### Dropped Features
- **nameOrig**: Sender ID (not predictive, too many unique values)
- **nameDest**: Recipient ID (not predictive, too many unique values)

## Usage Examples

### Tabular Only
```python
preprocessor = FraudDataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_data()
preprocessor.save_scaler('scaler.pkl')
```

### Multimodal
```python
preprocessor = MultimodalDataPreprocessor(image_size=(128, 128))
data = preprocessor.prepare_train_test_data(n_samples=5000)

# Access train data
X_train_tabular = data['X_train_tabular']
X_train_images = data['X_train_images']
y_train = data['y_train']

preprocessor.save_preprocessors('preprocessor.pkl')
```

## Data Statistics
After preprocessing:
- **Shape**: (n_samples, 1, 8) for tabular
- **Shape**: (n_samples, 128, 128, 3) for images
- **Mean**: ~0 (standardized)
- **Std**: ~1 (standardized)
- **Fraud ratio**: Configurable (default 2-10%)
