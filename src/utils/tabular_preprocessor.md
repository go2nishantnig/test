# tabular_preprocessor.py - Tabular Data Preprocessor Documentation

## Overview
This module provides specialized preprocessing for tabular transaction data from the Online Payments Fraud Detection dataset. It handles feature encoding, scaling, and synthetic data generation for fraud detection.

## `FraudDataPreprocessor` Class

### Purpose
Transforms raw transaction data into model-ready features through encoding, scaling, and reshaping.

### Initialization
```python
class FraudDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
```
**Components**:
- **scaler**: StandardScaler for normalizing numeric features
- **label_encoders**: Dictionary mapping column names to LabelEncoders
- **feature_names**: List of feature names after preprocessing

### Transaction Features

#### Input Features
Based on the Online Payments Fraud Detection dataset:
```python
{
    'step': 1,                    # Hour in simulation (1-744)
    'type': 'PAYMENT',            # Transaction type
    'amount': 50.00,              # Transaction amount
    'nameOrig': 'C1234567890',    # Sender ID
    'oldbalanceOrg': 50000.0,     # Sender balance before
    'newbalanceOrig': 49950.0,    # Sender balance after
    'nameDest': 'M1234567890',    # Recipient ID
    'oldbalanceDest': 10000.0,    # Recipient balance before
    'newbalanceDest': 10050.0,    # Recipient balance after
    'isFlaggedFraud': 0,          # Business rule flag
    'isFraud': 0                  # Ground truth label
}
```

#### Transaction Types
```python
transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
```
**Fraud patterns**:
- **PAYMENT**: Low fraud rate (4%)
- **TRANSFER**: Higher fraud rate (used in fraud schemes)
- **CASH_OUT**: Higher fraud rate (cash extraction)
- **DEBIT**: Low fraud rate
- **CASH_IN**: Lowest fraud rate (adding money)

### Synthetic Data Generation

#### Normal Transactions
```python
normal_data = {
    'step': np.random.randint(1, 744, size=n_normal),
    'type': np.random.choice(transaction_types, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
    'amount': np.random.exponential(scale=100, size=n_normal),
    'oldbalanceOrg': np.abs(np.random.normal(50000, 30000, size=n_normal)),
    'newbalanceOrig': np.abs(np.random.normal(49000, 30000, size=n_normal)),
}
```
**Characteristics**:
- **Amounts**: Exponential distribution (scale=100) → mostly small transactions
- **Types**: Weighted towards PAYMENT (40%)
- **Balances**: Normal distribution around $50,000
- **Balance changes**: Consistent with transaction amounts

**Why exponential distribution?**
Real-world transaction amounts follow exponential/log-normal distribution:
- Many small transactions ($10-$100)
- Fewer medium transactions ($1000-$10,000)
- Very few large transactions ($100,000+)

#### Fraudulent Transactions
```python
fraud_data = {
    'type': np.random.choice(['TRANSFER', 'CASH_OUT'], p=[0.5, 0.5]),
    'amount': np.random.exponential(scale=50000, size=n_fraud),  # Larger scale
    'oldbalanceOrg': np.abs(np.random.normal(100000, 50000, size=n_fraud)),
    'newbalanceOrig': np.abs(np.random.normal(10000, 10000, size=n_fraud)),  # Much lower
    'oldbalanceDest': np.abs(np.random.normal(50000, 30000, size=n_fraud)),
    'newbalanceDest': np.abs(np.random.normal(150000, 50000, size=n_fraud)),  # Much higher
}
```
**Characteristics**:
- **Types**: Only TRANSFER or CASH_OUT (fraud patterns)
- **Amounts**: Larger scale (50000 vs 100) → bigger transactions
- **Sender balance**: Drops significantly after transaction
- **Recipient balance**: Increases significantly
- **Flags**: Only 10% flagged (most frauds evade detection)

**Fraud Indicators**:
1. **Large amount**: $50,000+ (vs typical $100)
2. **Suspicious types**: TRANSFER/CASH_OUT
3. **Balance anomalies**: Sender depleted, recipient enriched
4. **Rarely flagged**: Sophisticated frauds bypass business rules

### Preprocessing Pipeline

#### Step 1: Separate Labels
```python
if 'isFraud' in df.columns:
    y = df['isFraud'].values
    df = df.drop('isFraud', axis=1)
```
**What it does**: Extracts target variable for supervised learning.

#### Step 2: Encode Categoricals
```python
categorical_cols = ['type']
for col in categorical_cols:
    if fit:
        self.label_encoders[col] = LabelEncoder()
        df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
    else:
        df[col] = self.label_encoders[col].transform(df[col].astype(str))
```
**What it does**: Converts transaction types to integers:
```
'PAYMENT'  → 0
'TRANSFER' → 1
'CASH_OUT' → 2
'DEBIT'    → 3
'CASH_IN'  → 4
```

**Why fit/transform split?**
- **fit=True** (training): Learn encoding from training data
- **fit=False** (test/inference): Apply learned encoding
- Prevents data leakage from test set

#### Step 3: Drop ID Columns
```python
id_cols = ['nameOrig', 'nameDest']
for col in id_cols:
    if col in df.columns:
        df = df.drop(col, axis=1)
```
**Why drop IDs?**
- Millions of unique values (high cardinality)
- No predictive power (random identifiers)
- Would cause overfitting if encoded
- Not generalizable to new users

#### Step 4: Feature Scaling
```python
if fit:
    X_scaled = self.scaler.fit_transform(df)
else:
    X_scaled = self.scaler.transform(df)
```
**What it does**: Standardizes features to zero mean and unit variance.

**Before scaling**:
```
Feature          Min        Max        Mean       Std
-------------------------------------------------------
step             1          744        372        215
amount           0.01       500000     5000       25000
oldbalanceOrg    0          10000000   500000     800000
```

**After scaling**:
```
Feature          Min        Max        Mean       Std
-------------------------------------------------------
step             -1.73      1.73       0.0        1.0
amount           -0.20      19.80      0.0        1.0
oldbalanceOrg    -0.63      11.88      0.0        1.0
```

**Benefits**:
- Equal feature importance (no domination by large-scale features)
- Faster gradient descent convergence
- Better numerical stability

#### Step 5: Reshape for Transformer
```python
X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
```
**What it does**: Adds sequence dimension.

**Shape transformation**:
```
Before: (n_samples, 8)
After:  (n_samples, 1, 8)
```
Where:
- **n_samples**: Number of transactions
- **1**: Sequence length (single time step)
- **8**: Number of features

**Why sequence dimension?**
Transformers expect sequence input (batch, seq_len, features). Even though we have single transactions, we maintain this format for consistency.

### Persistence

#### Saving
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
**What it saves**:
- **scaler**: Fitted StandardScaler (mean/std for each feature)
- **label_encoders**: Encoding mappings (type → int)
- **feature_names**: Column order for consistency

#### Loading
```python
def load_scaler(self, filepath):
    with open(filepath, 'rb') as f:
        save_dict = pickle.load(f)
    self.scaler = save_dict.get('scaler')
    self.label_encoders = save_dict.get('label_encoders')
    self.feature_names = save_dict.get('feature_names')
```
**Why needed?**
At inference time, new data must be transformed identically to training data:
- Same encoding (PAYMENT must always → 0)
- Same scaling (using training mean/std)
- Same feature order

### Train/Test Split

```python
def prepare_train_test_data(self, test_size=0.2, random_state=42):
    data = self.generate_synthetic_data()
    
    # Stratified split
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state, 
        stratify=data['isFraud']
    )
    
    # Preprocess (fit on train, transform test)
    X_train, y_train = self.preprocess_data(train_data, fit=True)
    X_test, y_test = self.preprocess_data(test_data, fit=False)
```
**Stratified split**: Maintains fraud ratio in both sets.

**Example**:
```
Original data: 10% fraud
Train set (80%): 10% fraud
Test set (20%): 10% fraud
```

## Usage Examples

### Training
```python
preprocessor = FraudDataPreprocessor()

# Generate and prepare data
X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_data(
    test_size=0.2, 
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Feature shape: {X_train.shape}")  # (8000, 1, 8)
print(f"Fraud ratio: {y_train.mean():.2%}")  # 2%

# Save for inference
preprocessor.save_scaler('models/saved_models/scaler.pkl')
```

### Inference
```python
preprocessor = FraudDataPreprocessor()
preprocessor.load_scaler('models/saved_models/scaler.pkl')

# New transaction
new_transaction = pd.DataFrame([{
    'step': 100,
    'type': 'TRANSFER',
    'amount': 50000.0,
    'nameOrig': 'C9876543210',
    'oldbalanceOrg': 100000.0,
    'newbalanceOrig': 50000.0,
    'nameDest': 'C5555555555',
    'oldbalanceDest': 10000.0,
    'newbalanceDest': 60000.0,
    'isFlaggedFraud': 0
}])

# Preprocess (fit=False uses saved scaler)
X_new, _ = preprocessor.preprocess_data(new_transaction, fit=False)
```

## Final Feature Set
After preprocessing, 8 features remain:
1. **step**: Time step (normalized)
2. **type**: Transaction type (0-4)
3. **amount**: Amount (normalized)
4. **oldbalanceOrg**: Sender balance before (normalized)
5. **newbalanceOrig**: Sender balance after (normalized)
6. **oldbalanceDest**: Recipient balance before (normalized)
7. **newbalanceDest**: Recipient balance after (normalized)
8. **isFlaggedFraud**: Business rule flag (0/1)
