# Test Data

This directory contains test data files for the fraud detection system.

## test_transactions.csv

A small test dataset with 100 lines (1 header + 99 data rows) for local testing and development.

### File Structure

The CSV file contains the following columns:

1. **step**: Hour of simulation (1-9)
2. **type**: Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT)
3. **amount**: Transaction amount
4. **nameOrig**: Origin account identifier (e.g., C1231006815)
5. **oldbalanceOrg**: Origin account balance before transaction
6. **newbalanceOrig**: Origin account balance after transaction
7. **nameDest**: Destination account identifier (e.g., M1979787155 for merchants, C553264065 for customers)
8. **oldbalanceDest**: Destination account balance before transaction
9. **newbalanceDest**: Destination account balance after transaction
10. **isFraud**: Binary indicator (1 = fraud, 0 = legitimate)
11. **isFlaggedFraud**: Binary indicator for system-flagged transactions

### Statistics

- **Total transactions**: 99
- **Fraud cases**: 9 (~9.1%)
- **Transaction types**:
  - PAYMENT: 50
  - DEBIT: 17
  - TRANSFER: 16
  - CASH_OUT: 16

### Usage

This file can be used for:
- Quick testing of data preprocessing pipelines
- Validating model input/output formats
- Development and debugging without large datasets
- Unit tests for data loading and validation

### Example Usage

```python
import pandas as pd

# Load the test data
df = pd.read_csv('data/test_transactions.csv')

# Display basic information
print(df.head())
print(df.info())
```

### Notes

- This is synthetic test data based on the Online Payments Fraud Detection dataset format
- Includes both fraudulent (isFraud=1) and legitimate (isFraud=0) transactions
- Account identifiers starting with 'C' are customers, 'M' are merchants
- Fraudulent transactions typically involve TRANSFER or CASH_OUT types with larger amounts
