#!/usr/bin/env python3
"""
Comprehensive validation script for CSV column compatibility.

This script verifies that the code correctly handles all columns in the CSV file
and produces a detailed report.
"""
import sys
import os
sys.path.insert(0, '/home/runner/work/test/test')
os.chdir('/home/runner/work/test/test')

import pandas as pd
import numpy as np
from src.utils.tabular_preprocessor import FraudDataPreprocessor


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def validate_csv_columns():
    """Comprehensive validation of CSV columns against code expectations"""
    
    print_section("CSV Column Compatibility Validation Report")
    print("Generated to verify code matches CSV file structure")
    
    csv_path = 'data/test_transactions.csv'
    
    # 1. CSV File Analysis
    print_section("1. CSV File Analysis")
    df = pd.read_csv(csv_path)
    print(f"File: {csv_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    
    # Show sample data
    print("\nSample data (first 3 rows):")
    print(df.head(3).to_string())
    
    # 2. Column Verification
    print_section("2. Column Verification")
    
    expected_columns = [
        'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
        'newbalanceOrig', 'nameDest', 'oldbalanceDest', 
        'newbalanceDest', 'isFraud', 'isFlaggedFraud'
    ]
    
    print("Expected columns (11):")
    for i, col in enumerate(expected_columns, 1):
        status = "✓" if col in df.columns else "✗"
        print(f"  {status} {i:2d}. {col}")
    
    missing = set(expected_columns) - set(df.columns)
    extra = set(df.columns) - set(expected_columns)
    
    if missing:
        print(f"\n⚠️  Missing columns: {missing}")
        return False
    if extra:
        print(f"\n⚠️  Extra columns: {extra}")
        return False
    
    print("\n✓ All columns present and match expected format")
    
    # 3. Preprocessing Test
    print_section("3. Preprocessing Test")
    
    preprocessor = FraudDataPreprocessor()
    
    print("Loading CSV with FraudDataPreprocessor...")
    loaded_df = preprocessor.load_from_csv(csv_path)
    print(f"✓ Successfully loaded {len(loaded_df)} rows")
    
    print("\nPreprocessing data...")
    X, y = preprocessor.preprocess_data(loaded_df, fit=True)
    
    print(f"✓ Preprocessing completed successfully")
    print(f"  - Input shape: {X.shape}")
    print(f"  - Labels shape: {y.shape}")
    print(f"  - Features after preprocessing: {len(preprocessor.feature_names)}")
    
    # 4. Feature Analysis
    print_section("4. Feature Analysis After Preprocessing")
    
    print("ID columns dropped (as expected):")
    print("  • nameOrig - Customer/Merchant origin identifier")
    print("  • nameDest - Customer/Merchant destination identifier")
    
    print("\nRemaining features (8):")
    for i, feature in enumerate(preprocessor.feature_names, 1):
        print(f"  {i}. {feature}")
    
    expected_features = [
        'step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud'
    ]
    
    if preprocessor.feature_names == expected_features:
        print("\n✓ Features match expected configuration")
    else:
        print("\n✗ Feature mismatch!")
        print(f"Expected: {expected_features}")
        print(f"Got: {preprocessor.feature_names}")
        return False
    
    # 5. Data Type Verification
    print_section("5. Data Type Verification")
    
    print("Column data types:")
    for col in df.columns:
        dtype = df[col].dtype
        sample_val = df[col].iloc[0]
        print(f"  • {col:20s}: {str(dtype):10s} (e.g., {sample_val})")
    
    # 6. Label Distribution
    print_section("6. Label Distribution")
    
    fraud_count = (y == 1).sum()
    legitimate_count = (y == 0).sum()
    fraud_ratio = fraud_count / len(y) * 100
    
    print(f"Total transactions: {len(y)}")
    print(f"Fraudulent: {fraud_count} ({fraud_ratio:.1f}%)")
    print(f"Legitimate: {legitimate_count} ({100-fraud_ratio:.1f}%)")
    
    # 7. Final Summary
    print_section("7. Validation Summary")
    
    print("✓ CSV file structure is correct")
    print("✓ All 11 expected columns are present")
    print("✓ Column names match code expectations exactly")
    print("✓ No typos or naming mismatches detected")
    print("✓ Preprocessing works correctly")
    print("✓ ID columns (nameOrig, nameDest) are properly dropped")
    print("✓ 8 features remain after preprocessing")
    print("✓ Labels (isFraud) are correctly extracted")
    
    print("\n" + "=" * 70)
    print("RESULT: ✓ All validations passed successfully!")
    print("The code correctly matches the CSV file columns.")
    print("=" * 70)
    
    return True


if __name__ == '__main__':
    try:
        success = validate_csv_columns()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
