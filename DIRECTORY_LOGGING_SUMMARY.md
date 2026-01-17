# Directory Logging Feature - Implementation Summary

## Problem Statement
The user requested to show which directories are used for CSV and images when running:
```bash
python3 src/train.py --mode multimodal
```

The system should display:
- CSV data directory (e.g., `/home/ec2-user/csvdata/PS_20174392719_1491204439457_log.csv`)
- Image directories with subdirectories:
  - Benign: `/home/ec2-user/qrimages/QR codes/benign/benign/benign_2.png`
  - Malicious: `/home/ec2-user/qrimages/QR codes/malicious/malicious/malicious_316254.png`

## Changes Made

### 1. Updated `src/train.py`
**Location**: `train_multimodal_model()` function (lines 241-253)

**Change**: Added directory path logging at the start of multimodal training, before GPU configuration.

**What it displays**:
```
Configured Data Paths:
----------------------------------------------------------------------
CSV Data Directory: /home/ec2-user/csvdata
  Example: /home/ec2-user/csvdata/PS_20174392719_1491204439457_log.csv

Image Data Directory: /home/ec2-user/qrimages/QR codes
  Benign images: /home/ec2-user/qrimages/QR codes/benign/benign/
    Example: /home/ec2-user/qrimages/QR codes/benign/benign/benign_2.png
  Malicious images: /home/ec2-user/qrimages/QR codes/malicious/malicious/
    Example: /home/ec2-user/qrimages/QR codes/malicious/malicious/malicious_316254.png
----------------------------------------------------------------------
```

### 2. Enhanced `src/utils/multimodal_preprocessor.py`
**Location**: `generate_synthetic_multimodal_data()` method (lines 61-66)

**Change**: Enhanced the existing logging to include example file paths.

**What it displays**:
```
======================================================================
MULTIMODAL DATA GENERATION
======================================================================
CSV Data Directory: /home/ec2-user/csvdata
  Example CSV file: /home/ec2-user/csvdata/PS_20174392719_1491204439457_log.csv

Image Data Directory: /home/ec2-user/qrimages/QR codes
  Benign images: /home/ec2-user/qrimages/QR codes/benign/benign/
  Malicious images: /home/ec2-user/qrimages/QR codes/malicious/malicious/
======================================================================
```

### 3. Updated Documentation (`README.md`)
**Changes**:
- Added directory logging examples to the "Running Locally" section
- Added directory logging examples to the "Running on AWS EC2" section
- Included a note explaining that paths vary based on environment configuration

## How It Works

1. **At Training Start**: When the user runs `python3 src/train.py --mode multimodal`, the script immediately displays the configured paths from `config/config.py`.

2. **During Data Preparation**: When the multimodal preprocessor starts generating/loading data, it displays the same paths again with additional context.

3. **Configuration-Based**: The paths shown are automatically determined by the `DATA_BASE_PATH` setting in `config/config.py`:
   - For AWS EC2: `/home/ec2-user`
   - For local/Codespaces: Repository directory or custom path

## Example Output

When running `python3 src/train.py --mode multimodal`, the user will see:

```
======================================================================
Multimodal Fraud Detection Transformer Model Training
(Combining Tabular + Image Data)
======================================================================

Configured Data Paths:
----------------------------------------------------------------------
CSV Data Directory: /home/ec2-user/csvdata
  Example: /home/ec2-user/csvdata/PS_20174392719_1491204439457_log.csv

Image Data Directory: /home/ec2-user/qrimages/QR codes
  Benign images: /home/ec2-user/qrimages/QR codes/benign/benign/
    Example: /home/ec2-user/qrimages/QR codes/benign/benign/benign_2.png
  Malicious images: /home/ec2-user/qrimages/QR codes/malicious/malicious/
    Example: /home/ec2-user/qrimages/QR codes/malicious/malicious/malicious_316254.png
----------------------------------------------------------------------

[... GPU configuration output ...]

1. Preparing multimodal data...

======================================================================
MULTIMODAL DATA GENERATION
======================================================================
CSV Data Directory: /home/ec2-user/csvdata
  Example CSV file: /home/ec2-user/csvdata/PS_20174392719_1491204439457_log.csv

Image Data Directory: /home/ec2-user/qrimages/QR codes
  Benign images: /home/ec2-user/qrimages/QR codes/benign/benign/
  Malicious images: /home/ec2-user/qrimages/QR codes/malicious/malicious/
======================================================================

[... rest of training output ...]
```

## Files Modified

1. `src/train.py` - Added initial directory logging
2. `src/utils/multimodal_preprocessor.py` - Enhanced existing logging
3. `README.md` - Updated documentation with examples

## Testing

A validation script was created to verify the functionality:
- ✓ Configuration paths are correctly loaded
- ✓ Directory logging displays correct paths
- ✓ Example file paths match the problem statement format
- ✓ Output is clear and informative

## Benefits

1. **Transparency**: Users immediately see which directories the system is using
2. **Debugging**: Easy to verify if paths are configured correctly
3. **Documentation**: Example file paths help users understand the expected structure
4. **Environment Awareness**: Shows paths appropriate to the current environment (EC2 vs local)
