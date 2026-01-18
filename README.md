# Multimodal Fraud Detection Transformer

A multimodal machine learning system for fraud detection that combines tabular transaction data with QR code image analysis using transformer-based neural networks.

## Overview

This project implements a state-of-the-art fraud detection system using:
- **Tabular Transformer**: Processes online payment transaction features
- **Vision Transformer**: Analyzes QR code images for malicious patterns
- **Cross-Modal Fusion**: Combines insights from both modalities for improved detection

The system can run in two modes:
1. **Multimodal Mode**: Uses both tabular and image data (default)
2. **Tabular-Only Mode**: Uses only transaction data (backward compatible)

## Features

- 🔄 **Dual-mode training**: Multimodal or tabular-only
- 🚀 **GPU acceleration**: Optimized for NVIDIA GPUs (tested on A10G)
- 📊 **TensorFlow 2.x**: Built with modern TensorFlow/Keras
- 🔍 **Transformer architecture**: Attention-based models for both modalities
- 📈 **TensorBoard integration**: Real-time training visualization
- 📊 **Automatic plotting**: Generates training and inference visualization plots
- ☁️ **AWS EC2 ready**: Pre-configured for cloud deployment

---

## Table of Contents

- [Quick Configuration: Switching Between Environments](#️-quick-configuration-switching-between-environments)
- [Prerequisites](#prerequisites)
- [Running Locally (Without Jupyter Notebooks)](#running-locally-without-jupyter-notebooks)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Set Up Python Environment](#2-set-up-python-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Run Training](#4-run-training)
  - [5. View Training Results](#5-view-training-results)
- [Running on AWS EC2 Instance](#running-on-aws-ec2-instance)
  - [1. Launch EC2 Instance](#1-launch-ec2-instance)
  - [2. Connect to EC2](#2-connect-to-ec2)
  - [3. Clone Repository](#3-clone-repository)
  - [4. Run Setup Script](#4-run-setup-script)
  - [5. Run Training on EC2](#5-run-training-on-ec2)
  - [6. (Optional) Use Jupyter Notebook](#6-optional-use-jupyter-notebook)
- [Making Predictions with predict.py](#making-predictions-with-predictpy)
  - [Prerequisites](#prerequisites-1)
  - [Where to Use predict.py](#where-to-use-predictpy)
  - [When to Use predict.py](#when-to-use-predictpy)
  - [How to Use predict.py](#how-to-use-predictpy)
  - [Using predict.py in GitHub Codespaces](#using-predictpy-in-github-codespaces)
  - [Using predict.py on AWS EC2](#using-predictpy-on-aws-ec2)
  - [Troubleshooting predict.py](#troubleshooting-predictpy)
  - [Performance Considerations](#performance-considerations)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## ⚙️ Quick Configuration: Switching Between Environments

**IMPORTANT**: To switch between AWS EC2 and GitHub Codespaces, you only need to change **ONE LINE** in the configuration file.

### Where to Change the Path

Edit the file: **`config/config.py`**

Look for these lines near the top of the file (around line 21-25):

```python
# ============================================================================
# CONFIGURABLE BASE PATH - CHANGE THIS TO SWITCH ENVIRONMENTS
# ============================================================================
# For AWS EC2, use: DATA_BASE_PATH = '/home/ec2-user'
# For GitHub Codespaces, use: DATA_BASE_PATH = '/workspaces/test/data'
DATA_BASE_PATH = '/home/ec2-user'
# ============================================================================
```

### To Use in GitHub Codespaces:
Change the `DATA_BASE_PATH` line to:
```python
DATA_BASE_PATH = '/workspaces/test/data'
```

### To Use in AWS EC2:
Change the `DATA_BASE_PATH` line to:
```python
DATA_BASE_PATH = '/home/ec2-user'
```

That's it! The entire application will automatically use the correct paths for data directories, models, and logs based on this single setting.

---

## Prerequisites

### Local Development
- **Python**: 3.8 or higher
- **pip**: Latest version
- **Git**: For cloning the repository
- **Optional**: NVIDIA GPU with CUDA support for accelerated training

### AWS EC2 Deployment
- **AWS Account** with EC2 access
- **EC2 Instance**: Recommended G5.xlarge (NVIDIA A10G GPU)
- **AMI**: Amazon Linux 2 or Ubuntu 20.04 LTS
- **Security Group**: Port 8888 open (if using Jupyter)
- **Storage**: At least 20GB EBS volume

---

## Running Locally (Without Jupyter Notebooks)

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/go2nishantnig/test.git
cd test
```

> **Note**: If you forked this repository, replace the URL with your fork's URL: `git clone https://github.com/YOUR-USERNAME/test.git`

### 2. Set Up Python Environment

It's recommended to use a virtual environment to isolate dependencies:

**Using venv (built-in):**
```bash
# Create virtual environment
python3 -m venv fraud_env

# Activate virtual environment
# On Linux/macOS:
source fraud_env/bin/activate
# On Windows:
fraud_env\Scripts\activate
```

**Using conda (alternative):**
```bash
conda create -n fraud_env python=3.9
conda activate fraud_env
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

This will install:
- TensorFlow (2.13.0+) with GPU support
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (for visualization)
- Pillow (for image processing)
- And other dependencies

### 4. Run Training

The repository provides Python scripts that can be run directly without Jupyter notebooks:

#### Option A: Run Multimodal Training (Default)

Trains the model using both tabular transaction data and QR code images:

```bash
python src/train.py --mode multimodal
```

#### Option B: Run Tabular-Only Training

Trains the model using only tabular transaction data:

```bash
python src/train.py --mode tabular
```

**What happens during training:**
1. **Loads ALL actual data from files**:
   - All CSV files from `data/csvdata/` directory (or EC2 equivalent)
   - All QR code images from `data/qrimages/QR codes/` directory
   - If counts don't match, images are replicated with augmentation to match CSV count
2. Preprocesses tabular and/or image data
3. Builds the transformer model architecture
4. Trains the model with early stopping and checkpointing
5. Evaluates on test set
6. Saves trained models to `models/saved_models/`
7. Saves preprocessors for inference

**Training output:**
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

======================================================================
USING ACTUAL DATA FROM FILES
======================================================================

1. Preparing multimodal data...
   Loading all CSV files from directory...
   Found 1 CSV file(s)
   ✓ Total CSV records loaded: 99
   
   Loading all images from directory...
   ✓ Loaded 8 images (4 benign, 4 malicious)
   
   Handling data size mismatch...
   Strategy: Replicate images with augmentation to match CSV count
   ✓ Images replicated to 99 samples
   
   Training samples: 79
   Testing samples: 20
   Fraud ratio in training: 9.09%
   Fraud ratio in testing: 9.09%
   Tabular shape: (79, 1, 8)
   Image shape: (79, 128, 128, 3)

2. Building multimodal transformer model...
...
```

**Note**: The paths shown will vary based on your environment configuration in `config/config.py`. The example above shows AWS EC2 paths. For local development, the paths will point to your local data directory.

### 5. View Training Results

#### Monitor Training with TensorBoard

```bash
# Start TensorBoard (runs in background)
tensorboard --logdir logs/ --port 6006

# Open in browser: http://localhost:6006
```

#### View Generated Plots

Training automatically generates visualization plots saved to organized directories:

```bash
# View training plots
ls -lh /home/ec2-user/graphs/training/
# Or locally: ls -lh graphs/training/

# View inference plots (after running predictions)
ls -lh /home/ec2-user/graphs/inference/
# Or locally: ls -lh graphs/inference/
```

Generated plots include:
- Training loss and accuracy curves
- Confusion matrix
- ROC curve with AUC score
- Prediction probability distributions

For detailed information, see [PLOTTING_GUIDE.md](PLOTTING_GUIDE.md).

#### Check Saved Models

```bash
ls -lh models/saved_models/
```

You should see:
- `*_best.keras` - Best model based on validation loss
- `*_final.keras` - Final model after all epochs
- `*_preprocessor.pkl` - Saved preprocessors for inference

---

## Running on AWS EC2 Instance

### 1. Launch EC2 Instance

**Recommended Configuration:**
- **Instance Type**: `g5.xlarge` (NVIDIA A10G GPU, 24GB VRAM)
  - For CPU-only: `t3.xlarge` or `m5.xlarge`
- **AMI**: Amazon Linux 2 with Deep Learning Base AMI
  - Or Ubuntu 20.04 LTS with manual NVIDIA driver setup
- **Storage**: 30GB gp3 EBS volume
- **Security Group Rules**:
  - SSH (port 22): Your IP address
  - Custom TCP (port 8888): Your IP address (for Jupyter)
  - HTTPS (port 443): Optional, for production

**Launch via AWS Console:**
1. Go to EC2 Dashboard → Launch Instance
2. Choose "Deep Learning Base AMI (Amazon Linux 2)" or "Ubuntu Server 20.04 LTS"
3. Select instance type: `g5.xlarge`
4. Configure security group with rules above
5. Create or select an existing key pair
6. Launch instance

**Launch via AWS CLI:**
```bash
# Note: Replace the placeholder values below with your actual AWS resource IDs
# - ami-xxxxxxxxx: Your chosen AMI ID (find in AWS Console or via 'aws ec2 describe-images')
# - your-key-pair: Your EC2 key pair name
# - sg-xxxxxxxxx: Your security group ID
# - subnet-xxxxxxxxx: Your VPC subnet ID

aws ec2 run-instances \
    --image-id ami-xxxxxxxxx \
    --instance-type g5.xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxxx \
    --subnet-id subnet-xxxxxxxxx \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=30,VolumeType=gp3}'
```

### 2. Connect to EC2

```bash
# SSH into your EC2 instance
ssh -i /path/to/your-key.pem ec2-user@<EC2-PUBLIC-IP>

# For Ubuntu instances, use 'ubuntu' instead of 'ec2-user'
```

### 3. Clone Repository

```bash
cd /home/ec2-user
git clone https://github.com/go2nishantnig/test.git
cd test
```

### 4. Run Setup Script

The repository includes an automated setup script for EC2:

```bash
# Make script executable (if not already)
chmod +x setup_ec2_jupyter.sh

# Run the setup script
./setup_ec2_jupyter.sh
```

**What the setup script does:**
1. Verifies Python and Git installation
2. Creates required directories:
   - `/home/ec2-user/qrdata` - QR code images
   - `/home/ec2-user/csv-data` - Transaction CSV data
   - `/home/ec2-user/model` - Trained models and logs
3. Creates Python virtual environment at `/home/ec2-user/fraud_detection_env`
4. Installs all dependencies from `requirements.txt`
5. Configures Jupyter Notebook for remote access
6. Checks GPU availability
7. Creates a test notebook for verification

**Expected output:**
```
======================================
AWS EC2 G5 XLarge Setup for Multimodal Fraud Detection
======================================

→ Checking environment...
✓ Python 3.9.x found
✓ Git 2.x.x found
✓ Directories created
✓ Virtual environment created
✓ Dependencies installed
✓ Jupyter Notebook configured
✓ NVIDIA GPU detected

======================================
Setup Complete!
======================================
```

### 5. Run Training on EC2

#### Activate the Virtual Environment

```bash
source /home/ec2-user/fraud_detection_env/bin/activate
```

#### Method 1: Quick Start Script (Recommended for EC2)

The repository provides an EC2-optimized quick start script:

```bash
# Navigate to repository
cd /home/ec2-user/test

# Run multimodal training (default)
python ec2_quick_start.py --mode multimodal

# Or run tabular-only training
python ec2_quick_start.py --mode tabular
```

This script automatically:
- Detects EC2 environment
- Uses EC2-specific directory paths
- Configures GPU settings (if available)
- Enables mixed precision training (FP16) for faster training
- Saves models to `/home/ec2-user/model/`

#### Method 2: Standard Training Script

```bash
cd /home/ec2-user/test

# Set environment variable to use EC2 configuration
export USE_EC2_CONFIG=1

# Run training
python src/train.py --mode multimodal
```

#### Training on EC2

**Expected output:**
```
======================================================================
AWS EC2 G5 XLarge - Fraud Detection Quick Start
======================================================================

Mode: MULTIMODAL
TensorFlow version: 2.13.0

======================================================================
GPU Configuration
======================================================================

✓ Found 1 GPU(s):
  GPU 0: /physical_device:GPU:0
✓ GPU memory growth enabled
✓ Mixed precision (FP16) enabled

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
...
```

#### Monitor Training Progress

You can monitor GPU usage in real-time:

```bash
# In a separate SSH session
watch -n 1 nvidia-smi
```

#### Run Training in Background (Long Training Sessions)

```bash
# Run training in background with nohup
nohup python ec2_quick_start.py --mode multimodal > training.log 2>&1 &

# Check progress
tail -f training.log

# Check if still running
ps aux | grep python
```

### 6. (Optional) Use Jupyter Notebook

If you prefer using Jupyter notebooks on EC2:

#### Set a Password (Recommended for Security)

```bash
jupyter notebook password
# Enter your password when prompted
```

#### Start Jupyter Notebook

```bash
# Activate virtual environment
source /home/ec2-user/fraud_detection_env/bin/activate

# Start Jupyter Notebook
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

#### Access Jupyter from Your Browser

```
http://<EC2-PUBLIC-IP>:8888
```

**Important Security Notes:**
- Ensure port 8888 is restricted to your IP in the security group
- Always set a strong password with `jupyter notebook password`
- For better security, use SSH tunneling instead:

```bash
# On your local machine, create SSH tunnel
ssh -i /path/to/key.pem -L 8888:localhost:8888 ec2-user@<EC2-PUBLIC-IP>

# Then access Jupyter at: http://localhost:8888
```

#### Run the Pipeline Notebook

Once Jupyter is running, open:
```
notebooks/multimodal_fraud_detection_pipeline.ipynb
```

This notebook contains the complete end-to-end pipeline with visualization and detailed explanations.

---

## Making Predictions with predict.py

After training your model, you can use the `predict.py` script to make fraud predictions on new transactions. This script works in both **multimodal** and **tabular-only** modes, matching your training configuration.

### Prerequisites

Before running predictions, you must have:
1. ✅ **Trained Model**: Run training first using `python src/train.py --mode [multimodal|tabular]`
2. ✅ **Saved Model Files**: The script looks for model files in the saved models directory:
   - **Local**: `models/saved_models/fraud_detection_v1_final.keras`
   - **EC2**: `/home/ec2-user/model/fraud_detection_v1_final.keras`
3. ✅ **Preprocessor Files**: Saved preprocessors for data normalization

### Where to Use predict.py

The script automatically adapts to your environment:

| Environment | Model Location | Configuration |
|------------|----------------|---------------|
| **GitHub Codespaces** | `models/saved_models/` | Set `DATA_BASE_PATH = '/workspaces/test/data'` in `config/config.py` |
| **AWS EC2** | `/home/ec2-user/model/` | Set `DATA_BASE_PATH = '/home/ec2-user'` in `config/config.py` |
| **Local Development** | `models/saved_models/` | Use repository-relative paths |

### When to Use predict.py

Use `predict.py` for:
- 🔍 **Testing trained models** with sample data
- 📊 **Batch predictions** on multiple transactions
- 🧪 **Model validation** before deployment
- 💡 **Demonstrating model capabilities** with synthetic data
- 🚀 **Development and debugging** of inference pipelines

### How to Use predict.py

#### Option 1: Demo Mode (Quick Test)

The easiest way to test your trained model is to run the demo mode, which generates synthetic test data:

**Multimodal Mode** (Tabular + Image):
```bash
# In GitHub Codespaces or Local
python src/predict.py --mode multimodal

# On AWS EC2
cd /home/ec2-user/test
source /home/ec2-user/fraud_detection_env/bin/activate
python src/predict.py --mode multimodal
```

**Tabular-Only Mode**:
```bash
# In GitHub Codespaces or Local
python src/predict.py --mode tabular

# On AWS EC2
cd /home/ec2-user/test
source /home/ec2-user/fraud_detection_env/bin/activate
python src/predict.py --mode tabular
```

**Demo Output Example:**
```
======================================================================
Multimodal Fraud Detection Transformer - Inference Demo
======================================================================

Loading model from: models/saved_models/fraud_detection_v1_final.keras
Loading preprocessor from: models/saved_models/fraud_detection_v1_preprocessor.pkl
Model and preprocessor loaded successfully!

1. Generating sample multimodal test data...

2. Making predictions...

======================================================================
Prediction Results:
======================================================================

Sample predictions:
   step    type    amount  isFraud  predicted_fraud  fraud_probability
0     1    CASH    5234.5        0                0             0.0234
1     2 PAYMENT   15678.2        1                1             0.9876
2     3 TRANSFER   8923.4        0                0             0.1234
...

======================================================================
Performance Metrics:
======================================================================
Accuracy:  0.9500
Precision: 0.8750
Recall:    0.9333
F1 Score:  0.9032
```

#### Option 2: Python API (Programmatic Use)

Use the predictor classes in your own Python scripts:

**Multimodal Predictions:**
```python
from src.predict import MultimodalFraudDetectionPredictor
import numpy as np
import pandas as pd

# Initialize predictor
predictor = MultimodalFraudDetectionPredictor()

# Prepare your data
transaction_data = pd.DataFrame({
    'step': [1],
    'type': ['TRANSFER'],
    'amount': [181.0],
    'oldbalanceOrg': [181.0],
    'newbalanceOrig': [0.0],
    'oldbalanceDest': [0.0],
    'newbalanceDest': [181.0],
    'isFlaggedFraud': [0]
})

# Load or create QR code image (normalized to [0, 1])
qr_image = np.random.rand(128, 128, 3)  # Example: random image

# Make prediction for single transaction
result = predictor.predict_single_transaction(
    tabular_features=transaction_data,
    image=qr_image,
    threshold=0.5
)

print(f"Is Fraud: {result['is_fraud']}")
print(f"Fraud Probability: {result['fraud_probability']:.4f}")
print(f"Prediction: {result['prediction']}")
```

**Tabular-Only Predictions:**
```python
from src.predict import FraudDetectionPredictor
import pandas as pd

# Initialize predictor
predictor = FraudDetectionPredictor()

# Single transaction prediction
transaction = {
    'step': 1,
    'type': 'TRANSFER',
    'amount': 181.0,
    'oldbalanceOrg': 181.0,
    'newbalanceOrig': 0.0,
    'oldbalanceDest': 0.0,
    'newbalanceDest': 181.0,
    'isFlaggedFraud': 0
}

result = predictor.predict_single_transaction(transaction)
print(f"Fraud Probability: {result['fraud_probability']:.4f}")
```

**Batch Predictions:**
```python
from src.predict import MultimodalFraudDetectionPredictor
import pandas as pd
import numpy as np

predictor = MultimodalFraudDetectionPredictor()

# Load your CSV data
transactions_df = pd.read_csv('your_transactions.csv')

# Load corresponding QR images
qr_images = np.load('your_qr_images.npy')  # Shape: (n_samples, 128, 128, 3)

# Make batch predictions
results = predictor.predict(
    tabular_data=transactions_df,
    images=qr_images,
    threshold=0.5
)

# Access results
print(f"Predictions: {results['predictions']}")
print(f"Probabilities: {results['probabilities']}")
print(f"Fraud flags: {results['is_fraud']}")
```

#### Option 3: Integration into Production Pipeline

For production deployments, integrate the predictor classes into your application:

```python
# Example: Flask API endpoint
from flask import Flask, request, jsonify
from src.predict import MultimodalFraudDetectionPredictor
import numpy as np
import pandas as pd

app = Flask(__name__)
predictor = MultimodalFraudDetectionPredictor()

@app.route('/predict', methods=['POST'])
def predict_fraud():
    data = request.json
    
    # Extract transaction features
    transaction = pd.DataFrame([data['transaction']])
    
    # Decode QR image (assuming base64 encoded)
    import base64
    from PIL import Image
    import io
    
    qr_bytes = base64.b64decode(data['qr_image'])
    qr_image = Image.open(io.BytesIO(qr_bytes))
    qr_array = np.array(qr_image.resize((128, 128))) / 255.0
    
    # Make prediction
    result = predictor.predict_single_transaction(transaction, qr_array)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Using predict.py in GitHub Codespaces

**Step-by-Step:**

1. **Configure Environment**:
   ```bash
   # Edit config/config.py and set:
   DATA_BASE_PATH = '/workspaces/test/data'
   ```

2. **Train Model** (if not already trained):
   ```bash
   python src/train.py --mode multimodal
   ```

3. **Run Prediction Demo**:
   ```bash
   python src/predict.py --mode multimodal
   ```

4. **Use in Jupyter Notebook**:
   - The Codespaces environment supports Jupyter notebooks
   - Open `notebooks/multimodal_fraud_detection_pipeline.ipynb`
   - Import and use the predictor classes interactively

**Codespaces Advantages:**
- ✅ No setup required - everything is pre-configured
- ✅ Direct integration with VS Code
- ✅ Easy to share and collaborate
- ✅ Version control built-in

### Using predict.py on AWS EC2

**Step-by-Step:**

1. **SSH into EC2**:
   ```bash
   ssh -i /path/to/your-key.pem ec2-user@<EC2-PUBLIC-IP>
   ```

2. **Navigate to Repository**:
   ```bash
   cd /home/ec2-user/test
   ```

3. **Activate Virtual Environment**:
   ```bash
   source /home/ec2-user/fraud_detection_env/bin/activate
   ```

4. **Verify Configuration** (should already be set for EC2):
   ```bash
   # Check that config/config.py has:
   # DATA_BASE_PATH = '/home/ec2-user'
   cat config/config.py | grep DATA_BASE_PATH
   ```

5. **Train Model** (if not already trained):
   ```bash
   python ec2_quick_start.py --mode multimodal
   # OR
   python src/train.py --mode multimodal
   ```

6. **Run Prediction Demo**:
   ```bash
   python src/predict.py --mode multimodal
   ```

7. **Run Predictions in Background** (for long-running jobs):
   ```bash
   nohup python src/predict.py --mode multimodal > predictions.log 2>&1 &
   
   # Check progress
   tail -f predictions.log
   ```

**EC2 Advantages:**
- ✅ GPU acceleration (G5.xlarge with NVIDIA A10G)
- ✅ Faster inference for large batches
- ✅ Can handle production workloads
- ✅ Persistent storage for models and data

### Troubleshooting predict.py

#### Error: "No trained model found"

**Solution**: Train the model first before running predictions:
```bash
# For multimodal mode
python src/train.py --mode multimodal

# For tabular-only mode
python src/train.py --mode tabular
```

#### Error: "ModuleNotFoundError"

**Solution**: Ensure dependencies are installed and virtual environment is activated:
```bash
# Activate virtual environment
source fraud_env/bin/activate  # Local/Codespaces
# OR
source /home/ec2-user/fraud_detection_env/bin/activate  # EC2

# Install dependencies
pip install -r requirements.txt
```

#### Error: "Model and preprocessor paths don't match"

**Solution**: Ensure you're using the correct mode (multimodal vs tabular) that matches your trained model:
```bash
# Check what models are available
ls -l models/saved_models/  # Local/Codespaces
# OR
ls -l /home/ec2-user/model/  # EC2

# Use the correct mode
python src/predict.py --mode multimodal  # or --mode tabular
```

#### Error: "Cannot load model - custom objects not found"

**Solution**: The script automatically imports custom layers. If you still see this error:
```python
# The predict.py already handles this, but if needed manually:
from src.models.transformer_model import (
    MultiHeadSelfAttention,
    TransformerBlock,
    CrossModalAttention,
    CrossModalTransformerBlock,
    PatchEmbedding
)
```

#### Error: Path-related issues

**Solution**: Verify your `DATA_BASE_PATH` in `config/config.py` matches your environment:
- **Codespaces**: `/workspaces/test/data`
- **EC2**: `/home/ec2-user`
- **Local**: Use repository-relative paths (default)

### Performance Considerations

**Inference Speed:**
- **GPU (EC2 G5.xlarge)**: ~10-20ms per sample
- **CPU**: ~50-100ms per sample
- **Batch processing**: More efficient than individual predictions

**Memory Usage:**
- **Multimodal model**: ~500MB-1GB RAM
- **Tabular-only model**: ~200-500MB RAM
- **QR images**: ~50KB per image (128x128x3)

**Optimization Tips:**
1. Use batch predictions for multiple samples
2. Enable GPU if available for faster inference
3. Adjust `threshold` parameter to balance precision/recall
4. Cache the predictor instance for repeated predictions

---

## Project Structure

```
test/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup_ec2_jupyter.sh              # EC2 automated setup script
├── ec2_quick_start.py                # EC2-optimized training script
├── check_ec2_env.py                  # EC2 environment checker
├── config/
│   ├── __init__.py
│   └── config.py                     # Configuration (auto-detects EC2/local)
├── src/
│   ├── __init__.py
│   ├── train.py                      # Main training script
│   ├── predict.py                    # Inference/prediction script
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transformer_model.py      # Multimodal transformer models
│   │   ├── transformer.py            # Core transformer implementation
│   │   ├── attention.py              # Multi-head attention
│   │   ├── blocks.py                 # Transformer blocks
│   │   ├── layers.py                 # Custom layers
│   │   └── embeddings.py             # Embedding layers
│   └── utils/
│       ├── __init__.py
│       ├── data_preprocessing.py     # Data preprocessing utilities
│       ├── tabular_preprocessor.py   # Tabular data preprocessing
│       ├── image_preprocessor.py     # Image preprocessing
│       └── multimodal_preprocessor.py # Multimodal data handling
├── notebooks/
│   └── multimodal_fraud_detection_pipeline.ipynb  # Complete pipeline notebook
├── models/
│   └── saved_models/                 # Trained models saved here (gitignored)
├── logs/                             # TensorBoard logs (gitignored)
└── data/                             # Local data directory (gitignored)
```

### EC2 Directory Structure

When running on EC2, the following directories are created:

```
/home/ec2-user/
├── test/                             # Cloned repository
├── qrdata/                           # QR code images directory
├── csv-data/                         # Transaction CSV data directory
├── model/                            # Trained models and artifacts
│   ├── logs/                         # TensorBoard logs
│   └── saved_models/                 # Model checkpoints
└── fraud_detection_env/              # Python virtual environment
```

---

## Configuration

### Environment Detection

The configuration system automatically detects the environment:
- **EC2**: If `/home/ec2-user` directory exists OR `USE_EC2_CONFIG=1` is set
- **Local**: Otherwise, uses local development paths

### Manual Configuration Override

To force EC2 configuration on a non-EC2 machine:

```bash
export USE_EC2_CONFIG=1
python src/train.py --mode multimodal
```

### Model Hyperparameters

Edit `config/config.py` to adjust:

```python
MODEL_CONFIG = {
    'd_model': 64,              # Embedding dimension
    'num_heads': 4,             # Attention heads
    'num_layers': 2,            # Transformer layers per modality
    'cross_modal_layers': 2,    # Cross-modal fusion layers
    'dff': 128,                 # Feed-forward dimension
    'dropout_rate': 0.1,        # Dropout rate
}

TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'early_stopping_patience': 5,
}
```

### GPU Configuration

For GPU-enabled training, the system automatically:
- Enables memory growth (prevents OOM errors)
- Uses mixed precision (FP16) for faster training on modern GPUs
- Configures optimal batch sizes

Disable mixed precision if needed:

```python
# In config/config.py
GPU_CONFIG = {
    'memory_growth': True,
    'mixed_precision': False,  # Set to False to disable
}
```

---

## Troubleshooting

### Local Issues

#### Issue: `ModuleNotFoundError: No module named 'tensorflow'`
**Solution**: Ensure virtual environment is activated and dependencies are installed:
```bash
source fraud_env/bin/activate
pip install -r requirements.txt
```

#### Issue: GPU not detected
**Solution**: 
1. Check CUDA installation: `nvidia-smi`
2. Verify TensorFlow installation (GPU support is included by default in TensorFlow 2.x): `pip install --upgrade tensorflow`
3. Verify GPU detection: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

#### Issue: Out of memory (OOM) errors
**Solution**:
- Reduce batch size in `config/config.py`: `TRAINING_CONFIG['batch_size'] = 16`
- Enable memory growth (should be default)
- Close other GPU applications

### EC2 Issues

#### Issue: Cannot connect to EC2 instance
**Solution**:
1. Check security group allows SSH (port 22) from your IP
2. Verify key pair permissions: `chmod 400 /path/to/key.pem`
3. Check instance is running: AWS Console → EC2 → Instances

#### Issue: Cannot access Jupyter on EC2
**Solution**:
1. Check security group allows port 8888 from your IP
2. Verify Jupyter is running: `ps aux | grep jupyter`
3. Use SSH tunneling instead of direct access (more secure)

#### Issue: NVIDIA driver not found on EC2
**Solution**:
For Amazon Linux 2:
```bash
sudo yum install -y kernel-devel-$(uname -r) gcc
sudo yum install -y nvidia-driver-latest-dkms
sudo reboot
```

For Ubuntu:
```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-525
sudo reboot
```

Or use AWS Deep Learning AMI which includes pre-installed drivers.

#### Issue: Permission denied when running setup script
**Solution**:
```bash
chmod +x setup_ec2_jupyter.sh
./setup_ec2_jupyter.sh
```

#### Issue: Setup script fails to create directories
**Solution**: Run with appropriate permissions or create manually:
```bash
sudo mkdir -p /home/ec2-user/{qrdata,csv-data,model}
sudo chown -R ec2-user:ec2-user /home/ec2-user/{qrdata,csv-data,model}
```

### Training Issues

#### Issue: Training very slow
**Solution**:
1. Verify GPU is being used:
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```
2. Enable mixed precision (should be default for EC2)
3. Increase batch size if you have available GPU memory
4. Use G5 instance instead of CPU instance

#### Issue: Loss not decreasing
**Solution**:
1. Check learning rate (try 0.001 to 0.0001)
2. Verify data preprocessing is correct
3. Check for data imbalance
4. Try training longer or adjusting model architecture

#### Issue: Model files not saved
**Solution**:
1. Check directory permissions: `ls -ld models/saved_models/`
2. Create directory manually: `mkdir -p models/saved_models`
3. For EC2: Ensure `/home/ec2-user/model` exists and is writable

---

## Additional Information

### Data Requirements

The system now loads ALL actual data from files for training:

**Tabular Data (CSV):**
- Place CSV files in `data/csvdata/` (local) or `/home/ec2-user/csvdata/` (EC2)
- **ALL CSV files in the directory will be loaded and combined**
- Expected columns: `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`
- Note: `nameOrig` and `nameDest` are ID columns that are automatically dropped during preprocessing

**Image Data (QR Codes):**
- Place QR code images in `data/qrimages/QR codes/` (local) or `/home/ec2-user/qrimages/QR codes/` (EC2)
- Directory structure: `benign/benign/` and `malicious/malicious/` subdirectories
- **ALL images in both subdirectories will be loaded**
- Supported formats: PNG, JPG, JPEG
- Images will be resized to 128x128

**Data Matching:**
- If CSV count ≠ image count, the smaller dataset is automatically replicated with augmentation to match
- This ensures every training sample has both tabular features and an associated image

### Model Deployment

After training, models can be deployed using:

1. **TensorFlow Serving**:
   ```bash
   tensorflow_model_server --model_base_path=/path/to/saved_model --rest_api_port=8501
   ```

2. **Flask/FastAPI**: Create REST API wrapper
3. **AWS SageMaker**: Deploy for production scale
4. **Docker**: Containerize for consistent deployment

### Performance Benchmarks

**G5.xlarge (NVIDIA A10G):**
- Training time: ~30-60 seconds per epoch (5000 samples)
- Inference: ~10-20ms per sample
- Memory usage: ~4-6GB GPU VRAM

**CPU (t3.xlarge):**
- Training time: ~5-10 minutes per epoch (5000 samples)
- Inference: ~50-100ms per sample

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### License

[Add your license information here]

### Support

For issues, questions, or contributions, please open an issue on GitHub.

---

## Quick Command Reference

### Local Development
```bash
# Setup
python3 -m venv fraud_env
source fraud_env/bin/activate
pip install -r requirements.txt

# Train
python src/train.py --mode multimodal

# Predict (after training)
python src/predict.py --mode multimodal

# Monitor
tensorboard --logdir logs/
```

### AWS EC2
```bash
# Initial setup
./setup_ec2_jupyter.sh
source /home/ec2-user/fraud_detection_env/bin/activate

# Train
python ec2_quick_start.py --mode multimodal

# Predict (after training)
python src/predict.py --mode multimodal

# Background training
nohup python ec2_quick_start.py --mode multimodal > training.log 2>&1 &

# Background predictions
nohup python src/predict.py --mode multimodal > predictions.log 2>&1 &

# Monitor
tail -f training.log
tail -f predictions.log
watch -n 1 nvidia-smi
```

### Jupyter (EC2)
```bash
# Set password
jupyter notebook password

# Start (EC2)
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Access
http://<EC2-IP>:8888
```

---

**Last Updated**: January 2026
