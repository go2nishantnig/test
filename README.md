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
- ☁️ **AWS EC2 ready**: Pre-configured for cloud deployment

---

## Table of Contents

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
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

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
1. Generates synthetic fraud detection dataset (or loads real data if available)
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

1. Preparing multimodal data...
   Training samples: 4000
   Testing samples: 1000
   Fraud ratio in training: 10.00%
   Fraud ratio in testing: 10.00%
   Tabular shape: (4000, 8)
   Image shape: (4000, 128, 128, 3)

2. Building multimodal transformer model...
...
```

### 5. View Training Results

#### Monitor Training with TensorBoard

```bash
# Start TensorBoard (runs in background)
tensorboard --logdir logs/ --port 6006

# Open in browser: http://localhost:6006
```

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

The system works with synthetic data by default for demonstration purposes. For production use:

**Tabular Data (CSV):**
- Place CSV files in `data/` (local) or `/home/ec2-user/csv-data/` (EC2)
- Expected columns: `step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `isFlaggedFraud`, `isFraud`

**Image Data (QR Codes):**
- Place QR code images in `data/qr_codes/` (local) or `/home/ec2-user/qrdata/` (EC2)
- Supported formats: PNG, JPG, JPEG
- Images will be resized to 128x128

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

# Background training
nohup python ec2_quick_start.py --mode multimodal > training.log 2>&1 &

# Monitor
tail -f training.log
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
