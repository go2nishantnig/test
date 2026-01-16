# AWS EC2 G5 XLarge Setup Guide

This guide explains how to set up and run the Multimodal Fraud Detection Transformer model on an AWS EC2 G5 XLarge instance with Jupyter Notebook.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [EC2 Instance Setup](#ec2-instance-setup)
- [Installation](#installation)
- [Running Jupyter Notebook](#running-jupyter-notebook)
- [Using the Model](#using-the-model)
- [Directory Structure](#directory-structure)
- [Troubleshooting](#troubleshooting)

## Overview

This setup is designed for **AWS EC2 G5 XLarge** instances, which feature:
- **NVIDIA A10G GPU** with 24GB VRAM
- Optimized for deep learning workloads
- CUDA-compatible for TensorFlow GPU acceleration

The configuration uses the following EC2 directories:
- **QR Code Data**: `/home/ec2-user/qrdata`
- **CSV Data**: `/home/ec2-user/csv-data`
- **Models**: `/home/ec2-user/model`
- **Logs**: `/home/ec2-user/model/logs`

## Prerequisites

Your EC2 G5 XLarge instance should have:
- ✅ **Git** (already installed)
- ✅ **Python 3.8+** (already installed)
- ⚠️ **NVIDIA Drivers** (required for GPU support)

### Installing NVIDIA Drivers (if not already installed)

If you launched a G5 instance with a Deep Learning AMI, NVIDIA drivers are pre-installed. Otherwise, install them:

```bash
# For Amazon Linux 2
sudo yum install -y nvidia-driver-latest-dkms

# Verify installation
nvidia-smi
```

You should see output showing your NVIDIA A10G GPU.

## EC2 Instance Setup

### 1. Security Group Configuration

Ensure your EC2 security group allows:
- **Port 22** (SSH) - for terminal access
- **Port 8888** (Jupyter) - for notebook access

Add inbound rule:
```
Type: Custom TCP
Port Range: 8888
Source: Your IP address or 0.0.0.0/0 (for public access - less secure)
```

### 2. Connect to Your Instance

```bash
ssh -i your-key.pem ec2-user@your-ec2-public-ip
```

## Installation

### Step 1: Clone the Repository

```bash
cd /home/ec2-user
git clone https://github.com/go2nishantnig/test.git
cd test
```

### Step 2: Run the Setup Script

The setup script will:
- Create required directories
- Install Python dependencies
- Set up virtual environment
- Configure Jupyter Notebook
- Verify GPU availability

```bash
bash setup_ec2_jupyter.sh
```

**This script installs:**
- TensorFlow 2.13+ (with GPU support)
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- Jupyter Lab/Notebook
- Pillow (for image processing)
- Additional EC2 utilities (boto3, tqdm, ipywidgets)

The script will create a virtual environment at `/home/ec2-user/fraud_detection_env`.

### Step 3: Activate Virtual Environment

```bash
source /home/ec2-user/fraud_detection_env/bin/activate
```

## Running Jupyter Notebook

### Start Jupyter Notebook

```bash
cd /home/ec2-user/test
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

Or use Jupyter Lab (modern interface):

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

### Access Jupyter from Your Browser

1. The terminal will display a URL with a token, like:
   ```
   http://0.0.0.0:8888/?token=abc123...
   ```

2. Replace `0.0.0.0` with your **EC2 public IP address**:
   ```
   http://your-ec2-public-ip:8888/?token=abc123...
   ```

3. Open this URL in your browser

### Set a Password (Optional)

For easier access without tokens:

```bash
jupyter notebook password
```

Then restart Jupyter - you can now login with your password instead of the token.

## Using the Model

### Quick Start Notebook

Open the **EC2 G5 XLarge Pipeline notebook**:
```
/home/ec2-user/test/notebooks/ec2_g5_xlarge_pipeline.ipynb
```

This notebook provides:
- ✅ GPU verification and configuration
- ✅ Data loading from EC2 directories
- ✅ Full training pipeline with multimodal data
- ✅ Model evaluation and visualization
- ✅ Model saving to `/home/ec2-user/model`

### Using Your Own Data

#### CSV Data (Transaction Data)

Place your CSV file in the data directory:
```bash
cp your_fraud_data.csv /home/ec2-user/csv-data/online_payments_fraud.csv
```

Expected columns:
- `step`, `type`, `amount`
- `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`
- `nameDest`, `oldbalanceDest`, `newbalanceDest`
- `isFraud` or `is_fraud` (label)
- `isFlaggedFraud`

#### QR Code Images

Organize images in subdirectories:
```bash
/home/ec2-user/qrdata/
  ├── benign/          # Normal QR codes
  │   ├── img1.png
  │   ├── img2.png
  │   └── ...
  └── malicious/       # Fraudulent QR codes
      ├── img1.png
      ├── img2.png
      └── ...
```

Supported formats: PNG, JPG, JPEG

### Training the Model

#### Option 1: Using Jupyter Notebook (Recommended)

Open and run `notebooks/ec2_g5_xlarge_pipeline.ipynb` cell by cell.

#### Option 2: Using Command Line

```bash
cd /home/ec2-user/test
source /home/ec2-user/fraud_detection_env/bin/activate

# Set environment variable to use EC2 config
export USE_EC2_CONFIG=1

python src/train.py --mode multimodal
```

Or create a custom training script:

```python
import os
os.environ['USE_EC2_CONFIG'] = '1'

from config.ec2_config import MODEL_CONFIG, TRAINING_CONFIG
from src.models.transformer_model import MultimodalFraudDetectionTransformer
from src.utils.data_preprocessing import MultimodalDataPreprocessor

# Your training code here
```

### Making Predictions

After training, use the model for inference:

```python
from tensorflow import keras
import numpy as np

# Load trained model
model = keras.models.load_model('/home/ec2-user/model/multimodal_fraud_detection_final.keras')

# Load preprocessors
from src.utils.data_preprocessing import MultimodalDataPreprocessor
preprocessor = MultimodalDataPreprocessor.load_preprocessors('/home/ec2-user/model/preprocessor.pkl')

# Make predictions
prediction = model.predict([tabular_data, image_data])
```

## Directory Structure

After setup, your EC2 instance will have:

```
/home/ec2-user/
├── test/                              # Repository
│   ├── config/
│   │   ├── config.py                 # Default config
│   │   └── ec2_config.py             # EC2-specific config ⭐
│   ├── notebooks/
│   │   └── ec2_g5_xlarge_pipeline.ipynb  # Main notebook ⭐
│   ├── src/
│   │   ├── models/
│   │   ├── utils/
│   │   ├── train.py
│   │   └── predict.py
│   ├── setup_ec2_jupyter.sh          # Setup script ⭐
│   └── requirements.txt
│
├── qrdata/                            # QR code images ⭐
│   ├── benign/
│   └── malicious/
│
├── csv-data/                          # Transaction data ⭐
│   └── online_payments_fraud.csv
│
├── model/                             # Trained models ⭐
│   ├── multimodal_fraud_detection_best.keras
│   ├── multimodal_fraud_detection_final.keras
│   ├── preprocessor.pkl
│   ├── model_config.json
│   ├── training_history.png
│   ├── evaluation_metrics.png
│   └── logs/                         # TensorBoard logs
│       └── 20240116-120000/
│
└── fraud_detection_env/              # Virtual environment
    ├── bin/
    ├── lib/
    └── ...
```

## Monitoring Training

### TensorBoard

View training progress in real-time:

```bash
tensorboard --logdir=/home/ec2-user/model/logs --host=0.0.0.0 --port=6006
```

Then access in browser:
```
http://your-ec2-public-ip:6006
```

**Note**: Add port 6006 to your security group inbound rules.

### GPU Utilization

Monitor GPU usage:

```bash
# Watch GPU in real-time
watch -n 1 nvidia-smi

# Or detailed stats
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv -l 1
```

### System Resources

```bash
# CPU and memory
htop

# Disk usage
df -h

# Check directory sizes
du -sh /home/ec2-user/model/*
```

## Configuration

### GPU Configuration

The EC2 config enables optimal GPU usage:

```python
# config/ec2_config.py
GPU_CONFIG = {
    'memory_growth': True,       # Allocate memory as needed
    'mixed_precision': True,     # Use FP16 for faster training
}
```

### Model Configuration

Adjust hyperparameters in `config/ec2_config.py`:

```python
MODEL_CONFIG = {
    'd_model': 64,               # Embedding dimension
    'num_heads': 4,              # Attention heads
    'num_layers': 2,             # Transformer layers
    'batch_size': 32,            # Batch size
    'epochs': 10,                # Training epochs
    # ... more options
}
```

## Troubleshooting

### Issue: GPU Not Detected

**Check NVIDIA drivers:**
```bash
nvidia-smi
```

**If command not found:**
```bash
# Amazon Linux 2
sudo yum install -y nvidia-driver-latest-dkms

# Reboot instance
sudo reboot
```

**Verify TensorFlow GPU:**
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### Issue: Jupyter Notebook Not Accessible

**Check if Jupyter is running:**
```bash
ps aux | grep jupyter
```

**Check port 8888 in security group:**
- AWS Console → EC2 → Security Groups
- Ensure inbound rule for port 8888 exists

**Try different port:**
```bash
jupyter notebook --ip=0.0.0.0 --port=8889 --no-browser
```

### Issue: Out of Memory (OOM)

**Reduce batch size in config:**
```python
TRAINING_CONFIG = {
    'batch_size': 16,  # Reduce from 32
    # ...
}
```

**Enable memory growth:**
```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Issue: Slow Training

**Verify GPU is being used:**
```bash
watch -n 1 nvidia-smi
# Check GPU-Util column - should be >0%
```

**Enable mixed precision:**
```python
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

**Increase batch size (if memory allows):**
```python
TRAINING_CONFIG = {
    'batch_size': 64,  # Increase from 32
    # ...
}
```

### Issue: Permission Denied

**Fix directory permissions:**
```bash
sudo chown -R ec2-user:ec2-user /home/ec2-user/qrdata
sudo chown -R ec2-user:ec2-user /home/ec2-user/csv-data
sudo chown -R ec2-user:ec2-user /home/ec2-user/model
```

## Best Practices

### 1. Use Virtual Environment

Always activate the virtual environment:
```bash
source /home/ec2-user/fraud_detection_env/bin/activate
```

### 2. Save Work Regularly

- Jupyter auto-saves, but manually save important cells
- Model checkpoints save best weights automatically
- Copy important results to S3 for backup

### 3. Monitor Costs

- Stop (don't terminate) instance when not in use
- G5 XLarge costs ~$1/hour (prices vary by region)
- Use AWS Budgets to set spending alerts

### 4. Backup Models

```bash
# Copy to S3
aws s3 sync /home/ec2-user/model s3://your-bucket/fraud-detection-models/

# Or download to local machine
scp -i your-key.pem -r ec2-user@your-ec2-ip:/home/ec2-user/model ./local_backup
```

### 5. Use tmux for Long Training

Keep training running even if SSH disconnects:

```bash
# Install tmux
sudo yum install tmux -y

# Start tmux session
tmux new -s training

# Run Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

## Additional Resources

- **TensorFlow GPU Guide**: https://www.tensorflow.org/install/gpu
- **AWS G5 Instances**: https://aws.amazon.com/ec2/instance-types/g5/
- **Jupyter Documentation**: https://jupyter-notebook.readthedocs.io/
- **Model Architecture**: See `README.md` in repository root

## Support

For issues with:
- **Setup/Installation**: Check this guide's Troubleshooting section
- **Model Architecture**: See main `README.md`
- **AWS EC2**: AWS Support or documentation
- **Repository Issues**: Open issue on GitHub

## Quick Reference

```bash
# Setup (one-time)
bash setup_ec2_jupyter.sh
source /home/ec2-user/fraud_detection_env/bin/activate

# Start Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Train model (CLI)
cd /home/ec2-user/test
python src/train.py --mode multimodal

# Monitor GPU
watch -n 1 nvidia-smi

# TensorBoard
tensorboard --logdir=/home/ec2-user/model/logs --host=0.0.0.0
```

---

**Happy Training! 🚀**
