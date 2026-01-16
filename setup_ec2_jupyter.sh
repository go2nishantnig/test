#!/bin/bash
# Setup script for AWS EC2 G5 XLarge instance with Jupyter Notebook
# Prerequisites: Git and Python 3.8+ already installed
# This script installs all required dependencies and sets up Jupyter notebook

set -e  # Exit on error

echo "======================================"
echo "AWS EC2 G5 XLarge Setup for Multimodal Fraud Detection"
echo "======================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if running on EC2
print_info "Checking environment..."
if [ ! -d "/home/ec2-user" ]; then
    print_info "Warning: /home/ec2-user not found. Are you running on AWS EC2?"
    echo "This script is designed for AWS EC2 instances with ec2-user."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Verify Python installation
print_info "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_success "Python $PYTHON_VERSION found"

# Verify Git installation
print_info "Checking Git installation..."
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install Git."
    exit 1
fi

GIT_VERSION=$(git --version | awk '{print $3}')
print_success "Git $GIT_VERSION found"

# Create required directories
print_info "Creating required directories..."
mkdir -p /home/ec2-user/qrdata
mkdir -p /home/ec2-user/csv-data
mkdir -p /home/ec2-user/model
mkdir -p /home/ec2-user/model/logs
mkdir -p /home/ec2-user/model/saved_models

# Ensure proper ownership (in case script is run as root)
if [ "$EUID" -eq 0 ]; then
    chown -R ec2-user:ec2-user /home/ec2-user/qrdata
    chown -R ec2-user:ec2-user /home/ec2-user/csv-data
    chown -R ec2-user:ec2-user /home/ec2-user/model
fi

print_success "Directories created: /home/ec2-user/qrdata, /home/ec2-user/csv-data, /home/ec2-user/model"

# Update pip
print_info "Upgrading pip..."
python3 -m pip install --upgrade pip --user

# Install virtualenv if not present
print_info "Checking for virtualenv..."
if ! python3 -m pip show virtualenv &> /dev/null; then
    print_info "Installing virtualenv..."
    python3 -m pip install virtualenv --user
fi
print_success "virtualenv is available"

# Create virtual environment
print_info "Creating virtual environment..."
if [ -d "/home/ec2-user/fraud_detection_env" ]; then
    print_info "Virtual environment already exists. Removing old one..."
    rm -rf /home/ec2-user/fraud_detection_env
fi

python3 -m venv /home/ec2-user/fraud_detection_env
print_success "Virtual environment created at /home/ec2-user/fraud_detection_env"

# Activate virtual environment
print_info "Activating virtual environment..."
source /home/ec2-user/fraud_detection_env/bin/activate

# Upgrade pip in virtual environment
print_info "Upgrading pip in virtual environment..."
pip install --upgrade pip

# Install Python dependencies from requirements.txt
print_info "Installing Python dependencies..."
if [ -f "requirements_ec2.txt" ]; then
    pip install -r requirements_ec2.txt
    print_success "Dependencies from requirements_ec2.txt installed"
elif [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "Dependencies from requirements.txt installed"
else
    print_error "requirements files not found. Installing dependencies manually..."
    pip install tensorflow>=2.13.0 numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0 \
                matplotlib>=3.7.0 seaborn>=0.12.0 jupyter>=1.0.0 Pillow>=10.0.0
    print_success "Core dependencies installed"
fi

# Install additional useful packages for EC2
print_info "Installing additional packages for EC2..."
pip install jupyterlab ipywidgets tqdm boto3

print_success "Additional packages installed"

# Configure GPU for TensorFlow (G5 XLarge has NVIDIA A10G)
print_info "Checking NVIDIA GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    print_success "NVIDIA GPU detected"
    
    # Install CUDA-compatible packages if needed
    print_info "Verifying TensorFlow GPU support..."
    python3 -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))" || true
else
    print_info "No NVIDIA GPU detected (nvidia-smi not found)"
    echo "Note: G5 XLarge instances should have NVIDIA drivers installed."
    echo "If you're on a G5 instance, you may need to install NVIDIA drivers:"
    echo "  sudo yum install -y nvidia-driver-latest-dkms"
fi

# Setup Jupyter Notebook
print_info "Configuring Jupyter Notebook..."

# Generate Jupyter config if it doesn't exist
jupyter notebook --generate-config -y 2>/dev/null || true

# Configure Jupyter to accept connections from any IP (for EC2)
JUPYTER_CONFIG="/home/ec2-user/.jupyter/jupyter_notebook_config.py"
if [ -f "$JUPYTER_CONFIG" ]; then
    # Backup existing config
    cp "$JUPYTER_CONFIG" "$JUPYTER_CONFIG.backup.$(date +%Y%m%d_%H%M%S)"
fi

cat > "$JUPYTER_CONFIG" << 'EOF'
# Jupyter Notebook Configuration for AWS EC2
# Compatible with both Jupyter Notebook 6.x and 7.x

# Jupyter Notebook 7.0+ settings (ServerApp)
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_remote_access = True
c.ServerApp.root_dir = '/home/ec2-user'

# Increase limits for large datasets
c.ServerApp.iopub_data_rate_limit = 1000000000
c.ServerApp.iopub_msg_rate_limit = 1000000

# Allow root (if needed)
c.ServerApp.allow_root = True

# Backward compatibility with Jupyter Notebook 6.x (NotebookApp)
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.NotebookApp.allow_remote_access = True
c.NotebookApp.notebook_dir = '/home/ec2-user'
c.NotebookApp.iopub_data_rate_limit = 1000000000
c.NotebookApp.iopub_msg_rate_limit = 1000000
c.NotebookApp.allow_root = True
EOF

print_success "Jupyter Notebook configured"

# Create a quick test notebook
print_info "Creating test notebook..."
cat > /home/ec2-user/test_setup.ipynb << 'EOF'
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# EC2 Setup Test\n",
        "This notebook verifies that your EC2 environment is correctly configured."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "import sys\n",
        "import tensorflow as tf\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import os\n",
        "\n",
        "print(\"Python version:\", sys.version)\n",
        "print(\"TensorFlow version:\", tf.__version__)\n",
        "print(\"NumPy version:\", np.__version__)\n",
        "print(\"Pandas version:\", pd.__version__)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Check GPU availability\n",
        "print(\"GPU Available:\", tf.config.list_physical_devices('GPU'))\n",
        "if tf.config.list_physical_devices('GPU'):\n",
        "    print(\"\\nGPU Details:\")\n",
        "    for gpu in tf.config.list_physical_devices('GPU'):\n",
        "        print(f\"  {gpu}\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Verify directories exist\n",
        "dirs = [\n",
        "    '/home/ec2-user/qrdata',\n",
        "    '/home/ec2-user/csv-data',\n",
        "    '/home/ec2-user/model'\n",
        "]\n",
        "\n",
        "print(\"Directory Check:\")\n",
        "for dir_path in dirs:\n",
        "    exists = os.path.exists(dir_path)\n",
        "    print(f\"  {dir_path}: {'✓ EXISTS' if exists else '✗ MISSING'}\")"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}
EOF

print_success "Test notebook created at /home/ec2-user/test_setup.ipynb"

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
print_success "All dependencies installed successfully"
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source /home/ec2-user/fraud_detection_env/bin/activate"
echo ""
echo "  2. Start Jupyter Notebook:"
echo "     jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser"
echo ""
echo "  3. Access Jupyter from your browser:"
echo "     http://<your-ec2-public-ip>:8888"
echo ""
echo "  4. Make sure port 8888 is open in your EC2 security group"
echo ""
echo "  5. Run the test notebook to verify setup:"
echo "     /home/ec2-user/test_setup.ipynb"
echo ""
echo "Directories created:"
echo "  - QR Data:   /home/ec2-user/qrdata"
echo "  - CSV Data:  /home/ec2-user/csv-data"
echo "  - Models:    /home/ec2-user/model"
echo ""
echo "Virtual environment: /home/ec2-user/fraud_detection_env"
echo ""
print_info "For GPU training, ensure NVIDIA drivers are installed on your G5 instance"
echo ""
