# EC2 G5 XLarge Quick Reference

Quick commands and tips for running the Multimodal Fraud Detection model on AWS EC2 G5 XLarge.

## One-Time Setup

```bash
# 1. Clone repository
cd /home/ec2-user
git clone https://github.com/go2nishantnig/test.git
cd test

# 2. Run setup script
bash setup_ec2_jupyter.sh

# 3. Activate virtual environment
source /home/ec2-user/fraud_detection_env/bin/activate
```

## Daily Usage

### Start Jupyter Notebook

```bash
# Activate virtual environment
source /home/ec2-user/fraud_detection_env/bin/activate

# Start Jupyter
cd /home/ec2-user/test
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Access in browser: http://<your-ec2-ip>:8888
```

### Train Model

**Option 1: Jupyter Notebook (Recommended)**
```bash
# Open in browser: notebooks/ec2_g5_xlarge_pipeline.ipynb
```

**Option 2: Command Line**
```bash
source /home/ec2-user/fraud_detection_env/bin/activate
cd /home/ec2-user/test
python ec2_quick_start.py --mode multimodal
```

### Check Environment

```bash
cd /home/ec2-user/test
python check_ec2_env.py
```

## Directory Structure

```
/home/ec2-user/
├── qrdata/              # Place QR code images here
│   ├── benign/         # Normal QR codes
│   └── malicious/      # Fraudulent QR codes
│
├── csv-data/            # Place CSV data here
│   └── online_payments_fraud.csv
│
├── model/               # Trained models saved here
│   ├── *.keras         # Model files
│   ├── *.pkl           # Preprocessors
│   └── logs/           # TensorBoard logs
│
└── test/               # Repository
    └── ...
```

## Monitor Training

### GPU Usage
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Detailed GPU stats
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv -l 1
```

### TensorBoard
```bash
# Start TensorBoard (in new terminal)
source /home/ec2-user/fraud_detection_env/bin/activate
tensorboard --logdir=/home/ec2-user/model/logs --host=0.0.0.0 --port=6006

# Access in browser: http://<your-ec2-ip>:6006
# (Add port 6006 to security group)
```

## Data Preparation

### CSV Data
```bash
# Upload CSV file
scp -i your-key.pem local_fraud_data.csv ec2-user@your-ec2-ip:/home/ec2-user/csv-data/online_payments_fraud.csv

# Or use AWS CLI
aws s3 cp s3://your-bucket/fraud_data.csv /home/ec2-user/csv-data/online_payments_fraud.csv
```

### QR Code Images
```bash
# Upload images (local to EC2)
scp -i your-key.pem -r local_qr_images/* ec2-user@your-ec2-ip:/home/ec2-user/qrdata/

# Or sync from S3
aws s3 sync s3://your-bucket/qr-images/ /home/ec2-user/qrdata/
```

## Jupyter as a Service (Optional)

Run Jupyter permanently as a systemd service:

```bash
# Copy service file
sudo cp /home/ec2-user/test/jupyter.service /etc/systemd/system/

# Start service
sudo systemctl daemon-reload
sudo systemctl enable jupyter
sudo systemctl start jupyter

# Check status
sudo systemctl status jupyter

# View logs
sudo journalctl -u jupyter -f
```

## Backup Models

### To S3
```bash
# Backup to S3
aws s3 sync /home/ec2-user/model s3://your-bucket/fraud-detection-models/

# Restore from S3
aws s3 sync s3://your-bucket/fraud-detection-models/ /home/ec2-user/model/
```

### To Local Machine
```bash
# Download from EC2 to local
scp -i your-key.pem -r ec2-user@your-ec2-ip:/home/ec2-user/model ./local_backup
```

## Security Group Ports

Required inbound rules:
- **Port 22** (SSH): Your IP
- **Port 8888** (Jupyter): Your IP or 0.0.0.0/0
- **Port 6006** (TensorBoard): Your IP (optional)

## Troubleshooting

### GPU Not Working
```bash
# Check NVIDIA driver
nvidia-smi

# If not found, install driver
sudo yum install -y nvidia-driver-latest-dkms
sudo reboot

# Verify TensorFlow sees GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Out of Memory
```python
# In Python/Jupyter, enable memory growth:
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Or reduce batch size in config/ec2_config.py:
TRAINING_CONFIG = {
    'batch_size': 16,  # Reduce from 32
    ...
}
```

### Jupyter Not Accessible
```bash
# Check if Jupyter is running
ps aux | grep jupyter

# Check security group has port 8888 open

# Try different port
jupyter notebook --ip=0.0.0.0 --port=8889 --no-browser
```

### Permission Issues
```bash
# Fix directory permissions
sudo chown -R ec2-user:ec2-user /home/ec2-user/qrdata
sudo chown -R ec2-user:ec2-user /home/ec2-user/csv-data
sudo chown -R ec2-user:ec2-user /home/ec2-user/model
```

## Performance Optimization

### Enable Mixed Precision (FP16)
Already enabled in `config/ec2_config.py`. Provides ~2x speedup on A10G GPU.

### Increase Batch Size
If you have extra GPU memory:
```python
# In config/ec2_config.py
TRAINING_CONFIG = {
    'batch_size': 64,  # Increase from 32
    ...
}
```

### Monitor Resource Usage
```bash
# CPU and memory
htop

# Disk space
df -h

# Directory sizes
du -sh /home/ec2-user/model/*
```

## Cost Optimization

### Stop Instance When Not Training
```bash
# From AWS CLI
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Or use AWS Console
```

**Note**: Stopping (not terminating) preserves your data and setup.

### Pricing Reference
- G5 XLarge: ~$1.006/hour (us-east-1, varies by region)
- Stop instance when not in use to save costs
- Data on EBS is charged separately (~$0.10/GB-month)

## Files Added for EC2

- `config/ec2_config.py` - EC2-specific configuration
- `setup_ec2_jupyter.sh` - Automated setup script
- `ec2_quick_start.py` - Quick training script
- `check_ec2_env.py` - Environment verification
- `requirements_ec2.txt` - EC2 dependencies
- `jupyter.service` - Systemd service file
- `notebooks/ec2_g5_xlarge_pipeline.ipynb` - Main training notebook
- `EC2_SETUP_GUIDE.md` - Comprehensive guide

## Support

- **Full Documentation**: See `EC2_SETUP_GUIDE.md`
- **Model Details**: See `README.md`
- **Issues**: Check `check_ec2_env.py` output

---

**Quick Start Checklist:**
- [ ] Run `setup_ec2_jupyter.sh`
- [ ] Activate virtual environment
- [ ] Place data in `/home/ec2-user/csv-data` and `/home/ec2-user/qrdata`
- [ ] Open `notebooks/ec2_g5_xlarge_pipeline.ipynb`
- [ ] Train model
- [ ] Models saved to `/home/ec2-user/model`
