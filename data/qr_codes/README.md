# QR Code Images

This directory contains QR code images for the multimodal fraud detection system.

## Purpose

The multimodal fraud detection system uses Vision Transformer models to analyze QR code images for malicious patterns. This directory serves as the storage location for QR code images used in training and inference.

## Directory Structure

```
qr_codes/
├── README.md           # This file
└── (QR code images)    # PNG, JPG, or JPEG files
```

## Usage

### Local Development

Place QR code image files in this directory for local development and testing:

```python
import os
from config.config import QRCODE_DATASET_PATH

# The path points to this directory in local mode
print(QRCODE_DATASET_PATH)  # Output: /path/to/repo/data/qr_codes
```

### Supported Formats

- **File types**: PNG, JPG, JPEG
- **Preprocessing**: Images are automatically resized to 128x128 pixels
- **Color space**: RGB (3 channels)

### Example Usage

```python
from PIL import Image
import os

# Load a QR code image
qr_image_path = os.path.join('data/qr_codes', 'sample_qr.png')
image = Image.open(qr_image_path)

# Images are processed by the image preprocessor
# which handles resizing and normalization automatically
```

## Notes

- For AWS EC2 deployment, QR code images should be placed in `/home/ec2-user/qrimages/QR codes/` instead
- This directory is tracked by git, but image files can be excluded via `.gitignore` patterns if needed
- The system can generate synthetic QR code data for testing if real images are not available
