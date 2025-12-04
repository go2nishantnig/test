# image_preprocessor.py - Image Preprocessor Documentation

## Overview
This module handles preprocessing of QR code images for fraud detection. It provides synthetic QR code generation for testing and a framework for loading real QR code datasets.

## `QRCodePreprocessor` Class

### Purpose
Processes QR code images from the "Benign and Malicious QR Codes" dataset concept, providing both synthetic data generation and real image loading capabilities.

### Initialization
```python
class QRCodePreprocessor:
    def __init__(self, image_size=(128, 128)):
        self.image_size = image_size
```
**What it does**: Sets target image dimensions for resizing and generation.

### QR Code Structure

#### Real QR Code Components
1. **Finder patterns**: Three corner squares for orientation
2. **Alignment patterns**: Smaller squares for grid alignment
3. **Timing patterns**: Alternating black/white modules
4. **Data modules**: Actual encoded information
5. **Error correction**: Reed-Solomon codes

#### Synthetic Generation
```python
def _generate_qr_pattern(self, malicious=False):
    h, w = self.image_size  # 128, 128
    img = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background
```
**What it does**: Creates base 128×128×3 white canvas.

### Position Detection Patterns

#### Finder Pattern Structure
```python
def _add_position_pattern(self, img, row, col, size):
    # Outer black square
    img[row:row+size, col:col+size] = 0
    
    # Middle white square
    inner_start = size // 7
    inner_size = size - 2 * inner_start
    img[row+inner_start:row+inner_start+inner_size, 
        col+inner_start:col+inner_start+inner_size] = 255
    
    # Inner black square
    center_start = size // 3
    center_size = size // 3
    img[row+center_start:row+center_start+center_size,
        col+center_start:col+center_start+center_size] = 0
```

**Visual representation** (24×24 pixels):
```
████████████████████████  ← Outer black (24×24)
█                      █
█  ████████████████  █  ← Middle white (17×17)
█  █              █  █
█  █  ████████  █  █  ← Inner black (8×8)
█  █  ████████  █  █
█  █  ████████  █  █
█  █              █  █
█  ████████████████  █
█                      █
████████████████████████
```

**Position in QR code**:
```
┌─────────────────────┐
│[F]            [F]   │  [F] = Finder pattern
│                     │
│                     │
│                     │
│                     │
│[F]                  │
└─────────────────────┘
```

### Data Module Generation

```python
block_size = 8
for i in range(block_size * 3, h - block_size * 3, block_size):
    for j in range(block_size * 3, w - block_size * 3, block_size):
        if np.random.random() > 0.5:
            img[i:i+block_size, j:j+block_size] = 0  # Black block
```
**What it does**: Fills center area with random 8×8 black/white blocks simulating data encoding.

**Pattern example**:
```
After position patterns:
┌────────────────┐
│■■■  ?????  ■■■│  ■ = Position pattern
│■■■  ?????  ■■■│  ? = Random data area
│     ?????     │
│     ?????     │
│■■■  ?????     │
└────────────────┘
```

### Malicious QR Code Characteristics

#### Noise Addition
```python
if malicious:
    # Add random noise
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    img = np.clip(img.astype(np.int32) + noise - 25, 0, 255).astype(np.uint8)
```
**What it does**: Adds random noise in range [-25, +25] to each pixel.

**Effect**:
- Clean QR: Sharp black (0) and white (255)
- Noisy QR: Grayish values (e.g., black becomes 25, white becomes 230)
- Simulates printing defects, camera blur, tampering

#### Color Tint
```python
if malicious:
    tint = np.random.choice([0, 1, 2])  # R, G, or B channel
    img[:, :, tint] = np.clip(img[:, :, tint] + 30, 0, 255)
```
**What it does**: Adds color bias to one channel.

**Visual effect**:
```
Channel 0 (Red) tint:   Reddish QR code
Channel 1 (Green) tint: Greenish QR code
Channel 2 (Blue) tint:  Blueish QR code
```

**Why these features?**
Malicious QR codes may have:
- **Overlay tampering**: Additional layers obscuring original
- **Screen capture**: Color casts from displays
- **Photocopying**: Noise and distortion
- **Deliberate obfuscation**: Making legitimate QR unreadable

### Synthetic Data Generation

```python
def generate_synthetic_qr_images(self, n_samples=1000, malicious_ratio=0.3):
    n_malicious = int(n_samples * malicious_ratio)
    n_benign = n_samples - n_malicious
    
    # Generate benign QR codes
    benign_images = [self._generate_qr_pattern(malicious=False) 
                     for _ in range(n_benign)]
    
    # Generate malicious QR codes
    malicious_images = [self._generate_qr_pattern(malicious=True) 
                        for _ in range(n_malicious)]
    
    # Combine and shuffle
    images = np.array(benign_images + malicious_images)
    labels = np.array([0] * n_benign + [1] * n_malicious)
    
    indices = np.random.permutation(n_samples)
    images = images[indices]
    labels = labels[indices]
    
    # Normalize to [0, 1]
    images = images.astype(np.float32) / 255.0
```

**Output**:
- **images**: (n_samples, 128, 128, 3) float32 array in [0, 1]
- **labels**: (n_samples,) binary array (0=benign, 1=malicious)

**Why normalize?**
- Neural networks prefer inputs in [0, 1] or [-1, 1]
- Prevents gradient explosion
- Standardizes across different image sources

### Real Data Loading (Framework)

#### Directory Structure
```
qr_codes/
├── benign/
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── malicious/
    ├── image001.png
    ├── image002.png
    └── ...
```

#### Loading Function
```python
def load_images_from_directory(self, directory, target_size=None):
    if not os.path.exists(directory):
        print("Directory not found. Using synthetic data.")
        return self.generate_synthetic_qr_images()
    
    # Load benign images
    benign_dir = os.path.join(directory, 'benign')
    for filename in os.listdir(benign_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = self._load_and_resize_image(
                os.path.join(benign_dir, filename), target_size
            )
    
    # Load malicious images
    malicious_dir = os.path.join(directory, 'malicious')
    # ... similar loading
```
**What it does**: Provides template for loading real QR code datasets.

**Current implementation**: Placeholder that returns synthetic data (PIL/OpenCV not imported to keep dependencies minimal).

**Production implementation**:
```python
from PIL import Image

def _load_and_resize_image(self, filepath, target_size):
    img = Image.open(filepath).resize(target_size).convert('RGB')
    return np.array(img) / 255.0
```

### Train/Test Split

```python
def prepare_train_test_data(self, test_size=0.2, random_state=42):
    images, labels = self.generate_synthetic_qr_images()
    
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=labels
    )
    
    return X_train, X_test, y_train, y_test
```
**Stratified split**: Maintains malicious ratio in both train and test sets.

## Image Data Flow

### Generation Pipeline
```
1. Create white canvas (128×128×3)
        ↓
2. Add position patterns (3 corners)
        ↓
3. Add random data blocks (center area)
        ↓
4. If malicious: Add noise + color tint
        ↓
5. Convert to uint8 [0, 255]
        ↓
6. Normalize to float32 [0, 1]
        ↓
7. Output ready for model
```

### Shape Transformations
```
Single image:
Raw generation:  (128, 128, 3) uint8 [0, 255]
After normalize: (128, 128, 3) float32 [0, 1]

Batch:
(n_samples, 128, 128, 3) float32 [0, 1]
```

## Usage Examples

### Basic Generation
```python
preprocessor = QRCodePreprocessor(image_size=(128, 128))

# Generate dataset
images, labels = preprocessor.generate_synthetic_qr_images(
    n_samples=1000, 
    malicious_ratio=0.3
)

print(f"Images shape: {images.shape}")  # (1000, 128, 128, 3)
print(f"Labels shape: {labels.shape}")  # (1000,)
print(f"Value range: [{images.min()}, {images.max()}]")  # [0.0, 1.0]
print(f"Malicious ratio: {labels.mean():.1%}")  # 30%
```

### Train/Test Split
```python
preprocessor = QRCodePreprocessor()
X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_data()

print(f"Train: {len(X_train)} samples")
print(f"Test: {len(X_test)} samples")
print(f"Train malicious: {y_train.mean():.1%}")
print(f"Test malicious: {y_test.mean():.1%}")
```

### Loading Real Data (When Available)
```python
preprocessor = QRCodePreprocessor(image_size=(128, 128))

# Attempts to load from directory, falls back to synthetic
images, labels = preprocessor.load_images_from_directory(
    directory='data/qr_codes',
    target_size=(128, 128)
)
```

## QR Code Dataset Notes

### Real-World Datasets
Potential sources for real QR code data:
1. **QR Code datasets on Kaggle**
2. **Security research papers** with malicious QR examples
3. **Generated datasets** using QR code libraries

### Malicious QR Code Types
Real malicious QR codes may:
1. **Phishing**: Link to fake websites
2. **Malware**: Trigger app downloads
3. **Scam**: Fraudulent payment requests
4. **Overlay**: Legitimate QR with malicious overlay

Our synthetic data simulates visual artifacts that might indicate tampering or suspicious generation methods.
