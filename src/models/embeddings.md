# embeddings.py - Embedding Layers Documentation

## Overview
This module provides embedding layers that convert different input types (images) into sequence representations suitable for transformer processing. The key implementation is the Vision Transformer (ViT) style patch embedding.

## `PatchEmbedding` Class
**Purpose**: Converts images into sequences of embedded patches for Vision Transformer processing.

### Concept
Vision Transformers treat images as sequences by:
1. Dividing the image into non-overlapping patches
2. Flattening each patch into a vector
3. Linearly projecting each patch to an embedding dimension
4. Adding position information

### Initialization
```python
class PatchEmbedding(layers.Layer):
    def __init__(self, image_size, patch_size, d_model):
        self.image_size = image_size  # (128, 128)
        self.patch_size = patch_size  # 16
        self.d_model = d_model  # 64
        
        # Calculate number of patches
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        # For 128x128 image with 16x16 patches: (128/16) * (128/16) = 8 * 8 = 64 patches
        
        self.projection = layers.Dense(d_model)
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches,  # 64 positions
            output_dim=d_model  # 64 dimensions
        )
```
**What it does**: Sets up the patch extraction and embedding mechanism.

### Patch Extraction
```python
def call(self, images):
    batch_size = tf.shape(images)[0]
    
    # Extract patches using TensorFlow's built-in operation
    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, self.patch_size, self.patch_size, 1],
        strides=[1, self.patch_size, self.patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
```
**What it does**: Extracts non-overlapping patches from the image.

**Visual Example** (simplified 4x4 image → 2x2 patches):
```
Original 4x4 Image:
┌─────────┐
│ A A B B │
│ A A B B │
│ C C D D │
│ C C D D │
└─────────┘

After patch extraction (2x2 patches):
Patches: [A, B, C, D]
```

**Actual Example** (128x128 image → 16x16 patches):
```
128x128 QR Code Image
        ↓
Divided into 8×8 grid
        ↓
64 patches of 16×16 pixels each
        ↓
Each patch: 16×16×3 = 768 values
```

### Patch Reshaping
```python
# Reshape patches
patch_dims = patches.shape[-1]  # 768 for 16x16x3 patches
patches = tf.reshape(patches, [batch_size, -1, patch_dims])
# Shape: (batch_size, 64, 768)
```
**What it does**: Flattens the spatial grid of patches into a sequence.

### Linear Projection
```python
embedded_patches = self.projection(patches)
# Projects from 768 to d_model (64)
# Shape: (batch_size, 64, 64)
```
**What it does**: 
- Reduces dimensionality from raw patch size (768) to model dimension (64)
- Learns which pixel patterns are most relevant
- Creates compact representations

### Position Embeddings
```python
positions = tf.range(start=0, limit=self.num_patches, delta=1)
# positions = [0, 1, 2, ..., 63]

position_embeddings = self.position_embedding(positions)
# Shape: (64, 64) - learnable position encodings

embedded_patches = embedded_patches + position_embeddings
# Broadcasting adds position info to each batch
```
**What it does**: Adds spatial position information to each patch.

**Why needed?** 
- Transformers have no inherent notion of position
- Without position embeddings, patches could be shuffled without changing the output
- Position embeddings tell the model "this patch is in the top-left corner"

**Visual Representation**:
```
Patch Positions in 8×8 Grid:
┌────────────────────────┐
│  0   1   2   3  ...  7 │
│  8   9  10  11  ... 15 │
│ 16  17  18  19  ... 23 │
│ ...                    │
│ 56  57  58  59  ... 63 │
└────────────────────────┘

Each position gets a unique 64-dimensional learned embedding
```

## Complete Processing Pipeline

### Input to Output Flow
```
Input QR Code Image: (batch, 128, 128, 3)
        ↓
Extract Patches: (batch, 8, 8, 768)
        ↓
Reshape: (batch, 64, 768)
        ↓
Linear Projection: (batch, 64, 64)
        ↓
Add Position Embeddings: (batch, 64, 64)
        ↓
Output: Sequence of 64 patch embeddings
```

### Mathematical View
For patch at position i:
```
patch_embedding[i] = Linear(Flatten(patch[i])) + PositionEmbedding(i)
```

## Why This Works

### Treating Images as Sequences
**Traditional CNN**: Processes images with convolutions
- Fixed receptive fields
- Locality bias (nearby pixels are related)
- Translation invariance

**Vision Transformer (ViT)**: Processes images as sequences
- Global receptive field (self-attention over all patches)
- Learns spatial relationships
- More flexible patterns

### Advantages for QR Codes
QR codes have:
- **Global patterns**: Position markers in corners affect interpretation
- **Long-range dependencies**: Data encoding spans the entire image
- **Structured layouts**: Transformers can learn QR code structure

### Patch Size Trade-offs

**Smaller patches (8×8)**:
- ✓ More patches → More detailed representation
- ✓ Better for fine-grained patterns
- ✗ More computation (longer sequences)
- ✗ More memory usage

**Larger patches (32×32)**:
- ✓ Fewer patches → Faster processing
- ✓ Less memory
- ✗ Less detail
- ✗ Might miss small malicious patterns

**Current choice (16×16)**: Balanced trade-off

## Integration with Transformer

```python
# In MultimodalFraudDetectionTransformer
def build_image_encoder(self, image_inputs):
    # Step 1: Convert image to patches
    x = PatchEmbedding(
        image_size=(128, 128),
        patch_size=16,
        d_model=64
    )(image_inputs)
    # x shape: (batch, 64, 64)
    
    # Step 2: Process with transformer blocks
    for _ in range(self.config['num_layers']):
        x = TransformerBlock(
            d_model=64,
            num_heads=4,
            dff=128,
            dropout_rate=0.1
        )(x)
    
    return x  # (batch, 64, 64) - encoded patches
```

**What happens**:
1. Image → 64 patch embeddings
2. Each patch attends to all other patches (self-attention)
3. Learn relationships between QR code regions
4. Detect global fraud patterns

## Comparison with CNN Embedding

**CNN Approach**:
```python
x = Conv2D(32, (3, 3))(image)
x = MaxPooling2D()(x)
x = Conv2D(64, (3, 3))(x)
x = GlobalAveragePooling2D()(x)
```

**ViT Approach (PatchEmbedding)**:
```python
x = PatchEmbedding()(image)
x = TransformerBlock()(x)
x = GlobalAveragePooling1D()(x)
```

**Key Differences**:
- CNN: Local convolutions → hierarchical features
- ViT: Patches → global attention → learned patterns

## Serialization
```python
@keras.utils.register_keras_serializable(package='FraudDetection')
def get_config(self):
    return {
        'image_size': self.image_size,
        'patch_size': self.patch_size,
        'd_model': self.d_model,
    }
```
**What it does**: Enables saving and loading models with custom PatchEmbedding layer.
