# transformer.py - Transformer Model Builders Documentation

## Overview
This module provides alternative/simplified implementations of the transformer model builders. It contains similar functionality to `transformer_model.py` but in a more streamlined format.

## Note
This file appears to be an alternative implementation or earlier version of the model builders. The main comprehensive implementation is in `transformer_model.py` which includes all the core transformer components (attention layers, blocks, feed-forward networks, etc.).

## Key Classes

### `MultimodalFraudDetectionTransformer`
**Purpose**: Builds multimodal transformer combining tabular and image data.

**Key Methods**:

#### `build_tabular_encoder(inputs)`
```python
def build_tabular_encoder(self, inputs):
    x = layers.Dense(self.config['d_model'])(inputs)
    position_ids = tf.constant([list(range(max_seq_len))])
    position_embedding_layer = layers.Embedding(...)
    x = x + position_embeddings
    
    for _ in range(self.config['num_layers']):
        x = TransformerBlock(...)(x)
```
**What it does**: Creates encoder branch for processing transaction features through transformer blocks with position embeddings.

#### `build_image_encoder(image_inputs)`
```python
def build_image_encoder(self, image_inputs):
    patch_embedding = PatchEmbedding(
        image_size=image_size,
        patch_size=patch_size,
        d_model=self.config['d_model']
    )
    x = patch_embedding(image_inputs)
    
    for _ in range(self.config['num_layers']):
        x = TransformerBlock(...)(x)
```
**What it does**: Creates Vision Transformer encoder for processing QR code images through patch embedding and transformer blocks.

#### `build_model()`
```python
def build_model():
    tabular_encoded = self.build_tabular_encoder(tabular_inputs)
    image_encoded = self.build_image_encoder(image_inputs)
    
    for _ in range(cross_modal_layers):
        cross_modal_block = CrossModalTransformerBlock(...)
        tabular_encoded, image_encoded = cross_modal_block(...)
    
    fused = layers.Concatenate()([tabular_pooled, image_pooled])
    outputs = layers.Dense(1, activation='sigmoid')(x)
```
**What it does**: Assembles complete multimodal architecture with cross-modal fusion and classification head.

### `FraudDetectionTransformer`
**Purpose**: Simplified tabular-only transformer for backward compatibility.

**Key Method**:

#### `build_model()`
```python
def build_model():
    x = layers.Dense(self.config['d_model'])(inputs)
    x = x + position_embeddings
    
    for _ in range(self.config['num_layers']):
        x = TransformerBlock(...)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
```
**What it does**: Creates single-encoder transformer processing only transaction features.

## Relationship to Other Files
- Uses `TransformerBlock` and `CrossModalTransformerBlock` from `blocks.py`
- Uses `PatchEmbedding` from `embeddings.py`
- Provides similar functionality to `transformer_model.py`

## Usage
This file provides model builders that can be used similarly to those in `transformer_model.py`:

```python
from src.models.transformer import MultimodalFraudDetectionTransformer

config = {
    'd_model': 64,
    'num_heads': 4,
    'num_layers': 2,
    # ... other config
}

model_builder = MultimodalFraudDetectionTransformer(config)
model = model_builder.build_model()
model = model_builder.compile_model()
```

## Documentation Reference
For complete, detailed documentation of the transformer architecture including all components, see:
- **transformer_model.md** - Comprehensive transformer architecture documentation
- **attention.md** - Detailed attention mechanism explanations
- **blocks.md** - Transformer block implementations
- **embeddings.md** - Patch embedding documentation
- **layers.md** - Core layer implementations
