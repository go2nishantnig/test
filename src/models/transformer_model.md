# transformer_model.py - Transformer Model Architecture Documentation

## Overview
This module implements the complete multimodal transformer architecture for fraud detection, combining Vision Transformer (ViT) for QR code images with a tabular transformer for transaction features, unified through cross-modal attention.

## Core Components

### `FeedForward` Layer
```python
class FeedForward(layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        self.dense1 = layers.Dense(dff, activation='relu')
        self.dense2 = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout_rate)
```
**Purpose**: Position-wise feed-forward network (FFN) used in every transformer block.

**What it does**: 
- **First layer**: Projects from d_model (64) to dff (128) with ReLU activation
- **Dropout**: Applies regularization
- **Second layer**: Projects back to d_model (64)
- **Formula**: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

### `ResidualConnection` Layer
```python
class ResidualConnection(layers.Layer):
    def call(self, x, sublayer_output, training=False):
        sublayer_output = self.dropout(sublayer_output, training=training)
        return self.layer_norm(x + sublayer_output)
```
**Purpose**: Implements residual connections with layer normalization (Add & Norm).

**What it does**:
- Adds original input to sublayer output (residual connection)
- Applies layer normalization for stable training
- Helps gradients flow through deep networks
- **Formula**: LayerNorm(x + Dropout(Sublayer(x)))

### `MultiHeadSelfAttention` Layer
```python
class MultiHeadSelfAttention(layers.Layer):
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
```
**Purpose**: Core attention mechanism allowing the model to focus on different positions.

**What it does**:
1. **Linear projections**: Projects input to Query, Key, Value using learned weights
2. **Split heads**: Divides d_model into num_heads separate attention heads
3. **Scaled dot-product**: Computes attention scores: softmax(QK^T / √d_k)V
4. **Concatenate**: Merges all heads back together
5. **Final projection**: Linear layer to get final output

**Key Code**:
```python
matmul_qk = tf.matmul(q, k, transpose_b=True)
dk = tf.cast(tf.shape(k)[-1], tf.float32)
scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
output = tf.matmul(attention_weights, v)
```
**What it does**: Computes scaled dot-product attention with √d_k scaling to prevent gradient saturation.

### `MaskedMultiHeadAttention` Layer
```python
class MaskedMultiHeadAttention(layers.Layer):
    def create_causal_mask(self, seq_len):
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return mask * MASK_VALUE
```
**Purpose**: Attention layer with causal masking for decoder (prevents looking ahead).

**What it does**:
- Creates lower triangular mask to prevent attention to future positions
- Essential for autoregressive generation tasks
- Mask values set to -1e9, becoming ~0 after softmax

### `CrossModalAttention` Layer
```python
class CrossModalAttention(layers.Layer):
    def call(self, query_input, key_value_input):
        q = self.wq(query_input)
        k = self.wk(key_value_input)
        v = self.wv(key_value_input)
```
**Purpose**: Enables attention between two different modalities (e.g., tabular ↔ image).

**What it does**:
- Query comes from one modality (e.g., tabular features)
- Key and Value come from another modality (e.g., image patches)
- Allows one modality to "attend to" and extract information from another
- Critical for multimodal fusion

### `TransformerBlock` Layer
```python
class TransformerBlock(layers.Layer):
    def call(self, inputs, training=False):
        # Multi-Head Self-Attention + Residual Connection
        attn_output = self.attention(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-Forward Network + Residual Connection
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
```
**Purpose**: Complete transformer encoder block with two sublayers.

**What it does**:
1. **Self-attention sublayer**: Captures relationships between positions
2. **Residual + LayerNorm**: Adds original input and normalizes
3. **FFN sublayer**: Non-linear transformations
4. **Residual + LayerNorm**: Adds and normalizes again

### `CrossModalTransformerBlock` Layer
```python
class CrossModalTransformerBlock(layers.Layer):
    def call(self, tabular_input, image_input, training=False):
        # Tabular attends to image
        cross_attn_tab = self.cross_attn_tab_to_img(tabular_input, image_input)
        tabular_out = self.layernorm_tab1(tabular_input + cross_attn_tab)
        
        # Image attends to tabular
        cross_attn_img = self.cross_attn_img_to_tab(image_input, tabular_input)
        image_out = self.layernorm_img1(image_input + cross_attn_img)
        
        return tabular_out, image_out
```
**Purpose**: Bidirectional cross-modal fusion block.

**What it does**:
- Tabular features attend to image patches (learns visual patterns)
- Image patches attend to tabular features (learns transaction context)
- Each modality processes the other's information through FFN
- Enables rich multimodal feature interaction

### `PatchEmbedding` Layer
```python
class PatchEmbedding(layers.Layer):
    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        embedded_patches = self.projection(patches)
        embedded_patches = embedded_patches + self.position_embedding(positions)
```
**Purpose**: Vision Transformer (ViT) style image to sequence converter.

**What it does**:
1. **Extract patches**: Divides 128×128 image into 8×8 grid of 16×16 patches
2. **Flatten patches**: Each patch becomes a vector
3. **Linear projection**: Projects patch vectors to d_model dimension
4. **Add position embeddings**: Injects spatial information about patch locations

## Model Classes

### `MultimodalFraudDetectionTransformer`
**Purpose**: Complete multimodal architecture combining tabular and image encoders.

#### Tabular Encoder
```python
def build_tabular_encoder(self, inputs):
    x = layers.Dense(self.config['d_model'])(inputs)
    x = x + position_embeddings
    
    for _ in range(self.config['num_layers']):
        x = TransformerBlock(...)(x)
    
    return x
```
**What it does**: Processes transaction features through transformer blocks to create contextual representations.

#### Image Encoder
```python
def build_image_encoder(self, image_inputs):
    x = PatchEmbedding(...)(image_inputs)
    
    for _ in range(self.config['num_layers']):
        x = TransformerBlock(...)(x)
    
    return x
```
**What it does**: Processes QR code images through Vision Transformer style encoder.

#### Cross-Modal Fusion
```python
for _ in range(self.config.get('cross_modal_layers', 2)):
    cross_modal_block = CrossModalTransformerBlock(...)
    tabular_encoded, image_encoded = cross_modal_block(tabular_encoded, image_encoded)
```
**What it does**: Fuses tabular and image representations through bidirectional cross-attention layers.

#### Classification Head
```python
tabular_pooled = layers.GlobalAveragePooling1D()(tabular_encoded)
image_pooled = layers.GlobalAveragePooling1D()(image_encoded)
fused = layers.Concatenate()([tabular_pooled, image_pooled])
x = layers.Dense(128, activation='relu')(fused)
outputs = layers.Dense(1, activation='sigmoid', name='fraud_prediction')(x)
```
**What it does**:
1. Pools sequence representations to fixed-size vectors
2. Concatenates multimodal features
3. Passes through dense layers for classification
4. Sigmoid activation for binary fraud probability

### `FraudDetectionTransformer`
**Purpose**: Simplified tabular-only transformer (backward compatible).

**What it does**: Single-encoder architecture processing only transaction features, without image branch or cross-modal fusion.

## Model Compilation
```python
def compile_model(self, learning_rate=0.001):
    self.model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall', 'auc']
    )
```
**What it does**: Configures optimizer (Adam), loss function (binary cross-entropy for fraud/not fraud), and evaluation metrics.

## Architecture Summary
- **Input**: Dual inputs (tabular features + QR images)
- **Tabular Branch**: Dense projection → Position embeddings → 2 Transformer blocks
- **Image Branch**: Patch embedding → 2 Transformer blocks
- **Fusion**: 2 Cross-modal transformer blocks
- **Output**: Global pooling → Concatenation → Dense layers → Binary classification
