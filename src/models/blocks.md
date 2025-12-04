# blocks.py - Transformer Building Blocks Documentation

## Overview
This module provides complete transformer block implementations that combine attention mechanisms with feed-forward networks and residual connections. These are the fundamental building blocks stacked to create deep transformer models.

## Key Classes

### `TransformerBlock` - Encoder Block
**Purpose**: Standard transformer encoder block with self-attention and feed-forward sublayers.

#### Architecture
```python
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dff, dropout_rate=0)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
```
**What it does**: Sets up two sublayers with their own layer normalization and dropout.

#### Forward Pass
```python
def call(self, inputs, training=False):
    # Sublayer 1: Multi-Head Self-Attention + Residual Connection
    attn_output = self.attention(inputs)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(inputs + attn_output)  # Add & Norm
    
    # Sublayer 2: Feed-Forward Network + Residual Connection
    ffn_output = self.ffn(out1, training=training)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # Add & Norm
    
    return out2
```
**What it does**:
1. **Self-Attention**: Captures relationships between all positions
2. **Dropout**: Regularization during training
3. **Residual + LayerNorm**: Adds original input and normalizes
4. **FFN**: Applies position-wise transformations
5. **Dropout**: More regularization
6. **Residual + LayerNorm**: Final add and normalize

**Visual Flow**:
```
Input (batch, seq_len, 64)
    ↓
Multi-Head Self-Attention
    ↓
Dropout
    ↓
Add & LayerNorm ← [residual connection]
    ↓
Feed-Forward Network (64 → 128 → 64)
    ↓
Dropout
    ↓
Add & LayerNorm ← [residual connection]
    ↓
Output (batch, seq_len, 64)
```

#### Why Residual Connections?
```python
out1 = self.layernorm1(inputs + attn_output)  # Not just attn_output!
```
**Benefits**:
- **Gradient flow**: Allows gradients to flow directly through the network
- **Training stability**: Makes deep networks trainable (16+ layers)
- **Information preservation**: Maintains original features alongside learned ones

### `TransformerDecoderBlock`
**Purpose**: Transformer decoder block with masked self-attention and cross-attention.

#### Architecture
```python
class TransformerDecoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        self.masked_attention = MaskedMultiHeadAttention(d_model, num_heads)
        self.cross_attention = CrossModalAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dff, dropout_rate=0)
        # 3 layer norms and 3 dropouts
```
**What it does**: Sets up three sublayers for autoregressive decoding.

#### Forward Pass
```python
def call(self, inputs, encoder_output, training=False, mask=None):
    # Sublayer 1: Masked Self-Attention (prevents looking ahead)
    masked_attn_output = self.masked_attention(inputs, mask=mask)
    masked_attn_output = self.dropout1(masked_attn_output, training=training)
    out1 = self.layernorm1(inputs + masked_attn_output)
    
    # Sublayer 2: Cross-Attention to encoder output
    cross_attn_output = self.cross_attention(out1, encoder_output)
    cross_attn_output = self.dropout2(cross_attn_output, training=training)
    out2 = self.layernorm2(out1 + cross_attn_output)
    
    # Sublayer 3: Feed-Forward Network
    ffn_output = self.ffn(out2, training=training)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(out2 + ffn_output)
    
    return out3
```
**What it does**:
1. **Masked Self-Attention**: Attends only to previous positions (causal)
2. **Cross-Attention**: Queries information from encoder output
3. **FFN**: Position-wise transformations

**Use Case**: Sequence-to-sequence tasks like translation or text generation.

**Visual Flow**:
```
Decoder Input                    Encoder Output
    ↓                                 ↓
Masked Self-Attention                |
    ↓                                 |
Add & LayerNorm                      |
    ↓                                 |
Cross-Attention ← [queries encoder] ←┘
    ↓
Add & LayerNorm
    ↓
Feed-Forward Network
    ↓
Add & LayerNorm
    ↓
Decoder Output
```

### `CrossModalTransformerBlock`
**Purpose**: Specialized block for bidirectional multimodal fusion.

#### Architecture
```python
class CrossModalTransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        # Two cross-attention layers (bidirectional)
        self.cross_attn_tab_to_img = CrossModalAttention(d_model, num_heads)
        self.cross_attn_img_to_tab = CrossModalAttention(d_model, num_heads)
        
        # Two FFNs (one per modality)
        self.ffn_tabular = FeedForward(d_model, dff, dropout_rate=0)
        self.ffn_image = FeedForward(d_model, dff, dropout_rate=0)
        
        # Four layer norms (two per modality)
        self.layernorm_tab1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_tab2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_img1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_img2 = layers.LayerNormalization(epsilon=1e-6)
```
**What it does**: Creates separate processing paths for each modality with cross-modal interaction.

#### Forward Pass - Bidirectional Fusion
```python
def call(self, tabular_input, image_input, training=False):
    # Tabular attends to image
    cross_attn_tab = self.cross_attn_tab_to_img(tabular_input, image_input)
    cross_attn_tab = self.dropout_tab1(cross_attn_tab, training=training)
    tabular_out = self.layernorm_tab1(tabular_input + cross_attn_tab)
    
    # Feed-forward for tabular
    ffn_tab = self.ffn_tabular(tabular_out, training=training)
    ffn_tab = self.dropout_tab2(ffn_tab, training=training)
    tabular_out = self.layernorm_tab2(tabular_out + ffn_tab)
    
    # Image attends to tabular
    cross_attn_img = self.cross_attn_img_to_tab(image_input, tabular_input)
    cross_attn_img = self.dropout_img1(cross_attn_img, training=training)
    image_out = self.layernorm_img1(image_input + cross_attn_img)
    
    # Feed-forward for image
    ffn_img = self.ffn_image(image_out, training=training)
    ffn_img = self.dropout_img2(ffn_img, training=training)
    image_out = self.layernorm_img2(image_out + ffn_img)
    
    return tabular_out, image_out
```

**What it does**: Processes both modalities in parallel with cross-modal information exchange.

**Visual Flow**:
```
Tabular Features              Image Features
       ↓                            ↓
       ↓ ← Cross-Attention ← ← ← ← ↓
       ↓                            ↓
   Add & Norm                  Add & Norm
       ↓                            ↓
     FFN                          FFN
       ↓                            ↓
   Add & Norm                  Add & Norm
       ↓                            ↓
Enhanced Tabular          Enhanced Image
```

#### Why Bidirectional?
**Tabular → Image**:
- Transaction features query QR code patches
- "Given this large transaction, are there suspicious visual patterns?"

**Image → Tabular**:
- QR code patches query transaction context
- "This QR pattern is unusual, does the transaction amount match?"

**Together**: Both modalities inform each other, creating richer representations.

## Stacking Blocks

### Encoder Stack
```python
x = input
for _ in range(num_layers):
    x = TransformerBlock(d_model, num_heads, dff, dropout_rate)(x)
```
**What it does**: Stacks multiple encoder blocks to create deeper representations. Each block refines the representation further.

### Cross-Modal Stack
```python
tabular_encoded = tabular_encoder(tabular_input)
image_encoded = image_encoder(image_input)

for _ in range(cross_modal_layers):
    tabular_encoded, image_encoded = CrossModalTransformerBlock(...)(
        tabular_encoded, image_encoded
    )
```
**What it does**: Iteratively exchanges information between modalities, allowing deeper multimodal understanding.

## Configuration Example
```python
config = {
    'd_model': 64,        # Embedding dimension
    'num_heads': 4,       # Attention heads
    'num_layers': 2,      # Encoder blocks per modality
    'cross_modal_layers': 2,  # Fusion blocks
    'dff': 128,          # FFN hidden size (2x d_model)
    'dropout_rate': 0.1  # Regularization
}
```

## Block Comparison

| Block Type | Inputs | Outputs | Key Feature | Use Case |
|------------|--------|---------|-------------|----------|
| TransformerBlock | 1 sequence | 1 sequence | Self-attention | Encoder |
| TransformerDecoderBlock | 2 sequences | 1 sequence | Masked + cross attention | Decoder |
| CrossModalTransformerBlock | 2 sequences | 2 sequences | Bidirectional cross attention | Multimodal fusion |

## Training vs Inference
```python
def call(self, inputs, training=False):
    x = self.dropout(x, training=training)
```
**What it does**:
- **Training mode (training=True)**: Applies dropout for regularization
- **Inference mode (training=False)**: Skips dropout, uses all neurons
