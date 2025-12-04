# layers.py - Core Transformer Layers Documentation

## Overview
This module provides fundamental building blocks used in transformer architectures. These are the basic components that are combined to create more complex transformer blocks.

## `FeedForward` Layer

### Purpose
Position-wise feed-forward network (FFN) that applies the same transformation to each position independently. This is a key component in every transformer block.

### Architecture
```python
class FeedForward(layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        self.d_model = d_model  # Input/output dimension (64)
        self.dff = dff  # Hidden layer dimension (128)
        self.dropout_rate = dropout_rate  # 0.1
        
        self.dense1 = layers.Dense(dff, activation='relu')  # 64 → 128
        self.dense2 = layers.Dense(d_model)  # 128 → 64
        self.dropout = layers.Dropout(dropout_rate)
```

### Mathematical Formula
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```
Where:
- **W₁**: First weight matrix (d_model × dff)
- **W₂**: Second weight matrix (dff × d_model)
- **max(0, ·)**: ReLU activation
- **x**: Input of shape (batch, seq_len, d_model)

### Forward Pass
```python
def call(self, x, training=False):
    # First linear transformation with ReLU
    x = self.dense1(x)  # (batch, seq_len, 64) → (batch, seq_len, 128)
    
    # Dropout for regularization
    x = self.dropout(x, training=training)
    
    # Second linear transformation
    x = self.dense2(x)  # (batch, seq_len, 128) → (batch, seq_len, 64)
    
    return x
```

### What It Does
**Step-by-step example** with d_model=64, dff=128:

**Input**: Transaction embedding [0.2, -0.1, 0.5, ..., 0.3] (64 values)
```
↓ Dense(128) + ReLU
```
**Hidden**: [1.2, 0.0, 0.8, ..., 1.5] (128 values, ReLU zeros negatives)
```
↓ Dropout (training only)
```
**Regularized**: [1.2, 0.0, 0.0, ..., 1.5] (some values randomly dropped)
```
↓ Dense(64)
```
**Output**: [0.4, 0.1, 0.6, ..., 0.2] (64 values)

### Why Two Layers?
1. **Expansion**: First layer expands to higher dimension (64 → 128)
   - Creates more capacity to learn complex patterns
   - Allows richer representations
   
2. **Compression**: Second layer projects back (128 → 64)
   - Returns to model dimension for residual connection
   - Distills learned features

### Position-Wise Processing
"Position-wise" means the same FFN is applied to each position independently:
```
Sequence: [emb₀, emb₁, emb₂, ..., emb_n]
           ↓     ↓     ↓          ↓
         FFN   FFN   FFN  ...   FFN  (same FFN weights)
           ↓     ↓     ↓          ↓
         [out₀, out₁, out₂, ..., out_n]
```

### Role in Transformer
```python
# In TransformerBlock
attn_output = self.attention(x)  # Captures relationships
x = x + attn_output  # Residual connection

ffn_output = self.ffn(x)  # Non-linear transformations
x = x + ffn_output  # Residual connection
```
**Attention** captures "what relates to what"
**FFN** captures "what to do with those relationships"

## `ResidualConnection` Layer

### Purpose
Implements residual connections with layer normalization (Add & Norm), a fundamental component for training deep networks.

### Architecture
```python
class ResidualConnection(layers.Layer):
    def __init__(self, d_model, dropout_rate=0.1):
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)
```

### Forward Pass
```python
def call(self, x, sublayer_output, training=False):
    # Apply dropout to sublayer output
    sublayer_output = self.dropout(sublayer_output, training=training)
    
    # Add residual connection and normalize
    return self.layer_norm(x + sublayer_output)
```

### Mathematical Formula
```
Output = LayerNorm(x + Dropout(Sublayer(x)))
```

### What It Does

**Example Flow**:
```
Original Input (x):     [1.0, 2.0, 3.0, 4.0]
                             ↓
                        Sublayer (e.g., Attention)
                             ↓
Sublayer Output:        [0.5, 0.3, -0.2, 0.1]
                             ↓
                        Dropout (training)
                             ↓
Dropped Output:         [0.5, 0.0, -0.2, 0.1]
                             ↓
                        Add Residual (x + output)
                             ↓
Sum:                    [1.5, 2.0, 2.8, 4.1]
                             ↓
                        Layer Normalization
                             ↓
Final Output:           [0.2, 0.5, 0.3, 0.9]  (normalized)
```

### Why Residual Connections?

#### 1. Gradient Flow
Without residual:
```
Input → Layer₁ → Layer₂ → ... → Layer_n → Output
```
Gradients must flow through all layers (vanishing gradient problem)

With residual:
```
Input → Layer₁ → Layer₂ → ... → Layer_n → Output
  ↓_________________________________↑
         (Shortcut connection)
```
Gradients can flow directly through shortcuts

#### 2. Identity Preservation
If a layer learns nothing useful, residual connection preserves the input:
```
output = x + 0 = x  (identity)
```
Network can always fall back to identity mapping

#### 3. Deep Network Training
Enables training very deep networks:
- Original ResNet paper: 152 layers
- Transformers: 12-24 layers common, up to 96+ in large models

### Layer Normalization

#### What It Does
Normalizes across the feature dimension (d_model):
```python
# For each sample and position
mean = average(features)  # Mean of 64 values
variance = var(features)  # Variance of 64 values
normalized = (features - mean) / sqrt(variance + epsilon)
output = gamma * normalized + beta  # Learnable scaling and shifting
```

#### Why Layer Norm?
- **Stabilizes training**: Keeps activations in reasonable range
- **Speeds up training**: Reduces internal covariate shift
- **Improves generalization**: Acts as regularization

#### Batch Norm vs Layer Norm
**Batch Normalization**: Normalizes across batch dimension
- Works well for CNNs
- Problematic for variable-length sequences

**Layer Normalization**: Normalizes across feature dimension
- Works well for transformers
- Consistent regardless of batch size

### Integration Example
```python
class TransformerBlock(layers.Layer):
    def call(self, inputs, training=False):
        # Use residual connection with attention
        attn_output = self.attention(inputs)
        residual_conn = ResidualConnection(d_model=64, dropout_rate=0.1)
        out1 = residual_conn(inputs, attn_output, training=training)
        
        # Use residual connection with FFN
        ffn_output = self.ffn(out1, training=training)
        out2 = residual_conn(out1, ffn_output, training=training)
        
        return out2
```

## Design Patterns

### Pre-Norm vs Post-Norm

**Post-Norm** (used in original Transformer):
```python
out = LayerNorm(x + Sublayer(x))
```

**Pre-Norm** (often better for deep models):
```python
out = x + Sublayer(LayerNorm(x))
```

Our implementation uses **Post-Norm** for consistency with original paper.

### Dropout Placement
```python
sublayer_output = self.dropout(sublayer_output, training=training)
return self.layer_norm(x + sublayer_output)
```
Dropout is applied **before** the residual addition, following standard practice.

## Configuration
```python
config = {
    'd_model': 64,      # Model dimension
    'dff': 128,         # FFN hidden size (typically 2-4× d_model)
    'dropout_rate': 0.1 # Dropout probability
}
```

## Common Dimensions

| Model | d_model | dff | Ratio |
|-------|---------|-----|-------|
| Our model | 64 | 128 | 2× |
| BERT-Base | 768 | 3072 | 4× |
| GPT-2 | 768 | 3072 | 4× |
| Transformer (original) | 512 | 2048 | 4× |

**Typical ratio**: dff = 4 × d_model, but we use 2× for efficiency.
