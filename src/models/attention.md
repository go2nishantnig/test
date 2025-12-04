# attention.py - Attention Mechanisms Documentation

## Overview
This module implements the core attention mechanisms used in transformer architectures. Attention allows the model to focus on relevant parts of the input sequence when processing each element.

## Key Concepts

### Attention Formula
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```
Where:
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What information do I have?"
- **V (Value)**: "What is the actual content?"
- **d_k**: Dimension of key vectors (for scaling)

## Classes

### `MultiHeadSelfAttention`
**Purpose**: Standard multi-head self-attention for transformer encoders.

#### Initialization
```python
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        self.d_model = d_model  # 64
        self.num_heads = num_heads  # 4
        self.depth = d_model // num_heads  # 16
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
```
**What it does**: 
- Sets up 4 attention heads, each with dimension 16 (64/4)
- Creates weight matrices for Query, Key, Value projections
- Final dense layer combines all heads

#### Split Heads
```python
def split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
```
**What it does**: 
- Reshapes from (batch, seq_len, d_model) to (batch, seq_len, num_heads, depth)
- Transposes to (batch, num_heads, seq_len, depth)
- Allows each head to compute attention independently

#### Attention Computation
```python
def call(self, inputs):
    # Linear projections
    q = self.wq(inputs)  # Query
    k = self.wk(inputs)  # Key
    v = self.wv(inputs)  # Value
    
    # Split heads
    q = self.split_heads(q, batch_size)
    k = self.split_heads(k, batch_size)
    v = self.split_heads(v, batch_size)
    
    # Scaled dot-product attention
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
```
**What it does**:
1. **Projects** input to Q, K, V using learned weights
2. **Splits** into multiple heads for parallel attention
3. **Computes** attention scores by multiplying Q and K^T
4. **Scales** by √d_k (=√16=4) to prevent gradient issues
5. **Softmax** normalizes scores to probabilities
6. **Multiplies** attention weights by V to get weighted sum

#### Why Multi-Head?
Different heads can learn different types of relationships:
- **Head 1**: Might focus on transaction amounts
- **Head 2**: Might focus on balance changes
- **Head 3**: Might focus on transaction types
- **Head 4**: Might focus on temporal patterns

### `MaskedMultiHeadAttention`
**Purpose**: Attention with causal masking for transformer decoders.

#### Causal Mask Creation
```python
def create_causal_mask(self, seq_len):
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return mask * MASK_VALUE  # MASK_VALUE = -1e9
```
**What it does**: Creates a mask matrix:
```
Position:    0      1      2      3
    0    [   0,  -inf,  -inf,  -inf]
    1    [   0,     0,  -inf,  -inf]
    2    [   0,     0,     0,  -inf]
    3    [   0,     0,     0,     0]
```
**Purpose**: Prevents position i from attending to positions > i (future tokens).

#### Masked Attention
```python
def call(self, inputs, mask=None):
    # ... compute attention scores ...
    
    if mask is None:
        causal_mask = self.create_causal_mask(seq_len)
        scaled_attention_logits += causal_mask
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
```
**What it does**:
- Adds large negative values (-1e9) to future positions
- After softmax, these become ~0, effectively preventing attention
- Essential for autoregressive models (e.g., language generation)

**Example**: When processing position 1:
- Can attend to positions 0, 1 ✓
- Cannot attend to positions 2, 3, ... ✗

### `CrossModalAttention`
**Purpose**: Enables attention between two different modalities or sequences.

#### Cross-Modal Mechanism
```python
class CrossModalAttention(layers.Layer):
    def call(self, query_input, key_value_input):
        # Query from one modality
        q = self.wq(query_input)
        
        # Key and Value from another modality
        k = self.wk(key_value_input)
        v = self.wv(key_value_input)
```
**What it does**: 
- **Query** comes from modality A (e.g., tabular features)
- **Key/Value** comes from modality B (e.g., image patches)
- Allows modality A to "query" information from modality B

#### Use Cases

**1. Tabular → Image Attention**
```python
tabular_attends_to_image = CrossModalAttention()(tabular_features, image_features)
```
**What it does**: Transaction features query relevant information from QR code patches.
- Example: "Is there a suspicious pattern in the QR code that correlates with this large transfer?"

**2. Image → Tabular Attention**
```python
image_attends_to_tabular = CrossModalAttention()(image_features, tabular_features)
```
**What it does**: QR code patches query transaction context.
- Example: "Given this transaction amount, how should I interpret this QR pattern?"

**3. Encoder-Decoder Attention**
In sequence-to-sequence models:
```python
decoder_attends_to_encoder = CrossModalAttention()(decoder_state, encoder_output)
```
**What it does**: Decoder queries encoder for relevant source information.

## Attention Visualization

### Self-Attention Example
For transaction sequence: [amount, type, balance_change]
```
Attention weights might show:
  amount          type        balance
amount  [0.5,        0.3,        0.2     ]  # Amount attends mostly to itself
type    [0.2,        0.6,        0.2     ]  # Type attends mostly to itself
balance [0.4,        0.1,        0.5     ]  # Balance attends to itself and amount
```

### Cross-Modal Attention Example
For tabular → image:
```
Transaction feature    QR Patch 1    Patch 2    Patch 3    ...
Large amount      →    [0.1,         0.7,       0.1,       ...]  # Focuses on suspicious patch
Normal amount     →    [0.3,         0.2,       0.3,       ...]  # Distributed attention
```

## Key Differences

| Feature | Self-Attention | Masked Self-Attention | Cross-Modal Attention |
|---------|----------------|----------------------|----------------------|
| Q, K, V source | Same input | Same input | Q from one, K/V from another |
| Masking | None | Causal mask | None (usually) |
| Use case | Encoder | Decoder | Multimodal fusion |
| Attention direction | Within sequence | Within sequence (causal) | Between sequences |

## Scaling Factor (√d_k)

```python
scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
```
**Why scale?** Without scaling, dot products grow large in magnitude for high dimensions, pushing softmax into regions with extremely small gradients (saturation).

**Example**:
- Without scaling: softmax([50, 48, 2]) ≈ [0.88, 0.12, 0.00] (saturated)
- With √16=4 scaling: softmax([12.5, 12.0, 0.5]) ≈ [0.55, 0.43, 0.02] (better gradients)

## Serialization
```python
@keras.utils.register_keras_serializable(package='FraudDetection')
```
**What it does**: Registers custom layers with Keras so models can be saved and loaded properly. Without this, loading a saved model would fail.
