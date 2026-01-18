"""
Multimodal Transformer-based model for fraud detection

This model combines:
1. Tabular data encoder for transaction features (Online Payments Fraud Detection)
2. Vision Transformer encoder for QR code images (Benign vs Malicious)
3. Cross-modal attention for fusion
4. Shared classification head

Transformer Components:
- MultiHeadSelfAttention: Standard multi-head self-attention for encoder
- MaskedMultiHeadAttention: Masked multi-head attention for decoder (causal masking)
- CrossModalAttention: Cross multi-head attention for cross-modal fusion
- FeedForward: Position-wise feed-forward network
- ResidualConnection: Residual connection with layer normalization (Add & Norm)
- TransformerBlock: Complete transformer encoder block
- CrossModalTransformerBlock: Cross-modal transformer block for multimodal fusion
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# Constant for attention masking (large negative value that becomes ~0 after softmax)
MASK_VALUE = -1e9


@keras.utils.register_keras_serializable(package='FraudDetection')
class FeedForward(layers.Layer):
    """
    Position-wise Feed-Forward Network (FFN)
    
    This is a key component of the Transformer architecture.
    It consists of two linear transformations with a ReLU activation in between:
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Args:
        d_model: Dimension of the model (input and output dimension)
        dff: Dimension of the feed-forward hidden layer (typically 4 * d_model)
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(self, d_model, dff, dropout_rate=0.1, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.d_model = d_model
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        # First linear transformation: d_model -> dff
        self.dense1 = layers.Dense(dff, activation='relu')
        # Second linear transformation: dff -> d_model
        self.dense2 = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False):
        """
        Forward pass through the feed-forward network
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            training: Boolean flag for training mode (affects dropout)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
        })
        return config


@keras.utils.register_keras_serializable(package='FraudDetection')
class ResidualConnection(layers.Layer):
    """
    Residual Connection with Layer Normalization (Add & Norm)
    
    This is a fundamental component of the Transformer architecture.
    It implements: LayerNorm(x + Sublayer(x))
    
    The residual connection helps with:
    - Gradient flow during backpropagation
    - Training deeper networks
    - Preserving information from earlier layers
    
    Args:
        d_model: Dimension of the model
        dropout_rate: Dropout rate applied to sublayer output before addition
    """
    
    def __init__(self, d_model, dropout_rate=0.1, **kwargs):
        super(ResidualConnection, self).__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, sublayer_output, training=False):
        """
        Apply residual connection: LayerNorm(x + Dropout(sublayer_output))
        
        Args:
            x: Original input tensor
            sublayer_output: Output from the sublayer (attention or FFN)
            training: Boolean flag for training mode
            
        Returns:
            Output tensor after residual connection and layer normalization
        """
        sublayer_output = self.dropout(sublayer_output, training=training)
        return self.layer_norm(x + sublayer_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate,
        })
        return config


@keras.utils.register_keras_serializable(package='FraudDetection')
class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-Head Self-Attention Layer
    
    This is a core component of the Transformer encoder.
    It allows the model to jointly attend to information from different 
    representation subspaces at different positions.
    
    Multi-head attention computes:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
    """
    
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        # Linear projections
        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)
        
        # Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], matmul_qk.dtype)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        # Concatenate heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        # Final linear projection
        output = self.dense(output)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
        })
        return config


@keras.utils.register_keras_serializable(package='FraudDetection')
class MaskedMultiHeadAttention(layers.Layer):
    """
    Masked Multi-Head Attention Layer
    
    This is a core component of the Transformer decoder.
    It prevents positions from attending to subsequent positions (causal masking),
    which is essential for autoregressive generation tasks.
    
    The masking ensures that the prediction for position i can depend only on
    the known outputs at positions less than i.
    
    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
    """
    
    def __init__(self, d_model, num_heads, **kwargs):
        super(MaskedMultiHeadAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def create_causal_mask(self, seq_len):
        """
        Create a causal (look-ahead) mask for autoregressive decoding
        
        The mask prevents attention to future positions:
        [[0, -inf, -inf, ...],
         [0,    0, -inf, ...],
         [0,    0,    0, ...],
         ...]
        
        Args:
            seq_len: Length of the sequence
            
        Returns:
            Causal mask tensor of shape (seq_len, seq_len)
        """
        # Create a lower triangular matrix
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        # Convert to large negative values for masking (will become ~0 after softmax)
        return mask * MASK_VALUE
    
    def call(self, inputs, mask=None):
        """
        Forward pass with optional masking
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional external mask. If None, creates causal mask automatically
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Linear projections
        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)
        
        # Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], matmul_qk.dtype)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Apply causal mask if no external mask provided
        if mask is None:
            causal_mask = self.create_causal_mask(seq_len)
            scaled_attention_logits += causal_mask
        else:
            scaled_attention_logits += mask
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        # Concatenate heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        # Final linear projection
        output = self.dense(output)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
        })
        return config


@keras.utils.register_keras_serializable(package='FraudDetection')
class CrossModalAttention(layers.Layer):
    """
    Cross Multi-Head Attention Layer (Cross-Modal Attention)
    
    This layer enables attention between two different modalities or sequences.
    Unlike self-attention where Q, K, V come from the same source,
    cross-attention uses Query from one modality and Key/Value from another.
    
    Use cases:
    - Multimodal learning (tabular + image fusion)
    - Encoder-decoder attention in sequence-to-sequence models
    - Cross-attention between different feature representations
    
    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
    """
    
    def __init__(self, d_model, num_heads, **kwargs):
        super(CrossModalAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.depth = d_model // num_heads
        
        # Query from one modality, Key/Value from another
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, query_input, key_value_input):
        """
        Compute cross-modal attention
        
        Args:
            query_input: Query from one modality (batch_size, seq_len_q, d_model)
            key_value_input: Key and Value from another modality (batch_size, seq_len_kv, d_model)
            
        Returns:
            Output tensor attending from query modality to key/value modality
        """
        batch_size = tf.shape(query_input)[0]
        
        # Linear projections
        q = self.wq(query_input)
        k = self.wk(key_value_input)
        v = self.wv(key_value_input)
        
        # Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], matmul_qk.dtype)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        # Concatenate heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        # Final linear projection
        output = self.dense(output)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
        })
        return config


@keras.utils.register_keras_serializable(package='FraudDetection')
class TransformerBlock(layers.Layer):
    """
    Transformer Encoder Block
    
    This is the standard transformer encoder block consisting of:
    1. Multi-Head Self-Attention sublayer
    2. Residual connection + Layer Normalization (Add & Norm)
    3. Position-wise Feed-Forward Network (FFN) sublayer  
    4. Residual connection + Layer Normalization (Add & Norm)
    
    The block implements: 
    x = LayerNorm(x + MultiHeadAttention(x))
    x = LayerNorm(x + FeedForward(x))
    
    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        dff: Dimension of the feed-forward hidden layer
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        # Multi-Head Self-Attention sublayer
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        
        # Feed-Forward Network sublayer (dropout disabled here, applied after in residual)
        self.ffn = FeedForward(d_model, dff, dropout_rate=0)
        
        # Layer Normalization for residual connections
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout layers applied before residual addition (standard transformer pattern)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=False):
        """
        Forward pass through the transformer block
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_len, d_model)
            training: Boolean flag for training mode
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Multi-Head Self-Attention + Residual Connection
        attn_output = self.attention(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        # Cast to input dtype for mixed precision compatibility
        attn_output = tf.cast(attn_output, inputs.dtype)
        out1 = self.layernorm1(inputs + attn_output)  # Residual connection
        
        # Feed-Forward Network + Residual Connection
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        # Cast to input dtype for mixed precision compatibility
        ffn_output = tf.cast(ffn_output, out1.dtype)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection
        
        return out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
        })
        return config


@keras.utils.register_keras_serializable(package='FraudDetection')
class TransformerDecoderBlock(layers.Layer):
    """
    Transformer Decoder Block
    
    This is the standard transformer decoder block consisting of:
    1. Masked Multi-Head Self-Attention (prevents looking at future tokens)
    2. Residual connection + Layer Normalization
    3. Cross Multi-Head Attention (attends to encoder output)
    4. Residual connection + Layer Normalization
    5. Position-wise Feed-Forward Network
    6. Residual connection + Layer Normalization
    
    The decoder block is used for autoregressive generation tasks.
    
    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        dff: Dimension of the feed-forward hidden layer
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(TransformerDecoderBlock, self).__init__(**kwargs)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        # Masked Multi-Head Self-Attention (for decoder)
        self.masked_attention = MaskedMultiHeadAttention(d_model, num_heads)
        
        # Cross Multi-Head Attention (attends to encoder output)
        self.cross_attention = CrossModalAttention(d_model, num_heads)
        
        # Feed-Forward Network (dropout disabled here, applied after in residual)
        self.ffn = FeedForward(d_model, dff, dropout_rate=0)
        
        # Layer Normalization for residual connections
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout layers applied before residual addition (standard transformer pattern)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)
        
    def call(self, inputs, encoder_output, training=False, mask=None):
        """
        Forward pass through the decoder block
        
        Args:
            inputs: Decoder input tensor (batch_size, target_seq_len, d_model)
            encoder_output: Encoder output tensor (batch_size, source_seq_len, d_model)
            training: Boolean flag for training mode
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, target_seq_len, d_model)
        """
        # Masked Multi-Head Self-Attention + Residual Connection
        masked_attn_output = self.masked_attention(inputs, mask=mask)
        masked_attn_output = self.dropout1(masked_attn_output, training=training)
        # Cast to input dtype for mixed precision compatibility
        masked_attn_output = tf.cast(masked_attn_output, inputs.dtype)
        out1 = self.layernorm1(inputs + masked_attn_output)
        
        # Cross Multi-Head Attention + Residual Connection
        cross_attn_output = self.cross_attention(out1, encoder_output)
        cross_attn_output = self.dropout2(cross_attn_output, training=training)
        # Cast to input dtype for mixed precision compatibility
        cross_attn_output = tf.cast(cross_attn_output, out1.dtype)
        out2 = self.layernorm2(out1 + cross_attn_output)
        
        # Feed-Forward Network + Residual Connection
        ffn_output = self.ffn(out2, training=training)
        ffn_output = self.dropout3(ffn_output, training=training)
        # Cast to input dtype for mixed precision compatibility
        ffn_output = tf.cast(ffn_output, out2.dtype)
        out3 = self.layernorm3(out2 + ffn_output)
        
        return out3
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
        })
        return config


@keras.utils.register_keras_serializable(package='FraudDetection')
class CrossModalTransformerBlock(layers.Layer):
    """
    Cross-Modal Transformer Block for Multimodal Fusion
    
    This block enables bidirectional cross-attention between two modalities.
    It allows each modality to attend to the other, enabling rich feature fusion.
    
    Structure:
    - Tabular modality attends to Image modality (Cross Multi-Head Attention)
    - Residual connection + Layer Normalization
    - Feed-Forward Network for Tabular
    - Residual connection + Layer Normalization
    - Image modality attends to Tabular modality (Cross Multi-Head Attention)
    - Residual connection + Layer Normalization
    - Feed-Forward Network for Image
    - Residual connection + Layer Normalization
    
    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        dff: Dimension of the feed-forward hidden layer
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(CrossModalTransformerBlock, self).__init__(**kwargs)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        # Cross attention: tabular attends to image
        self.cross_attn_tab_to_img = CrossModalAttention(d_model, num_heads)
        # Cross attention: image attends to tabular
        self.cross_attn_img_to_tab = CrossModalAttention(d_model, num_heads)
        
        # Feed-Forward Networks for each modality (dropout disabled, applied after in residual)
        self.ffn_tabular = FeedForward(d_model, dff, dropout_rate=0)
        self.ffn_image = FeedForward(d_model, dff, dropout_rate=0)
        
        # Layer Normalization for residual connections
        self.layernorm_tab1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_tab2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_img1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_img2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout layers applied before residual addition (standard transformer pattern)
        self.dropout_tab1 = layers.Dropout(dropout_rate)
        self.dropout_tab2 = layers.Dropout(dropout_rate)
        self.dropout_img1 = layers.Dropout(dropout_rate)
        self.dropout_img2 = layers.Dropout(dropout_rate)
        
    def call(self, tabular_input, image_input, training=False):
        """
        Forward pass through the cross-modal transformer block
        
        Args:
            tabular_input: Tabular features (batch_size, seq_len_tab, d_model)
            image_input: Image features (batch_size, num_patches, d_model)
            training: Boolean flag for training mode
            
        Returns:
            Tuple of (tabular_output, image_output) with cross-modal attention applied
        """
        # Cross Multi-Head Attention: tabular attends to image + Residual
        cross_attn_tab = self.cross_attn_tab_to_img(tabular_input, image_input)
        cross_attn_tab = self.dropout_tab1(cross_attn_tab, training=training)
        # Cast to input dtype for mixed precision compatibility
        cross_attn_tab = tf.cast(cross_attn_tab, tabular_input.dtype)
        tabular_out = self.layernorm_tab1(tabular_input + cross_attn_tab)  # Residual connection
        
        # Feed-Forward Network for tabular + Residual
        ffn_tab = self.ffn_tabular(tabular_out, training=training)
        ffn_tab = self.dropout_tab2(ffn_tab, training=training)
        # Cast to input dtype for mixed precision compatibility
        ffn_tab = tf.cast(ffn_tab, tabular_out.dtype)
        tabular_out = self.layernorm_tab2(tabular_out + ffn_tab)  # Residual connection
        
        # Cross Multi-Head Attention: image attends to tabular + Residual
        cross_attn_img = self.cross_attn_img_to_tab(image_input, tabular_input)
        cross_attn_img = self.dropout_img1(cross_attn_img, training=training)
        # Cast to input dtype for mixed precision compatibility
        cross_attn_img = tf.cast(cross_attn_img, image_input.dtype)
        image_out = self.layernorm_img1(image_input + cross_attn_img)  # Residual connection
        
        # Feed-Forward Network for image + Residual
        ffn_img = self.ffn_image(image_out, training=training)
        ffn_img = self.dropout_img2(ffn_img, training=training)
        # Cast to input dtype for mixed precision compatibility
        ffn_img = tf.cast(ffn_img, image_out.dtype)
        image_out = self.layernorm_img2(image_out + ffn_img)  # Residual connection
        
        return tabular_out, image_out
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
        })
        return config


@keras.utils.register_keras_serializable(package='FraudDetection')
class PatchEmbedding(layers.Layer):
    """Extract patches from images and embed them (Vision Transformer style)"""
    
    def __init__(self, image_size, patch_size, d_model, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model
        
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.projection = layers.Dense(d_model)
        
        # Learnable position embeddings
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches,
            output_dim=d_model
        )
    
    def call(self, images):
        # Get batch size dynamically
        batch_size = tf.shape(images)[0]
        
        # Extract patches using tf.image.extract_patches
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        
        # Reshape patches: (batch, h_patches, w_patches, patch_dim) -> (batch, num_patches, patch_dim)
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        
        # Project to d_model dimension
        embedded_patches = self.projection(patches)
        
        # Add position embeddings
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        position_embeddings = self.position_embedding(positions)
        embedded_patches = embedded_patches + position_embeddings
        
        return embedded_patches
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'd_model': self.d_model,
        })
        return config


class MultimodalFraudDetectionTransformer:
    """Multimodal Transformer for fraud detection combining tabular and image data"""
    
    def __init__(self, config):
        """
        Initialize the multimodal fraud detection transformer
        
        Args:
            config: Dictionary containing model configuration
        """
        self.config = config
        self.model = None
        
    def build_tabular_encoder(self, inputs):
        """Build encoder for tabular data (transaction features)"""
        # Project input to d_model dimension
        x = layers.Dense(self.config['d_model'])(inputs)
        
        # Add positional encoding (simple learned embeddings)
        # Note: position_ids is created as constant during model build, not each forward pass
        max_seq_len = self.config.get('max_sequence_length', 1)
        position_ids = tf.constant([list(range(max_seq_len))])
        position_embedding_layer = layers.Embedding(
            input_dim=max_seq_len,
            output_dim=self.config['d_model']
        )
        position_embeddings = position_embedding_layer(position_ids)
        x = x + position_embeddings
        
        # Stack transformer blocks for tabular data
        for _ in range(self.config['num_layers']):
            x = TransformerBlock(
                d_model=self.config['d_model'],
                num_heads=self.config['num_heads'],
                dff=self.config['dff'],
                dropout_rate=self.config['dropout_rate']
            )(x)
        
        return x
    
    def build_image_encoder(self, image_inputs):
        """Build encoder for image data (QR codes) using Vision Transformer approach"""
        image_size = self.config.get('image_size', (128, 128))
        patch_size = self.config.get('patch_size', 16)
        
        # Extract and embed patches
        patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            d_model=self.config['d_model']
        )
        x = patch_embedding(image_inputs)
        
        # Stack transformer blocks for image patches
        for _ in range(self.config['num_layers']):
            x = TransformerBlock(
                d_model=self.config['d_model'],
                num_heads=self.config['num_heads'],
                dff=self.config['dff'],
                dropout_rate=self.config['dropout_rate']
            )(x)
        
        return x
    
    def build_model(self):
        """Build the multimodal transformer model"""
        
        # Tabular input (transaction features)
        tabular_inputs = layers.Input(
            shape=(self.config.get('max_sequence_length', 1), self.config['tabular_num_features']),
            name='tabular_features'
        )
        
        # Image input (QR code images)
        image_size = self.config.get('image_size', (128, 128))
        image_channels = self.config.get('image_channels', 3)
        image_inputs = layers.Input(
            shape=(*image_size, image_channels),
            name='image_features'
        )
        
        # Encode each modality
        tabular_encoded = self.build_tabular_encoder(tabular_inputs)
        image_encoded = self.build_image_encoder(image_inputs)
        
        # Cross-modal attention fusion
        for _ in range(self.config.get('cross_modal_layers', 2)):
            cross_modal_block = CrossModalTransformerBlock(
                d_model=self.config['d_model'],
                num_heads=self.config['num_heads'],
                dff=self.config['dff'],
                dropout_rate=self.config['dropout_rate']
            )
            tabular_encoded, image_encoded = cross_modal_block(tabular_encoded, image_encoded)
        
        # Pool representations
        tabular_pooled = layers.GlobalAveragePooling1D()(tabular_encoded)
        image_pooled = layers.GlobalAveragePooling1D()(image_encoded)
        
        # Concatenate multimodal representations
        fused = layers.Concatenate()([tabular_pooled, image_pooled])
        
        # Classification head
        x = layers.Dense(128, activation='relu')(fused)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        # Output layer (binary classification: fraud or not)
        outputs = layers.Dense(1, activation='sigmoid', name='fraud_prediction')(x)
        
        # Create model
        self.model = keras.Model(
            inputs=[tabular_inputs, image_inputs],
            outputs=outputs,
            name='multimodal_fraud_detection_transformer'
        )
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss function"""
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
            ]
        )
        
        return self.model
    
    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()


class FraudDetectionTransformer:
    """Transformer model for fraud detection (tabular data only - backward compatible)"""
    
    def __init__(self, config):
        """
        Initialize the fraud detection transformer
        
        Args:
            config: Dictionary containing model configuration
        """
        self.config = config
        self.model = None
        
    def build_model(self):
        """Build the transformer model"""
        
        # Input layer
        inputs = layers.Input(
            shape=(self.config['max_sequence_length'], self.config['num_features']),
            name='transaction_features'
        )
        
        # Project input to d_model dimension
        x = layers.Dense(self.config['d_model'])(inputs)
        
        # Add positional encoding (simple learned embeddings)
        # Note: position_ids is created as constant during model build, not each forward pass
        max_seq_len = self.config['max_sequence_length']
        position_ids = tf.constant([list(range(max_seq_len))])
        position_embedding_layer = layers.Embedding(
            input_dim=max_seq_len,
            output_dim=self.config['d_model']
        )
        position_embeddings = position_embedding_layer(position_ids)
        x = x + position_embeddings
        
        # Stack transformer blocks
        for _ in range(self.config['num_layers']):
            x = TransformerBlock(
                d_model=self.config['d_model'],
                num_heads=self.config['num_heads'],
                dff=self.config['dff'],
                dropout_rate=self.config['dropout_rate']
            )(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers for classification
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        # Output layer (binary classification: fraud or not)
        outputs = layers.Dense(1, activation='sigmoid', name='fraud_prediction')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='fraud_detection_transformer')
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss function"""
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
            ]
        )
        
        return self.model
    
    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()
