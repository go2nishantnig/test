"""
Attention mechanisms for transformer models.

This module provides attention layer implementations:
- MultiHeadSelfAttention: Standard multi-head self-attention for encoder
- MaskedMultiHeadAttention: Masked multi-head attention for decoder (causal masking)
- CrossModalAttention: Cross multi-head attention for cross-modal fusion
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Constant for attention masking (large negative value that becomes ~0 after softmax)
MASK_VALUE = -1e9


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
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
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
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
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
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
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
