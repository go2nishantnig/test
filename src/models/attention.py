import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Constant for attention masking (large negative value that becomes ~0 after softmax)
MASK_VALUE = -1e9


@keras.utils.register_keras_serializable(package='FraudDetection')
class MultiHeadSelfAttention(layers.Layer):

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
        # Create a lower triangular matrix
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        # Convert to large negative values for masking (will become ~0 after softmax)
        return mask * MASK_VALUE
    
    def call(self, inputs, mask=None):
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
