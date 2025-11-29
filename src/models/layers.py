"""
Core transformer layers.

This module provides fundamental transformer layer implementations:
- FeedForward: Position-wise feed-forward network (FFN)
- ResidualConnection: Residual connection with layer normalization (Add & Norm)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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
