import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package='FraudDetection')
class FeedForward(layers.Layer):

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

    def __init__(self, d_model, dropout_rate=0.1, **kwargs):
        super(ResidualConnection, self).__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, sublayer_output, training=False):
        sublayer_output = self.dropout(sublayer_output, training=training)
        return self.layer_norm(x + sublayer_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate,
        })
        return config
