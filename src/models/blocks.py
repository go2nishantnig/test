import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.attention import MultiHeadSelfAttention, MaskedMultiHeadAttention, CrossModalAttention
from src.models.layers import FeedForward


@keras.utils.register_keras_serializable(package='FraudDetection')
class TransformerBlock(layers.Layer):

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
        # Multi-Head Self-Attention + Residual Connection
        attn_output = self.attention(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Residual connection
        
        # Feed-Forward Network + Residual Connection
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
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
        # Masked Multi-Head Self-Attention + Residual Connection
        masked_attn_output = self.masked_attention(inputs, mask=mask)
        masked_attn_output = self.dropout1(masked_attn_output, training=training)
        out1 = self.layernorm1(inputs + masked_attn_output)
        
        # Cross Multi-Head Attention + Residual Connection
        cross_attn_output = self.cross_attention(out1, encoder_output)
        cross_attn_output = self.dropout2(cross_attn_output, training=training)
        out2 = self.layernorm2(out1 + cross_attn_output)
        
        # Feed-Forward Network + Residual Connection
        ffn_output = self.ffn(out2, training=training)
        ffn_output = self.dropout3(ffn_output, training=training)
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
        # Cross Multi-Head Attention: tabular attends to image + Residual
        cross_attn_tab = self.cross_attn_tab_to_img(tabular_input, image_input)
        cross_attn_tab = self.dropout_tab1(cross_attn_tab, training=training)
        tabular_out = self.layernorm_tab1(tabular_input + cross_attn_tab)  # Residual connection
        
        # Feed-Forward Network for tabular + Residual
        ffn_tab = self.ffn_tabular(tabular_out, training=training)
        ffn_tab = self.dropout_tab2(ffn_tab, training=training)
        tabular_out = self.layernorm_tab2(tabular_out + ffn_tab)  # Residual connection
        
        # Cross Multi-Head Attention: image attends to tabular + Residual
        cross_attn_img = self.cross_attn_img_to_tab(image_input, tabular_input)
        cross_attn_img = self.dropout_img1(cross_attn_img, training=training)
        image_out = self.layernorm_img1(image_input + cross_attn_img)  # Residual connection
        
        # Feed-Forward Network for image + Residual
        ffn_img = self.ffn_image(image_out, training=training)
        ffn_img = self.dropout_img2(ffn_img, training=training)
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
