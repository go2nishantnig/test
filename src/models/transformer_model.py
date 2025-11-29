"""
Multimodal Transformer-based model for fraud detection

This model combines:
1. Tabular data encoder for transaction features (Online Payments Fraud Detection)
2. Vision Transformer encoder for QR code images (Benign vs Malicious)
3. Cross-modal attention for fusion
4. Shared classification head
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


@keras.utils.register_keras_serializable(package='FraudDetection')
class MultiHeadSelfAttention(layers.Layer):
    """Multi-head self-attention layer"""
    
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
class CrossModalAttention(layers.Layer):
    """Cross-modal attention layer for fusing tabular and image features"""
    
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
        Args:
            query_input: Query from one modality
            key_value_input: Key and Value from another modality
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


@keras.utils.register_keras_serializable(package='FraudDetection')
class TransformerBlock(layers.Layer):
    """Transformer block with attention and feed-forward network"""
    
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model),
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=False):
        # Multi-head attention
        attn_output = self.attention(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
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
class CrossModalTransformerBlock(layers.Layer):
    """Cross-modal transformer block for fusing two modalities"""
    
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
        
        self.ffn_tabular = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model),
        ])
        self.ffn_image = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model),
        ])
        
        self.layernorm_tab1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_tab2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_img1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_img2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout_tab1 = layers.Dropout(dropout_rate)
        self.dropout_tab2 = layers.Dropout(dropout_rate)
        self.dropout_img1 = layers.Dropout(dropout_rate)
        self.dropout_img2 = layers.Dropout(dropout_rate)
        
    def call(self, tabular_input, image_input, training=False):
        # Cross attention: tabular attends to image
        cross_attn_tab = self.cross_attn_tab_to_img(tabular_input, image_input)
        cross_attn_tab = self.dropout_tab1(cross_attn_tab, training=training)
        tabular_out = self.layernorm_tab1(tabular_input + cross_attn_tab)
        
        # FFN for tabular
        ffn_tab = self.ffn_tabular(tabular_out)
        ffn_tab = self.dropout_tab2(ffn_tab, training=training)
        tabular_out = self.layernorm_tab2(tabular_out + ffn_tab)
        
        # Cross attention: image attends to tabular
        cross_attn_img = self.cross_attn_img_to_tab(image_input, tabular_input)
        cross_attn_img = self.dropout_img1(cross_attn_img, training=training)
        image_out = self.layernorm_img1(image_input + cross_attn_img)
        
        # FFN for image
        ffn_img = self.ffn_image(image_out)
        ffn_img = self.dropout_img2(ffn_img, training=training)
        image_out = self.layernorm_img2(image_out + ffn_img)
        
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
        batch_size = tf.shape(images)[0]
        
        # Extract patches
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        
        # Reshape patches
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
        max_seq_len = self.config.get('max_sequence_length', 1)
        position_embedding_layer = layers.Embedding(
            input_dim=max_seq_len,
            output_dim=self.config['d_model']
        )
        position_ids = tf.constant([list(range(max_seq_len))])
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
        # Create position indices as a constant
        position_ids = tf.constant([list(range(self.config['max_sequence_length']))])
        position_embedding_layer = layers.Embedding(
            input_dim=self.config['max_sequence_length'],
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
