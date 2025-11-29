"""
Fraud detection transformer model builders.

This module provides complete model implementations:
- FraudDetectionTransformer: Tabular-only model (backward compatible)
- MultimodalFraudDetectionTransformer: Multimodal model combining tabular and image data
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.blocks import TransformerBlock, CrossModalTransformerBlock
from src.models.embeddings import PatchEmbedding


class MultimodalFraudDetectionTransformer:
    """
    Multimodal Transformer for Fraud Detection
    
    This model combines tabular transaction data with QR code images
    using cross-modal attention for comprehensive fraud detection.
    
    Architecture:
    1. Tabular Encoder: Processes transaction features through transformer blocks
    2. Image Encoder: Processes QR code images using Vision Transformer approach
    3. Cross-Modal Fusion: Bidirectional attention between modalities
    4. Classification Head: Dense layers with sigmoid output
    
    Args:
        config: Dictionary containing model configuration including:
            - tabular_num_features: Number of input features for tabular data
            - image_size: Tuple of (height, width) for images
            - image_channels: Number of image channels
            - patch_size: Size of image patches for ViT
            - d_model: Model dimension
            - num_heads: Number of attention heads
            - num_layers: Number of transformer blocks per modality
            - cross_modal_layers: Number of cross-modal attention layers
            - dff: Feed-forward network dimension
            - dropout_rate: Dropout rate
    """
    
    def __init__(self, config):
        """
        Initialize the multimodal fraud detection transformer
        
        Args:
            config: Dictionary containing model configuration
        """
        self.config = config
        self.model = None
        
    def build_tabular_encoder(self, inputs):
        """
        Build encoder for tabular data (transaction features)
        
        Args:
            inputs: Input tensor for tabular features
            
        Returns:
            Encoded tabular features
        """
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
        """
        Build encoder for image data (QR codes) using Vision Transformer approach
        
        Args:
            image_inputs: Input tensor for images
            
        Returns:
            Encoded image features
        """
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
        """
        Build the multimodal transformer model
        
        Returns:
            Compiled Keras model
        """
        
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
        """
        Compile the model with optimizer and loss function
        
        Args:
            learning_rate: Learning rate for Adam optimizer
            
        Returns:
            Compiled Keras model
        """
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
    """
    Transformer Model for Fraud Detection (Tabular Only)
    
    This model provides backward compatibility for tabular-only fraud detection.
    It uses transformer blocks to process transaction features.
    
    Architecture:
    1. Feature Embedding: Projects input features to d_model dimension
    2. Positional Encoding: Adds learnable position embeddings
    3. Transformer Blocks: Stacked self-attention and feed-forward layers
    4. Global Pooling: Aggregates sequence information
    5. Classification Head: Dense layers with sigmoid output
    
    Args:
        config: Dictionary containing model configuration including:
            - num_features: Number of input features
            - max_sequence_length: Maximum sequence length
            - d_model: Model dimension
            - num_heads: Number of attention heads
            - num_layers: Number of transformer blocks
            - dff: Feed-forward network dimension
            - dropout_rate: Dropout rate
    """
    
    def __init__(self, config):
        """
        Initialize the fraud detection transformer
        
        Args:
            config: Dictionary containing model configuration
        """
        self.config = config
        self.model = None
        
    def build_model(self):
        """
        Build the transformer model
        
        Returns:
            Keras model
        """
        
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
        """
        Compile the model with optimizer and loss function
        
        Args:
            learning_rate: Learning rate for Adam optimizer
            
        Returns:
            Compiled Keras model
        """
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
