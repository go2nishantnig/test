"""
Embedding layers for transformer models.

This module provides embedding layer implementations:
- PatchEmbedding: Vision Transformer style patch embedding for images
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package='FraudDetection')
class PatchEmbedding(layers.Layer):
    """
    Patch Embedding Layer (Vision Transformer Style)
    
    Extracts patches from images and embeds them into a vector space.
    This is a key component for processing images in a transformer architecture.
    
    The process:
    1. Divide image into non-overlapping patches
    2. Flatten each patch
    3. Project to d_model dimensions
    4. Add learnable position embeddings
    
    Args:
        image_size: Tuple of (height, width) of input images
        patch_size: Size of each square patch
        d_model: Dimension of the embedding space
    """
    
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
        """
        Extract and embed patches from images
        
        Args:
            images: Input images of shape (batch_size, height, width, channels)
            
        Returns:
            Embedded patches of shape (batch_size, num_patches, d_model)
        """
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
