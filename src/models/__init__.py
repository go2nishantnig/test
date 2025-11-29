# Models module
"""
Transformer-based models for multimodal fraud detection.

This module provides the following transformer components:

Attention Layers:
- MultiHeadSelfAttention: Standard multi-head self-attention for encoder
- MaskedMultiHeadAttention: Masked multi-head attention for decoder (causal masking)
- CrossModalAttention: Cross multi-head attention for cross-modal fusion

Core Components:
- FeedForward: Position-wise feed-forward network
- ResidualConnection: Residual connection with layer normalization (Add & Norm)

Transformer Blocks:
- TransformerBlock: Complete transformer encoder block
- TransformerDecoderBlock: Complete transformer decoder block with masked attention
- CrossModalTransformerBlock: Cross-modal transformer block for multimodal fusion

Image Processing:
- PatchEmbedding: Vision Transformer style patch embedding

Complete Models:
- MultimodalFraudDetectionTransformer: Multimodal model combining tabular and image data
- FraudDetectionTransformer: Tabular-only model (backward compatible)
"""

from src.models.transformer_model import (
    # Attention layers
    MultiHeadSelfAttention,
    MaskedMultiHeadAttention,
    CrossModalAttention,
    
    # Core components
    FeedForward,
    ResidualConnection,
    
    # Transformer blocks
    TransformerBlock,
    TransformerDecoderBlock,
    CrossModalTransformerBlock,
    
    # Image processing
    PatchEmbedding,
    
    # Complete models
    MultimodalFraudDetectionTransformer,
    FraudDetectionTransformer,
)

__all__ = [
    # Attention layers
    'MultiHeadSelfAttention',
    'MaskedMultiHeadAttention',
    'CrossModalAttention',
    
    # Core components
    'FeedForward',
    'ResidualConnection',
    
    # Transformer blocks
    'TransformerBlock',
    'TransformerDecoderBlock',
    'CrossModalTransformerBlock',
    
    # Image processing
    'PatchEmbedding',
    
    # Complete models
    'MultimodalFraudDetectionTransformer',
    'FraudDetectionTransformer',
]
