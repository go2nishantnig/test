# Models module
"""
Transformer-based models for multimodal fraud detection.

This package provides modular transformer components organized into separate modules
for better code organization and understanding.

Package Structure:
==================
src/models/
├── __init__.py           # Package exports (this file)
├── attention.py          # Attention mechanism implementations
├── layers.py             # Core transformer layers (FFN, Residual)
├── blocks.py             # Complete transformer blocks
├── embeddings.py         # Embedding layers (PatchEmbedding)
├── transformer.py        # Model builders
└── transformer_model.py  # Legacy single-file module (backward compatible)

Attention Layers (attention.py):
- MultiHeadSelfAttention: Standard multi-head self-attention for encoder
- MaskedMultiHeadAttention: Masked multi-head attention for decoder (causal masking)
- CrossModalAttention: Cross multi-head attention for cross-modal fusion

Core Components (layers.py):
- FeedForward: Position-wise feed-forward network
- ResidualConnection: Residual connection with layer normalization (Add & Norm)

Transformer Blocks (blocks.py):
- TransformerBlock: Complete transformer encoder block
- TransformerDecoderBlock: Complete transformer decoder block with masked attention
- CrossModalTransformerBlock: Cross-modal transformer block for multimodal fusion

Image Processing (embeddings.py):
- PatchEmbedding: Vision Transformer style patch embedding

Complete Models (transformer.py):
- MultimodalFraudDetectionTransformer: Multimodal model combining tabular and image data
- FraudDetectionTransformer: Tabular-only model (backward compatible)
"""

# Import from new modular structure
from src.models.attention import (
    MultiHeadSelfAttention,
    MaskedMultiHeadAttention,
    CrossModalAttention,
    MASK_VALUE,
)

from src.models.layers import (
    FeedForward,
    ResidualConnection,
)

from src.models.blocks import (
    TransformerBlock,
    TransformerDecoderBlock,
    CrossModalTransformerBlock,
)

from src.models.embeddings import (
    PatchEmbedding,
)

from src.models.transformer import (
    MultimodalFraudDetectionTransformer,
    FraudDetectionTransformer,
)

__all__ = [
    # Constants
    'MASK_VALUE',
    
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
