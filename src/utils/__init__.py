# Utilities module
"""
Data preprocessing utilities for multimodal fraud detection.

This package provides modular preprocessing classes organized into separate modules
for better code organization and understanding.

Package Structure:
==================
src/utils/
├── __init__.py                   # Package exports (this file)
├── tabular_preprocessor.py       # Tabular data preprocessing
├── image_preprocessor.py         # QR code image preprocessing
├── multimodal_preprocessor.py    # Combined multimodal preprocessing
└── data_preprocessing.py         # Legacy single-file module (backward compatible)

Tabular Preprocessing (tabular_preprocessor.py):
- FraudDataPreprocessor: Handles Online Payments Fraud Detection dataset

Image Preprocessing (image_preprocessor.py):
- QRCodePreprocessor: Handles Benign and Malicious QR Codes dataset

Multimodal Preprocessing (multimodal_preprocessor.py):
- MultimodalDataPreprocessor: Combines tabular and image preprocessing
"""

# Import from new modular structure
from src.utils.tabular_preprocessor import FraudDataPreprocessor
from src.utils.image_preprocessor import QRCodePreprocessor
from src.utils.multimodal_preprocessor import MultimodalDataPreprocessor

__all__ = [
    'FraudDataPreprocessor',
    'QRCodePreprocessor',
    'MultimodalDataPreprocessor',
]
