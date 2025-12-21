# Colab Notebooks

This folder contains Google Colab notebooks for interactive demonstrations and experiments with the Fraud Detection Transformer Model with **Google Drive integration** for dataset storage and persistence.

## Available Notebooks

### 1. `complete_pipeline_demo.ipynb` ⭐ **NEW - Recommended**

**Complete end-to-end pipeline demonstrating the full fraud detection system with Google Drive integration:**

**Pipeline Flow:**
```
Online Payment Fraud Dataset (Google Drive) → Tabular Data
                                                    ↓
                                         FraudDataPreprocessor
                                                    ↓
QR Code Images Dataset (Google Drive) → Image Data |
                                                    ↓
                                         QRCodePreprocessor
                                                    ↓
                                    MultimodalDataPreprocessor
                                                    ↓
                                  Processed Data (Google Drive)
                                                    ↓
                    ┌───────────────────────────────┴───────────────┐
                    ↓                                               ↓
         TabularTransformerEncoder                    VisionTransformerEncoder
                    ↓                                               ↓
                    └───────────────────┬───────────────────────────┘
                                        ↓
                                  Fraud Detection
```

**Key Features:**
- ✅ **Google Drive Integration**: Mount and use Google Drive for all data storage
- ✅ **Complete Data Pipeline**: Load datasets → Preprocess → Store processed data
- ✅ **FraudDataPreprocessor**: Process online payment fraud transactions (tabular data)
- ✅ **QRCodePreprocessor**: Process QR code images (image data)
- ✅ **MultimodalDataPreprocessor**: Combine both modalities with correlation
- ✅ **Processed Data Storage**: Save all preprocessed data to Google Drive
- ✅ **TabularTransformerEncoder Demo**: See how transaction features are encoded
- ✅ **VisionTransformerEncoder Demo**: See how QR codes are processed
- ✅ **Complete Model Demo**: End-to-end multimodal fraud detection
- ✅ **Visualizations**: Comprehensive plots and analysis at each step
- ✅ **Persistence**: All data and models stored in Google Drive

**What You'll Learn:**
- Complete data preprocessing pipeline for multimodal fraud detection
- How to use Google Drive for dataset storage in Colab
- FraudDataPreprocessor for tabular transaction data
- QRCodePreprocessor for QR code image data
- MultimodalDataPreprocessor for combining modalities
- TabularTransformerEncoder architecture and operation
- VisionTransformerEncoder (ViT) architecture and operation
- Cross-modal fusion for multimodal learning
- End-to-end inference pipeline

### 2. `transformer_encoders_demo.ipynb`

**Demonstrates the core encoder components with Google Drive integration:**

- **Google Drive Setup**: Mount and configure Google Drive paths
- **Data Loading**: Load datasets from Google Drive or generate synthetic data
- **Tabular Transformer Encoder**: Shows how transaction features are processed through self-attention layers
  - Input: Transaction features (amount, distance, merchant info, etc.)
  - Architecture: Feature projection → Positional encoding → Transformer blocks
  - Output: Contextualized feature representations

- **Vision Transformer Encoder**: Shows how QR code images are processed using patch-based encoding
  - Input: QR code images (128x128 RGB)
  - Architecture: Patch extraction → Patch embedding → Positional encoding → Transformer blocks  
  - Output: Encoded patch features

**Key Features:**
- ✅ Google Drive integration for data persistence
- ✅ Complete working examples with sample data
- ✅ Detailed visualizations of encoder outputs
- ✅ Side-by-side comparison of both encoders
- ✅ Architecture explanations and statistics
- ✅ Ready to run in Google Colab or locally

## How to Use

### Running in Google Colab

1. Click on the notebook file in GitHub
2. Click "Open in Colab" button (or manually open it in [Google Colab](https://colab.research.google.com/))
3. Uncomment the installation and repository cloning cells
4. Run all cells sequentially

### Running Locally

1. Ensure you have the required dependencies installed:
   ```bash
   pip install tensorflow numpy matplotlib
   ```

2. Open the notebook in Jupyter:
   ```bash
   cd colab
   jupyter notebook transformer_encoders_demo.ipynb
   ```

3. Run all cells sequentially

## What You'll Learn

- How transformer encoders process different data modalities
- The difference between tabular and vision transformer architectures
- How patch-based encoding works for images
- How to visualize transformer encoder outputs
- Statistics and distributions of encoded features

## Related Files

- `src/models/transformer.py` - Model builders including encoder implementations
- `src/models/blocks.py` - Transformer block implementations
- `src/models/embeddings.py` - Patch embedding layer
- `src/models/attention.py` - Multi-head attention mechanisms
- `notebooks/fraud_detection_demo.ipynb` - Complete end-to-end fraud detection demo

## Contributing

To add new Colab notebooks:

1. Create your notebook in this folder
2. Ensure it's well-documented with markdown cells
3. Include sample outputs and visualizations
4. Update this README with a description

---

**For more information, see the main [README](../README.md)**
