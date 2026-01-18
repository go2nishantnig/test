# Plotting and Visualization Guide

## Overview

The fraud detection system now includes comprehensive plotting and visualization capabilities for both training and inference phases. Plots are automatically generated and saved to organized directories.

## Features

### Training Visualizations
During model training, the following plots are automatically generated:
- **Training Loss & Validation Loss**: Track model convergence over epochs
- **Training Accuracy & Validation Accuracy**: Monitor performance improvements
- **All Metrics Dashboard**: Combined view of all training metrics
- **Confusion Matrix**: Evaluate classification performance on test set
- **ROC Curve**: Assess model's discrimination ability with AUC score
- **Prediction Distribution**: Visualize probability distributions for fraud/non-fraud cases

### Inference Visualizations
During model inference/prediction, the following plots are generated:
- **Confusion Matrix**: Compare predicted vs actual labels
- **ROC Curve**: Evaluate prediction quality with AUC metric
- **Prediction Distribution**: Show distribution of predicted probabilities

## Directory Structure

Plots are saved to the following directory structure:

```
/home/ec2-user/graphs/
├── training/
│   ├── training_loss_<timestamp>.png
│   ├── training_accuracy_<timestamp>.png
│   ├── all_metrics_<timestamp>.png
│   ├── confusion_matrix_<timestamp>.png
│   ├── roc_curve_<timestamp>.png
│   └── prediction_distribution_<timestamp>.png
└── inference/
    ├── confusion_matrix_<timestamp>.png
    ├── roc_curve_<timestamp>.png
    └── prediction_distribution_<timestamp>.png
```

**Note**: If `/home/ec2-user/graphs/` is not accessible (e.g., on local development), the system automatically falls back to `./graphs/` in the current working directory.

## Usage

### Training Mode

When you run the training script, plots are automatically generated at the end:

```bash
# Multimodal training (tabular + image)
python src/train.py --mode multimodal

# Tabular-only training
python src/train.py --mode tabular
```

The training script will:
1. Train the model and save metrics to TensorBoard
2. Evaluate the model on the test set
3. Generate predictions on test data
4. Create and save all visualization plots
5. Print paths to saved plots

### Inference Mode

When you run the prediction script, plots are automatically generated:

```bash
# Multimodal inference
python src/predict.py --mode multimodal

# Tabular-only inference
python src/predict.py --mode tabular
```

The prediction script will:
1. Load the trained model
2. Generate predictions on test data
3. Create and save visualization plots
4. Print paths to saved plots

## Using the Plotting Module

You can also use the plotting utilities directly in your own scripts:

```python
from src.utils.plotting import (
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_prediction_distribution,
    generate_training_report,
    generate_inference_report
)

# Example: Plot training history
history = model.fit(X_train, y_train, ...)
plot_training_history(history, output_dir='/home/ec2-user/graphs/training')

# Example: Generate complete training report
generate_training_report(
    history=history,
    y_true=y_test,
    y_pred=predictions,
    y_pred_proba=probabilities
)

# Example: Generate inference report
generate_inference_report(
    y_true=y_test,
    y_pred=predictions,
    y_pred_proba=probabilities
)
```

## Plot Descriptions

### Training Loss Plot
Shows how the model's loss decreases over training epochs for both training and validation sets. A decreasing trend indicates the model is learning, while divergence between training and validation suggests overfitting.

### Accuracy Plot
Displays model accuracy improvements over epochs. Helps identify when the model reaches peak performance and whether early stopping is triggered.

### Confusion Matrix
A 2x2 matrix showing:
- True Negatives (TN): Correctly predicted non-fraud
- False Positives (FP): Incorrectly predicted as fraud
- False Negatives (FN): Incorrectly predicted as non-fraud
- True Positives (TP): Correctly predicted fraud

### ROC Curve
Receiver Operating Characteristic curve plotting True Positive Rate vs False Positive Rate at various threshold settings. The Area Under Curve (AUC) metric indicates overall model performance:
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random classifier
- AUC > 0.8: Good classifier

### Prediction Distribution
Histogram showing the distribution of predicted fraud probabilities, separated by true class labels. Helps understand model confidence and decision boundaries.

## Customization

### Custom Output Directory

```python
from src.utils.plotting import ensure_plot_directory

# Create custom directory
custom_dir = ensure_plot_directory(base_dir='/custom/path', subdirectory='my_plots')

# Use in plotting functions
plot_training_history(history, output_dir=custom_dir)
```

### Custom Timestamps

```python
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_confusion_matrix(y_true, y_pred, timestamp=f'experiment_{timestamp}')
```

## Dependencies

The plotting functionality requires the following Python packages:
- `matplotlib>=3.7.0`: Core plotting library
- `seaborn>=0.12.0`: Enhanced visualization
- `scikit-learn>=1.3.0`: Metrics computation
- `numpy>=1.24.0`: Numerical operations

These are already included in `requirements.txt`.

## Troubleshooting

### Permission Errors

If you encounter permission errors when saving to `/home/ec2-user/graphs/`:
```
PermissionError: [Errno 13] Permission denied: '/home/ec2-user/graphs'
```

The system automatically falls back to `./graphs/` in your current directory. You'll see a warning message:
```
⚠ Cannot create /home/ec2-user/graphs/training, using fallback: /path/to/current/graphs/training
```

### Memory Issues

If generating plots causes memory issues with large datasets:
1. Reduce the dataset size for visualization
2. Use sampling for plot generation
3. Ensure matplotlib is using the 'Agg' backend (non-interactive)

### Display Issues

If you encounter display-related errors:
```python
import matplotlib
matplotlib.use('Agg')  # This is already set in the plotting module
```

## Integration with TensorBoard

While the plotting utilities generate static PNG files, TensorBoard logs are still created during training for interactive exploration:

```bash
# View TensorBoard logs
tensorboard --logdir /home/ec2-user/model/logs

# Or locally
tensorboard --logdir logs/
```

TensorBoard provides real-time monitoring during training, while the static plots provide a permanent record and are easier to share.

## Best Practices

1. **Review plots after training**: Check for overfitting, underfitting, or other issues
2. **Compare across experiments**: Use timestamps to track different training runs
3. **Share plots in reports**: PNG files are easy to include in documentation
4. **Monitor disk space**: Regular cleanup of old plots may be needed
5. **Version control**: Keep plot directories in `.gitignore` to avoid committing generated files

## Examples

### Example 1: Review Training Results
```bash
# After training
ls /home/ec2-user/graphs/training/
# training_loss_20260118_143022.png
# training_accuracy_20260118_143022.png
# confusion_matrix_20260118_143022.png
# ...

# Copy to local machine for review
scp ec2-user@instance:/home/ec2-user/graphs/training/*.png ./local_plots/
```

### Example 2: Batch Processing
```python
for experiment in experiments:
    model, history = train_model(experiment.config)
    
    # Generate plots with experiment-specific directory
    output_dir = f'/home/ec2-user/graphs/training/{experiment.name}'
    generate_training_report(
        history=history,
        y_true=y_test,
        y_pred=predictions,
        y_pred_proba=probabilities,
        output_dir=output_dir
    )
```

### Example 3: A/B Testing Comparison
```python
# Train two models
model_a, history_a = train_model(config_a)
model_b, history_b = train_model(config_b)

# Generate separate plots
generate_training_report(history_a, ..., 
                        output_dir='/home/ec2-user/graphs/training/model_a')
generate_training_report(history_b, ..., 
                        output_dir='/home/ec2-user/graphs/training/model_b')

# Compare results visually
```

## Future Enhancements

Potential future additions:
- Interactive HTML plots using Plotly
- Animated training progress GIFs
- Model architecture visualization
- Feature importance plots
- Attention weight visualization for transformers
- Real-time plot updates during training
- Automatic plot comparison across runs
