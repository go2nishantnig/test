"""
Plotting utilities for training and inference visualization

This module provides functions to generate and save plots for:
- Training metrics (loss, accuracy curves)
- Confusion matrices
- ROC curves and AUC
- Prediction distributions
"""
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc


def ensure_plot_directory(base_dir='/home/ec2-user/graphs', subdirectory='training'):
    """
    Ensure the plot directory exists
    
    Args:
        base_dir: Base directory for plots (default: /home/ec2-user/graphs)
        subdirectory: Subdirectory for specific plot type (training or inference)
        
    Returns:
        Path to the plot directory
    """
    plot_dir = os.path.join(base_dir, subdirectory)
    try:
        os.makedirs(plot_dir, exist_ok=True)
    except (PermissionError, FileNotFoundError):
        # Fall back to current directory if we don't have permission
        fallback_dir = os.path.join(os.getcwd(), 'graphs', subdirectory)
        print(f"⚠ Cannot create {plot_dir}, using fallback: {fallback_dir}")
        os.makedirs(fallback_dir, exist_ok=True)
        plot_dir = fallback_dir
    return plot_dir


def plot_training_history(history, output_dir=None, timestamp=None):
    """
    Plot training history (loss and metrics over epochs)
    
    Args:
        history: Keras History object or dict with 'loss', 'val_loss', etc.
        output_dir: Directory to save plots (if None, uses default)
        timestamp: Timestamp string for filename (if None, uses current time)
        
    Returns:
        List of saved plot paths
    """
    if output_dir is None:
        output_dir = ensure_plot_directory(subdirectory='training')
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract history data
    if hasattr(history, 'history'):
        history_dict = history.history
    else:
        history_dict = history
    
    saved_plots = []
    
    # Plot training & validation loss
    plt.figure(figsize=(10, 6))
    if 'loss' in history_dict:
        plt.plot(history_dict['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history_dict:
        plt.plot(history_dict['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    loss_path = os.path.join(output_dir, f'training_loss_{timestamp}.png')
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_plots.append(loss_path)
    print(f"✓ Saved loss plot: {loss_path}")
    
    # Plot training & validation accuracy (if available)
    if 'accuracy' in history_dict or 'val_accuracy' in history_dict:
        plt.figure(figsize=(10, 6))
        if 'accuracy' in history_dict:
            plt.plot(history_dict['accuracy'], label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history_dict:
            plt.plot(history_dict['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        acc_path = os.path.join(output_dir, f'training_accuracy_{timestamp}.png')
        plt.savefig(acc_path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_plots.append(acc_path)
        print(f"✓ Saved accuracy plot: {acc_path}")
    
    # Plot all metrics together
    plt.figure(figsize=(12, 8))
    metrics_to_plot = [key for key in history_dict.keys() if not key.startswith('val_')]
    n_metrics = len(metrics_to_plot)
    
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(int(np.ceil(n_metrics / 2)), 2, i)
        plt.plot(history_dict[metric], label=f'Training {metric}', linewidth=2)
        val_metric = f'val_{metric}'
        if val_metric in history_dict:
            plt.plot(history_dict[val_metric], label=f'Validation {metric}', linewidth=2)
        plt.title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=10)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    all_metrics_path = os.path.join(output_dir, f'all_metrics_{timestamp}.png')
    plt.savefig(all_metrics_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_plots.append(all_metrics_path)
    print(f"✓ Saved all metrics plot: {all_metrics_path}")
    
    return saved_plots


def plot_confusion_matrix(y_true, y_pred, output_dir=None, timestamp=None, title='Confusion Matrix'):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_dir: Directory to save plot (if None, uses default)
        timestamp: Timestamp string for filename (if None, uses current time)
        title: Title for the plot
        
    Returns:
        Path to saved plot
    """
    if output_dir is None:
        output_dir = ensure_plot_directory(subdirectory='inference')
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    cm_path = os.path.join(output_dir, f'confusion_matrix_{timestamp}.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix: {cm_path}")
    
    return cm_path


def plot_roc_curve(y_true, y_pred_proba, output_dir=None, timestamp=None):
    """
    Plot ROC curve and calculate AUC
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_dir: Directory to save plot (if None, uses default)
        timestamp: Timestamp string for filename (if None, uses current time)
        
    Returns:
        Path to saved plot
    """
    if output_dir is None:
        output_dir = ensure_plot_directory(subdirectory='inference')
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    roc_path = os.path.join(output_dir, f'roc_curve_{timestamp}.png')
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved ROC curve: {roc_path}")
    
    return roc_path


def plot_prediction_distribution(y_true, y_pred_proba, output_dir=None, timestamp=None):
    """
    Plot distribution of predicted probabilities for fraud and non-fraud cases
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_dir: Directory to save plot (if None, uses default)
        timestamp: Timestamp string for filename (if None, uses current time)
        
    Returns:
        Path to saved plot
    """
    if output_dir is None:
        output_dir = ensure_plot_directory(subdirectory='inference')
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Separate probabilities by true class
    fraud_probs = y_pred_proba[y_true == 1]
    non_fraud_probs = y_pred_proba[y_true == 0]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(non_fraud_probs, bins=50, alpha=0.6, label='Non-Fraud (True)', color='blue', edgecolor='black')
    plt.hist(fraud_probs, bins=50, alpha=0.6, label='Fraud (True)', color='red', edgecolor='black')
    plt.xlabel('Predicted Fraud Probability', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    dist_path = os.path.join(output_dir, f'prediction_distribution_{timestamp}.png')
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved prediction distribution: {dist_path}")
    
    return dist_path


def plot_feature_importance(feature_names, importance_scores, output_dir=None, timestamp=None, top_n=10):
    """
    Plot feature importance
    
    Args:
        feature_names: List of feature names
        importance_scores: Importance scores for each feature
        output_dir: Directory to save plot (if None, uses default)
        timestamp: Timestamp string for filename (if None, uses current time)
        top_n: Number of top features to display
        
    Returns:
        Path to saved plot
    """
    if output_dir is None:
        output_dir = ensure_plot_directory(subdirectory='training')
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Validate inputs
    if len(feature_names) != len(importance_scores):
        raise ValueError(
            f"Length mismatch: feature_names has {len(feature_names)} elements "
            f"but importance_scores has {len(importance_scores)} elements"
        )
    
    # Sort features by importance
    indices = np.argsort(importance_scores)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores = [importance_scores[i] for i in indices]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_scores, color='steelblue')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    
    feat_path = os.path.join(output_dir, f'feature_importance_{timestamp}.png')
    plt.savefig(feat_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved feature importance: {feat_path}")
    
    return feat_path


def generate_training_report(history, y_true, y_pred, y_pred_proba, output_dir=None):
    """
    Generate a comprehensive training report with all plots
    
    Args:
        history: Keras History object
        y_true: True labels for test set
        y_pred: Predicted labels for test set
        y_pred_proba: Predicted probabilities for test set
        output_dir: Directory to save plots (if None, uses default training dir)
        
    Returns:
        Dictionary with paths to all generated plots
    """
    if output_dir is None:
        output_dir = ensure_plot_directory(subdirectory='training')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "=" * 70)
    print("Generating Training Visualizations")
    print("=" * 70)
    
    plots = {}
    
    # Training history plots
    plots['history'] = plot_training_history(history, output_dir, timestamp)
    
    # Confusion matrix
    plots['confusion_matrix'] = plot_confusion_matrix(y_true, y_pred, output_dir, timestamp, 
                                                     title='Training Evaluation - Confusion Matrix')
    
    # ROC curve
    plots['roc_curve'] = plot_roc_curve(y_true, y_pred_proba, output_dir, timestamp)
    
    # Prediction distribution
    plots['prediction_dist'] = plot_prediction_distribution(y_true, y_pred_proba, output_dir, timestamp)
    
    print("\n" + "=" * 70)
    print(f"All plots saved to: {output_dir}")
    print("=" * 70)
    
    return plots


def generate_inference_report(y_true, y_pred, y_pred_proba, output_dir=None):
    """
    Generate a comprehensive inference report with all plots
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        output_dir: Directory to save plots (if None, uses default inference dir)
        
    Returns:
        Dictionary with paths to all generated plots
    """
    if output_dir is None:
        output_dir = ensure_plot_directory(subdirectory='inference')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "=" * 70)
    print("Generating Inference Visualizations")
    print("=" * 70)
    
    plots = {}
    
    # Confusion matrix
    plots['confusion_matrix'] = plot_confusion_matrix(y_true, y_pred, output_dir, timestamp,
                                                     title='Inference - Confusion Matrix')
    
    # ROC curve
    plots['roc_curve'] = plot_roc_curve(y_true, y_pred_proba, output_dir, timestamp)
    
    # Prediction distribution
    plots['prediction_dist'] = plot_prediction_distribution(y_true, y_pred_proba, output_dir, timestamp)
    
    print("\n" + "=" * 70)
    print(f"All plots saved to: {output_dir}")
    print("=" * 70)
    
    return plots
