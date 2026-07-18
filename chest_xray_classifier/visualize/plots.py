"""
Visualization module for model training, evaluation, and explainability.

Provides Plotly/Matplotlib functions for training curves, confusion matrix,
ROC/PR curves, and other evaluation visualizations.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from chest_xray_classifier.config.config import CLASSES, CLASS_COLORS

logger = logging.getLogger(__name__)


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Training history dict from model.fit()
        save_path: Path to save interactive HTML plot
        
    Returns:
        Plotly figure object
    """
    logger.info("Creating training history plot...")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Loss", "Accuracy")
    )
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss
    fig.add_trace(
        go.Scatter(x=list(epochs), y=history['loss'], name='Train Loss', mode='lines'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(epochs), y=history['val_loss'], name='Val Loss', mode='lines'),
        row=1, col=1
    )
    
    # Accuracy
    fig.add_trace(
        go.Scatter(x=list(epochs), y=history['accuracy'], name='Train Accuracy', mode='lines'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=list(epochs), y=history['val_accuracy'], name='Val Accuracy', mode='lines'),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    
    fig.update_layout(
        title="Training History",
        hovermode="x unified",
        height=500,
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Training history plot saved to {save_path}")
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    normalize: bool = False
) -> go.Figure:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        save_path: Path to save plot
        normalize: Whether to normalize by true label count
        
    Returns:
        Plotly figure
    """
    logger.info("Creating confusion matrix plot...")
    
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        text_format = '.2%'
    else:
        text_format = 'd'
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=CLASSES,
        y=CLASSES,
        text=cm,
        texttemplate=f'%{text_format}',
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        height=600,
        width=600
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Plot ROC curves for each class (one-vs-rest).
    
    Args:
        y_true: True class labels
        y_pred_probs: Predicted probabilities (num_samples, num_classes)
        save_path: Path to save plot
        
    Returns:
        Plotly figure
    """
    logger.info("Creating ROC curves plot...")
    
    fig = go.Figure()
    
    y_true_onehot = np.eye(len(CLASSES))[y_true]
    
    for i, class_name in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{class_name} (AUC={roc_auc:.3f})",
            mode='lines'
        ))
    
    # Diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name="Random",
        mode='lines',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=600,
        width=700
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"ROC curves saved to {save_path}")
    
    return fig


def plot_pr_curves(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Plot Precision-Recall curves for each class.
    
    Args:
        y_true: True class labels
        y_pred_probs: Predicted probabilities
        save_path: Path to save plot
        
    Returns:
        Plotly figure
    """
    logger.info("Creating PR curves plot...")
    
    fig = go.Figure()
    
    y_true_onehot = np.eye(len(CLASSES))[y_true]
    
    for i, class_name in enumerate(CLASSES):
        precision, recall, _ = precision_recall_curve(
            y_true_onehot[:, i],
            y_pred_probs[:, i]
        )
        ap = auc(recall, precision)
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            name=f"{class_name} (AP={ap:.3f})",
            mode='lines'
        ))
    
    fig.update_layout(
        title="Precision-Recall Curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=600,
        width=700
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"PR curves saved to {save_path}")
    
    return fig


def plot_confidence_distribution(
    predictions: np.ndarray,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Plot distribution of prediction confidence scores.
    
    Args:
        predictions: Array of max class probabilities
        save_path: Path to save plot
        
    Returns:
        Plotly figure
    """
    logger.info("Creating confidence distribution plot...")
    
    fig = go.Figure(data=[
        go.Histogram(x=predictions, nbinsx=30, name="Confidence")
    ])
    
    fig.add_vline(x=np.mean(predictions), line_dash="dash", name="Mean")
    fig.add_vline(x=np.median(predictions), line_dash="dot", name="Median")
    
    fig.update_layout(
        title="Confidence Score Distribution",
        xaxis_title="Confidence",
        yaxis_title="Frequency",
        height=500
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Confidence distribution saved to {save_path}")
    
    return fig


def plot_uncertainty_distribution(
    uncertainties: list,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Plot distribution of uncertainty estimates.
    
    Args:
        uncertainties: List of total uncertainty scores
        save_path: Path to save plot
        
    Returns:
        Plotly figure
    """
    logger.info("Creating uncertainty distribution plot...")
    
    fig = go.Figure(data=[
        go.Histogram(x=uncertainties, nbinsx=30, name="Uncertainty")
    ])
    
    fig.add_vline(x=np.mean(uncertainties), line_dash="dash", name="Mean")
    
    fig.update_layout(
        title="Uncertainty Distribution",
        xaxis_title="Total Uncertainty",
        yaxis_title="Frequency",
        height=500
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Uncertainty distribution saved to {save_path}")
    
    return fig


def plot_class_distribution(
    class_counts: Dict[str, int],
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Plot class distribution as bar chart.
    
    Args:
        class_counts: Dict of class names to counts
        save_path: Path to save plot
        
    Returns:
        Plotly figure
    """
    logger.info("Creating class distribution plot...")
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = [CLASS_COLORS.get(c, '#1f77b4') for c in classes]
    
    fig = go.Figure(data=[
        go.Bar(x=classes, y=counts, marker_color=colors)
    ])
    
    fig.update_layout(
        title="Class Distribution",
        xaxis_title="Class",
        yaxis_title="Count",
        height=500
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Class distribution saved to {save_path}")
    
    return fig
