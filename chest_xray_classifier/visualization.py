"""
Visualization module for Chest X-Ray Disease Classification.

Provides standard plotting utilities for training curves, confusion matrices,
ROC curves, per-class metrics comparison, class distribution, and error analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from chest_xray_classifier.config.config import CLASSES, CLASS_COLORS

logger = logging.getLogger(__name__)

def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: str | Path
) -> Path:
    """Plot and save training and validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    epochs_range = range(1, len(history['loss']) + 1)
    
    # 1. Loss Curves
    axes[0].plot(epochs_range, history['loss'], label='Training Loss', color='#38bdf8', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(epochs_range, history['val_loss'], label='Validation Loss', color='#f97316', linewidth=2)
    axes[0].set_title('Model Loss Over Epochs', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # 2. Accuracy Curves
    axes[1].plot(epochs_range, history['accuracy'], label='Training Accuracy', color='#22c55e', linewidth=2)
    if 'val_accuracy' in history:
        axes[1].plot(epochs_range, history['val_accuracy'], label='Validation Accuracy', color='#a855f7', linewidth=2)
    axes[1].set_title('Model Accuracy Over Epochs', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output_path

def plot_per_class_metrics(
    per_class_metrics: Dict[str, Dict[str, float]],
    output_path: str | Path
) -> Path:
    """Plot and save a comparison bar chart for per-class Precision, Recall, and F1 Score."""
    data = []
    for class_name, metrics in per_class_metrics.items():
        for metric_name in ['precision', 'recall', 'f1_score']:
            data.append({
                "Class": class_name,
                "Metric": metric_name.capitalize().replace("_", " "),
                "Value": metrics[metric_name]
            })
            
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df, x="Class", y="Value", hue="Metric",
        palette=["#38bdf8", "#22c55e", "#ef9f27"], ax=ax
    )
    ax.set_title('Performance Metrics by Diagnosis Class', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Diagnosis Class', fontsize=11, labelpad=10)
    ax.set_ylabel('Score (0.0 - 1.0)', fontsize=11, labelpad=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.xticks(rotation=15)
    plt.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output_path

def plot_class_distribution(
    report: Dict[str, Any],
    output_path: str | Path
) -> Path:
    """Plot and save a bar chart of class distributions across splits."""
    data = []
    for split_name, split_data in report["splits"].items():
        for class_name, count in split_data["class_counts"].items():
            data.append({
                "Split": split_name.capitalize(),
                "Class": class_name,
                "Count": count
            })
            
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df, x="Class", y="Count", hue="Split",
        palette=["#3b82f6", "#a855f7", "#6b7280"], ax=ax
    )
    ax.set_title('Dataset Class Distribution by Split', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Diagnosis Class', fontsize=11, labelpad=10)
    ax.set_ylabel('Number of Images', fontsize=11, labelpad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.xticks(rotation=15)
    plt.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output_path

def generate_misclassified_gallery(
    worst_predictions: List[Dict[str, Any]],
    dataset_dir: str | Path,
    output_path: str | Path
) -> Path:
    """
    Generate and save a visual grid gallery of misclassified X-ray examples
    with confidence scores and labels.
    
    Args:
        worst_predictions: List of prediction dictionaries (e.g. from run_error_analysis)
        dataset_dir: Base validation dataset folder (contains class folders)
        output_path: Output PNG image path
    """
    output_path = Path(output_path)
    dataset_dir = Path(dataset_dir)
    
    # We display at most 9 worst predictions in a 3x3 grid
    num_to_show = min(len(worst_predictions), 9)
    if num_to_show == 0:
        # Save a placeholder empty plot if no errors exist
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No misclassified images detected.", ha='center', va='center', fontsize=14)
        ax.axis('off')
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return output_path
        
    num_cols = 3
    num_rows = int(np.ceil(num_to_show / num_cols))
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 4.5 * num_rows))
    axes = np.array(axes).flatten()  # Ensure 1D array even if 1 row/col
    
    for idx in range(len(axes)):
        if idx >= num_to_show:
            axes[idx].axis('off')
            continue
            
        entry = worst_predictions[idx]
        entry_fn = Path(entry["filename"]).name
        image_file = dataset_dir / entry["actual"] / entry_fn
        
        # Fallback if image file doesn't exist under validation/class
        if not image_file.exists():
            # Check other splits
            for split in ('validation', 'train', 'test'):
                alt_file = dataset_dir.parent / split / entry["actual"] / entry_fn
                if alt_file.exists():
                    image_file = alt_file
                    break
                    
        ax = axes[idx]
        if image_file.exists():
            try:
                img = Image.open(image_file).convert('RGB')
                ax.imshow(img, cmap='gray')
            except Exception:
                ax.text(0.5, 0.5, "Image Error", ha='center', va='center')
        else:
            ax.text(0.5, 0.5, f"Image Not Found\n{entry['filename']}", ha='center', va='center')
            
        title_text = (
            f"File: {entry['filename']}\n"
            f"Actual: {entry['actual']}\n"
            f"Predicted: {entry['predicted']}\n"
            f"Confidence: {entry['confidence']:.2%}"
        )
        ax.set_title(title_text, fontsize=10, color='red', pad=8)
        ax.axis('off')
        
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output_path
