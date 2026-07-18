"""
Metrics computation functions for model evaluation.

Provides functions to compute confusion matrix, ROC/PR curves,
classification metrics, and generate formatted reports.
"""

import logging
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score
)
import pandas as pd

from chest_xray_classifier.config.config import CLASSES

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred_classes: np.ndarray,
    y_pred_probs: np.ndarray
) -> Dict:
    """
    Compute comprehensive metrics for multi-class classification.
    
    Args:
        y_true: True class labels (class indices)
        y_pred_classes: Predicted class indices
        y_pred_probs: Predicted class probabilities (num_samples, num_classes)
        
    Returns:
        Dict with all metrics:
        - 'accuracy': Overall accuracy
        - 'precision': Macro-averaged precision
        - 'recall': Macro-averaged recall
        - 'f1': Macro-averaged F1-score
        - 'confusion_matrix': Confusion matrix
        - 'per_class': Dict with per-class metrics
    """
    logger.info("Computing classification metrics...")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes, labels=range(len(CLASSES)))
    
    # Compute overall metrics
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred_classes, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred_classes, average='macro', zero_division=0)
    
    # Compute per-class metrics
    per_class_report = classification_report(
        y_true, y_pred_classes,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )
    
    # Extract per-class precision, recall, f1
    per_class_metrics = {
        class_name: {
            'precision': per_class_report[class_name]['precision'],
            'recall': per_class_report[class_name]['recall'],
            'f1_score': per_class_report[class_name]['f1-score'],
            'support': int(per_class_report[class_name]['support'])
        }
        for class_name in CLASSES
    }
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm,
        'per_class': per_class_metrics
    }
    
    logger.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return metrics


def compute_roc_curves(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray
) -> Dict:
    """
    Compute ROC curves and AUC scores for each class (one-vs-rest).
    
    Args:
        y_true: True class labels (class indices)
        y_pred_probs: Predicted probabilities (num_samples, num_classes)
        
    Returns:
        Dict with ROC curves and AUC scores:
        - 'auc_scores': Dict of AUC per class
        - 'roc_curves': Dict of (fpr, tpr) per class
        - 'macro_auc': Macro-averaged AUC
    """
    logger.info("Computing ROC curves...")
    
    # One-hot encode true labels
    y_true_onehot = np.eye(len(CLASSES))[y_true]
    
    roc_curves = {}
    auc_scores = {}
    
    for i, class_name in enumerate(CLASSES):
        try:
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            roc_curves[class_name] = {'fpr': fpr, 'tpr': tpr}
            auc_scores[class_name] = float(roc_auc)
        except Exception as e:
            logger.warning(f"Failed to compute ROC for {class_name}: {e}")
            auc_scores[class_name] = 0.0
    
    macro_auc = np.mean(list(auc_scores.values()))
    
    return {
        'auc_scores': auc_scores,
        'roc_curves': roc_curves,
        'macro_auc': float(macro_auc)
    }


def compute_pr_curves(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray
) -> Dict:
    """
    Compute Precision-Recall curves for each class.
    
    Args:
        y_true: True class labels
        y_pred_probs: Predicted probabilities
        
    Returns:
        Dict with PR curves:
        - 'pr_curves': Dict of (precision, recall) per class
        - 'ap_scores': Dict of Average Precision per class
    """
    logger.info("Computing Precision-Recall curves...")
    
    y_true_onehot = np.eye(len(CLASSES))[y_true]
    
    pr_curves = {}
    ap_scores = {}
    
    for i, class_name in enumerate(CLASSES):
        try:
            precision, recall, _ = precision_recall_curve(
                y_true_onehot[:, i],
                y_pred_probs[:, i]
            )
            
            # Compute AP
            ap = auc(recall, precision)
            
            pr_curves[class_name] = {'precision': precision, 'recall': recall}
            ap_scores[class_name] = float(ap)
        except Exception as e:
            logger.warning(f"Failed to compute PR for {class_name}: {e}")
            ap_scores[class_name] = 0.0
    
    return {
        'pr_curves': pr_curves,
        'ap_scores': ap_scores
    }


def format_metrics_table(metrics: Dict) -> str:
    """
    Format metrics dict into readable table string.
    
    Args:
        metrics: Metrics dictionary from compute_all_metrics()
        
    Returns:
        Formatted string table
    """
    rows = [
        f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}",
        "-" * 75
    ]
    
    for class_name, class_metrics in metrics['per_class'].items():
        row = (
            f"{class_name:<25} "
            f"{class_metrics['precision']:<12.4f} "
            f"{class_metrics['recall']:<12.4f} "
            f"{class_metrics['f1_score']:<12.4f} "
            f"{class_metrics['support']:<10}"
        )
        rows.append(row)
    
    rows.extend([
        "-" * 75,
        f"{'Macro Averages':<25} "
        f"{metrics['precision']:<12.4f} "
        f"{metrics['recall']:<12.4f} "
        f"{metrics['f1']:<12.4f}",
        f"\nOverall Accuracy: {metrics['accuracy']:.4f}"
    ])
    
    return '\n'.join(rows)


def metrics_to_dataframe(metrics: Dict) -> pd.DataFrame:
    """
    Convert per-class metrics to pandas DataFrame.
    
    Args:
        metrics: Metrics dictionary
        
    Returns:
        DataFrame with columns: Class, Precision, Recall, F1-Score, Support
    """
    data = []
    for class_name, class_metrics in metrics['per_class'].items():
        data.append({
            'Class': class_name,
            'Precision': class_metrics['precision'],
            'Recall': class_metrics['recall'],
            'F1-Score': class_metrics['f1_score'],
            'Support': class_metrics['support']
        })
    
    return pd.DataFrame(data)
