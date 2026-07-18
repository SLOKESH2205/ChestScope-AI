"""
Evaluation module for Chest X-Ray Disease Classification.

Computes comprehensive metrics, handles error analysis (misclassified gallery,
worst-confidence predictions), and exports validation results to the outputs folder.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from chest_xray_classifier.config.config import CLASSES

logger = logging.getLogger(__name__)

def evaluate_model_on_dataset(
    model: Any,
    generator: Any,
    model_name: str,
    model_size_mb: float
) -> Dict[str, Any]:
    """
    Evaluate a model on a dataset generator, computing all standard metrics,
    confusion matrices, classification reports, and tracking predictions.
    
    Args:
        model: Trained Keras model
        generator: Data generator (should have shuffle=False)
        model_name: Name of the model
        model_size_mb: File size of model in MB
        
    Returns:
        Dictionary containing evaluation results
    """
    generator.reset()
    start_time = time.perf_counter()
    
    # Run predictions
    y_prob = model.predict(generator, verbose=0)
    total_time = time.perf_counter() - start_time
    
    y_true = generator.classes
    y_pred = np.argmax(y_prob, axis=1)
    filenames = generator.filenames
    
    # Overall metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    recall = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    
    # ROC-AUC (handling multi-class)
    y_true_onehot = np.eye(len(CLASSES))[y_true]
    try:
        roc_auc = float(roc_auc_score(y_true_onehot, y_prob, average='macro', multi_class='ovr'))
    except Exception as exc:
        logger.warning(f"Failed to calculate ROC AUC: {exc}")
        roc_auc = 0.0
        
    # Calculate specificity and sensitivity per class from confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp
    tn = float(len(y_true)) - (tp + fp + fn)
    
    sensitivity_per_class = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    specificity_per_class = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) != 0)
    
    sensitivity = float(np.mean(sensitivity_per_class))
    specificity = float(np.mean(specificity_per_class))
    
    avg_inference_ms = float((total_time / len(y_true)) * 1000.0)
    
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc,
        "Specificity": specificity,
        "Sensitivity": sensitivity,
        "Balanced Accuracy": balanced_acc,
        "Matthews Correlation Coefficient": mcc,
        "Avg Inference Time (ms/image)": avg_inference_ms,
        "Model Size (MB)": model_size_mb
    }
    
    # Per-class classification report
    report_dict = classification_report(
        y_true, y_pred, target_names=CLASSES, zero_division=0, output_dict=True
    )
    
    per_class = {}
    for i, class_name in enumerate(CLASSES):
        per_class[class_name] = {
            "precision": float(report_dict[class_name]["precision"]),
            "recall": float(report_dict[class_name]["recall"]),
            "f1_score": float(report_dict[class_name]["f1-score"]),
            "specificity": float(specificity_per_class[i]),
            "sensitivity": float(sensitivity_per_class[i]),
            "support": int(report_dict[class_name]["support"])
        }
        
    # Build list of prediction entries for error analysis
    predictions_log = []
    for idx in range(len(y_true)):
        true_lbl = CLASSES[y_true[idx]]
        pred_lbl = CLASSES[y_pred[idx]]
        conf = float(y_prob[idx][y_pred[idx]])
        
        # Calculate uncertainty as simple normalized Shannon entropy
        prob_vec = y_prob[idx]
        entropy = -np.sum(prob_vec * np.log2(prob_vec + 1e-12))
        uncertainty = float(entropy / np.log2(len(CLASSES)))
        
        predictions_log.append({
            "index": idx,
            "filename": filenames[idx],
            "actual": true_lbl,
            "predicted": pred_lbl,
            "is_correct": bool(y_true[idx] == y_pred[idx]),
            "confidence": conf,
            "uncertainty": uncertainty,
            "probabilities": {CLASSES[c]: float(y_prob[idx][c]) for c in range(len(CLASSES))}
        })
        
    return {
        "model_name": model_name,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "predictions": predictions_log,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist()
    }

def run_error_analysis(
    eval_result: Dict[str, Any],
    output_dir: Path | str
) -> Dict[str, Any]:
    """
    Perform automatic error analysis on evaluation results.
    Identifies misclassified examples and the worst predictions.
    
    Args:
        eval_result: Result dictionary returned by evaluate_model_on_dataset
        output_dir: Folder to save outputs
        
    Returns:
        Analysis summary dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = eval_result["predictions"]
    df_preds = pd.DataFrame(predictions)
    
    # 1. Misclassified gallery list
    misclassified = df_preds[~df_preds["is_correct"]].copy()
    
    # 2. Worst predictions (misclassified cases with HIGHEST confidence)
    worst_predictions = misclassified.sort_values(by="confidence", ascending=False).head(10)
    
    # 3. Most uncertain predictions (cases with HIGHEST uncertainty/entropy)
    uncertain_predictions = df_preds.sort_values(by="uncertainty", ascending=False).head(10)
    
    analysis_report = {
        "model_name": eval_result["model_name"],
        "total_evaluated": len(predictions),
        "correct_count": int(df_preds["is_correct"].sum()),
        "misclassified_count": len(misclassified),
        "error_rate": float(len(misclassified) / len(predictions)),
        "worst_predictions": worst_predictions.to_dict(orient="records"),
        "most_uncertain_predictions": uncertain_predictions.to_dict(orient="records")
    }
    
    # Save JSON report
    model_key = eval_result["model_name"].lower().replace(" ", "_")
    report_path = output_dir / f"{model_key}_error_analysis.json"
    report_path.write_text(json.dumps(analysis_report, indent=2), encoding="utf-8")
    
    logger.info(f"Saved error analysis report to {report_path}")
    return analysis_report

def save_confusion_matrix(
    eval_result: Dict[str, Any],
    output_dir: Path | str
) -> Path:
    """Save confusion matrix heatmap as PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cm = np.array(eval_result["confusion_matrix"])
    model_name = eval_result["model_name"]
    model_key = model_name.lower().replace(" ", "_")
    
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
        cbar=True, annot_kws={"size": 12}
    )
    ax.set_title(f"{model_name} Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    plt.xticks(rotation=25, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = output_dir / f"{model_key}_confusion_matrix.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output_path
