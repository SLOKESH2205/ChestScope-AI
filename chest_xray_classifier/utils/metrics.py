"""Dashboard metric loading and formatting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chest_xray_classifier.utils.evaluation_pipeline import (
    CHARTS_DIR,
    CONFUSION_DIR,
    MODEL_COMPARISON_CSV,
    MODEL_METRICS_CSV,
    evaluate_all_models,
    load_saved_model_metrics,
)
from chest_xray_classifier.utils.model_loader import MODEL_SPECS, resolve_model_path


PRIMARY_CARD_METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "Balanced Accuracy",
    "Sensitivity",
    "Specificity",
    "ROC AUC",
]

SECONDARY_CARD_METRICS = [
    "Positive Predictive Value",
    "Negative Predictive Value",
    "Log Loss",
    "Matthews Correlation Coefficient",
    "Cohen Kappa",
    "Avg Prediction Confidence",
    "Avg Inference Time (ms/image)",
    "Model Size (MB)",
]


def _normalize_single(model_name: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"Model": model_name, "Status": "Missing"}

    metrics = payload.get("metrics", payload)
    normalized = {"Model": model_name, "Status": payload.get("Status", metrics.get("Status", "Loaded"))}
    normalized.update(metrics)
    normalized["per_class"] = payload.get("per_class", {})
    normalized["confusion_matrix"] = payload.get("confusion_matrix", [])
    normalized["classification_report"] = payload.get("classification_report", {})
    normalized["roc_curves"] = payload.get("roc_curves", {})
    normalized["model_key"] = payload.get("model_key", MODEL_SPECS[model_name]["key"])
    return normalized


def compute_metrics(model_name: str, force: bool = False, train_missing: bool = False) -> dict[str, Any]:
    """Load one model's saved metrics, or regenerate the full suite when needed."""
    payload = None if force else load_saved_model_metrics(model_name)
    if payload is None:
        if resolve_model_path(model_name) is None and not train_missing:
            return {"Model": model_name, "Status": "Missing"}
        evaluate_all_models(train_missing=train_missing)
        payload = load_saved_model_metrics(model_name)
    return _normalize_single(model_name, payload)


def load_metrics_for_all(force: bool = False, train_missing: bool = False) -> pd.DataFrame:
    """Return the row-wise comparison dataframe for all registered models."""
    if force or not MODEL_METRICS_CSV.exists():
        evaluate_all_models(train_missing=train_missing)
    if MODEL_METRICS_CSV.exists():
        return pd.read_csv(MODEL_METRICS_CSV)

    rows = []
    for model_name in MODEL_SPECS:
        rows.append(compute_metrics(model_name, force=force, train_missing=train_missing))
    return pd.DataFrame(rows)


def load_comparison_table(force: bool = False, train_missing: bool = False) -> pd.DataFrame:
    """Return the metric-by-model comparison table."""
    if force or not MODEL_COMPARISON_CSV.exists():
        evaluate_all_models(train_missing=train_missing)
    if MODEL_COMPARISON_CSV.exists():
        return pd.read_csv(MODEL_COMPARISON_CSV)
    return pd.DataFrame()


def chart_path(filename: str) -> Path:
    """Return a generated chart path."""
    return CHARTS_DIR / filename


def confusion_matrix_path(model_key: str) -> Path:
    """Return a generated confusion matrix image path."""
    return CONFUSION_DIR / f"{model_key}_confusion_matrix.png"


def format_metric_value(label: str, value: Any) -> str:
    """Format metrics for cards and tables."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if label in {"Avg Prediction Confidence"}:
        return f"{float(value):.2%}"
    if label in {"Avg Inference Time (ms/image)"}:
        return f"{float(value):.2f} ms"
    if label in {"Total Evaluation Time (s)"}:
        return f"{float(value):.2f} s"
    if label in {"Model Size (MB)"}:
        return f"{float(value):.2f} MB"
    if label in {"Parameters", "Trainable Parameters"}:
        return f"{int(value):,}"
    return f"{float(value):.4f}"


def render_metrics_cards(st_module, metrics: dict[str, Any]) -> None:
    """Render primary and secondary metric cards."""
    for group in [PRIMARY_CARD_METRICS, SECONDARY_CARD_METRICS]:
        columns = st_module.columns(4)
        for index, label in enumerate(group):
            with columns[index % 4]:
                st_module.markdown(
                    (
                        '<div class="metric-card">'
                        f"<span>{label}</span>"
                        f"<strong>{format_metric_value(label, metrics.get(label))}</strong>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
