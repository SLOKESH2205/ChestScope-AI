"""Unified training, evaluation, and dashboard artifact pipeline."""

from __future__ import annotations

import builtins
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

builtins.tf = tf

from chest_xray_classifier.config.config import (
    BATCH_SIZE,
    CLASSES,
    CUSTOM_CNN_H5_PATH,
    EFFICIENTNETB0_H5_PATH,
    IMG_SIZE,
    MOBILENETV2_H5_PATH,
    RESULTS_DIR,
    ROOT_CUSTOM_CNN_H5_PATH,
    ROOT_EFFICIENTNETB0_H5_PATH,
    ROOT_MOBILENETV2_H5_PATH,
    SEED,
    TEST_DIR,
    VAL_DIR,
)
from chest_xray_classifier.data.loader import DataLoader
from chest_xray_classifier.train.trainer import Trainer
from chest_xray_classifier.utils.model_loader import (
    MODEL_SPECS,
    iter_existing_model_paths,
    load_model_from_path,
    resolve_model_path,
)


logger = logging.getLogger(__name__)

RESULTS_ROOT = Path(RESULTS_DIR)
PER_MODEL_DIR = RESULTS_ROOT / "model_metrics"
CONFUSION_DIR = RESULTS_ROOT / "confusion_matrices"
CHARTS_DIR = RESULTS_ROOT / "charts"
MODEL_METRICS_CSV = RESULTS_ROOT / "model_metrics.csv"
MODEL_COMPARISON_CSV = RESULTS_ROOT / "model_comparison.csv"
SPLIT_INFO_JSON = RESULTS_ROOT / "evaluation_split_info.json"

MODEL_OUTPUTS = {
    "Custom CNN": {
        "key": "custom_cnn",
        "package_h5": Path(CUSTOM_CNN_H5_PATH),
        "root_h5": Path(ROOT_CUSTOM_CNN_H5_PATH),
        "train_method": "train_custom_cnn",
    },
    "EfficientNetB0": {
        "key": "efficientnetb0",
        "package_h5": Path(EFFICIENTNETB0_H5_PATH),
        "root_h5": Path(ROOT_EFFICIENTNETB0_H5_PATH),
        "train_method": "train_efficientnet",
    },
    "MobileNetV2": {
        "key": "mobilenetv2",
        "package_h5": Path(MOBILENETV2_H5_PATH),
        "root_h5": Path(ROOT_MOBILENETV2_H5_PATH),
        "train_method": "train_mobilenetv2",
    },
}

LOWER_IS_BETTER = {
    "Log Loss",
    "Avg Inference Time (ms/image)",
    "Total Evaluation Time (s)",
    "Model Size (MB)",
}


def ensure_output_dirs() -> None:
    """Create the full dashboard artifact directory structure."""
    for directory in [RESULTS_ROOT, PER_MODEL_DIR, CONFUSION_DIR, CHARTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _to_builtin(value: Any) -> Any:
    """Convert numpy-heavy objects into JSON-safe primitives."""
    if isinstance(value, dict):
        return {str(key): _to_builtin(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        if np.isnan(value):
            return None
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _copy_if_needed(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)


def export_model_artifacts(model_name: str, model: keras.Model | None = None) -> Path:
    """Ensure the canonical H5 artifact exists for a model."""
    output_info = MODEL_OUTPUTS[model_name]
    package_h5 = output_info["package_h5"]
    root_h5 = output_info["root_h5"]
    existing_path = resolve_model_path(model_name)

    if package_h5.exists():
        _copy_if_needed(package_h5, root_h5)
        return package_h5

    if existing_path is None and model is None:
        raise FileNotFoundError(f"No saved model exists yet for {model_name}.")

    if model is None:
        model = load_model_from_path(existing_path)

    model.save(str(package_h5))
    _copy_if_needed(package_h5, root_h5)
    logger.info("Exported %s to %s and %s", model_name, package_h5, root_h5)
    return package_h5


def prepare_shared_test_split() -> tuple[Path, dict[str, Any]]:
    """Resolve the one fair evaluation split used for every model."""
    ensure_output_dirs()

    def _has_images(directory: Path) -> bool:
        return directory.exists() and any(
            path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            for path in directory.rglob("*")
            if path.is_file()
        )

    test_dir = Path(TEST_DIR)
    if _has_images(test_dir):
        split_info = {
            "evaluation_source": "dataset/test",
            "evaluation_dir": str(test_dir),
            "classes": CLASSES,
            "image_size": list(IMG_SIZE),
            "preprocessing": "rescale=1/255",
        }
        SPLIT_INFO_JSON.write_text(json.dumps(split_info, indent=2), encoding="utf-8")
        return test_dir, split_info

    val_dir = Path(VAL_DIR)
    if not _has_images(val_dir):
        raise FileNotFoundError("No evaluation images were found in dataset/test or dataset/validation.")

    split_info = {
        "evaluation_source": "dataset/validation",
        "evaluation_dir": str(val_dir),
        "classes": CLASSES,
        "image_size": list(IMG_SIZE),
        "preprocessing": "rescale=1/255",
        "note": "dataset/test is empty, so dataset/validation is used as the common evaluator split.",
    }
    SPLIT_INFO_JSON.write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    return val_dir, split_info


def build_eval_generator(directory: Path):
    """Create the deterministic generator shared across evaluations."""
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    return datagen.flow_from_directory(
        directory=str(directory),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        seed=SEED,
        classes=CLASSES,
    )


def _train_missing_model(model_name: str) -> keras.Model:
    """Train a missing model on the existing train/validation split."""
    data_loader = DataLoader()
    if not data_loader.verify_dataset_structure():
        raise RuntimeError("Dataset structure is invalid. Training cannot continue.")

    trainer = Trainer(experiment_name="chest_xray_dashboard_suite")
    train_gen = data_loader.get_train_generator()
    val_gen = data_loader.get_val_generator()
    train_method = getattr(trainer, MODEL_OUTPUTS[model_name]["train_method"])
    model, _ = train_method(train_gen, val_gen)
    return model


def get_or_create_model(model_name: str, train_missing: bool = False) -> tuple[keras.Model, Path]:
    """Load a model, or train and export it when required."""
    existing_path = resolve_model_path(model_name)
    if existing_path is None:
        if not train_missing:
            raise FileNotFoundError(f"Missing saved weights for {model_name}.")
        logger.info("Training missing model %s", model_name)
        model = _train_missing_model(model_name)
        saved_path = export_model_artifacts(model_name, model=model)
        return model, saved_path

    last_error = None
    for candidate in iter_existing_model_paths(model_name):
        try:
            model = load_model_from_path(candidate)
            saved_path = export_model_artifacts(model_name, model=model)
            return model, saved_path
        except Exception as exc:  # pragma: no cover - runtime fallback
            last_error = exc
            logger.warning("Failed to load %s from %s: %s", model_name, candidate, exc)

    raise RuntimeError(f"Could not load any saved artifact for {model_name}: {last_error}") from last_error


def _model_size_mb(path: Path) -> float:
    return float(path.stat().st_size / (1024 * 1024))


def _parameter_counts(model: keras.Model) -> tuple[int, int]:
    total = int(model.count_params())
    trainable = int(np.sum([keras.backend.count_params(weight) for weight in model.trainable_weights]))
    return total, trainable


def _medical_vectors(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = float(cm.sum())
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp
    tn = total - (tp + fp + fn)
    return tp, fp, fn, tn


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)


def evaluate_single_model(model_name: str, model: keras.Model, model_path: Path, generator) -> dict[str, Any]:
    """Compute the full evaluator-ready metric set for one model."""
    generator.reset()
    start = time.perf_counter()
    y_prob = model.predict(generator, verbose=0)
    total_eval_time = time.perf_counter() - start

    y_true = generator.classes
    y_pred = np.argmax(y_prob, axis=1)
    y_true_one_hot = np.eye(len(CLASSES))[y_true]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    tp, fp, fn, tn = _medical_vectors(cm)

    sensitivity_per_class = _safe_divide(tp, tp + fn)
    specificity_per_class = _safe_divide(tn, tn + fp)
    ppv_per_class = _safe_divide(tp, tp + fp)
    npv_per_class = _safe_divide(tn, tn + fn)

    total_params, trainable_params = _parameter_counts(model)

    try:
        roc_auc = float(roc_auc_score(y_true_one_hot, y_prob, average="macro", multi_class="ovr"))
    except ValueError:
        roc_auc = None

    metrics = {
        "Model": model_name,
        "Status": "Loaded",
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1 Score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Sensitivity": float(np.mean(sensitivity_per_class)),
        "Specificity": float(np.mean(specificity_per_class)),
        "Positive Predictive Value": float(np.mean(ppv_per_class)),
        "Negative Predictive Value": float(np.mean(npv_per_class)),
        "ROC AUC": roc_auc,
        "Log Loss": float(log_loss(y_true, y_prob, labels=list(range(len(CLASSES))))),
        "Matthews Correlation Coefficient": float(matthews_corrcoef(y_true, y_pred)),
        "Cohen Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "Avg Inference Time (ms/image)": float((total_eval_time / max(len(y_true), 1)) * 1000.0),
        "Total Evaluation Time (s)": float(total_eval_time),
        "Avg Prediction Confidence": float(np.max(y_prob, axis=1).mean()),
        "Parameters": total_params,
        "Trainable Parameters": trainable_params,
        "Model Size (MB)": _model_size_mb(model_path),
    }

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        zero_division=0,
        output_dict=True,
    )

    per_class = {}
    roc_curves = {}
    for class_index, class_name in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_true_one_hot[:, class_index], y_prob[:, class_index])
        per_class[class_name] = {
            "precision": float(report[class_name]["precision"]),
            "recall": float(report[class_name]["recall"]),
            "f1_score": float(report[class_name]["f1-score"]),
            "specificity": float(specificity_per_class[class_index]),
            "sensitivity": float(sensitivity_per_class[class_index]),
            "positive_predictive_value": float(ppv_per_class[class_index]),
            "negative_predictive_value": float(npv_per_class[class_index]),
            "support": int(report[class_name]["support"]),
        }
        roc_curves[class_name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }

    result = {
        "model_name": model_name,
        "model_key": MODEL_SPECS[model_name]["key"],
        "model_path": str(model_path),
        "evaluation_split": str(generator.directory),
        "class_names": CLASSES,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": _to_builtin(report),
        "per_class": per_class,
        "roc_curves": roc_curves,
        "filenames": list(generator.filenames),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
    }

    output_path = PER_MODEL_DIR / f"{MODEL_SPECS[model_name]['key']}_metrics.json"
    output_path.write_text(json.dumps(_to_builtin(result), indent=2), encoding="utf-8")
    return result


def load_saved_model_metrics(model_name: str) -> dict[str, Any] | None:
    """Load a previously generated metrics JSON for one model."""
    path = PER_MODEL_DIR / f"{MODEL_SPECS[model_name]['key']}_metrics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_confusion_matrix_chart(model_result: dict[str, Any]) -> Path:
    """Persist one confusion matrix PNG."""
    cm = np.array(model_result["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 5.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(f"{model_result['model_name']} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    output_path = CONFUSION_DIR / f"{model_result['model_key']}_confusion_matrix.png"
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_roc_chart(model_result: dict[str, Any]) -> Path:
    """Persist one ROC curve PNG."""
    fig, ax = plt.subplots(figsize=(7, 5.8))
    for class_name, curve in model_result["roc_curves"].items():
        ax.plot(curve["fpr"], curve["tpr"], label=class_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(f"{model_result['model_name']} ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    output_path = CHARTS_DIR / f"{model_result['model_key']}_roc_curve.png"
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _comparison_metric_list() -> list[str]:
    return [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Balanced Accuracy",
        "Sensitivity",
        "Specificity",
        "Positive Predictive Value",
        "Negative Predictive Value",
        "ROC AUC",
        "Log Loss",
        "Matthews Correlation Coefficient",
        "Cohen Kappa",
        "Avg Inference Time (ms/image)",
        "Total Evaluation Time (s)",
        "Avg Prediction Confidence",
        "Parameters",
        "Trainable Parameters",
        "Model Size (MB)",
    ]


def build_summary_frames(model_results: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create both row-wise and metric-wise comparison tables."""
    model_rows = []
    for model_name, result in model_results.items():
        row = {"Model": model_name}
        row.update(result["metrics"])
        model_rows.append(row)

    metrics_df = pd.DataFrame(model_rows)

    comparison_rows = []
    for metric_name in _comparison_metric_list():
        row = {"Metric": metric_name}
        metric_values = {}
        for model_name, result in model_results.items():
            value = result["metrics"][metric_name]
            row[model_name] = value
            metric_values[model_name] = value

        best_value = (
            min(metric_values.values()) if metric_name in LOWER_IS_BETTER else max(metric_values.values())
        )
        row["Best Model"] = ", ".join(
            [
                model_name
                for model_name, value in metric_values.items()
                if np.isclose(value, best_value)
            ]
        )
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    return metrics_df, comparison_df


def _save_bar_chart(metrics_df: pd.DataFrame, metric_name: str, output_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(metrics_df["Model"], metrics_df[metric_name], color=["#0ea5e9", "#22c55e", "#f59e0b"])
    ax.set_title(f"{metric_name} Comparison")
    ax.set_ylabel(metric_name)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(metrics_df[metric_name]):
        ax.text(idx, value, f"{value:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / output_name, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _save_radar_chart(metrics_df: pd.DataFrame) -> None:
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "Specificity", "ROC AUC"]
    labels = np.array(metrics)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw={"projection": "polar"})
    palette = ["#0ea5e9", "#22c55e", "#f59e0b"]

    for color, (_, row) in zip(palette, metrics_df.iterrows()):
        values = row[metrics].astype(float).to_list()
        values.append(values[0])
        ax.plot(angles, values, linewidth=2.2, label=row["Model"], color=color)
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Overall Model Strengths Radar Chart")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12))
    fig.savefig(CHARTS_DIR / "overall_strengths_radar_chart.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_comparison_artifacts(model_results: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save CSV tables and chart assets for the dashboard."""
    metrics_df, comparison_df = build_summary_frames(model_results)
    metrics_df.to_csv(MODEL_METRICS_CSV, index=False)
    comparison_df.to_csv(MODEL_COMPARISON_CSV, index=False)

    _save_bar_chart(metrics_df, "Accuracy", "accuracy_comparison.png")
    _save_bar_chart(metrics_df, "Precision", "precision_comparison.png")
    _save_bar_chart(metrics_df, "Recall", "recall_comparison.png")
    _save_bar_chart(metrics_df, "F1 Score", "f1_score_comparison.png")
    _save_bar_chart(metrics_df, "Avg Inference Time (ms/image)", "inference_time_comparison.png")
    _save_radar_chart(metrics_df)
    return metrics_df, comparison_df


def evaluate_all_models(train_missing: bool = False) -> dict[str, dict[str, Any]]:
    """Train missing models if requested, then evaluate every model on one shared split."""
    ensure_output_dirs()
    eval_dir, _ = prepare_shared_test_split()
    generator = build_eval_generator(eval_dir)

    model_results = {}
    for model_name in MODEL_SPECS:
        model, model_path = get_or_create_model(model_name, train_missing=train_missing)
        result = evaluate_single_model(model_name, model, model_path, generator)
        model_results[model_name] = result
        save_confusion_matrix_chart(result)
        save_roc_chart(result)
        generator.reset()

    save_comparison_artifacts(model_results)
    return model_results
