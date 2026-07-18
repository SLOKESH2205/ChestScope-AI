"""Model discovery and cached loading helpers for the Streamlit dashboard."""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import streamlit as st
import tensorflow as tf
from tensorflow import keras

builtins.tf = tf

from chest_xray_classifier.config.config import (
    BASE_DIR,
    CUSTOM_CNN_H5_PATH,
    EFFICIENTNETB0_H5_PATH,
    EFFNET_PATH,
    MOBILENETV2_H5_PATH,
    MOBILENET_PATH,
    MODEL_PATH,
    ROOT_CUSTOM_CNN_H5_PATH,
    ROOT_EFFICIENTNETB0_H5_PATH,
    ROOT_MOBILENETV2_H5_PATH,
)


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "Custom CNN": {
        "key": "custom_cnn",
        "candidates": [
            Path(CUSTOM_CNN_H5_PATH),
            Path(ROOT_CUSTOM_CNN_H5_PATH),
            Path(MODEL_PATH),
            BASE_DIR / "model" / "custom_cnn.h5",
        ],
        "description": "A project-trained convolutional neural network baseline.",
    },
    "EfficientNetB0": {
        "key": "efficientnetb0",
        "candidates": [
            Path(EFFICIENTNETB0_H5_PATH),
            Path(ROOT_EFFICIENTNETB0_H5_PATH),
            BASE_DIR / "model" / "efficientnetb0.h5",
            Path(EFFNET_PATH),
        ],
        "description": "A transfer-learning model using EfficientNetB0 features.",
    },
    "MobileNetV2": {
        "key": "mobilenetv2",
        "candidates": [
            Path(MOBILENETV2_H5_PATH),
            Path(ROOT_MOBILENETV2_H5_PATH),
            BASE_DIR / "model" / "mobilenetv2.h5",
            Path(MOBILENET_PATH),
        ],
        "description": "A lightweight transfer-learning model optimized for fast inference.",
    },
}


def resolve_model_path(name: str) -> Path | None:
    """Return the first existing weight file for a model."""
    for candidate in MODEL_SPECS[name]["candidates"]:
        if Path(candidate).exists():
            return Path(candidate)
    return None


def iter_existing_model_paths(name: str) -> list[Path]:
    """Return every existing saved path for a model in fallback order."""
    return [Path(candidate) for candidate in MODEL_SPECS[name]["candidates"] if Path(candidate).exists()]


def load_model_from_path(path: Path):
    """Load one Keras model from a trusted local path."""
    return keras.models.load_model(
        str(path),
        compile=False,
        safe_mode=False,
        custom_objects={"tf": tf},
    )


def get_model_status() -> list[dict[str, str]]:
    """Scan model folders and report actual availability."""
    rows: list[dict[str, str]] = []
    for name, spec in MODEL_SPECS.items():
        path = resolve_model_path(name)
        rows.append(
            {
                "Model": name,
                "Status": "Loaded" if path else "Missing",
                "Model File": path.name if path else Path(spec["candidates"][0]).name,
            }
        )
    return rows


def get_available_model_names() -> list[str]:
    """Return display names for models with existing weights."""
    return [name for name in MODEL_SPECS if resolve_model_path(name) is not None]


@st.cache_resource(show_spinner=False)
def load_single_model(name: str):
    """Load one Keras model with Streamlit resource caching."""
    paths = iter_existing_model_paths(name)
    if not paths:
        raise FileNotFoundError(f"{name} weights were not found.")
    last_error = None
    for path in paths:
        try:
            return load_model_from_path(path)
        except Exception as exc:  # pragma: no cover - runtime fallback
            last_error = exc
    raise RuntimeError(f"Failed to load any saved artifact for {name}: {last_error}") from last_error


@st.cache_resource(show_spinner=False)
def load_all_models() -> dict[str, keras.Model]:
    """Load every model whose weights are available."""
    loaded = {}
    for name in get_available_model_names():
        try:
            loaded[name] = load_single_model(name)
        except Exception:
            continue
    return loaded


def get_model_file_size_mb(name: str) -> float | None:
    """Return saved model size in MB when the file exists."""
    path = resolve_model_path(name)
    if path is None:
        return None
    return path.stat().st_size / (1024 * 1024)
