"""
Model loader component.
Caches loaded models to prevent duplicate memory usage and speed up inference.
"""

from __future__ import annotations

from pathlib import Path
import streamlit as st
from chest_xray_classifier.predict import load_cached_model

@st.cache_resource
def load_prediction_model(model_path: str | Path):
    """Load and cache Keras model."""
    path = Path(model_path)
    if not path.exists():
        return None
    return load_cached_model(path)
