"""
Predict module for Chest X-Ray Disease Classification.

Handles single image and batch predictions with Monte Carlo Dropout
for uncertainty estimation and confidence-based warning flags.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from chest_xray_classifier.config.config import CLASSES, CONFIDENCE_THRESHOLD
from chest_xray_classifier.preprocessing import preprocess_image

logger = logging.getLogger(__name__)

def load_cached_model(model_path: str | Path) -> keras.Model:
    """
    Load a Keras model from disk.
    
    Args:
        model_path: Path to the Keras model weight file (.h5 or .keras)
        
    Returns:
        Loaded Keras model
    """
    try:
        model = keras.models.load_model(
            str(model_path),
            compile=False,
            safe_mode=False,
            custom_objects={"tf": tf}
        )
        # Ensure model is built by running a dummy forward pass
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        model(dummy, training=False)
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {model_path}: {exc}")

def predict_with_uncertainty(
    model: keras.Model,
    image_tensor: np.ndarray,
    num_mc_passes: int = 15
) -> Tuple[Dict[str, float], float, float]:
    """
    Make predictions using Monte Carlo Dropout to estimate predictive uncertainty.
    
    Args:
        model: Loaded Keras model
        image_tensor: Preprocessed image tensor of shape (1, 224, 224, 3)
        num_mc_passes: Number of forward passes with dropout enabled
        
    Returns:
        Tuple of:
        - Mean class probabilities: dict of {class_name: probability}
        - Overall confidence: float (max probability)
        - Predictive uncertainty: float (Shannon entropy scaled to [0, 1])
    """
    mc_preds = []
    
    # Run multiple forward passes with training=True to keep dropout active
    for _ in range(num_mc_passes):
        pred = model(image_tensor, training=True)
        mc_preds.append(pred.numpy()[0])
        
    mc_preds = np.array(mc_preds)  # Shape: (num_mc_passes, num_classes)
    
    # Compute mean prediction across passes
    mean_probs = np.mean(mc_preds, axis=0)
    
    # Calculate overall confidence
    class_index = np.argmax(mean_probs)
    confidence = float(mean_probs[class_index])
    
    # Calculate Shannon entropy as predictive uncertainty: H(p) = -sum(p_i * log2(p_i))
    # Normalized to [0, 1] by dividing by log2(num_classes)
    entropy = -np.sum(mean_probs * np.log2(mean_probs + 1e-12))
    max_entropy = np.log2(len(CLASSES))
    uncertainty = float(entropy / max_entropy)
    
    mean_probs_dict = {CLASSES[i]: float(mean_probs[i]) for i in range(len(CLASSES))}
    
    return mean_probs_dict, confidence, uncertainty

def run_single_inference(
    model: keras.Model,
    image_path: str | Path,
    model_name: str,
    confidence_threshold: float = CONFIDENCE_THRESHOLD
) -> Dict[str, Any]:
    """
    Perform complete inference on a single image and return detailed prediction,
    confidence, uncertainty, and execution latency.
    
    Args:
        model: Loaded Keras model
        image_path: Path to the target image
        model_name: Display name of the model
        confidence_threshold: Threshold below which warning is flagged
        
    Returns:
        Inference results dictionary
    """
    start_time = time.perf_counter()
    
    # Preprocess image
    image_tensor = preprocess_image(image_path)
    
    # Run prediction with uncertainty estimation
    probs_dict, confidence, uncertainty = predict_with_uncertainty(model, image_tensor)
    
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    
    predicted_class = max(probs_dict, key=probs_dict.get)
    class_index = CLASSES.index(predicted_class)
    
    # Check if confidence falls below threshold
    requires_review = confidence < confidence_threshold
    status_message = (
        "This prediction has low confidence and should be reviewed by a clinician."
        if requires_review else "Confidence level is acceptable."
    )
    
    return {
        "filename": Path(image_path).name,
        "model_name": model_name,
        "prediction": predicted_class,
        "class_index": class_index,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "probabilities": probs_dict,
        "inference_ms": latency_ms,
        "requires_review": requires_review,
        "status_message": status_message,
        "image_size": "224x224"
    }
