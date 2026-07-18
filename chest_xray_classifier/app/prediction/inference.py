"""
Inference component.
Handles MC Dropout predictions, uncertainty calculation, and clinical warning threshold checks.
"""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
from tensorflow import keras

from chest_xray_classifier.config.config import CLASSES
from chest_xray_classifier.preprocessing import preprocess_image
from chest_xray_classifier.predict import predict_with_uncertainty

def run_dashboard_inference(
    model: keras.Model,
    image_path: str | Path,
    model_name: str,
    threshold: float = 0.60
) -> dict:
    """
    Run prediction on the uploaded scan with Monte Carlo Dropout uncertainty estimation.
    """
    # 1. Preprocess
    preprocessed_img = preprocess_image(image_path)
    
    # 2. Timing
    start_time = time.perf_counter()
    probs_dict, confidence, uncertainty = predict_with_uncertainty(model, preprocessed_img, num_mc_passes=15)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    
    # 3. Determine class
    predicted_class = max(probs_dict, key=probs_dict.get)
    
    # 4. Low-confidence alert check based on user threshold
    requires_review = confidence < threshold
    
    if requires_review:
        status_message = f"Prediction confidence ({confidence:.1%}) is below the configured threshold of {threshold:.1%}. Clinical specialist review is recommended."
    else:
        status_message = f"Prediction confidence ({confidence:.1%}) is nominal. Proceeding with standard review."
        
    return {
        "filename": Path(image_path).name,
        "prediction": predicted_class,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "inference_ms": elapsed_ms,
        "probabilities": probs_dict,
        "requires_review": requires_review,
        "status_message": status_message,
        "preprocessed_tensor": preprocessed_img
    }
