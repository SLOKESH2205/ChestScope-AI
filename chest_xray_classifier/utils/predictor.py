"""Prediction utilities shared by the dashboard tabs."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

from chest_xray_classifier.config.config import CLASSES, IMG_SIZE
from chest_xray_classifier.utils.model_loader import load_all_models, load_single_model


def preprocess_uploaded_image(image: Image.Image) -> np.ndarray:
    """Resize and normalize a PIL image for Keras inference."""
    rgb = image.convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict_image(model: keras.Model, image: Image.Image | np.ndarray, filename: str = "uploaded_image") -> dict:
    """Predict a single image and return formatted probabilities plus latency."""
    model_input = preprocess_uploaded_image(image) if isinstance(image, Image.Image) else image
    start = time.perf_counter()
    probabilities = model.predict(model_input, verbose=0)[0].astype(float)
    inference_ms = (time.perf_counter() - start) * 1000
    class_index = int(np.argmax(probabilities))
    return {
        "filename": Path(filename).name,
        "prediction": CLASSES[class_index],
        "class_index": class_index,
        "confidence": float(probabilities[class_index]),
        "inference_ms": float(inference_ms),
        "probabilities": {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))},
        "image_size": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}",
    }


def _consensus(results: list[dict]) -> dict | None:
    """Majority vote with highest-confidence tie breaking."""
    if not results:
        return None
    counts = Counter(result["prediction"] for result in results)
    max_votes = max(counts.values())
    tied_classes = {label for label, votes in counts.items() if votes == max_votes}
    eligible = [result for result in results if result["prediction"] in tied_classes]
    winner = max(eligible, key=lambda result: result["confidence"])
    return {
        "prediction": winner["prediction"],
        "confidence": winner["confidence"],
        "model": winner["model"],
        "votes": counts[winner["prediction"]],
    }


def compare_models(image: Image.Image | np.ndarray, filename: str = "uploaded_image") -> tuple[list[dict], dict | None]:
    """Run all loaded models on the same image and return consensus."""
    loaded = load_all_models()
    results: list[dict] = []
    model_input = preprocess_uploaded_image(image) if isinstance(image, Image.Image) else image
    for model_name, model in loaded.items():
        result = predict_image(model, model_input, filename)
        result["model"] = model_name
        results.append(result)
    return results, _consensus(results)


def predict_with_model_name(model_name: str, image: Image.Image, filename: str) -> dict:
    """Load a named model and predict an image."""
    model = load_single_model(model_name)
    result = predict_image(model, image, filename)
    result["model"] = model_name
    return result

