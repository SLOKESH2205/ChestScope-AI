import tensorflow as tf
import numpy as np
from typing import Dict, Any
import streamlit as st

# Class names in the order used by the model
CLASS_NAMES = ['Bacterial Pneumonia', 'Covid-19', 'Normal', 'Viral Pneumonia']

@st.cache_resource
def load_model(model_path: str) -> tf.keras.Model:
    """
    Load and cache the Keras model to avoid reloading on every prediction.

    Args:
        model_path: Path to the .keras model file

    Returns:
        Loaded Keras model

    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: For other loading errors
    """
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

def predict(model: tf.keras.Model, preprocessed_image: np.ndarray) -> Dict[str, Any]:
    """
    Make prediction using the loaded model.

    Args:
        model: Loaded Keras model
        preprocessed_image: Preprocessed image array (shape: (1, height, width, 3))

    Returns:
        Dictionary containing prediction results:
        {
            "predicted_class": str,
            "confidence": float,
            "all_probabilities": dict of class_name: probability
        }
    """
    # Make prediction
    predictions = model.predict(preprocessed_image, verbose=0)

    # Get predicted class index and confidence
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index])

    # Create probabilities dictionary
    all_probabilities = {
        class_name: float(prob)
        for class_name, prob in zip(CLASS_NAMES, predictions[0])
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": all_probabilities
    }