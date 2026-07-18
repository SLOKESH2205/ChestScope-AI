"""
SHAP (SHapley Additive exPlanations) for model interpretability.

Uses SHAP DeepExplainer to attribute model predictions to input features,
with visualization of feature importance across the image.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not installed. Some features will be unavailable.")

from chest_xray_classifier.config.config import CLASSES, IMG_SHAPE, IMG_SIZE

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based explainability for medical image classification.
    
    Uses DeepExplainer to compute feature importance and generate
    SHAP value visualizations.
    """
    
    def __init__(self, model: keras.Model, background_data: Optional[np.ndarray] = None):
        """
        Initialize SHAP Explainer.
        
        Args:
            model: Keras model
            background_data: Background data for SHAP (uses zeros if None)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")
        
        self.model = model
        
        # Use provided background or default to zeros
        if background_data is None:
            background_data = np.zeros((10, *IMG_SHAPE), dtype=np.float32)
        
        self.background = background_data
        self.explainer = None
        
        logger.info("SHAP Explainer initialized")
    
    def get_explainer(self):
        """
        Get or create SHAP DeepExplainer instance.
        
        Returns:
            SHAP DeepExplainer
        """
        if self.explainer is None:
            logger.info("Creating SHAP DeepExplainer...")
            self.explainer = shap.DeepExplainer(self.model, self.background)
        
        return self.explainer
    
    def explain_instance(
        self,
        image: np.ndarray,
        class_index: Optional[int] = None
    ) -> dict:
        """
        Generate SHAP explanation for a single image.
        
        Args:
            image: Input image (224, 224, 3) or (1, 224, 224, 3)
            class_index: Class to explain (uses argmax if None)
            
        Returns:
            Dict with SHAP values and prediction info
        """
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        image = image.astype(np.float32)
        
        # Get prediction
        pred = self.model.predict(image, verbose=0)
        pred_class = np.argmax(pred[0])
        pred_confidence = float(np.max(pred[0]))
        
        if class_index is None:
            class_index = pred_class
        
        # Compute SHAP values
        logger.info(f"Computing SHAP values for image (class {class_index})...")
        explainer = self.get_explainer()
        shap_values = explainer.shap_values(image)

        shap_image = self._to_importance_map(shap_values, int(class_index))
        shap_image = shap_image / (np.max(shap_image) + 1e-8)
        
        return {
            'shap_values': shap_image,
            'prediction_class': int(pred_class),
            'prediction_confidence': pred_confidence,
            'explained_class': int(class_index),
            'all_predictions': pred[0].tolist()
        }

    def _to_importance_map(self, shap_values, class_index: int) -> np.ndarray:
        """
        Convert SHAP outputs from different SHAP/TensorFlow versions into a 2D map.

        SHAP may return:
        - a list of arrays, one per class
        - a single array with class logits on the last axis: (N, H, W, C, K)
        - a single batch tensor without explicit class axis: (N, H, W, C)
        """
        if isinstance(shap_values, list):
            class_values = np.asarray(shap_values[class_index])
        else:
            shap_array = np.asarray(shap_values)

            if shap_array.ndim == 5 and shap_array.shape[-1] == len(CLASSES):
                class_values = shap_array[..., class_index]
            elif shap_array.ndim == 5 and shap_array.shape[0] == len(CLASSES):
                class_values = shap_array[class_index]
            else:
                class_values = shap_array

        class_values = np.asarray(class_values)

        if class_values.ndim == 4:
            class_values = class_values[0]

        if class_values.ndim == 3:
            return np.mean(np.abs(class_values), axis=-1)

        if class_values.ndim == 2:
            return np.abs(class_values)

        raise ValueError(
            f"Unsupported SHAP output shape {class_values.shape}; expected a 2D or 3D class explanation."
        )
    
    def explain_batch(
        self,
        images: np.ndarray,
        class_indices: Optional[list] = None
    ) -> list:
        """
        Generate SHAP explanations for batch of images.
        
        Args:
            images: Batch of images (N, 224, 224, 3)
            class_indices: Classes to explain (uses argmax if None)
            
        Returns:
            List of explanation dicts (one per image)
        """
        explanations = []
        
        for i, image in enumerate(images):
            class_idx = class_indices[i] if class_indices else None
            explana = self.explain_instance(image, class_idx)
            explanations.append(explana)
        
        return explanations
    
    def visualize_shap_values(
        self,
        shap_values: np.ndarray,
        image: np.ndarray,
        alpha: float = 0.6
    ) -> np.ndarray:
        """
        Overlay SHAP values on original image for visualization.
        
        Args:
            shap_values: SHAP importance map (224, 224)
            image: Original image (224, 224, 3) in [0, 1]
            alpha: Overlay transparency
            
        Returns:
            Overlaid visualization (224, 224, 3)
        """
        import matplotlib.cm as cm

        shap_values = np.asarray(shap_values)

        # Be defensive here as some SHAP versions can still surface a class axis.
        if shap_values.ndim == 4 and shap_values.shape[-1] == len(CLASSES):
            shap_values = np.mean(np.abs(shap_values), axis=(-1, -2))
        elif shap_values.ndim == 3:
            shap_values = np.mean(np.abs(shap_values), axis=-1)
        elif shap_values.ndim != 2:
            raise ValueError(f"Unsupported SHAP map shape for visualization: {shap_values.shape}")
        
        # Normalize image
        if image.max() > 1:
            image = image / 255.0
        
        # Get colormap
        cmap = cm.get_cmap('hot')
        shap_colored = cmap(shap_values)[:, :, :3]
        
        # Blend
        blended = alpha * shap_colored + (1 - alpha) * image
        blended = (blended * 255).astype(np.uint8)
        
        return blended


def compare_class_explanations(
    explainer: SHAPExplainer,
    image: np.ndarray,
    class_indices: Optional[list] = None
) -> dict:
    """
    Generate SHAP explanations for multiple classes, useful for comparison.
    
    Args:
        explainer: SHAPExplainer instance
        image: Input image (224, 224, 3)
        class_indices: Classes to explain (uses top-3 if None)
        
    Returns:
        Dict with explanations for each class
    """
    # Get prediction to identify top classes
    if len(image.shape) == 3:
        img_batch = np.expand_dims(image, axis=0)
    else:
        img_batch = image
    
    pred = explainer.model.predict(img_batch, verbose=0)
    
    if class_indices is None:
        # Top 3 classes
        class_indices = np.argsort(pred[0])[-3:][::-1]
    
    explanations = {}
    for class_idx in class_indices:
        exp = explainer.explain_instance(image, int(class_idx))
        explanations[CLASSES[int(class_idx)]] = exp
    
    return explanations
