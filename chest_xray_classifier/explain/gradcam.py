"""
Gradient-weighted Class Activation Mapping (Grad-CAM) for model explainability.

Provides visual explanations of model predictions by highlighting
regions of input images that influence the predicted class.
"""

import logging
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from chest_xray_classifier.config.config import IMG_SIZE

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Computes Grad-CAM visualizations for CNN predictions.
    
    Shows which regions of an image are important for the model's decision.
    """
    
    def __init__(self, model: keras.Model, layer_name: str = None):
        """
        Initialize GradCAM.
        
        Args:
            model: Keras model
            layer_name: Name of layer to visualize (uses last conv layer if None)
        """
        self.model = model
        self._ensure_model_ready()
        
        # Find last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
        
        self.layer_name = layer_name
        logger.info(f"GradCAM initialized with layer: {layer_name}")

    def _ensure_model_ready(self):
        """Ensure loaded models have concrete inputs/outputs before Grad-CAM runs."""
        try:
            if not self.model.built:
                dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
                self.model(dummy, training=False)
                return

            _ = self.model.output
        except Exception:
            dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
            self.model(dummy, training=False)
    
    def compute_gradcam(
        self,
        image: np.ndarray,
        pred_index: int = None,
        eps: float = 1e-8
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for an image.
        
        Args:
            image: Input image (224, 224, 3) or (1, 224, 224, 3)
            pred_index: Class index to compute CAM for (uses argmax if None)
            eps: Small epsilon for numerical stability
            
        Returns:
            Heatmap of shape (224, 224)
        """
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        image = tf.cast(image, tf.float32)
        
        # Create model that outputs both predictions and conv layer output
        grad_model = keras.Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.outputs[0]
            ]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            class_channel = predictions[:, pred_index]
        
        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            logger.warning(f"Failed to compute gradients for class {pred_index}")
            return np.zeros((IMG_SIZE[0], IMG_SIZE[1]), dtype=np.float32)
        
        # Global average pooling on gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU and normalize
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / (tf.reduce_max(heatmap) + eps)
        
        # Resize to original image size
        heatmap = tf.image.resize(
            tf.expand_dims(heatmap, axis=-1),
            IMG_SIZE
        )
        heatmap = tf.squeeze(heatmap).numpy()
        
        return heatmap
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image.
        
        Args:
            image: Original image (224, 224, 3) with values in [0, 1] or [0, 255]
            heatmap: Grad-CAM heatmap (224, 224) with values in [0, 1]
            alpha: Overlay transparency (0=image only, 1=heatmap only)
            colormap: Matplotlib colormap name
            
        Returns:
            Overlaid image (224, 224, 3) with values in [0, 255]
        """
        import matplotlib.cm as cm
        
        # Ensure image is in [0, 1]
        if image.max() > 1:
            image = image / 255.0
        
        # Get colormap
        try:
            from matplotlib import colormaps
            cmap = colormaps[colormap]
        except ImportError:
            cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
        
        # Blend: alpha * heatmap + (1-alpha) * image
        blended = alpha * heatmap_colored + (1 - alpha) * image
        
        # Convert to [0, 255]
        blended = (blended * 255).astype(np.uint8)
        
        return blended
    
    def explain_prediction(
        self,
        image: np.ndarray,
        pred_index: int = None,
        return_heatmap: bool = True
    ) -> dict:
        """
        Generate Grad-CAM explanation for a model prediction.
        
        Args:
            image: Input image (224, 224, 3)
            pred_index: Class to explain (uses argmax if None)
            return_heatmap: Whether to return raw heatmap
            
        Returns:
            Dict with:
            - 'heatmap': Raw heatmap (224, 224)
            - 'overlay': Overlaid image (224, 224, 3)
            - 'pred_index': Predicted class index
        """
        # Compute heatmap
        heatmap = self.compute_gradcam(image, pred_index)
        
        # Ensure image is in [0, 1]
        if len(image.shape) == 4:
            image = image[0]
        if image.max() > 1:
            image = image / 255.0
        
        # Overlay
        overlay = self.overlay_heatmap(image, heatmap)
        
        # Get prediction if needed
        if pred_index is None:
            preds = self.model.predict(
                np.expand_dims(image, axis=0) if len(image.shape) == 3 else image,
                verbose=0
            )
            pred_index = np.argmax(preds[0])
        
        return {
            'heatmap': heatmap,
            'overlay': overlay,
            'pred_index': int(pred_index)
        }
