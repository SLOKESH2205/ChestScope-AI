"""
Models factory module for Chest X-Ray Disease Classification.

Provides a unified interface to build different deep learning architectures:
- Custom CNN
- EfficientNetB0
- MobileNetV2
"""

from __future__ import annotations

import logging
from tensorflow import keras

from chest_xray_classifier.model.custom_cnn import build_custom_cnn
from chest_xray_classifier.model.efficientnet import build_efficientnet, unfreeze_base_model as unfreeze_effnet
from chest_xray_classifier.model.mobilenetv2 import build_mobilenetv2, unfreeze_mobilenetv2 as unfreeze_mobilenet

logger = logging.getLogger(__name__)

def get_model_by_name(
    model_name: str,
    num_classes: int = 4,
    input_shape: tuple = (224, 224, 3),
    freeze_base: bool = True
) -> keras.Model:
    """
    Unified model factory method.
    
    Args:
        model_name: Name of the model ('Custom CNN', 'EfficientNetB0', or 'MobileNetV2')
        num_classes: Number of diagnosis classes
        input_shape: Resolution dimensions (224, 224, 3)
        freeze_base: If True, locks transfer learning weights (Phase 1 training)
        
    Returns:
        Built Keras model
    """
    logger.info(f"Factory building model: {model_name} (freeze_base={freeze_base})")
    
    name_lower = model_name.lower().replace(" ", "").replace("_", "")
    
    if "custom" in name_lower or "cnn" in name_lower:
        return build_custom_cnn(
            input_shape=input_shape,
            num_classes=num_classes
        )
    elif "efficient" in name_lower or "effnet" in name_lower:
        return build_efficientnet(
            input_shape=input_shape,
            num_classes=num_classes,
            freeze_base=freeze_base
        )
    elif "mobile" in name_lower:
        return build_mobilenetv2(
            input_shape=input_shape,
            num_classes=num_classes,
            freeze_base=freeze_base
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported models: 'Custom CNN', 'EfficientNetB0', 'MobileNetV2'")
