"""
EfficientNetB0 transfer learning model for Chest X-Ray Classification.

Implements fine-tuning strategy with two training phases:
- Phase 1: Frozen base network, train head only
- Phase 2: Unfreeze and fine-tune full network with lower learning rate
"""

import logging
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

from chest_xray_classifier.config.config import IMG_SHAPE, NUM_CLASSES

logger = logging.getLogger(__name__)


def build_efficientnet(
    input_shape: tuple = IMG_SHAPE,
    num_classes: int = NUM_CLASSES,
    freeze_base: bool = True
) -> models.Model:
    """
    Build EfficientNetB0 transfer learning model.
    
    Architecture:
    - EfficientNetB0 (ImageNet pre-trained, optionally frozen)
    - GlobalAveragePooling2D
    - Dense(256, relu, dropout=0.3)
    - Dense(num_classes, softmax)
    
    Args:
        input_shape: Input image shape (default: (224, 224, 3))
        num_classes: Number of output classes (default: 4)
        freeze_base: If True, base network weights are frozen (Phase 1)
        
    Returns:
        Keras model
    """
    logger.info(f"Building EfficientNetB0 with freeze_base={freeze_base}")
    
    # Load pre-trained EfficientNetB0
    try:
        base_model = EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    except Exception as exc:
        logger.warning(
            "Falling back to randomly initialized EfficientNetB0 because pretrained "
            "weights could not be loaded: %s",
            exc
        )
        base_model = EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights=None
        )
    
    # Freeze base layers if specified
    base_model.trainable = not freeze_base
    logger.info(f"Base model trainable: {base_model.trainable}")
    
    # Build full model
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Scale [0, 1] inputs back to [0, 255] since EfficientNetB0 base has a built-in Rescaling(1/255) layer
        layers.Rescaling(255.0),

        # Base model
        base_model,
        
        # Head
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ], name='efficientnet_b0')
    
    logger.info(f"EfficientNetB0 model built successfully")
    logger.debug(f"Total parameters: {model.count_params():,}")
    logger.debug(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    
    return model


def unfreeze_base_model(model: models.Model, num_layers_to_unfreeze: int = 50) -> models.Model:
    """
    Unfreeze (set trainable=True) the last N layers of base model for fine-tuning.
    
    Args:
        model: EfficientNet model
        num_layers_to_unfreeze: Number of layers from the end to unfreeze (default: 50)
        
    Returns:
        Modified model
    """
    logger.info(f"Unfreezing last {num_layers_to_unfreeze} layers for fine-tuning")
    
    # Locate the nested application model inside the Sequential wrapper.
    base_model = next((layer for layer in model.layers if hasattr(layer, 'layers')), None)
    if base_model is None:
        raise ValueError("EfficientNet base model could not be located for fine-tuning.")
    
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze all but last num_layers_to_unfreeze
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    logger.info(f"Trainable parameters after unfreezing: {trainable_count:,}")
    
    return model


def get_transfer_learning_model_info(model: models.Model) -> dict:
    """
    Get summary info about transfer learning model.
    
    Returns:
        Dict with model statistics
    """
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    return {
        'model_name': model.name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0
    }
