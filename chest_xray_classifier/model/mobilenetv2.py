"""
MobileNetV2 transfer learning model for Chest X-Ray Classification.

This model is intentionally lightweight, making it a good second model for
academic comparison against the existing custom CNN.
"""

import logging

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

from chest_xray_classifier.config.config import IMG_SHAPE, NUM_CLASSES

logger = logging.getLogger(__name__)


def build_mobilenetv2(
    input_shape: tuple = IMG_SHAPE,
    num_classes: int = NUM_CLASSES,
    freeze_base: bool = True,
    weights: str = 'imagenet'
) -> models.Model:
    """
    Build a MobileNetV2 transfer learning classifier.

    If ImageNet weights are unavailable in the current environment, the model
    falls back to random initialization so the training pipeline still runs.
    """
    logger.info(
        "Building MobileNetV2 with freeze_base=%s and weights=%s",
        freeze_base,
        weights
    )

    try:
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=weights
        )
    except Exception as exc:
        logger.warning(
            "Falling back to randomly initialized MobileNetV2 because pretrained "
            "weights could not be loaded: %s",
            exc
        )
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=None
        )

    base_model.trainable = not freeze_base

    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Scale [0, 1] inputs to [-1, 1] which is expected by MobileNetV2
        layers.Rescaling(scale=2.0, offset=-1.0),
        
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name='mobilenetv2')

    return model


def unfreeze_mobilenetv2(model: models.Model, num_layers_to_unfreeze: int = 40) -> models.Model:
    """Unfreeze the last N layers of the MobileNetV2 base model."""
    base_model = next((layer for layer in model.layers if hasattr(layer, 'layers')), None)
    if base_model is None:
        raise ValueError("MobileNetV2 base model could not be located for fine-tuning.")
    base_model.trainable = True

    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    return model


def get_transfer_learning_model_info(model: models.Model) -> dict:
    """Return total/trainable parameter counts for reporting."""
    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = total_params - trainable_params

    return {
        'model_name': model.name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'trainable_ratio': trainable_params / total_params if total_params else 0.0
    }
