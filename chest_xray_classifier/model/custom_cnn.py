"""
Custom CNN model architecture for Chest X-Ray Classification.

Implements a 4-block convolutional neural network with batch normalization,
dropout, and L2 regularization for robust learning.
"""

import logging
from typing import Callable

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, models

from chest_xray_classifier.config.config import IMG_SHAPE, NUM_CLASSES, SEED

logger = logging.getLogger(__name__)


def build_custom_cnn(
    input_shape: tuple = IMG_SHAPE,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.4,
    l2_factor: float = 0.001
) -> models.Model:
    """
    Build a custom 4-block CNN model.
    
    Architecture:
    - Block 1: Conv(32) -> Conv(32) -> Pool -> Dropout
    - Block 2: Conv(64) -> Conv(64) -> Pool -> Dropout
    - Block 3: Conv(128) -> Conv(128) -> Pool -> Dropout
    - Block 4: Conv(256) -> Conv(256) -> Pool -> Dropout
    - Head: GlobalAvgPool -> Dense(512) -> Dropout -> Dense(num_classes)
    
    All convolutions use ReLU activation, batch normalization, and L2 regularization.
    
    Args:
        input_shape: Input image shape (default: (224, 224, 3))
        num_classes: Number of output classes (default: 4)
        dropout_rate: Dropout rate (default: 0.4)
        l2_factor: L2 regularization factor (default: 0.001)
        
    Returns:
        Compiled Keras model
    """
    logger.info(f"Building Custom CNN with input_shape={input_shape}, num_classes={num_classes}")
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Block 1: Conv -> Conv -> Pool -> Dropout
        layers.Conv2D(
            32, (3, 3), padding='same',
            kernel_regularizer=regularizers.l2(l2_factor)
        ),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        layers.Conv2D(
            32, (3, 3), padding='same',
            kernel_regularizer=regularizers.l2(l2_factor)
        ),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),
        
        # Block 2: Conv -> Conv -> Pool -> Dropout
        layers.Conv2D(
            64, (3, 3), padding='same',
            kernel_regularizer=regularizers.l2(l2_factor)
        ),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        layers.Conv2D(
            64, (3, 3), padding='same',
            kernel_regularizer=regularizers.l2(l2_factor)
        ),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),
        
        # Block 3: Conv -> Conv -> Pool -> Dropout
        layers.Conv2D(
            128, (3, 3), padding='same',
            kernel_regularizer=regularizers.l2(l2_factor)
        ),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        layers.Conv2D(
            128, (3, 3), padding='same',
            kernel_regularizer=regularizers.l2(l2_factor)
        ),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),
        
        # Block 4: Conv -> Conv -> Pool -> Dropout
        layers.Conv2D(
            256, (3, 3), padding='same',
            kernel_regularizer=regularizers.l2(l2_factor)
        ),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        layers.Conv2D(
            256, (3, 3), padding='same',
            kernel_regularizer=regularizers.l2(l2_factor)
        ),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),
        
        # Head: Global Average Pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(512, kernel_regularizer=regularizers.l2(l2_factor)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ], name='custom_cnn')
    
    logger.info(f"Custom CNN model built successfully")
    logger.debug(f"Total parameters: {model.count_params():,}")
    
    return model


def get_model_summary(model: models.Model) -> str:
    """
    Get detailed model summary as string.
    
    Args:
        model: Keras model
        
    Returns:
        Model summary string
    """
    summary_list = []
    model.summary(print_fn=summary_list.append)
    return '\n'.join(summary_list)
