"""
Keras callbacks for training including MLflow integration.

Provides custom callbacks for early stopping, learning rate scheduling,
model checkpointing, and MLflow experiment tracking.
"""

import logging
from typing import Optional

import numpy as np
from tensorflow.keras.callbacks import (
    Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

try:
    import mlflow
except ImportError:  # pragma: no cover - runtime dependency fallback
    mlflow = None

from chest_xray_classifier.config.config import (
    PATIENCE_EARLY_STOP, PATIENCE_LR, LR_FACTOR, MIN_LR
)

logger = logging.getLogger(__name__)


class MLflowCallback(Callback):
    """
    Custom callback to log metrics and model artifacts to MLflow.
    
    Logs metrics at each epoch and saves model at best validation accuracy.
    """
    
    def __init__(
        self,
        log_artifacts: bool = True,
        log_model: bool = True
    ):
        """
        Initialize MLflow callback.
        
        Args:
            log_artifacts: Whether to log training plots as artifacts
            log_model: Whether to log model checkpoint to MLflow
        """
        super().__init__()
        self.log_artifacts = log_artifacts
        self.log_model = log_model
        self.best_val_acc = 0
        
        logger.info("MLflowCallback initialized")
    
    def on_epoch_end(self, epoch: int, logs: dict = None):
        """
        Log metrics to MLflow at end of each epoch.
        
        Args:
            epoch: Epoch number
            logs: Dict with training metrics
        """
        if logs is None:
            logs = {}
        if mlflow is None:
            return
        
        try:
            for metric_name, metric_value in logs.items():
                if isinstance(metric_value, (int, float, np.number)):
                    mlflow.log_metric(metric_name, float(metric_value), step=epoch)
            
            # Track best validation accuracy
            val_acc = logs.get('val_accuracy', 0)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                mlflow.log_metric('best_val_accuracy', float(self.best_val_acc))
        
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")
    
    def on_train_end(self, logs: dict = None):
        """
        Log training summary to MLflow at end of training.
        
        Args:
            logs: Dict with final metrics
        """
        if logs is None:
            logs = {}
        if mlflow is None:
            return
        
        try:
            mlflow.log_metric('final_train_loss', float(logs.get('loss', 0)))
            mlflow.log_metric('final_val_loss', float(logs.get('val_loss', 0)))
            mlflow.log_metric('final_train_accuracy', float(logs.get('accuracy', 0)))
            mlflow.log_metric('final_val_accuracy', float(logs.get('val_accuracy', 0)))
            
            logger.info("Training summary logged to MLflow")
        
        except Exception as e:
            logger.warning(f"Failed to log training summary: {e}")


def get_callbacks(
    model_checkpoint_path: str,
    patience_es: int = PATIENCE_EARLY_STOP,
    patience_lr: int = PATIENCE_LR,
    lr_factor: float = LR_FACTOR,
    min_lr: float = MIN_LR
) -> list:
    """
    Create standard callback list for training.
    
    Includes:
    - EarlyStopping: Stop if val_loss doesn't improve
    - ReduceLROnPlateau: Reduce learning rate if plateau detected
    - ModelCheckpoint: Save best model
    - MLflowCallback: Log to MLflow
    
    Args:
        model_checkpoint_path: Path to save best model
        patience_es: Early stopping patience (epochs)
        patience_lr: ReduceLR patience (epochs)
        lr_factor: Factor to reduce learning rate by
        min_lr: Minimum learning rate threshold
        
    Returns:
        List of Keras callbacks
    """
    logger.info(f"Creating callbacks: ES_patience={patience_es}, LR_patience={patience_lr}")
    
    callbacks = [
        # Early Stopping
        EarlyStopping(
            monitor='val_loss',
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        
        # Learning Rate Reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=lr_factor,
            patience=patience_lr,
            min_lr=min_lr,
            verbose=1,
            mode='min'
        ),
        
        # Model Checkpoint
        ModelCheckpoint(
            filepath=model_checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
    ]

    if mlflow is not None:
        callbacks.append(MLflowCallback(log_model=True))

    return callbacks
