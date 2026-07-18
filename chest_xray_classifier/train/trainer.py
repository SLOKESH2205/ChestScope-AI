"""
Trainer module for model training with MLflow experiment tracking.

Handles both custom CNN and EfficientNetB0 training with proper
experiment logging, two-phase training strategy, and model checkpointing.
"""

import logging
import shutil
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, AUC

try:
    import mlflow
except ImportError:  # pragma: no cover - runtime dependency fallback
    mlflow = None

from chest_xray_classifier.config.config import (
    CUSTOM_CNN_H5_PATH,
    SEED, EPOCHS_CNN, LR_CNN, EPOCHS_EFFNET_PHASE1, EPOCHS_EFFNET_PHASE2,
    LR_EFFNET_PHASE1, LR_EFFNET_PHASE2, MLFLOW_EXPERIMENT_CNN,
    MLFLOW_EXPERIMENT_EFFNET, MODEL_PATH, EFFNET_PATH, MOBILENET_PATH,
    EFFICIENTNETB0_H5_PATH, MOBILENETV2_H5_PATH, ROOT_CUSTOM_CNN_H5_PATH,
    ROOT_EFFICIENTNETB0_H5_PATH, ROOT_MOBILENETV2_H5_PATH,
    EPOCHS_MOBILENET_PHASE1, EPOCHS_MOBILENET_PHASE2,
    LR_MOBILENET_PHASE1, LR_MOBILENET_PHASE2
)
from chest_xray_classifier.data.loader import DataLoader
from chest_xray_classifier.model.custom_cnn import build_custom_cnn
from chest_xray_classifier.model.efficientnet import build_efficientnet, unfreeze_base_model
from chest_xray_classifier.model.mobilenetv2 import build_mobilenetv2, unfreeze_mobilenetv2
from chest_xray_classifier.train.callbacks import get_callbacks

logger = logging.getLogger(__name__)


class _NullMLflowRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _NullMLflow:
    """Minimal MLflow-compatible shim so training still works when MLflow is absent."""

    @staticmethod
    def set_experiment(*args, **kwargs):
        logger.warning("MLflow is not installed; training will continue without experiment tracking.")

    @staticmethod
    def start_run(*args, **kwargs):
        return _NullMLflowRun()

    @staticmethod
    def log_params(*args, **kwargs):
        return None

    @staticmethod
    def log_metric(*args, **kwargs):
        return None

    class keras:  # pylint: disable=too-few-public-methods
        @staticmethod
        def log_model(*args, **kwargs):
            return None


if mlflow is None:
    mlflow = _NullMLflow()


class Trainer:
    """
    Handles model training with MLflow experiment tracking.
    
    Supports both custom CNN and EfficientNetB0 architectures with
    proper learning rate scheduling and two-phase training strategy.
    """
    
    def __init__(self, experiment_name: str = None, run_name: str = None):
        """
        Initialize Trainer.
        
        Args:
            experiment_name: MLflow experiment name
            run_name: MLflow run name (auto-generated if None)
        """
        self.experiment_name = experiment_name or MLFLOW_EXPERIMENT_CNN
        self.run_name = run_name
        
        # Setup MLflow
        mlflow.set_experiment(self.experiment_name)
        
        logger.info(f"Trainer initialized with experiment: {self.experiment_name}")

    @staticmethod
    def _get_metrics():
        """Shared metrics list for all models."""
        return [
            CategoricalAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc')
        ]

    @staticmethod
    def _export_saved_model(model: keras.Model, target_path: Path, mirror_path: Optional[Path] = None) -> None:
        """Export a trained model to the evaluator-facing H5 path and optional mirror copy."""
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(target_path))
        logger.info("Exported model artifact to %s", target_path)

        if mirror_path is not None:
            mirror_path = Path(mirror_path)
            mirror_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target_path, mirror_path)
            logger.info("Mirrored model artifact to %s", mirror_path)
    
    def train_custom_cnn(
        self,
        train_generator,
        val_generator,
        epochs: int = EPOCHS_CNN,
        learning_rate: float = LR_CNN,
        dropout_rate: float = 0.4,
        l2_factor: float = 0.001,
        run_name: str = None
    ) -> Tuple[keras.Model, dict]:
        """
        Train custom CNN model.
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of epochs
            learning_rate: Initial learning rate
            dropout_rate: Dropout rate for regularization
            l2_factor: L2 regularization factor
            run_name: MLflow run name
            
        Returns:
            Tuple of (trained model, training history dict)
        """
        run_name = run_name or f"custom_cnn_{epochs}ep"
        
        logger.info(f"Starting Custom CNN training: {epochs} epochs, LR={learning_rate}")
        
        with mlflow.start_run(run_name=run_name):
            # Log hyperparameters
            mlflow.log_params({
                'model': 'custom_cnn',
                'epochs': epochs,
                'learning_rate': learning_rate,
                'dropout_rate': dropout_rate,
                'l2_factor': l2_factor,
                'batch_size': train_generator.batch_size
            })
            
            # Build model
            model = build_custom_cnn(
                dropout_rate=dropout_rate,
                l2_factor=l2_factor
            )
            
            # Compile
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=self._get_metrics()
            )
            
            logger.info("Model compiled")
            
            # Get callbacks
            callbacks = get_callbacks(str(MODEL_PATH))
            
            # Train
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save model
            model.save(str(MODEL_PATH))
            logger.info(f"Model saved to {MODEL_PATH}")
            self._export_saved_model(model, CUSTOM_CNN_H5_PATH, ROOT_CUSTOM_CNN_H5_PATH)
            
            # Log model to MLflow
            mlflow.keras.log_model(model, 'model')
            
            # Log metrics
            final_metrics = {
                'final_train_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'final_train_accuracy': float(history.history['accuracy'][-1]),
                'final_val_accuracy': float(history.history['val_accuracy'][-1]),
                'best_val_accuracy': float(np.max(history.history['val_accuracy']))
            }
            for metric_name, metric_value in final_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            logger.info(f"Training complete. Best val accuracy: {final_metrics['best_val_accuracy']:.4f}")
            
            return model, history.history
    
    def train_efficientnet(
        self,
        train_generator,
        val_generator,
        phase1_epochs: int = EPOCHS_EFFNET_PHASE1,
        phase2_epochs: int = EPOCHS_EFFNET_PHASE2,
        phase1_lr: float = LR_EFFNET_PHASE1,
        phase2_lr: float = LR_EFFNET_PHASE2,
        run_name: str = None
    ) -> Tuple[keras.Model, dict]:
        """
        Train EfficientNetB0 with two-phase strategy.
        
        Phase 1: Train head with frozen base (15 epochs)
        Phase 2: Fine-tune unfrozen base with lower LR (10 epochs)
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            phase1_epochs: Epochs for phase 1
            phase2_epochs: Epochs for phase 2
            phase1_lr: Learning rate for phase 1
            phase2_lr: Learning rate for phase 2
            run_name: MLflow run name
            
        Returns:
            Tuple of (trained model, combined history dict)
        """
        run_name = run_name or f"efficientnet_p1{phase1_epochs}_p2{phase2_epochs}"
        
        logger.info(f"Starting EfficientNetB0 two-phase training")
        
        # Change experiment
        mlflow.set_experiment(MLFLOW_EXPERIMENT_EFFNET)
        
        with mlflow.start_run(run_name=run_name):
            # Log overall hyperparameters
            mlflow.log_params({
                'model': 'efficientnet_b0',
                'phase1_epochs': phase1_epochs,
                'phase1_lr': phase1_lr,
                'phase2_epochs': phase2_epochs,
                'phase2_lr': phase2_lr,
                'batch_size': train_generator.batch_size
            })
            
            # ===== PHASE 1: Train head with frozen base =====
            logger.info("=== PHASE 1: Training head with frozen base ===")
            
            model = build_efficientnet(freeze_base=True)
            
            model.compile(
                optimizer=Adam(learning_rate=phase1_lr),
                loss='categorical_crossentropy',
                metrics=self._get_metrics()
            )
            
            callbacks_p1 = get_callbacks(str(EFFNET_PATH))
            
            history_p1 = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=phase1_epochs,
                callbacks=callbacks_p1,
                verbose=1
            )
            
            logger.info(f"Phase 1 complete. Best val accuracy: {np.max(history_p1.history['val_accuracy']):.4f}")
            
            # ===== PHASE 2: Fine-tune with unfrozen base =====
            logger.info("=== PHASE 2: Fine-tuning with unfrozen base ===")
            
            model = unfreeze_base_model(model, num_layers_to_unfreeze=50)
            
            model.compile(
                optimizer=Adam(learning_rate=phase2_lr),
                loss='categorical_crossentropy',
                metrics=self._get_metrics()
            )
            
            callbacks_p2 = get_callbacks(str(EFFNET_PATH))
            
            history_p2 = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=phase1_epochs + phase2_epochs,
                initial_epoch=phase1_epochs,
                callbacks=callbacks_p2,
                verbose=1
            )
            
            # Save final model
            model.save(str(EFFNET_PATH))
            logger.info(f"Model saved to {EFFNET_PATH}")
            self._export_saved_model(model, EFFICIENTNETB0_H5_PATH, ROOT_EFFICIENTNETB0_H5_PATH)
            
            # Log model to MLflow
            mlflow.keras.log_model(model, 'model')
            
            # Combine histories
            combined_history = {}
            for key in history_p1.history.keys():
                combined_history[key] = history_p1.history[key] + history_p2.history.get(key, [])
            
            # Log final metrics
            final_metrics = {
                'phase1_best_val_accuracy': float(np.max(history_p1.history['val_accuracy'])),
                'phase2_final_val_accuracy': float(history_p2.history['val_accuracy'][-1]),
                'overall_best_val_accuracy': float(np.max(combined_history['val_accuracy'])),
                'final_train_loss': float(combined_history['loss'][-1]),
                'final_val_loss': float(combined_history['val_loss'][-1])
            }
            
            for metric_name, metric_value in final_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            logger.info(f"EfficientNetB0 training complete. Best val accuracy: {final_metrics['overall_best_val_accuracy']:.4f}")
            
            return model, combined_history

    def train_mobilenetv2(
        self,
        train_generator,
        val_generator,
        phase1_epochs: int = EPOCHS_MOBILENET_PHASE1,
        phase2_epochs: int = EPOCHS_MOBILENET_PHASE2,
        phase1_lr: float = LR_MOBILENET_PHASE1,
        phase2_lr: float = LR_MOBILENET_PHASE2,
        run_name: str = None
    ) -> Tuple[keras.Model, dict]:
        """
        Train MobileNetV2 with a two-phase transfer learning strategy.
        """
        run_name = run_name or f"mobilenetv2_p1{phase1_epochs}_p2{phase2_epochs}"

        logger.info("Starting MobileNetV2 two-phase training")
        mlflow.set_experiment("chest_xray_mobilenetv2")

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                'model': 'mobilenetv2',
                'phase1_epochs': phase1_epochs,
                'phase1_lr': phase1_lr,
                'phase2_epochs': phase2_epochs,
                'phase2_lr': phase2_lr,
                'batch_size': train_generator.batch_size
            })

            model = build_mobilenetv2(freeze_base=True)
            model.compile(
                optimizer=Adam(learning_rate=phase1_lr),
                loss='categorical_crossentropy',
                metrics=self._get_metrics()
            )

            callbacks_p1 = get_callbacks(str(MOBILENET_PATH))
            history_p1 = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=phase1_epochs,
                callbacks=callbacks_p1,
                verbose=1
            )

            logger.info(
                "Phase 1 complete. Best val accuracy: %.4f",
                np.max(history_p1.history['val_accuracy'])
            )

            model = unfreeze_mobilenetv2(model, num_layers_to_unfreeze=40)
            model.compile(
                optimizer=Adam(learning_rate=phase2_lr),
                loss='categorical_crossentropy',
                metrics=self._get_metrics()
            )

            callbacks_p2 = get_callbacks(str(MOBILENET_PATH))
            history_p2 = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=phase1_epochs + phase2_epochs,
                initial_epoch=phase1_epochs,
                callbacks=callbacks_p2,
                verbose=1
            )

            model.save(str(MOBILENET_PATH))
            logger.info("Model saved to %s", MOBILENET_PATH)
            self._export_saved_model(model, MOBILENETV2_H5_PATH, ROOT_MOBILENETV2_H5_PATH)
            mlflow.keras.log_model(model, 'model')

            combined_history = {}
            for key in history_p1.history.keys():
                combined_history[key] = history_p1.history[key] + history_p2.history.get(key, [])

            final_metrics = {
                'phase1_best_val_accuracy': float(np.max(history_p1.history['val_accuracy'])),
                'phase2_final_val_accuracy': float(history_p2.history['val_accuracy'][-1]),
                'overall_best_val_accuracy': float(np.max(combined_history['val_accuracy'])),
                'final_train_loss': float(combined_history['loss'][-1]),
                'final_val_loss': float(combined_history['val_loss'][-1])
            }
            for metric_name, metric_value in final_metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            logger.info(
                "MobileNetV2 training complete. Best val accuracy: %.4f",
                final_metrics['overall_best_val_accuracy']
            )

            return model, combined_history
