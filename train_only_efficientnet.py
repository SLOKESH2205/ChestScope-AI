#!/usr/bin/env python3
"""
train_only_efficientnet.py

Utility script to train only the EfficientNetB0 model on CPU for debugging.
Applies:
- Model-specific preprocessing fix (Rescaling layer)
- Two-phase training (frozen base, then unfreeze last 50 layers)
- EarlyStopping (patience=3) and ReduceLROnPlateau
- Maximum 10 epochs (5 for Phase 1, 5 for Phase 2)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, AUC

from chest_xray_classifier.config.config import (
    EFFICIENTNETB0_H5_PATH, EFFNET_PATH,
    EFFNET_LR, FINETUNE_LR,
    EFFNET_PHASE1_EPOCHS, EFFNET_PHASE2_EPOCHS
)
from chest_xray_classifier.data.loader import DataLoader
from chest_xray_classifier.models import get_model_by_name, unfreeze_effnet
from chest_xray_classifier.train.callbacks import get_callbacks

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_only_efficientnet")

def main() -> None:
    logger.info("Initializing EfficientNetB0 training pipeline...")
    
    # Check GPU (native Windows TF >= 2.11 is CPU only)
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"GPUs detected: {gpus} (Running on CPU)")
    
    # Initialize DataLoader
    data_loader = DataLoader()
    if not data_loader.verify_dataset_structure():
        logger.error("Dataset structure validation failed.")
        return
        
    train_gen = data_loader.get_train_generator()
    val_gen = data_loader.get_val_generator()
    
    # Build EfficientNetB0 model with frozen base
    logger.info("Building EfficientNetB0 (Phase 1: Frozen Base)...")
    model = get_model_by_name("EfficientNetB0", freeze_base=True)
    
    metrics = [
        CategoricalAccuracy(name='accuracy'),
        Precision(name='precision'),
        Recall(name='recall'),
        AUC(name='auc')
    ]
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=EFFNET_LR),
        loss='categorical_crossentropy',
        metrics=metrics
    )
    
    # Callbacks (checks config.py where early stopping patience was set to 3)
    # Target path is EFFNET_PATH which is chest_xray_classifier/model/efficientnet_model.keras
    checkpoint_path = Path(EFFNET_PATH)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks_p1 = get_callbacks(str(checkpoint_path))
    
    # ===== Phase 1: Train Head =====
    logger.info(f"=== PHASE 1: Training head for up to {EFFNET_PHASE1_EPOCHS} epochs ===")
    start_p1 = time.perf_counter()
    history_p1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EFFNET_PHASE1_EPOCHS,
        callbacks=callbacks_p1,
        verbose=1
    )
    elapsed_p1 = time.perf_counter() - start_p1
    logger.info(f"Phase 1 completed in {elapsed_p1:.1f} seconds.")
    
    # ===== Phase 2: Fine-Tuning =====
    logger.info("=== PHASE 2: Fine-tuning last 50 layers ===")
    # Unfreeze last 50 layers
    model = unfreeze_effnet(model, num_layers_to_unfreeze=50)
    
    # Re-compile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=FINETUNE_LR),
        loss='categorical_crossentropy',
        metrics=[
            CategoricalAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc')
        ]
    )
    
    callbacks_p2 = get_callbacks(str(checkpoint_path))
    
    start_p2 = time.perf_counter()
    history_p2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EFFNET_PHASE2_EPOCHS,
        callbacks=callbacks_p2,
        verbose=1
    )
    elapsed_p2 = time.perf_counter() - start_p2
    logger.info(f"Phase 2 completed in {elapsed_p2:.1f} seconds.")
    
    # Save final model weights
    # 1. Package models folder
    Path(EFFICIENTNETB0_H5_PATH).parent.mkdir(parents=True, exist_ok=True)
    model.save(str(EFFICIENTNETB0_H5_PATH))
    logger.info(f"Saved H5 model to {EFFICIENTNETB0_H5_PATH}")
    
    # 2. Duplicate to root models directory for comparison loading
    root_h5_path = Path("models") / "efficientnetb0.h5"
    root_h5_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(EFFICIENTNETB0_H5_PATH, root_h5_path)
    logger.info(f"Mirrored H5 model to {root_h5_path}")
    
    # Save combined history to outputs/
    history_combined = {}
    for key in history_p1.history:
        history_combined[key] = history_p1.history[key] + history_p2.history.get(key, [])
        
    history_out_path = Path("outputs") / "efficientnetb0_history.json"
    history_out_path.parent.mkdir(parents=True, exist_ok=True)
    history_out_path.write_text(json.dumps(history_combined, indent=2), encoding="utf-8")
    logger.info(f"Saved combined history report to {history_out_path}")
    
    logger.info("EfficientNetB0 training pipeline execution complete.")

if __name__ == "__main__":
    main()
