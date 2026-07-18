#!/usr/bin/env python3
"""
tune_custom_cnn.py

Hyperparameter tuning script for Custom CNN.
Searches:
- Learning rate
- Dropout rate
- L2 weight decay
Limits search to 15 trials on a representative training subset for speed.
Saves the new model weights ONLY if validation F1 improves by >= 0.5% (0.005 absolute).
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping

from chest_xray_classifier.config.config import CUSTOM_CNN_H5_PATH
from chest_xray_classifier.data.loader import DataLoader
from chest_xray_classifier.model.custom_cnn import build_custom_cnn
from chest_xray_classifier.evaluation import evaluate_model_on_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tune_custom_cnn")

# Baseline F1 score from full evaluation
BASELINE_F1 = 0.82348

def main() -> None:
    logger.info("Starting hyperparameter tuning for Custom CNN...")
    
    # 1. Load Data
    data_loader = DataLoader()
    train_gen = data_loader.get_train_generator()
    val_gen = data_loader.get_val_generator()
    
    # Define search space
    learning_rates = [1e-4, 5e-4, 1e-3, 2e-3]
    dropout_rates = [0.3, 0.4, 0.5]
    l2_factors = [1e-4, 5e-4, 1e-3, 5e-3]
    
    num_trials = 15
    trials = []
    
    # Create random combinations
    random.seed(42)
    while len(trials) < num_trials:
        trial = {
            "learning_rate": random.choice(learning_rates),
            "dropout_rate": random.choice(dropout_rates),
            "l2_factor": random.choice(l2_factors)
        }
        if trial not in trials:
            trials.append(trial)
            
    logger.info(f"Generated {len(trials)} unique trials for hyperparameter search.")
    
    # Pre-load validation images to speed up evaluation in trials
    val_gen.reset()
    val_images = []
    val_labels = []
    for _ in range(len(val_gen)):
        x, y = next(val_gen)
        val_images.append(x)
        val_labels.append(y)
    x_val = np.vstack(val_images)
    y_val = np.vstack(val_labels)
    
    # Pre-load a subset (25%) of training images for fast trials
    train_gen.reset()
    train_images = []
    train_labels = []
    for _ in range(min(5, len(train_gen))):  # ~160 images
        x, y = next(train_gen)
        train_images.append(x)
        train_labels.append(y)
    x_train = np.vstack(train_images)
    y_train = np.vstack(train_labels)
    
    best_val_loss = float('inf')
    best_hyperparameters = None
    best_trial_index = -1
    
    results = []
    
    for i, trial in enumerate(trials):
        logger.info(f"--- Trial {i+1}/{num_trials}: LR={trial['learning_rate']}, Dropout={trial['dropout_rate']}, L2={trial['l2_factor']} ---")
        
        # Build model with hyperparameters
        model = build_custom_cnn(
            input_shape=(224, 224, 3),
            num_classes=4,
            dropout_rate=trial["dropout_rate"],
            l2_factor=trial["l2_factor"]
        )
        
        model.compile(
            optimizer=Adam(learning_rate=trial["learning_rate"]),
            loss='categorical_crossentropy',
            metrics=[CategoricalAccuracy()]
        )
        
        # Load baseline weights for meaningful tuning starting point
        try:
            if Path(CUSTOM_CNN_H5_PATH).exists():
                model.load_weights(str(CUSTOM_CNN_H5_PATH), by_name=True, skip_mismatch=True)
                logger.debug("Loaded baseline pre-trained weights for Trial.")
        except Exception as exc:
            logger.warning(f"Failed to load baseline weights: {exc}")
        
        # Train on subset for 3 epochs for fast screening
        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        
        start_time = time.perf_counter()
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=3,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        elapsed = time.perf_counter() - start_time
        
        val_loss = history.history['val_loss'][-1]
        val_acc = history.history['val_categorical_accuracy'][-1]
        
        logger.info(f"Completed in {elapsed:.1f}s. Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")
        
        trial_result = trial.copy()
        trial_result["val_loss"] = float(val_loss)
        trial_result["val_acc"] = float(val_acc)
        trial_result["elapsed_sec"] = float(elapsed)
        results.append(trial_result)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_hyperparameters = trial
            best_trial_index = i
            
    logger.info(f"Tuning complete. Best trial: Index={best_trial_index+1}, Loss={best_val_loss:.4f}")
    
    # 3. Retrain Best Hyperparameters on FULL dataset
    logger.info(f"Retraining best hyperparameters on full dataset: {best_hyperparameters}")
    
    best_model = build_custom_cnn(
        input_shape=(224, 224, 3),
        num_classes=4,
        dropout_rate=best_hyperparameters["dropout_rate"],
        l2_factor=best_hyperparameters["l2_factor"]
    )
    
    best_model.compile(
        optimizer=Adam(learning_rate=best_hyperparameters["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=[CategoricalAccuracy()]
    )
    
    early_stop_full = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Train full model for 5 epochs
    best_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5,
        callbacks=[early_stop_full],
        verbose=1
    )
    
    # 4. Evaluate and Compare F1
    logger.info("Evaluating tuned model on full validation generator...")
    eval_tuned = evaluate_model_on_dataset(best_model, val_gen, "Tuned Custom CNN", 508.0)
    tuned_f1 = eval_tuned["metrics"]["F1 Score"]
    
    logger.info(f"Baseline F1: {BASELINE_F1:.5f}")
    logger.info(f"Tuned F1: {tuned_f1:.5f}")
    
    improvement = tuned_f1 - BASELINE_F1
    logger.info(f"Absolute improvement: {improvement:.5f}")
    
    tuning_report = {
        "baseline_f1": BASELINE_F1,
        "tuned_f1": float(tuned_f1),
        "improvement": float(improvement),
        "best_hyperparameters": best_hyperparameters,
        "trials": results,
        "kept_new_model": False
    }
    
    # Check if improvement is >= 0.005 (0.5 percentage points)
    if improvement >= 0.005:
        logger.info(f"F1 improved by {improvement:.2%}. Saving new model weights.")
        Path(CUSTOM_CNN_H5_PATH).parent.mkdir(parents=True, exist_ok=True)
        best_model.save(str(CUSTOM_CNN_H5_PATH))
        
        # Mirror to models/
        root_path = Path("models") / "custom_cnn.h5"
        root_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(CUSTOM_CNN_H5_PATH, root_path)
        
        # Save keras model too
        best_model.save("chest_xray_classifier/model/chest_xray_model.keras")
        
        tuning_report["kept_new_model"] = True
    else:
        logger.info("F1 did not improve by >= 0.5 percentage points. Keeping baseline model weights.")
        
    # Save report
    report_path = Path("outputs") / "tuning_report.json"
    report_path.write_text(json.dumps(tuning_report, indent=2), encoding="utf-8")
    logger.info(f"Saved tuning report to {report_path}")

if __name__ == "__main__":
    main()
