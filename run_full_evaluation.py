#!/usr/bin/env python3
"""
run_full_evaluation.py

A unified evaluation script that:
1. Evaluates all trained models (Custom CNN, EfficientNetB0, MobileNetV2) on a shared validation split.
2. Computes Accuracy, Precision, Recall, F1, ROC-AUC, Specificity, Sensitivity, Balanced Accuracy, MCC, Latency, and Size.
3. Performs error analysis (generates misclassified examples gallery and worst-confidence lists).
4. Generates explainability maps (Grad-CAM, Grad-CAM++, Integrated Gradients) for example cases.
5. Saves all figures and a final metrics comparison JSON to the outputs/ directory.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Insert project path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from chest_xray_classifier.config.config import CLASSES, VAL_DIR
from chest_xray_classifier.data.loader import DataLoader
from chest_xray_classifier.predict import load_cached_model
from chest_xray_classifier.evaluation import evaluate_model_on_dataset, run_error_analysis, save_confusion_matrix
from chest_xray_classifier.visualization import plot_per_class_metrics, generate_misclassified_gallery
from chest_xray_classifier.explainability import generate_xai_plots
from chest_xray_classifier.utils import setup_project_directories, get_file_size_mb

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_full_evaluation")

# Model configurations
MODELS_TO_EVALUATE = {
    "Custom CNN": {
        "key": "custom_cnn",
        "path": project_root / "chest_xray_classifier" / "models" / "custom_cnn.h5"
    },
    "EfficientNetB0": {
        "key": "efficientnetb0",
        "path": project_root / "chest_xray_classifier" / "models" / "efficientnetb0.h5"
    },
    "MobileNetV2": {
        "key": "mobilenetv2",
        "path": project_root / "chest_xray_classifier" / "models" / "mobilenetv2.h5"
    }
}

def main() -> None:
    logger.info("Initializing full evaluation pipeline...")
    setup_project_directories()
    
    # Initialize DataLoader
    data_loader = DataLoader()
    if not data_loader.verify_dataset_structure():
        logger.error("Dataset structure is invalid. Cannot proceed.")
        sys.exit(1)
        
    val_generator = data_loader.get_val_generator()
    val_count = len(val_generator.filenames)
    logger.info(f"Loaded validation generator with {val_count} images.")
    
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    final_metrics = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_summary": {
            "validation_images": val_count,
            "classes": CLASSES
        },
        "models": {},
        "comparison_table": {},
        "best_model": None
    }
    
    evaluated_models = {}
    
    for display_name, spec in MODELS_TO_EVALUATE.items():
        model_path = spec["path"]
        logger.info(f"Evaluating {display_name}...")
        
        if not model_path.exists():
            logger.warning(f"Weights file not found for {display_name} at {model_path}. Skipping.")
            continue
            
        try:
            model_size = get_file_size_mb(model_path)
            model = load_cached_model(model_path)
            
            # Run evaluation
            eval_result = evaluate_model_on_dataset(model, val_generator, display_name, model_size)
            evaluated_models[display_name] = eval_result
            
            # Store overall metrics
            final_metrics["models"][spec["key"]] = {
                "metrics": eval_result["metrics"],
                "per_class": eval_result["per_class"]
            }
            
            # Save confusion matrix plot
            cm_path = save_confusion_matrix(eval_result, outputs_dir)
            logger.info(f"Saved confusion matrix to {cm_path}")
            
            # Plot per-class metrics
            pcm_path = outputs_dir / f"{spec['key']}_per_class_metrics.png"
            plot_per_class_metrics(eval_result["per_class"], pcm_path)
            logger.info(f"Saved per-class metrics plot to {pcm_path}")
            
            # Run error analysis
            error_report = run_error_analysis(eval_result, outputs_dir)
            
            # Generate misclassified examples gallery
            gallery_path = outputs_dir / f"{spec['key']}_misclassified_gallery.png"
            generate_misclassified_gallery(error_report["worst_predictions"], VAL_DIR, gallery_path)
            logger.info(f"Saved misclassified gallery to {gallery_path}")
            
            # Generate explainability (Grad-CAM, Grad-CAM++, Integrated Gradients)
            # Take the first image in the validation generator as a visual example
            val_generator.reset()
            first_batch_x, first_batch_y = next(val_generator)
            example_image = first_batch_x[0]  # Shape: (224, 224, 3)
            example_class = int(np.argmax(first_batch_y[0]))
            
            xai_results = generate_xai_plots(
                model, example_image, example_class,
                output_dir=outputs_dir / "heatmap_examples",
                prefix=f"{spec['key']}_"
            )
            
            # Copy main gradcam example to outputs folder as requested
            if spec["key"] == "mobilenetv2" or spec["key"] == "efficientnetb0":
                # Save as gradcam_example.png in outputs/
                import shutil
                shutil.copy2(xai_results["gradcam_example"], outputs_dir / f"gradcam_example.png")
                
            logger.info(f"Generated explainability plots for {display_name}")
            
        except Exception as exc:
            logger.error(f"Error evaluating {display_name}: {exc}", exc_info=True)
            
    if not evaluated_models:
        logger.error("No models were successfully evaluated.")
        sys.exit(1)
        
    # Build comparison dataframe
    comparison_data = []
    for model_name, res in evaluated_models.items():
        row = {"Model": model_name}
        row.update(res["metrics"])
        comparison_data.append(row)
        
    df_comparison = pd.DataFrame(comparison_data)
    logger.info("\n=== MODEL COMPARISON TABLE ===\n" + df_comparison.to_string(index=False))
    
    # Save comparison table to CSV
    csv_path = outputs_dir / "model_comparison.csv"
    df_comparison.to_csv(csv_path, index=False)
    logger.info(f"Saved comparison table CSV to {csv_path}")
    
    # Identify the best model based on F1 Score
    best_row = df_comparison.sort_values(by="F1 Score", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    logger.info(f"Best model selected: {best_model_name} with F1-Score of {best_row['F1 Score']:.4f}")
    
    final_metrics["comparison_table"] = df_comparison.to_dict(orient="records")
    final_metrics["best_model"] = {
        "name": best_model_name,
        "key": MODELS_TO_EVALUATE[best_model_name]["key"],
        "f1_score": float(best_row["F1 Score"]),
        "accuracy": float(best_row["Accuracy"])
    }
    
    # Save consolidated final_metrics.json
    final_metrics_path = outputs_dir / "final_metrics.json"
    final_metrics_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    logger.info(f"Saved final evaluation metrics JSON to {final_metrics_path}")
    
    logger.info("Evaluation pipeline execution complete.")

if __name__ == "__main__":
    main()
