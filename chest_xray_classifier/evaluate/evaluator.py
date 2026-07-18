"""
Evaluator module for comprehensive model evaluation on validation/test sets.

Orchestrates full evaluation pipeline: predictions, metrics computation,
result visualization, and report generation.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import json
from tensorflow import keras

from chest_xray_classifier.config.config import RESULTS_DIR
from chest_xray_classifier.data.loader import DataLoader
from chest_xray_classifier.evaluate.metrics import (
    compute_all_metrics, compute_roc_curves, compute_pr_curves,
    format_metrics_table
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Orchestrates comprehensive model evaluation.
    
    Handles predictions on validation/test data, metric computation,
    result saving, and report generation.
    """
    
    def __init__(self, model: keras.Model, model_name: str = 'model'):
        """
        Initialize Evaluator.
        
        Args:
            model: Keras model to evaluate
            model_name: Name for saving results
        """
        self.model = model
        self.model_name = model_name
        self.data_loader = DataLoader()
        
        logger.info(f"Evaluator initialized for model: {model_name}")
    
    def evaluate_on_generator(
        self,
        val_generator
    ) -> Dict:
        """
        Evaluate model on validation data generator.
        
        Args:
            val_generator: Validation data generator
            
        Returns:
            Dict with full evaluation results:
            - 'metrics': Dict with accuracy, precision, recall, F1, confusion matrix
            - 'roc': ROC curves and AUC scores
            - 'pr': PR curves and AP scores
            - 'test_loss': Loss on validation set
            - 'test_accuracy': Top-1 accuracy
        """
        logger.info("Evaluating on validation generator...")
        val_generator.reset()
        
        # Get predictions
        y_pred_probs = self.model.predict(val_generator)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_true = val_generator.classes
        
        # Compute metrics
        metrics = compute_all_metrics(y_true, y_pred_classes, y_pred_probs)
        roc_results = compute_roc_curves(y_true, y_pred_probs)
        pr_results = compute_pr_curves(y_true, y_pred_probs)
        
        # Evaluate on full generator (for loss/accuracy)
        val_generator.reset()
        eval_results = self.model.evaluate(val_generator, verbose=0)
        
        results = {
            'metrics': metrics,
            'roc': roc_results,
            'pr': pr_results,
            'test_loss': float(eval_results[0]),
            'test_accuracy': float(eval_results[1]),
            'y_true': y_true,
            'y_pred_classes': y_pred_classes,
            'y_pred_probs': y_pred_probs
        }
        
        logger.info(f"Evaluation complete. Test accuracy: {results['test_accuracy']:.4f}")
        
        return results
    
    def evaluate_on_test_images(
        self,
        test_images: list,
        test_labels: Optional[list] = None
    ) -> Dict:
        """
        Evaluate on a list of test images (without generator).
        
        Args:
            test_images: List of image arrays (num_samples, 224, 224, 3)
            test_labels: Optional true labels (class indices)
            
        Returns:
            Dict with predictions and metrics (if labels provided)
        """
        logger.info(f"Evaluating on {len(test_images)} test images...")
        
        # Stack images into batch
        test_batch = np.array(test_images)
        
        # Get predictions
        y_pred_probs = self.model.predict(test_batch, verbose=0)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        
        results = {
            'y_pred_classes': y_pred_classes,
            'y_pred_probs': y_pred_probs
        }
        
        # Compute metrics if true labels provided
        if test_labels is not None:
            test_labels = np.array(test_labels)
            metrics = compute_all_metrics(test_labels, y_pred_classes, y_pred_probs)
            roc_results = compute_roc_curves(test_labels, y_pred_probs)
            pr_results = compute_pr_curves(test_labels, y_pred_probs)
            
            results.update({
                'metrics': metrics,
                'roc': roc_results,
                'pr': pr_results,
                'y_true': test_labels
            })
        
        logger.info("Test image evaluation complete")
        
        return results
    
    def save_results(
        self,
        results: Dict,
        output_dir: Path = RESULTS_DIR,
        file_prefix: Optional[str] = None
    ) -> Dict:
        """
        Save evaluation results to JSON files.
        
        Saves:
        - metrics_summary.json: Overall metrics, per-class metrics
        - roc_auc_scores.json: ROC AUC scores
        - pr_ap_scores.json: PR average precision scores
        
        Does NOT save: ROC/PR curves (numpy arrays), y_true/pred (large arrays)
        
        Args:
            results: Results dict from evaluate_on_generator()
            output_dir: Directory to save results
            
        Returns:
            Dict with paths to saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to {output_dir}")
        file_prefix = file_prefix or self.model_name.lower().replace(' ', '_')
        
        # Prepare metrics for JSON
        metrics = results['metrics']
        metrics_summary = {
            'model_name': self.model_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'test_loss': results.get('test_loss'),
            'test_accuracy': results.get('test_accuracy'),
            'per_class': metrics['per_class']
        }
        
        # Save metrics
        metrics_path = output_dir / f'{file_prefix}_metrics_summary.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Save ROC AUC scores
        roc_auc_path = output_dir / f'{file_prefix}_roc_auc_scores.json'
        with open(roc_auc_path, 'w') as f:
            json.dump(results['roc']['auc_scores'], f, indent=2)
        logger.info(f"ROC AUC scores saved to {roc_auc_path}")
        
        # Save PR AP scores
        pr_ap_path = output_dir / f'{file_prefix}_pr_ap_scores.json'
        with open(pr_ap_path, 'w') as f:
            json.dump(results['pr']['ap_scores'], f, indent=2)
        logger.info(f"PR AP scores saved to {pr_ap_path}")
        
        return {
            'metrics': metrics_path,
            'roc_auc': roc_auc_path,
            'pr_ap': pr_ap_path
        }
    
    def print_summary(self, results: Dict):
        """
        Print evaluation summary to console.
        
        Args:
            results: Results dict from evaluate_on_generator()
        """
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)
        
        # Overall metrics
        logger.info(f"\nTest Loss: {results.get('test_loss', 'N/A'):.4f}")
        logger.info(f"Test Accuracy: {results.get('test_accuracy', 'N/A'):.4f}")
        
        # Metrics table
        logger.info("\n" + format_metrics_table(results['metrics']))
        
        # ROC AUC scores
        logger.info("\nROC AUC Scores:")
        for class_name, auc_score in results['roc']['auc_scores'].items():
            logger.info(f"  {class_name}: {auc_score:.4f}")
        logger.info(f"  Macro AUC: {results['roc']['macro_auc']:.4f}")
        
        # PR AP scores
        logger.info("\nPR Average Precision Scores:")
        for class_name, ap_score in results['pr']['ap_scores'].items():
            logger.info(f"  {class_name}: {ap_score:.4f}")
        
        logger.info("="*80 + "\n")
