"""
Unit tests for metrics computation module.

Tests metric calculation, confusion matrix, ROC/PR curves.

Run with: pytest tests/test_metrics.py
"""

import pytest
import numpy as np
import pandas as pd

from chest_xray_classifier.evaluate.metrics import (
    compute_all_metrics, compute_roc_curves, compute_pr_curves,
    format_metrics_table, metrics_to_dataframe
)
from chest_xray_classifier.config.config import CLASSES


class TestMetrics:
    """Test suite for metrics module."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample true labels and predictions."""
        num_samples = 100
        num_classes = len(CLASSES)
        
        # Random true labels
        y_true = np.random.randint(0, num_classes, num_samples)
        
        # Random predictions (high confidence)
        y_pred_probs = np.random.dirichlet(np.ones(num_classes), num_samples)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        
        return y_true, y_pred_classes, y_pred_probs
    
    # ===== compute_all_metrics Tests =====
    
    def test_compute_all_metrics_returns_dict(self, sample_predictions):
        """Test that compute_all_metrics returns valid dict."""
        y_true, y_pred_classes, y_pred_probs = sample_predictions
        
        metrics = compute_all_metrics(y_true, y_pred_classes, y_pred_probs)
        
        assert isinstance(metrics, dict)
    
    def test_compute_all_metrics_has_required_keys(self, sample_predictions):
        """Test that metrics dict has all required keys."""
        y_true, y_pred_classes, y_pred_probs = sample_predictions
        
        metrics = compute_all_metrics(y_true, y_pred_classes, y_pred_probs)
        
        required_keys = ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix', 'per_class']
        
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"
    
    def test_compute_all_metrics_value_ranges(self, sample_predictions):
        """Test that metric values are in valid ranges."""
        y_true, y_pred_classes, y_pred_probs = sample_predictions
        
        metrics = compute_all_metrics(y_true, y_pred_classes, y_pred_probs)
        
        # All metrics should be in [0, 1]
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    def test_compute_all_metrics_confusion_matrix_shape(self, sample_predictions):
        """Test confusion matrix shape."""
        y_true, y_pred_classes, y_pred_probs = sample_predictions
        
        metrics = compute_all_metrics(y_true, y_pred_classes, y_pred_probs)
        
        cm = metrics['confusion_matrix']
        assert cm.shape == (len(CLASSES), len(CLASSES))
    
    def test_compute_all_metrics_per_class_structure(self, sample_predictions):
        """Test per-class metrics structure."""
        y_true, y_pred_classes, y_pred_probs = sample_predictions
        
        metrics = compute_all_metrics(y_true, y_pred_classes, y_pred_probs)
        
        per_class = metrics['per_class']
        
        assert len(per_class) == len(CLASSES)
        
        for class_name in CLASSES:
            assert class_name in per_class
            assert 'precision' in per_class[class_name]
            assert 'recall' in per_class[class_name]
            assert 'f1_score' in per_class[class_name]
            assert 'support' in per_class[class_name]
    
    # ===== compute_roc_curves Tests =====
    
    def test_compute_roc_curves_returns_dict(self, sample_predictions):
        """Test that compute_roc_curves returns dict."""
        y_true, _, y_pred_probs = sample_predictions
        
        roc = compute_roc_curves(y_true, y_pred_probs)
        
        assert isinstance(roc, dict)
    
    def test_compute_roc_curves_has_required_keys(self, sample_predictions):
        """Test ROC curves dict structure."""
        y_true, _, y_pred_probs = sample_predictions
        
        roc = compute_roc_curves(y_true, y_pred_probs)
        
        assert 'auc_scores' in roc
        assert 'roc_curves' in roc
        assert 'macro_auc' in roc
    
    def test_compute_roc_curves_auc_ranges(self, sample_predictions):
        """Test AUC scores are in [0, 1]."""
        y_true, _, y_pred_probs = sample_predictions
        
        roc = compute_roc_curves(y_true, y_pred_probs)
        
        for class_name, auc_score in roc['auc_scores'].items():
            assert 0 <= auc_score <= 1, f"{class_name} AUC out of range: {auc_score}"
    
    def test_compute_roc_curves_macro_auc_in_range(self, sample_predictions):
        """Test macro AUC is in valid range."""
        y_true, _, y_pred_probs = sample_predictions
        
        roc = compute_roc_curves(y_true, y_pred_probs)
        
        assert 0 <= roc['macro_auc'] <= 1
    
    # ===== compute_pr_curves Tests =====
    
    def test_compute_pr_curves_returns_dict(self, sample_predictions):
        """Test that compute_pr_curves returns dict."""
        y_true, _, y_pred_probs = sample_predictions
        
        pr = compute_pr_curves(y_true, y_pred_probs)
        
        assert isinstance(pr, dict)
    
    def test_compute_pr_curves_has_required_keys(self, sample_predictions):
        """Test PR curves dict structure."""
        y_true, _, y_pred_probs = sample_predictions
        
        pr = compute_pr_curves(y_true, y_pred_probs)
        
        assert 'pr_curves' in pr
        assert 'ap_scores' in pr
    
    def test_compute_pr_curves_ap_ranges(self, sample_predictions):
        """Test AP scores are in [0, 1]."""
        y_true, _, y_pred_probs = sample_predictions
        
        pr = compute_pr_curves(y_true, y_pred_probs)
        
        for class_name, ap_score in pr['ap_scores'].items():
            assert 0 <= ap_score <= 1, f"{class_name} AP out of range: {ap_score}"
    
    # ===== Formatting Functions Tests =====
    
    def test_format_metrics_table_returns_string(self, sample_predictions):
        """Test that format_metrics_table returns string."""
        y_true, y_pred_classes, y_pred_probs = sample_predictions
        
        metrics = compute_all_metrics(y_true, y_pred_classes, y_pred_probs)
        table = format_metrics_table(metrics)
        
        assert isinstance(table, str)
        assert len(table) > 0
    
    def test_format_metrics_table_contains_classes(self, sample_predictions):
        """Test that formatted table contains class names."""
        y_true, y_pred_classes, y_pred_probs = sample_predictions
        
        metrics = compute_all_metrics(y_true, y_pred_classes, y_pred_probs)
        table = format_metrics_table(metrics)
        
        for class_name in CLASSES:
            assert class_name in table
    
    def test_metrics_to_dataframe_returns_df(self, sample_predictions):
        """Test that metrics_to_dataframe returns DataFrame."""
        y_true, y_pred_classes, y_pred_probs = sample_predictions
        
        metrics = compute_all_metrics(y_true, y_pred_classes, y_pred_probs)
        df = metrics_to_dataframe(metrics)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(CLASSES)
    
    def test_metrics_dataframe_has_required_columns(self, sample_predictions):
        """Test DataFrame columns."""
        y_true, y_pred_classes, y_pred_probs = sample_predictions
        
        metrics = compute_all_metrics(y_true, y_pred_classes, y_pred_probs)
        df = metrics_to_dataframe(metrics)
        
        required_cols = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
        
        for col in required_cols:
            assert col in df.columns
    
    # ===== Edge Cases =====
    
    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_probs = np.eye(4)[y_pred]  # One-hot
        
        metrics = compute_all_metrics(y_true, y_pred, y_probs)
        
        assert metrics['accuracy'] == 1.0
    
    def test_completely_wrong_predictions(self):
        """Test with completely wrong predictions."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 2, 3, 1])
        y_probs = np.eye(4)[y_pred]
        
        metrics = compute_all_metrics(y_true, y_pred, y_probs)
        
        assert metrics['accuracy'] == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
