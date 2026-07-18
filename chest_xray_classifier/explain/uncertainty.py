"""
Uncertainty quantification using Monte Carlo Dropout.

Estimates prediction uncertainty by running multiple stochastic forward passes
through dropout layers and analyzing the distribution of predictions.
"""

import logging
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from chest_xray_classifier.config.config import MC_DROPOUT_ITERS, CLASSES

logger = logging.getLogger(__name__)


class MCDropoutUncertainty:
    """
    Computes prediction uncertainty using Monte Carlo Dropout.
    
    By running multiple forward passes with dropout enabled during inference,
    we can estimate epistemic (model) uncertainty.
    """
    
    def __init__(self, model: keras.Model, n_iterations: int = MC_DROPOUT_ITERS):
        """
        Initialize MC Dropout uncertainty estimator.
        
        Args:
            model: Keras model with Dropout layers
            n_iterations: Number of stochastic forward passes
        """
        self.model = model
        self.n_iterations = n_iterations
        
        logger.info(f"MC Dropout Uncertainty initialized with {n_iterations} iterations")
    
    def predict_with_uncertainty(
        self,
        image: np.ndarray
    ) -> dict:
        """
        Get predictions with uncertainty estimates via MC Dropout.
        
        Args:
            image: Input image (224, 224, 3) or (1, 224, 224, 3)
            
        Returns:
            Dict with:
            - 'class_predictions': Mean class probabilities
            - 'predicted_class': Argmax class
            - 'confidence': Max probability
            - 'predictive_entropy': Entropy of mean predictions
            - 'variation_ratio': Ratio of non-majority votes
            - 'epistemic_uncertainty': Variance across MC samples
            - 'aleatoric_uncertainty': Expected variance (data uncertainty)
            - 'total_uncertainty': Sum of epistemic and aleatoric
            - 'mc_samples': All predictions from MC iterations
        """
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        image = image.astype(np.float32)
        
        # Enable dropout during inference
        mc_samples = []
        
        for i in range(self.n_iterations):
            # Use training=True to enable dropout
            pred = self.model(image, training=True)
            mc_samples.append(pred.numpy())
        
        # Stack all predictions
        mc_samples = np.array(mc_samples)  # Shape: (n_iter, batch_size, num_classes)
        
        # Compute statistics
        mean_pred = np.mean(mc_samples, axis=0)[0]  # Average across iterations
        std_pred = np.std(mc_samples, axis=0)[0]    # Std across iterations
        
        predicted_class = np.argmax(mean_pred)
        confidence = float(np.max(mean_pred))
        
        # 1. Predictive Entropy: -sum(p * log(p))
        predictive_entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10))
        
        # 2. Variation Ratio: ratio of samples that don't predict majority class
        mc_classes = np.argmax(mc_samples, axis=2)  # (n_iter, batch_size)
        mc_classes = mc_classes[:, 0]  # (n_iter,) for single image
        unique, counts = np.unique(mc_classes, return_counts=True)
        majority_count = np.max(counts)
        variation_ratio = 1.0 - (majority_count / self.n_iterations)
        
        # 3. Epistemic Uncertainty: mean variance across classes
        epistemic = np.mean(std_pred)
        
        # 4. Aleatoric (Data) Uncertainty: expected variance
        # Average variance of predicted distributions
        all_variances = np.var(mc_samples, axis=0)
        aleatoric = np.mean(all_variances[0])
        
        # 5. Total Uncertainty
        total_uncertainty = epistemic + aleatoric
        
        return {
            'class_predictions': mean_pred.tolist(),
            'predicted_class': int(predicted_class),
            'predicted_class_name': CLASSES[predicted_class],
            'confidence': confidence,
            'class_probabilities_std': std_pred.tolist(),
            'predictive_entropy': float(predictive_entropy),
            'variation_ratio': float(variation_ratio),
            'epistemic_uncertainty': float(epistemic),
            'aleatoric_uncertainty': float(aleatoric),
            'total_uncertainty': float(total_uncertainty),
            'mc_samples': mc_samples.tolist()  # All predictions
        }
    
    def batch_predict_with_uncertainty(
        self,
        images: np.ndarray
    ) -> list:
        """
        Get predictions with uncertainty for batch of images.
        
        Args:
            images: Batch of images (N, 224, 224, 3)
            
        Returns:
            List of uncertainty dicts (one per image)
        """
        logger.info(f"Computing uncertainties for {len(images)} images...")
        
        results = []
        for i, image in enumerate(images):
            result = self.predict_with_uncertainty(image)
            results.append(result)
            
            if (i + 1) % max(1, len(images) // 5) == 0:
                logger.info(f"  Processed {i + 1}/{len(images)} images")
        
        return results
    
    def is_high_uncertainty(
        self,
        uncertainty_dict: dict,
        threshold: float = 0.15
    ) -> bool:
        """
        Check if prediction has high uncertainty.
        
        Args:
            uncertainty_dict: Result from predict_with_uncertainty()
            threshold: Uncertainty threshold (default from config)
            
        Returns:
            True if total_uncertainty > threshold
        """
        return uncertainty_dict['total_uncertainty'] > threshold
    
    def confidence_score(
        self,
        uncertainty_dict: dict
    ) -> float:
        """
        Compute confidence score as inverse of uncertainty.
        
        Confidence = 1 / (1 + total_uncertainty)
        
        Args:
            uncertainty_dict: Result from predict_with_uncertainty()
            
        Returns:
            Confidence score in [0, 1]
        """
        total_unc = uncertainty_dict['total_uncertainty']
        return 1.0 / (1.0 + total_unc)


def ensemble_predictions(
    model_predictions: list,
    weights: list = None
) -> dict:
    """
    Aggregate predictions from multiple models.
    
    Args:
        model_predictions: List of prediction arrays (N, num_classes)
        weights: Optional weights for each model (default: equal)
        
    Returns:
        Dict with:
        - 'ensemble_prediction': Weighted average prediction
        - 'ensemble_class': Argmax class
        - 'agreement': How much models agree on predicted class
    """
    model_predictions = np.array(model_predictions)
    
    if weights is None:
        weights = np.ones(len(model_predictions)) / len(model_predictions)
    else:
        weights = np.array(weights) / np.sum(weights)
    
    # Weighted average
    ensemble_pred = np.average(model_predictions, axis=0, weights=weights)
    ensemble_class = np.argmax(ensemble_pred)
    
    # Compute agreement: how many models predict majority class
    model_classes = np.argmax(model_predictions, axis=1)
    agreement = np.sum(model_classes == ensemble_class) / len(model_classes)
    
    return {
        'ensemble_prediction': ensemble_pred.tolist(),
        'ensemble_class': int(ensemble_class),
        'ensemble_confidence': float(np.max(ensemble_pred)),
        'model_agreement': float(agreement),
        'num_models': len(model_predictions)
    }
