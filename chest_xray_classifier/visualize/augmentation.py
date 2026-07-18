"""
Data augmentation preview and robustness testing module.

Visualizes augmentation effects and tests model robustness against
various image transformations.
"""

import logging
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import plotly.graph_objects as go

from chest_xray_classifier.config.config import IMG_SIZE, AUGMENTATION_PARAMS

logger = logging.getLogger(__name__)


class AugmentationVisualizer:
    """
    Visualizes effects of data augmentation on images.
    """
    
    def __init__(self, augmentation_params: dict = None):
        """
        Initialize augmentation visualizer.
        
        Args:
            augmentation_params: Augmentation parameters dict
        """
        self.augmentation_params = augmentation_params or AUGMENTATION_PARAMS
        logger.info("AugmentationVisualizer initialized")
    
    def create_augmentation_preview(
        self,
        image: np.ndarray,
        num_variations: int = 9,
        save_path: str = None
    ) -> np.ndarray:
        """
        Create grid of augmented image variations.
        
        Args:
            image: Input image (224, 224, 3) in [0, 1] or [0, 255]
            num_variations: Number of augmented versions to generate
            save_path: Path to save grid image
            
        Returns:
            Grid image array (can be saved with PIL)
        """
        logger.info(f"Creating augmentation preview with {num_variations} variations...")
        
        # Ensure correct shape
        if len(image.shape) == 4:
            image = image[0]
        if image.max() > 1:
            image = image / 255.0
        
        # Create augmentation generator
        datagen = ImageDataGenerator(
            rescale=self.augmentation_params['rescale'],
            rotation_range=self.augmentation_params['rotation_range'],
            width_shift_range=self.augmentation_params['width_shift_range'],
            height_shift_range=self.augmentation_params['height_shift_range'],
            shear_range=self.augmentation_params['shear_range'],
            zoom_range=self.augmentation_params['zoom_range'],
            horizontal_flip=self.augmentation_params['horizontal_flip'],
            brightness_range=self.augmentation_params['brightness_range']
        )
        
        # Generate augmented images
        aug_images = [image]  # Original
        
        img_batch = np.expand_dims(image, axis=0)
        aug_iter = datagen.flow(img_batch, batch_size=1)
        
        for _ in range(num_variations - 1):
            aug_img = next(aug_iter)[0]
            aug_images.append(aug_img)
        
        # Create grid
        grid_size = int(np.ceil(np.sqrt(num_variations)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()
        
        for idx, aug_img in enumerate(aug_images):
            axes[idx].imshow(aug_img)
            axes[idx].set_title(f"Variation {idx}")
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(aug_images), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Augmentation preview saved to {save_path}")
        
        # Convert to array
        fig.canvas.draw()
        grid_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        grid_array = grid_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return grid_array
    
    def test_augmentation_robustness(
        self,
        model,
        image: np.ndarray,
        num_trials: int = 20
    ) -> dict:
        """
        Test model robustness by running predictions on augmented images.
        
        Args:
            model: Keras model
            image: Input image
            num_trials: Number of augmented versions to test
            
        Returns:
            Dict with robustness statistics:
            - 'mean_confidence': Average confidence across augmentations
            - 'std_confidence': Std of confidence
            - 'predicted_class_consistency': % of augmentations predicting same class
            - 'class_votes': Dict of class prediction counts
        """
        logger.info(f"Testing robustness with {num_trials} augmented versions...")
        
        # Prepare image
        if len(image.shape) == 4:
            image = image[0]
        if image.max() > 1:
            image = image / 255.0
        
        # Create augmentation generator
        datagen = ImageDataGenerator(
            rescale=self.augmentation_params['rescale'],
            rotation_range=self.augmentation_params['rotation_range'],
            width_shift_range=self.augmentation_params['width_shift_range'],
            height_shift_range=self.augmentation_params['height_shift_range'],
            shear_range=self.augmentation_params['shear_range'],
            zoom_range=self.augmentation_params['zoom_range'],
            horizontal_flip=self.augmentation_params['horizontal_flip'],
            brightness_range=self.augmentation_params['brightness_range']
        )
        
        # Run predictions on augmented images
        confidences = []
        predicted_classes = []
        
        img_batch = np.expand_dims(image, axis=0)
        
        # Original image
        pred = model.predict(img_batch, verbose=0)
        confidences.append(float(np.max(pred[0])))
        predicted_classes.append(int(np.argmax(pred[0])))
        
        # Augmented images
        aug_iter = datagen.flow(img_batch, batch_size=1)
        
        for _ in range(num_trials - 1):
            aug_img = next(aug_iter)
            pred = model.predict(aug_img, verbose=0)
            confidences.append(float(np.max(pred[0])))
            predicted_classes.append(int(np.argmax(pred[0])))
        
        # Compute statistics
        original_class = predicted_classes[0]
        class_votes = {}
        
        from chest_xray_classifier.config.config import CLASSES
        for class_idx, class_name in enumerate(CLASSES):
            class_votes[class_name] = predicted_classes.count(class_idx)
        
        consistency = class_votes[CLASSES[original_class]] / num_trials
        
        return {
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'predicted_class_consistency': float(consistency),
            'class_votes': class_votes,
            'num_trials': num_trials
        }


def visualize_specific_augmentations(
    image: np.ndarray,
    num_samples: int = 5,
    save_path: str = None
) -> dict:
    """
    Test model response to individual augmentation types.
    
    Applies: rotation, shift, zoom, rotation, flip, brightness.
    
    Args:
        image: Input image
        num_samples: Samples per augmentation type
        save_path: Path to save comparison plot
        
    Returns:
        Dict with augmentation test results
    """
    logger.info(f"Testing {6} augmentation types with {num_samples} samples each...")
    
    # Ensure correct shape
    if len(image.shape) == 4:
        image = image[0]
    if image.max() > 1:
        image = image / 255.0
    
    augmentation_types = {
        'rotation': ImageDataGenerator(rotation_range=45),
        'shift': ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2),
        'zoom': ImageDataGenerator(zoom_range=0.3),
        'shear': ImageDataGenerator(shear_range=0.2),
        'flip': ImageDataGenerator(horizontal_flip=True),
        'brightness': ImageDataGenerator(brightness_range=[0.7, 1.3])
    }
    
    results = {}
    img_batch = np.expand_dims(image, axis=0)
    
    for aug_name, datagen in augmentation_types.items():
        aug_iter = datagen.flow(img_batch, batch_size=1)
        images = []
        
        for _ in range(num_samples):
            aug_img = next(aug_iter)[0]
            images.append((aug_img * 255).astype(np.uint8) if aug_img.max() <= 1 else aug_img)
        
        results[aug_name] = images
    
    logger.info("Augmentation tests complete")
    return results
