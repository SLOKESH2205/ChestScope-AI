"""
Data loading module for Chest X-Ray Classifier.

Handles image loading, preprocessing, and data generator creation
with proper augmentation for training and validation datasets.
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from PIL import Image

from chest_xray_classifier.config.config import (
    CLASSES, IMG_SIZE, BATCH_SIZE, TRAIN_DIR, VAL_DIR, TEST_DIR,
    AUGMENTATION_PARAMS, SEED
)

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and preprocessing of chest X-ray images.
    
    Provides train/val data generators with appropriate augmentation,
    single image loading, and dataset analysis utilities.
    """
    
    def __init__(self, config=None):
        """
        Initialize DataLoader.
        
        Args:
            config: Configuration object (uses defaults from config.py if None)
        """
        self.config = config or type('Config', (), {
            'img_size': IMG_SIZE,
            'batch_size': BATCH_SIZE,
            'train_dir': TRAIN_DIR,
            'val_dir': VAL_DIR,
            'test_dir': TEST_DIR,
            'augmentation_params': AUGMENTATION_PARAMS,
            'seed': SEED,
            'classes': CLASSES
        })()
        
        logger.info(f"DataLoader initialized with IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE}")
    
    def get_train_generator(self) -> DirectoryIterator:
        """
        Create training data generator with augmentation.
        
        Returns:
            DirectoryIterator with augmented training data
        """
        logger.info(f"Creating training data generator from {self.config.train_dir}")
        
        train_datagen = ImageDataGenerator(
            rescale=self.config.augmentation_params['rescale'],
            rotation_range=self.config.augmentation_params['rotation_range'],
            width_shift_range=self.config.augmentation_params['width_shift_range'],
            height_shift_range=self.config.augmentation_params['height_shift_range'],
            shear_range=self.config.augmentation_params['shear_range'],
            zoom_range=self.config.augmentation_params['zoom_range'],
            horizontal_flip=self.config.augmentation_params['horizontal_flip'],
            brightness_range=self.config.augmentation_params['brightness_range'],
            fill_mode=self.config.augmentation_params['fill_mode']
        )
        
        train_generator = train_datagen.flow_from_directory(
            directory=str(self.config.train_dir),
            target_size=self.config.img_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=self.config.seed,
            classes=self.config.classes
        )
        
        logger.info(f"Training generator ready: {len(train_generator)} batches")
        return train_generator
    
    def get_val_generator(self) -> DirectoryIterator:
        """
        Create validation data generator (minimal augmentation).
        
        Returns:
            DirectoryIterator with validation data (rescale only)
        """
        logger.info(f"Creating validation data generator from {self.config.val_dir}")
        
        val_datagen = ImageDataGenerator(
            rescale=self.config.augmentation_params['rescale']
        )
        
        val_generator = val_datagen.flow_from_directory(
            directory=str(self.config.val_dir),
            target_size=self.config.img_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=self.config.seed,
            classes=self.config.classes
        )
        
        logger.info(f"Validation generator ready: {len(val_generator)} batches")
        return val_generator
    
    def get_class_distribution(self) -> pd.DataFrame:
        """
        Analyze class distribution in train/val splits.
        
        Returns:
            DataFrame with columns: Class, Train Count, Val Count, Total, % of Total
        """
        logger.info("Computing class distribution...")
        
        distribution = []
        total_train = 0
        total_val = 0
        
        for class_name in self.config.classes:
            train_count = len(list((self.config.train_dir / class_name).glob('*.jp*'))) + \
                         len(list((self.config.train_dir / class_name).glob('*.png')))
            val_count = len(list((self.config.val_dir / class_name).glob('*.jp*'))) + \
                       len(list((self.config.val_dir / class_name).glob('*.png')))
            
            total = train_count + val_count
            total_train += train_count
            total_val += val_count
            
            distribution.append({
                'Class': class_name,
                'Train Count': train_count,
                'Val Count': val_count,
                'Total': total
            })
        
        df = pd.DataFrame(distribution)
        grand_total = total_train + total_val
        df['% of Total'] = (df['Total'] / grand_total * 100).round(2)
        
        logger.info(f"\nClass Distribution:\n{df.to_string(index=False)}")
        return df
    
    def load_single_image(self, image_path: str) -> np.ndarray:
        """
        Load a single image, resize, and normalize.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image array of shape (1, 224, 224, 3) normalized to [0, 1]
            
        Raises:
            FileNotFoundError: If image file does not exist
            ValueError: If image cannot be loaded
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.config.img_size, Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            logger.debug(f"Loaded image: {image_path.name}, shape: {img_array.shape}")
            return img_array
            
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {str(e)}")
    
    def load_test_images(self, test_dir: Optional[str] = None) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load all images from test directory.
        
        Args:
            test_dir: Test directory path (uses config.test_dir if None)
            
        Returns:
            Tuple of (image_arrays list, filenames list)
        """
        test_dir = Path(test_dir or self.config.test_dir)
        
        if not test_dir.exists():
            logger.warning(f"Test directory not found: {test_dir}")
            return [], []
        
        logger.info(f"Loading test images from {test_dir}")
        
        image_arrays = []
        filenames = []
        
        for image_path in sorted(test_dir.glob('*.jp*')) + sorted(test_dir.glob('*.png')):
            try:
                img_array = self.load_single_image(str(image_path))
                image_arrays.append(img_array[0])  # Remove batch dim from single load
                filenames.append(image_path.name)
            except Exception as e:
                logger.warning(f"Failed to load {image_path.name}: {e}")
        
        logger.info(f"Loaded {len(image_arrays)} test images")
        return image_arrays, filenames
    
    def get_label_from_filename(self, filename: str) -> int:
        """
        Infer class label from filename pattern.
        
        Patterns:
        - 'bacteria' or 'bacterial' → 0 (Bacterial Pneumonia)
        - 'covid' → 1 (Covid-19)
        - 'normal' → 2 (Normal)
        - 'virus' or 'viral' → 3 (Viral Pneumonia)
        
        Args:
            filename: Image filename (case-insensitive)
            
        Returns:
            Class index (0-3) or -1 if unknown
        """
        filename_lower = filename.lower()
        
        if 'bacteria' in filename_lower:
            return 0
        elif 'covid' in filename_lower:
            return 1
        elif 'normal' in filename_lower:
            return 2
        elif 'virus' in filename_lower or 'viral' in filename_lower:
            return 3
        else:
            logger.debug(f"Could not infer label from filename: {filename}")
            return -1
    
    def verify_dataset_structure(self) -> bool:
        """
        Verify that dataset directories and classes exist.
        
        Returns:
            True if structure is valid, False otherwise
        """
        logger.info("Verifying dataset structure...")
        
        for split_dir in [self.config.train_dir, self.config.val_dir]:
            if not split_dir.exists():
                logger.error(f"Missing directory: {split_dir}")
                return False
            
            for class_name in self.config.classes:
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    logger.error(f"Missing class directory: {class_dir}")
                    return False
                
                imgs = list(class_dir.glob('*.jp*')) + list(class_dir.glob('*.png'))
                if len(imgs) == 0:
                    logger.warning(f"No images in {class_dir}")
        
        logger.info("Dataset structure verified successfully")
        return True
