"""
Preprocessing module for Chest X-Ray Disease Classification.

Handles automatic image validation, corruption detection, consistent resizing,
normalization, data augmentation parameters, and dataset report generation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import os

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Standard configurations
IMG_SIZE = (224, 224)
CLASSES = ['Bacterial Pneumonia', 'Covid-19', 'Normal', 'Viral Pneumonia']

def validate_image(image_path: str | Path) -> Tuple[bool, str]:
    """
    Validate an image file: checks if it exists, is not corrupted,
    can be opened by PIL, and has valid dimensions/channels.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (is_valid: bool, status_message: str)
    """
    path = Path(image_path)
    if not path.exists():
        return False, f"File does not exist: {image_path}"
    
    if not path.is_file():
        return False, f"Path is not a file: {image_path}"
        
    # Check file size (should not be empty)
    if path.stat().st_size == 0:
        return False, "File is empty (0 bytes)"
        
    try:
        with Image.open(path) as img:
            img.verify()  # Verify image integrity (catches truncation and corruption)
        
        # Re-open because verify() closes the file but can leave it in an unusable state
        with Image.open(path) as img:
            width, height = img.size
            if width < 32 or height < 32:
                return False, f"Image resolution too low: {width}x{height}"
            
            # Check channels/mode
            if img.mode not in ('RGB', 'L', 'RGBA'):
                return False, f"Unsupported image mode/channels: {img.mode}"
                
            return True, "Valid image"
    except Exception as exc:
        return False, f"Image corruption detected or failed to parse: {exc}"

def preprocess_image(image_path: str | Path, target_size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """
    Load, validate, convert to RGB, resize, and normalize a single image to [0, 1].
    
    Args:
        image_path: Path to the image file
        target_size: Target resolution as (width, height)
        
    Returns:
        Numpy array of shape (1, height, width, 3) normalized to [0, 1]
        
    Raises:
        ValueError: If image validation fails or preprocessing fails
    """
    is_valid, err_msg = validate_image(image_path)
    if not is_valid:
        raise ValueError(f"Invalid image: {err_msg}")
        
    try:
        with Image.open(image_path) as img:
            rgb_img = img.convert('RGB')
            resized_img = rgb_img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(resized_img, dtype=np.float32) / 255.0
            return np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as exc:
        raise ValueError(f"Preprocessing failed for {image_path}: {exc}")

def generate_dataset_report(
    train_dir: str | Path,
    val_dir: str | Path,
    test_dir: str | Path,
    output_path: str | Path,
    classes: list[str] = CLASSES
) -> Dict[str, Any]:
    """
    Generate dataset summary report and save to JSON.
    
    Args:
        train_dir: Path to training split
        val_dir: Path to validation split
        test_dir: Path to test split
        output_path: Output JSON file path
        classes: List of class directories to count
        
    Returns:
        Report summary dictionary
    """
    report: Dict[str, Any] = {
        "metadata": {
            "image_resolution": list(IMG_SIZE),
            "classes": classes,
        },
        "splits": {},
        "summary": {
            "total_images": 0,
            "corrupted_images_detected": 0,
            "corrupted_files": []
        }
    }
    
    splits = {
        "train": Path(train_dir),
        "validation": Path(val_dir),
        "test": Path(test_dir)
    }
    
    # Standard augmentation settings (re-extracted for reporting)
    report["augmentation_settings"] = {
        "rescale": "1.0 / 255.0",
        "rotation_range": 15,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "shear_range": 0.1,
        "zoom_range": 0.2,
        "horizontal_flip": True,
        "brightness_range": [0.8, 1.2],
        "fill_mode": "nearest"
    }
    
    for split_name, split_path in splits.items():
        report["splits"][split_name] = {
            "total_count": 0,
            "class_counts": {}
        }
        
        if not split_path.exists():
            continue
            
        # Count classes
        for class_name in classes:
            class_path = split_path / class_name
            if not class_path.exists():
                report["splits"][split_name]["class_counts"][class_name] = 0
                continue
                
            # Grab all image files
            files = []
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
                files.extend(class_path.glob(ext))
                files.extend(class_path.glob(ext.upper()))
                
            valid_count = 0
            for f in files:
                is_valid, msg = validate_image(f)
                if is_valid:
                    valid_count += 1
                else:
                    report["summary"]["corrupted_images_detected"] += 1
                    report["summary"]["corrupted_files"].append({
                        "file": str(f),
                        "split": split_name,
                        "class": class_name,
                        "reason": msg
                    })
                    
            report["splits"][split_name]["class_counts"][class_name] = valid_count
            report["splits"][split_name]["total_count"] += valid_count
            report["summary"]["total_images"] += valid_count
            
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    
    logger.info(f"Dataset summary report written to {output_path}")
    return report
