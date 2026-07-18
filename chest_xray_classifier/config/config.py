"""
Configuration module for Chest X-Ray Classifier.

This module contains all project constants, paths, hyperparameters, and settings.
No hardcoded values should appear anywhere else in the codebase.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# PROJECT PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"
MODEL_DIR = BASE_DIR / "model"
MODEL_EXPORT_DIR = BASE_DIR / "models"
ROOT_MODEL_EXPORT_DIR = BASE_DIR.parent / "models"
MODEL_PATH = MODEL_DIR / "chest_xray_model.keras"
EFFNET_PATH = MODEL_DIR / "efficientnet_model.keras"
MOBILENET_PATH = MODEL_DIR / "mobilenetv2_model.keras"
CUSTOM_CNN_H5_PATH = MODEL_EXPORT_DIR / "custom_cnn.h5"
EFFICIENTNETB0_H5_PATH = MODEL_EXPORT_DIR / "efficientnetb0.h5"
MOBILENETV2_H5_PATH = MODEL_EXPORT_DIR / "mobilenetv2.h5"
ROOT_CUSTOM_CNN_H5_PATH = ROOT_MODEL_EXPORT_DIR / "custom_cnn.h5"
ROOT_EFFICIENTNETB0_H5_PATH = ROOT_MODEL_EXPORT_DIR / "efficientnetb0.h5"
ROOT_MOBILENETV2_H5_PATH = ROOT_MODEL_EXPORT_DIR / "mobilenetv2.h5"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
DB_PATH = BASE_DIR / "predictions.db"
LOG_DIR = BASE_DIR / "logs"
ASSETS_DIR = BASE_DIR / "assets"
SAMPLE_IMAGES_DIR = ASSETS_DIR / "samples"

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    MODEL_DIR,
    MODEL_EXPORT_DIR,
    ROOT_MODEL_EXPORT_DIR,
    RESULTS_DIR,
    REPORTS_DIR,
    LOG_DIR,
    ASSETS_DIR,
    SAMPLE_IMAGES_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Disease classes
CLASSES: List[str] = [
    'Bacterial Pneumonia',
    'Covid-19',
    'Normal',
    'Viral Pneumonia'
]

NUM_CLASSES: int = len(CLASSES)

# Image configuration
IMG_SIZE: Tuple[int, int] = (224, 224)
IMG_SHAPE: Tuple[int, int, int] = (224, 224, 3)
BATCH_SIZE: int = 32

# Reproducibility
SEED: int = 42

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Custom CNN Training
EPOCHS: int = 10
LEARNING_RATE: float = 1e-3
DROPOUT_RATE: float = 0.4
L2_LAMBDA: float = 0.001

# EfficientNet Training
EFFNET_LR: float = 1e-4
EFFNET_PHASE1_EPOCHS: int = 5
EFFNET_PHASE2_EPOCHS: int = 5
FINETUNE_LR: float = 1e-5

# MobileNetV2 Training
MOBILENET_LR: float = 1e-4
MOBILENET_PHASE1_EPOCHS: int = 5
MOBILENET_PHASE2_EPOCHS: int = 3
MOBILENET_FINETUNE_LR: float = 1e-5

# Callback Configuration
PATIENCE_EARLY_STOP: int = 3
PATIENCE_LR: int = 2
LR_FACTOR: float = 0.5
MIN_LR: float = 1e-6

# Prediction Configuration
CONFIDENCE_THRESHOLD: float = 0.60
MC_DROPOUT_ITERS: int = 50

# ============================================================================
# MLflow CONFIGURATION
# ============================================================================

MLFLOW_EXPERIMENT_CNN: str = "chest_xray_cnn"
MLFLOW_EXPERIMENT_EFFNET: str = "chest_xray_efficientnet"

# ============================================================================
# VISUALIZATION & STYLING
# ============================================================================

# Class color mapping for consistent visualization
CLASS_COLORS: Dict[str, str] = {
    'Bacterial Pneumonia': '#E8593C',  # Red-orange
    'Covid-19': '#E24B4A',              # Red
    'Normal': '#1D9E75',                # Green
    'Viral Pneumonia': '#EF9F27'        # Orange
}

# Confidence level color mapping
CONFIDENCE_COLORS: Dict[str, str] = {
    'high': '#1D9E75',      # Green
    'medium': '#EF9F27',    # Orange
    'low': '#E24B4A'        # Red
}

# ============================================================================
# DATA AUGMENTATION CONFIGURATION
# ============================================================================

AUGMENTATION_PARAMS: Dict = {
    'rescale': 1.0 / 255.0,
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'shear_range': 0.1,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'brightness_range': [0.8, 1.2],
    'fill_mode': 'nearest'
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL: str = 'INFO'

# ============================================================================
# PDF REPORT CONFIGURATION
# ============================================================================

PDF_TITLE: str = "Chest X-Ray AI Diagnostic Report"
PDF_AUTHOR: str = "Chest X-Ray AI System"
PDF_FONT_SIZE_TITLE: int = 20
PDF_FONT_SIZE_NORMAL: int = 11
PDF_MARGIN: int = 10

# ============================================================================
# UNCERTAINTY QUANTIFICATION THRESHOLDS
# ============================================================================

HIGH_UNCERTAINTY_THRESHOLD: float = 0.15
MODERATE_UNCERTAINTY_THRESHOLD: float = 0.08

# ============================================================================
# INFERENCE BATCH SIZE (for faster processing)
# ============================================================================

INFERENCE_BATCH_SIZE: int = 64

# ============================================================================
# ALIASES FOR BACKWARD COMPATIBILITY
# ============================================================================

# Custom CNN aliases
EPOCHS_CNN = EPOCHS
LR_CNN = LEARNING_RATE

# EfficientNet aliases
EPOCHS_EFFNET_PHASE1 = EFFNET_PHASE1_EPOCHS
EPOCHS_EFFNET_PHASE2 = EFFNET_PHASE2_EPOCHS
LR_EFFNET_PHASE1 = EFFNET_LR
LR_EFFNET_PHASE2 = FINETUNE_LR

# MobileNetV2 aliases
EPOCHS_MOBILENET_PHASE1 = MOBILENET_PHASE1_EPOCHS
EPOCHS_MOBILENET_PHASE2 = MOBILENET_PHASE2_EPOCHS
LR_MOBILENET_PHASE1 = MOBILENET_LR
LR_MOBILENET_PHASE2 = MOBILENET_FINETUNE_LR

def get_class_index(class_name: str) -> int:
    """
    Get the numeric index of a class.
    
    Args:
        class_name: Name of the disease class
        
    Returns:
        Integer index (0-3) or -1 if not found
    """
    try:
        return CLASSES.index(class_name)
    except ValueError:
        return -1


def get_class_color(class_name: str) -> str:
    """
    Get the hex color for a given class.
    
    Args:
        class_name: Name of the disease class
        
    Returns:
        Hex color string
    """
    return CLASS_COLORS.get(class_name, '#999999')


def get_confidence_color(confidence: float) -> str:
    """
    Get the color label for a confidence score.
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Confidence level string: 'high', 'medium', or 'low'
    """
    if confidence >= CONFIDENCE_THRESHOLD:
        return 'high'
    elif confidence >= 0.50:
        return 'medium'
    else:
        return 'low'


def get_confidence_color_hex(confidence: float) -> str:
    """
    Get the hex color for a confidence score.
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Hex color string
    """
    level = get_confidence_color(confidence)
    return CONFIDENCE_COLORS.get(level, '#cccccc')
