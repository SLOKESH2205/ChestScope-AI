"""
Unit tests for the modular Chest X-Ray pipeline:
- Image preprocessing and validation
- Model loading and dummy prediction
- MC Dropout uncertainty estimation
- Invalid file and corruption handling
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from chest_xray_classifier.preprocessing import validate_image, preprocess_image
from chest_xray_classifier.predict import predict_with_uncertainty
from chest_xray_classifier.models import get_model_by_name

@pytest.fixture
def temp_images_dir():
    """Create a temporary directory for saving dummy images during test runs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

def test_valid_image_validation(temp_images_dir):
    """Test image validation logic with a clean valid PIL generated image."""
    img_path = temp_images_dir / "valid_image.png"
    img = Image.new("RGB", (256, 256), color="white")
    img.save(img_path)
    
    is_valid, msg = validate_image(img_path)
    assert is_valid is True
    assert "Valid" in msg

def test_invalid_image_validation(temp_images_dir):
    """Test validation on non-existent and empty files."""
    # 1. Non-existent file
    is_valid, msg = validate_image(temp_images_dir / "non_existent.jpg")
    assert is_valid is False
    assert "does not exist" in msg
    
    # 2. Empty file (0 bytes)
    empty_path = temp_images_dir / "empty.jpg"
    empty_path.write_bytes(b"")
    is_valid, msg = validate_image(empty_path)
    assert is_valid is False
    assert "empty" in msg

def test_corrupted_image_validation(temp_images_dir):
    """Test that a file filled with random non-image bytes fails validation."""
    corrupt_path = temp_images_dir / "corrupted.jpg"
    corrupt_path.write_bytes(b"random garbage bytes that are not an image header")
    
    is_valid, msg = validate_image(corrupt_path)
    assert is_valid is False
    assert "corruption" in msg or "failed to parse" in msg

def test_too_small_image_validation(temp_images_dir):
    """Test that validation fails for resolution below the minimum limit."""
    small_path = temp_images_dir / "too_small.png"
    img = Image.new("RGB", (16, 16), color="white")
    img.save(small_path)
    
    is_valid, msg = validate_image(small_path)
    assert is_valid is False
    assert "resolution too low" in msg

def test_image_preprocessing(temp_images_dir):
    """Test that preprocessing correctly resizes and scales an image to [0, 1]."""
    img_path = temp_images_dir / "test_preprocess.png"
    img = Image.new("RGB", (300, 300), color="blue")
    img.save(img_path)
    
    processed = preprocess_image(img_path, target_size=(224, 224))
    
    assert processed.shape == (1, 224, 224, 3)
    assert processed.max() <= 1.0
    assert processed.min() >= 0.0

def test_model_factory():
    """Verify that the model factory builds the architectures correctly."""
    model = get_model_by_name("Custom CNN")
    assert model.name == "custom_cnn"
    assert len(model.inputs) == 1
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 4)

def test_mc_dropout_prediction():
    """Test predict_with_uncertainty on a dummy model."""
    # Build a tiny custom cnn model for testing
    model = get_model_by_name("Custom CNN")
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    probs_dict, confidence, uncertainty = predict_with_uncertainty(model, dummy_input, num_mc_passes=3)
    
    assert len(probs_dict) == 4
    assert sum(probs_dict.values()) == pytest.approx(1.0, rel=1e-5)
    assert 0.0 <= confidence <= 1.0
    assert 0.0 <= uncertainty <= 1.0
