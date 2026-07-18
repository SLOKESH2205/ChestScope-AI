"""
Unit tests for Predictor module.

Tests single/batch prediction, confidence thresholding, and result formatting.

Run with: pytest tests/test_predictor.py
"""

import pytest
import numpy as np
from pathlib import Path
from tensorflow import keras

from chest_xray_classifier.predict.predictor import Predictor
from chest_xray_classifier.config.config import CLASSES, CONFIDENCE_THRESHOLD, IMG_SIZE, MODEL_PATH


class TestPredictor:
    """Test suite for Predictor class."""
    
    @pytest.fixture
    def model(self):
        """Create a dummy model for testing."""
        # Create simple sequential model
        model = keras.Sequential([
            keras.layers.Input(shape=IMG_SIZE),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(len(CLASSES), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model
    
    @pytest.fixture
    def predictor(self, model):
        """Create Predictor instance."""
        return Predictor(model, enable_uncertainty=False)  # Disable for faster tests
    
    @pytest.fixture
    def test_image(self):
        """Create dummy test image."""
        return np.random.rand(1, *IMG_SIZE).astype(np.float32)
    
    # ===== Single Prediction Tests =====
    
    def test_predict_single_returns_dict(self, predictor, test_image, tmp_path):
        """Test that predict_single returns valid dict."""
        # Save test image
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage
        img = PILImage.fromarray((test_image[0] * 255).astype(np.uint8))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)
        
        result = predictor.predict_single(str(img_path), include_uncertainty=False)
        
        assert isinstance(result, dict)
    
    def test_predict_single_has_required_keys(self, predictor, test_image, tmp_path):
        """Test that prediction dict has all required keys."""
        # Save test image
        from PIL import Image as PILImage
        img = PILImage.fromarray((test_image[0] * 255).astype(np.uint8))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)
        
        result = predictor.predict_single(str(img_path), include_uncertainty=False)
        
        required_keys = [
            'image_path', 'predicted_class', 'class_index',
            'confidence', 'all_probabilities', 'above_threshold'
        ]
        
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_predict_single_class_in_valid_range(self, predictor, test_image, tmp_path):
        """Test that predicted class index is valid."""
        from PIL import Image as PILImage
        img = PILImage.fromarray((test_image[0] * 255).astype(np.uint8))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)
        
        result = predictor.predict_single(str(img_path), include_uncertainty=False)
        
        assert 0 <= result['class_index'] < len(CLASSES)
        assert result['predicted_class'] in CLASSES
    
    def test_predict_single_confidence_range(self, predictor, test_image, tmp_path):
        """Test that confidence is in [0, 1]."""
        from PIL import Image as PILImage
        img = PILImage.fromarray((test_image[0] * 255).astype(np.uint8))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)
        
        result = predictor.predict_single(str(img_path), include_uncertainty=False)
        
        assert 0.0 <= result['confidence'] <= 1.0
    
    def test_predict_single_probabilities_sum(self, predictor, test_image, tmp_path):
        """Test that probabilities sum to 1."""
        from PIL import Image as PILImage
        img = PILImage.fromarray((test_image[0] * 255).astype(np.uint8))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)
        
        result = predictor.predict_single(str(img_path), include_uncertainty=False)
        
        prob_sum = sum(result['all_probabilities'].values())
        assert np.isclose(prob_sum, 1.0, atol=1e-5)
    
    def test_predict_single_file_not_found(self, predictor):
        """Test error handling for non-existent image."""
        with pytest.raises(FileNotFoundError):
            predictor.predict_single('/nonexistent/image.jpg')
    
    # ===== Batch Prediction Tests =====
    
    def test_predict_batch_returns_list(self, predictor, tmp_path):
        """Test that predict_batch returns list."""
        # Create test images
        from PIL import Image as PILImage
        
        image_paths = []
        for i in range(3):
            img = PILImage.fromarray(
                (np.random.rand(*IMG_SIZE) * 255).astype(np.uint8)
            )
            img_path = tmp_path / f"test_{i}.jpg"
            img.save(img_path)
            image_paths.append(img_path)
        
        results = predictor.predict_batch(image_paths, include_uncertainty=False)
        
        assert isinstance(results, list)
        assert len(results) == len(image_paths)
    
    def test_predict_batch_all_dicts(self, predictor, tmp_path):
        """Test that batch results are all dicts with valid keys."""
        from PIL import Image as PILImage
        
        image_paths = []
        for i in range(2):
            img = PILImage.fromarray(
                (np.random.rand(*IMG_SIZE) * 255).astype(np.uint8)
            )
            img_path = tmp_path / f"test_{i}.jpg"
            img.save(img_path)
            image_paths.append(img_path)
        
        results = predictor.predict_batch(image_paths, include_uncertainty=False)
        
        for result in results:
            assert isinstance(result, dict)
            assert 'predicted_class' in result
            assert 'confidence' in result
    
    def test_predict_batch_array_format(self, predictor, tmp_path):
        """Test predict_batch with array return format."""
        from PIL import Image as PILImage
        
        image_paths = []
        for i in range(2):
            img = PILImage.fromarray(
                (np.random.rand(*IMG_SIZE) * 255).astype(np.uint8)
            )
            img_path = tmp_path / f"test_{i}.jpg"
            img.save(img_path)
            image_paths.append(img_path)
        
        results = predictor.predict_batch(
            image_paths,
            include_uncertainty=False,
            return_format='array'
        )
        
        assert isinstance(results, np.ndarray)
        assert results.shape == (len(image_paths), len(CLASSES))
    
    # ===== Confidence Threshold Tests =====
    
    def test_above_threshold_flag(self, predictor, test_image, tmp_path):
        """Test above_threshold flag."""
        from PIL import Image as PILImage
        img = PILImage.fromarray((test_image[0] * 255).astype(np.uint8))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)
        
        result = predictor.predict_single(str(img_path), include_uncertainty=False)
        
        expected = result['confidence'] >= predictor.confidence_threshold
        assert result['above_threshold'] == expected
    
    # ===== Formatting Tests =====
    
    def test_ascii_card_format(self, predictor):
        """Test ASCII card formatting."""
        dummy_prediction = {
            'image_path': 'test.jpg',
            'predicted_class': 'Normal',
            'confidence': 0.95,
            'all_probabilities': {
                'Normal': 0.95,
                'Bacterial Pneumonia': 0.03,
                'Viral Pneumonia': 0.01,
                'Covid-19': 0.01
            }
        }
        
        card = predictor.format_to_ascii_card(dummy_prediction)
        
        assert isinstance(card, str)
        assert 'test.jpg' in card
        assert 'Normal' in card
        assert '95' in card  # Confidence percentage


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
