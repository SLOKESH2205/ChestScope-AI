"""
Unit tests for DataLoader module.

Tests image loading, preprocessing, and data generator functionality.

Run with: pytest tests/test_loader.py
"""

import pytest
import numpy as np
from pathlib import Path

from chest_xray_classifier.data.loader import DataLoader
from chest_xray_classifier.config.config import IMG_SIZE, CLASSES, TRAIN_DIR, VAL_DIR


class TestDataLoader:
    """Test suite for DataLoader class."""
    
    @pytest.fixture
    def loader(self):
        """Create DataLoader instance."""
        return DataLoader()
    
    # ===== Generator Tests =====
    
    def test_train_generator_creation(self, loader):
        """Test that training generator is created successfully."""
        train_gen = loader.get_train_generator()
        
        assert train_gen is not None
        assert train_gen.batch_size > 0
        assert len(train_gen) > 0
    
    def test_val_generator_creation(self, loader):
        """Test that validation generator is created successfully."""
        val_gen = loader.get_val_generator()
        
        assert val_gen is not None
        assert val_gen.batch_size > 0
        assert len(val_gen) > 0
    
    def test_generator_batch_shape(self, loader):
        """Test that generator outputs correct batch shape."""
        train_gen = loader.get_train_generator()
        batch_x, batch_y = next(iter(train_gen))
        
        assert batch_x.shape[1:3] == IMG_SIZE  # (batch, H, W, C)
        assert len(batch_y.shape) == 2  # (batch, num_classes)
        assert batch_y.shape[1] == len(CLASSES)
    
    def test_generator_image_range(self, loader):
        """Test that images are in [0, 1] range."""
        train_gen = loader.get_train_generator()
        batch_x, _ = next(iter(train_gen))
        
        assert np.min(batch_x) >= 0.0
        assert np.max(batch_x) <= 1.0 + 1e-6  # Allow small numerical error
    
    def test_generator_one_hot_labels(self, loader):
        """Test that labels are one-hot encoded."""
        train_gen = loader.get_train_generator()
        _, batch_y = next(iter(train_gen))
        
        # Sum of each label should be 1
        label_sums = np.sum(batch_y, axis=1)
        assert np.allclose(label_sums, 1.0)
    
    # ===== Class Distribution Tests =====
    
    def test_class_distribution(self, loader):
        """Test class distribution calculation."""
        dist_df = loader.get_class_distribution()
        
        assert len(dist_df) == len(CLASSES)
        assert 'Class' in dist_df.columns
        assert 'Total' in dist_df.columns
        assert dist_df['Total'].sum() > 0
    
    def test_class_distribution_balance(self, loader):
        """Test that dataset is reasonably balanced."""
        dist_df = loader.get_class_distribution()
        
        # Each class should have >0 samples
        assert (dist_df['Total'] > 0).all()
        
        # Check balance (all classes should have similar counts)
        mean_count = dist_df['Total'].mean()
        min_count = dist_df['Total'].min()
        
        # Min count should be at least 50% of mean
        assert min_count >= mean_count * 0.5
    
    # ===== Image Loading Tests =====
    
    def test_load_single_image(self, loader):
        """Test loading a single image."""
        # Get a valid image path
        class_dir = TRAIN_DIR / CLASSES[0]
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        if not image_files:
            pytest.skip("No test images found")
        
        image_path = image_files[0]
        image = loader.load_single_image(str(image_path))
        
        assert image.shape == (1, *IMG_SIZE)  # (1, 224, 224, 3)
        assert image.dtype == np.float32
        assert np.min(image) >= 0.0
        assert np.max(image) <= 1.0
    
    def test_load_single_image_not_found(self, loader):
        """Test error handling for non-existent image."""
        with pytest.raises(FileNotFoundError):
            loader.load_single_image('/nonexistent/image.jpg')
    
    def test_load_single_image_invalid_format(self, loader):
        """Test error handling for invalid image format."""
        # Create a temporary invalid file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b'not an image')
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                loader.load_single_image(temp_path)
        finally:
            Path(temp_path).unlink()
    
    # ===== Batch Loading Tests =====
    
    def test_load_test_images(self, loader):
        """Test loading batch of test images."""
        # This may not have actual test images, so we'll just check the function works
        images, filenames = loader.load_test_images()
        
        # Should be lists
        assert isinstance(images, list)
        assert isinstance(filenames, list)
        
        # If any images were found
        if len(images) > 0:
            assert images[0].shape == IMG_SIZE
            assert len(filenames) == len(images)
    
    # ===== Label Inference Tests =====
    
    def test_label_from_filename_bacteria(self, loader):
        """Test bacterial pneumonia label inference."""
        assert loader.get_label_from_filename('bacteria_001.jpg') == 0
        assert loader.get_label_from_filename('bacterial_pneumonia.png') == 0
    
    def test_label_from_filename_covid(self, loader):
        """Test COVID-19 label inference."""
        assert loader.get_label_from_filename('covid_patient_001.jpg') == 1
    
    def test_label_from_filename_normal(self, loader):
        """Test normal label inference."""
        assert loader.get_label_from_filename('normal_xray.jpg') == 2
        assert loader.get_label_from_filename('healthy_normal.png') == 2
    
    def test_label_from_filename_virus(self, loader):
        """Test viral pneumonia label inference."""
        assert loader.get_label_from_filename('viral_pneumonia.jpg') == 3
        assert loader.get_label_from_filename('virus_infection.png') == 3
    
    def test_label_from_filename_unknown(self, loader):
        """Test unknown label inference."""
        assert loader.get_label_from_filename('unknown_file.jpg') == -1
    
    # ===== Dataset Structure Tests =====
    
    def test_verify_dataset_structure(self, loader):
        """Test dataset structure verification."""
        is_valid = loader.verify_dataset_structure()
        
        # Should be True if dataset exists and is properly structured
        assert isinstance(is_valid, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
