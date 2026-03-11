"""
Unit tests for data preprocessing functionality
"""

import pytest
import numpy as np

pytest.importorskip("albumentations")  # src/data/data_preprocessing.py requires albumentations
pytest.importorskip("cv2")

import cv2

from src.data.data_preprocessing import ImagePreprocessor


class TestImagePreprocessor:
    """Test cases for ImagePreprocessor class"""
    
    def test_init(self):
        """Test ImagePreprocessor initialization"""
        preprocessor = ImagePreprocessor(target_size=(256, 256))
        assert preprocessor.target_size == (256, 256)
        
    def test_resize_image(self, sample_image):
        """Test image resizing"""
        preprocessor = ImagePreprocessor(target_size=(128, 128))
        resized = preprocessor.resize_image(sample_image)
        
        assert resized.shape[:2] == (128, 128)
        
    def test_normalize_image(self, sample_image):
        """Test image normalization"""
        preprocessor = ImagePreprocessor()
        normalized = preprocessor.normalize_image(sample_image)
        
        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        
    def test_denormalize_image(self):
        """Test image denormalization"""
        preprocessor = ImagePreprocessor()
        normalized = np.random.rand(224, 224, 3).astype(np.float32)
        denormalized = preprocessor.denormalize_image(normalized)
        
        assert denormalized.dtype == np.uint8
        assert denormalized.min() >= 0
        assert denormalized.max() <= 255
        
    def test_preprocess_batch(self, sample_batch):
        """Test batch preprocessing"""
        preprocessor = ImagePreprocessor(target_size=(128, 128))
        preprocessed = preprocessor.preprocess_batch(list(sample_batch))
        
        assert preprocessed.shape[0] == 4
        assert preprocessed.shape[1:3] == (128, 128)
        
    def test_apply_clahe(self, sample_image):
        """Test CLAHE application"""
        preprocessor = ImagePreprocessor()
        enhanced = preprocessor.apply_clahe(sample_image)
        
        assert enhanced.shape == sample_image.shape
        assert enhanced.dtype == sample_image.dtype
