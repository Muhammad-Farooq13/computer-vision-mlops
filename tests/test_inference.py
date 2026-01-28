"""
Unit tests for model inference functionality
"""

import pytest
import torch
import numpy as np

from src.models.predict import ModelInference
from src.models.train import CVModel


class TestModelInference:
    """Test cases for ModelInference class"""
    
    def test_init(self, sample_model):
        """Test ModelInference initialization"""
        inference = ModelInference(sample_model)
        assert inference.model is not None
        assert inference.device in ['cuda', 'cpu']
        
    def test_preprocess_image(self, sample_model, sample_image):
        """Test image preprocessing"""
        inference = ModelInference(sample_model)
        tensor = inference.preprocess_image(sample_image)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)
        
    def test_predict(self, sample_model, sample_image):
        """Test single image prediction"""
        inference = ModelInference(sample_model)
        result = inference.predict(sample_image, top_k=3)
        
        assert 'predictions' in result
        assert 'top_class' in result
        assert 'top_probability' in result
        assert len(result['predictions']) == 3
        
    def test_predict_batch(self, sample_model, sample_batch):
        """Test batch prediction"""
        inference = ModelInference(sample_model)
        results = inference.predict_batch(list(sample_batch))
        
        assert len(results) == 4
        assert all('predictions' in r for r in results)
        
    def test_visualize_prediction(self, sample_model, sample_image, temp_dir):
        """Test prediction visualization"""
        inference = ModelInference(sample_model)
        prediction = inference.predict(sample_image)
        
        save_path = temp_dir / 'prediction.jpg'
        annotated = inference.visualize_prediction(
            sample_image, 
            prediction, 
            save_path=str(save_path)
        )
        
        assert annotated.shape == sample_image.shape
        assert save_path.exists()
