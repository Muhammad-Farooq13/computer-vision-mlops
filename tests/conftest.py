"""
Test configuration and fixtures
"""

import pytest
import torch
import numpy as np
import cv2
from pathlib import Path


@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_batch():
    """Create a sample batch of images"""
    batch = np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8)
    return batch


@pytest.fixture
def sample_model():
    """Create a sample model for testing"""
    from src.models.train import CVModel
    model = CVModel(model_name='resnet18', num_classes=10, pretrained=False)
    return model


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for testing"""
    return tmp_path


@pytest.fixture
def mock_config():
    """Create mock configuration"""
    return {
        'data': {
            'image_size': [224, 224],
            'batch_size': 4
        },
        'model': {
            'architecture': 'resnet18',
            'num_classes': 10
        },
        'training': {
            'epochs': 2,
            'learning_rate': 0.001
        }
    }
