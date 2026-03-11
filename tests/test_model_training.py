"""
Unit tests for model training functionality
"""

import pytest

pytest.importorskip("torchvision")  # src/models/train.py requires torchvision

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.train import CVModel, ModelTrainer


class TestCVModel:
    """Test cases for CVModel class"""
    
    def test_init_resnet18(self):
        """Test CVModel initialization with ResNet18"""
        model = CVModel(model_name='resnet18', num_classes=10, pretrained=False)
        assert isinstance(model, nn.Module)
        
    def test_init_resnet50(self):
        """Test CVModel initialization with ResNet50"""
        model = CVModel(model_name='resnet50', num_classes=5, pretrained=False)
        assert isinstance(model, nn.Module)
        
    def test_forward_pass(self):
        """Test forward pass"""
        model = CVModel(model_name='resnet18', num_classes=10, pretrained=False)
        input_tensor = torch.randn(2, 3, 224, 224)
        output = model(input_tensor)
        
        assert output.shape == (2, 10)
        
    def test_invalid_model_name(self):
        """Test initialization with invalid model name"""
        with pytest.raises(ValueError):
            CVModel(model_name='invalid_model', num_classes=10)


class TestModelTrainer:
    """Test cases for ModelTrainer class"""
    
    def test_init(self, sample_model):
        """Test ModelTrainer initialization"""
        trainer = ModelTrainer(sample_model)
        assert trainer.model is not None
        assert trainer.device in ['cuda', 'cpu']
        
    def test_train_epoch(self, sample_model):
        """Test training for one epoch"""
        trainer = ModelTrainer(sample_model)
        
        # Create dummy data
        X = torch.randn(16, 3, 224, 224)
        y = torch.randint(0, 10, (16,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(sample_model.parameters(), lr=0.001)
        
        loss, acc = trainer.train_epoch(loader, criterion, optimizer)
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 100
        
    def test_validate(self, sample_model):
        """Test validation"""
        trainer = ModelTrainer(sample_model)
        
        # Create dummy data
        X = torch.randn(16, 3, 224, 224)
        y = torch.randint(0, 10, (16,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4)
        
        criterion = nn.CrossEntropyLoss()
        
        loss, acc = trainer.validate(loader, criterion)
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 100
