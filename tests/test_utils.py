"""
Unit tests for utility functions
"""

import pytest
import numpy as np

pytest.importorskip("torch")  # src/utils/helpers.py requires torch

import torch

from src.utils.helpers import (
    set_seed, save_json, load_json,
    count_parameters, get_device, format_time,
)
from src.utils.config import Config


class TestHelpers:
    """Test cases for helper functions"""
    
    def test_set_seed(self):
        """Test random seed setting"""
        set_seed(42)
        val1 = np.random.rand()
        
        set_seed(42)
        val2 = np.random.rand()
        
        assert val1 == val2
        
    def test_save_and_load_json(self, temp_dir):
        """Test JSON save and load"""
        data = {'key': 'value', 'number': 42}
        filepath = temp_dir / 'test.json'
        
        save_json(data, str(filepath))
        loaded = load_json(str(filepath))
        
        assert data == loaded
        
    def test_count_parameters(self):
        """Test parameter counting with a simple inline model"""
        import torch.nn as nn

        model = nn.Linear(10, 5)
        count = count_parameters(model)
        assert isinstance(count, int)
        assert count > 0
        
    def test_get_device(self):
        """Test device detection"""
        device = get_device()
        assert isinstance(device, torch.device)
        
    def test_format_time(self):
        """Test time formatting"""
        assert format_time(30) == "30s"
        assert format_time(90) == "1m 30s"
        assert format_time(3661) == "1h 1m 1s"


class TestConfig:
    """Test cases for Config class"""
    
    def test_init(self):
        """Test Config initialization"""
        config = Config()
        assert config.config is not None
        
    def test_get_default(self):
        """Test getting default values"""
        config = Config()
        value = config.get('data.batch_size')
        assert isinstance(value, int)
        
    def test_get_with_default(self):
        """Test getting with default value"""
        config = Config()
        value = config.get('nonexistent.key', default=42)
        assert value == 42
        
    def test_set(self):
        """Test setting values"""
        config = Config()
        config.set('test.key', 'test_value')
        assert config.get('test.key') == 'test_value'
        
    def test_save_and_load_config(self, temp_dir):
        """Test config save and load"""
        config = Config()
        config.set('custom.value', 123)
        
        filepath = temp_dir / 'config.yaml'
        config.save_config(str(filepath))
        
        new_config = Config(str(filepath))
        assert new_config.get('custom.value') == 123
