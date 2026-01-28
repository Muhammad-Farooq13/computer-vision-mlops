"""
Unit tests for data loading functionality
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from src.data.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class"""
    
    def test_init(self, temp_dir):
        """Test DataLoader initialization"""
        loader = DataLoader(str(temp_dir))
        assert loader.data_dir == temp_dir
        
    def test_get_dataset_stats_empty(self, temp_dir):
        """Test getting stats for empty dataset"""
        # Create split directories
        (temp_dir / 'train').mkdir()
        (temp_dir / 'test').mkdir()
        (temp_dir / 'valid').mkdir()
        
        loader = DataLoader(str(temp_dir))
        stats = loader.get_dataset_stats()
        
        assert stats['train'] == 0
        assert stats['test'] == 0
        assert stats['valid'] == 0
        
    def test_get_dataset_stats_with_images(self, temp_dir):
        """Test getting stats with images"""
        # Create split directories
        train_dir = temp_dir / 'train'
        train_dir.mkdir()
        
        # Create dummy images
        for i in range(3):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(train_dir / f'image_{i}.jpg'), img)
            
        loader = DataLoader(str(temp_dir))
        stats = loader.get_dataset_stats()
        
        assert stats['train'] == 3
        
    def test_load_annotations_missing_files(self, temp_dir):
        """Test loading annotations when files don't exist"""
        (temp_dir / 'train').mkdir()
        
        loader = DataLoader(str(temp_dir))
        annotations = loader.load_annotations('train')
        
        assert 'annotations' not in annotations
        assert 'classes' not in annotations
        
    def test_load_annotations_with_files(self, temp_dir):
        """Test loading annotations when files exist"""
        train_dir = temp_dir / 'train'
        train_dir.mkdir()
        
        # Create annotation files
        (train_dir / '_annotations.txt').write_text('test annotation')
        (train_dir / '_classes.txt').write_text('class1\nclass2\nclass3')
        
        loader = DataLoader(str(temp_dir))
        annotations = loader.load_annotations('train')
        
        assert 'annotations' in annotations
        assert 'classes' in annotations
        assert len(annotations['classes']) == 3
