"""
Data Loading Module
Handles loading of raw data from various sources
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Class for loading image data and annotations"""
    
    def __init__(self, data_dir: str):
        """
        Initialize DataLoader
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"
        self.valid_dir = self.data_dir / "valid"
        
    def load_annotations(self, split: str = "train") -> Dict:
        """
        Load annotations for a given split
        
        Args:
            split: Data split (train/test/valid)
            
        Returns:
            Dictionary containing annotations
        """
        split_dir = self.data_dir / split
        annotations_file = split_dir / "_annotations.txt"
        classes_file = split_dir / "_classes.txt"
        
        annotations = {}
        
        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                annotations['annotations'] = f.read()
                
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                annotations['classes'] = [line.strip() for line in f.readlines()]
                
        logger.info(f"Loaded annotations for {split} split")
        return annotations
    
    def load_images(self, split: str = "train", limit: int = None) -> List[np.ndarray]:
        """
        Load images from a split directory
        
        Args:
            split: Data split (train/test/valid)
            limit: Maximum number of images to load
            
        Returns:
            List of loaded images
        """
        split_dir = self.data_dir / split
        images = []
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in split_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if limit:
            image_files = image_files[:limit]
            
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            if img is not None:
                images.append(img)
                
        logger.info(f"Loaded {len(images)} images from {split} split")
        return images
    
    def get_dataset_stats(self) -> Dict:
        """
        Get statistics about the dataset
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {}
        
        for split in ['train', 'test', 'valid']:
            split_dir = self.data_dir / split
            if split_dir.exists():
                image_files = [f for f in split_dir.iterdir() 
                              if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                stats[split] = len(image_files)
                
        return stats


def move_raw_data_to_folder():
    """
    Move existing train/test/valid folders to data/raw directory
    """
    base_dir = Path(os.getcwd())
    raw_dir = base_dir / "data" / "raw"
    
    for split in ['train', 'test', 'valid']:
        source = base_dir / split
        destination = raw_dir / split
        
        if source.exists() and not destination.exists():
            logger.info(f"Note: Consider moving {split} folder to data/raw/ for better organization")
            
    return raw_dir


if __name__ == "__main__":
    # Example usage
    loader = DataLoader("./data/raw")
    stats = loader.get_dataset_stats()
    print(f"Dataset Statistics: {stats}")
