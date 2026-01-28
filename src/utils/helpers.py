"""
Helper Utilities
General utility functions for the project
"""

import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Any, List
import json
import pickle
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed}")


def save_json(data: Any, filepath: str):
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filepath: Path to save file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
        
    logger.info(f"Data saved to {filepath}")


def load_json(filepath: str) -> Any:
    """
    Load data from JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    logger.info(f"Data loaded from {filepath}")
    return data


def save_pickle(data: Any, filepath: str):
    """
    Save data to pickle file
    
    Args:
        data: Data to save
        filepath: Path to save file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        
    logger.info(f"Data saved to {filepath}")


def load_pickle(filepath: str) -> Any:
    """
    Load data from pickle file
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        
    logger.info(f"Data loaded from {filepath}")
    return data


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """
    Get available device (CUDA or CPU)
    
    Returns:
        PyTorch device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
    return device


def create_directories(dirs: List[str]):
    """
    Create multiple directories
    
    Args:
        dirs: List of directory paths to create
    """
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    logger.info(f"Created {len(dirs)} directories")


def get_file_size(filepath: str) -> str:
    """
    Get human-readable file size
    
    Args:
        filepath: Path to file
        
    Returns:
        File size as string
    """
    size = os.path.getsize(filepath)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
        
    return f"{size:.2f} TB"


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0, 
                 verbose: bool = True):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float):
        """
        Check if training should stop
        
        Args:
            val_loss: Current validation loss
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


if __name__ == "__main__":
    # Example usage
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
