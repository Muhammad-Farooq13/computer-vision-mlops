"""
Data Preprocessing Module
Handles cleaning and preprocessing of raw data
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Class for image preprocessing and augmentation"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size (width, height)
        """
        self.target_size = target_size
        
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        return cv2.resize(image, self.target_size)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Denormalize image from [0, 1] to [0, 255] range
        
        Args:
            image: Normalized image
            
        Returns:
            Denormalized image
        """
        return (image * 255.0).astype(np.uint8)
    
    def get_train_transforms(self):
        """
        Get training data augmentation transforms
        
        Returns:
            Albumentations transform pipeline
        """
        return A.Compose([
            A.Resize(height=self.target_size[1], width=self.target_size[0]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    
    def get_val_transforms(self):
        """
        Get validation data transforms
        
        Returns:
            Albumentations transform pipeline
        """
        return A.Compose([
            A.Resize(height=self.target_size[1], width=self.target_size[0]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    
    def preprocess_batch(self, images: List[np.ndarray], 
                        normalize: bool = True) -> np.ndarray:
        """
        Preprocess a batch of images
        
        Args:
            images: List of input images
            normalize: Whether to normalize images
            
        Returns:
            Preprocessed batch of images
        """
        preprocessed = []
        
        for img in images:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = self.resize_image(img)
            
            # Normalize
            if normalize:
                img = self.normalize_image(img)
                
            preprocessed.append(img)
            
        return np.array(preprocessed)
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def save_processed_data(images: np.ndarray, output_path: str):
    """
    Save processed data to disk
    
    Args:
        images: Processed images
        output_path: Output file path
    """
    np.save(output_path, images)
    logger.info(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    # Example usage
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    print("Image Preprocessor initialized")
