"""
Feature Extraction Module
Extracts features from images using various methods
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Class for extracting features from images"""
    
    def __init__(self, method: str = "resnet50"):
        """
        Initialize feature extractor
        
        Args:
            method: Feature extraction method (resnet50, vgg16, sift, orb)
        """
        self.method = method
        self.model = None
        
        if method in ["resnet50", "vgg16"]:
            self._load_deep_model()
            
    def _load_deep_model(self):
        """Load pretrained deep learning model"""
        if self.method == "resnet50":
            self.model = models.resnet50(pretrained=True)
            # Remove final classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        elif self.method == "vgg16":
            self.model = models.vgg16(pretrained=True)
            self.model.classifier = self.model.classifier[:-1]
            
        self.model.eval()
        logger.info(f"Loaded {self.method} model")
        
    def extract_deep_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features using deep learning model
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Feature vector
        """
        # Preprocessing
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess and add batch dimension
        input_tensor = preprocess(image_rgb).unsqueeze(0)
        
        # Extract features
        with torch.no_grad():
            features = self.model(input_tensor)
            
        return features.squeeze().numpy()
    
    def extract_sift_features(self, image: np.ndarray, 
                             n_features: int = 100) -> Tuple[List, np.ndarray]:
        """
        Extract SIFT features
        
        Args:
            image: Input image
            n_features: Number of features to extract
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(nfeatures=n_features)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def extract_orb_features(self, image: np.ndarray,
                            n_features: int = 500) -> Tuple[List, np.ndarray]:
        """
        Extract ORB features
        
        Args:
            image: Input image
            n_features: Number of features to extract
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=n_features)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def extract_color_histogram(self, image: np.ndarray, 
                               bins: int = 32) -> np.ndarray:
        """
        Extract color histogram features
        
        Args:
            image: Input image
            bins: Number of histogram bins
            
        Returns:
            Flattened histogram feature vector
        """
        hist_features = []
        
        for i in range(3):  # For each color channel
            hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)
            
        return np.array(hist_features)
    
    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HOG (Histogram of Oriented Gradients) features
        
        Args:
            image: Input image
            
        Returns:
            HOG feature vector
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 128))
        
        win_size = (64, 128)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, 
                               cell_size, nbins)
        features = hog.compute(resized)
        
        return features.flatten()
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features based on initialized method
        
        Args:
            image: Input image
            
        Returns:
            Feature vector
        """
        if self.method in ["resnet50", "vgg16"]:
            return self.extract_deep_features(image)
        elif self.method == "sift":
            _, descriptors = self.extract_sift_features(image)
            return descriptors
        elif self.method == "orb":
            _, descriptors = self.extract_orb_features(image)
            return descriptors
        elif self.method == "histogram":
            return self.extract_color_histogram(image)
        elif self.method == "hog":
            return self.extract_hog_features(image)
        else:
            raise ValueError(f"Unknown method: {self.method}")


if __name__ == "__main__":
    # Example usage
    extractor = FeatureExtractor(method="histogram")
    print("Feature Extractor initialized")
