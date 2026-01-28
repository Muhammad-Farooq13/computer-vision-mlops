"""
Model Inference Module
Handles model predictions and inference
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import torchvision.transforms as transforms
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInference:
    """Class for model inference"""
    
    def __init__(self, model: nn.Module, model_path: str = None, 
                 device: str = None, class_names: List[str] = None):
        """
        Initialize inference engine
        
        Args:
            model: PyTorch model
            model_path: Path to model checkpoint
            device: Device to run inference on
            class_names: List of class names
        """
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        
        if model_path:
            self.load_model(model_path)
            
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Inference engine initialized on {self.device}")
        
    def load_model(self, model_path: str):
        """
        Load model from checkpoint
        
        Args:
            model_path: Path to model checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {model_path}")
        
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        tensor = self.transform(image_rgb)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
        
    def predict(self, image: np.ndarray, top_k: int = 5) -> Dict:
        """
        Make prediction on a single image
        
        Args:
            image: Input image
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions
        """
        # Preprocess
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, min(top_k, probabilities.size(1)))
        
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # Format results
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            pred = {
                'class_id': int(idx),
                'probability': float(prob)
            }
            
            if self.class_names and idx < len(self.class_names):
                pred['class_name'] = self.class_names[idx]
            else:
                pred['class_name'] = f"Class_{idx}"
                
            predictions.append(pred)
            
        result = {
            'predictions': predictions,
            'top_class': predictions[0]['class_name'],
            'top_probability': predictions[0]['probability']
        }
        
        return result
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Make predictions on a batch of images
        
        Args:
            images: List of input images
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for image in images:
            result = self.predict(image)
            results.append(result)
            
        return results
    
    def predict_from_path(self, image_path: str, top_k: int = 5) -> Dict:
        """
        Make prediction on image from file path
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions
        """
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        return self.predict(image, top_k=top_k)
    
    def visualize_prediction(self, image: np.ndarray, 
                           prediction: Dict, save_path: str = None) -> np.ndarray:
        """
        Visualize prediction on image
        
        Args:
            image: Input image
            prediction: Prediction dictionary
            save_path: Optional path to save visualization
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Add text
        text = f"{prediction['top_class']}: {prediction['top_probability']:.2%}"
        
        # Add background rectangle for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        cv2.rectangle(annotated, (10, 10), (20 + text_width, 20 + text_height), 
                     (0, 0, 0), -1)
        cv2.putText(annotated, text, (15, 15 + text_height), font, 
                   font_scale, (255, 255, 255), thickness)
        
        # Add top-3 predictions
        y_offset = 60
        for i, pred in enumerate(prediction['predictions'][:3]):
            text = f"{i+1}. {pred['class_name']}: {pred['probability']:.2%}"
            cv2.putText(annotated, text, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30
            
        if save_path:
            cv2.imwrite(save_path, annotated)
            logger.info(f"Visualization saved to {save_path}")
            
        return annotated


if __name__ == "__main__":
    print("Model Inference module")
