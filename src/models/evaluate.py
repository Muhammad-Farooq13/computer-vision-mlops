"""
Model Evaluation Module
Handles model evaluation and metrics calculation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class for evaluating trained models"""
    
    def __init__(self, model: nn.Module, device: str = None):
        """
        Initialize evaluator
        
        Args:
            model: PyTorch model
            device: Device to evaluate on (cuda/cpu)
        """
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions for a dataset
        
        Args:
            data_loader: Data loader for the dataset
            
        Returns:
            Tuple of (predictions, ground truth labels)
        """
        predictions = []
        ground_truth = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                predictions.extend(predicted.cpu().numpy())
                ground_truth.extend(targets.numpy())
                
        return np.array(predictions), np.array(ground_truth)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate evaluation metrics
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def evaluate(self, data_loader: DataLoader, 
                class_names: List[str] = None) -> Dict:
        """
        Evaluate model on dataset
        
        Args:
            data_loader: Data loader for evaluation
            class_names: List of class names
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting evaluation...")
        
        predictions, ground_truth = self.predict(data_loader)
        metrics = self.calculate_metrics(ground_truth, predictions)
        
        # Confusion matrix
        cm = confusion_matrix(ground_truth, predictions)
        
        # Classification report
        if class_names:
            report = classification_report(ground_truth, predictions, 
                                          target_names=class_names, 
                                          output_dict=True)
        else:
            report = classification_report(ground_truth, predictions, 
                                          output_dict=True)
        
        results = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'predictions': predictions.tolist(),
            'ground_truth': ground_truth.tolist()
        }
        
        logger.info(f"Evaluation complete. Accuracy: {metrics['accuracy']:.4f}")
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str],
                             save_path: str = None):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    def plot_metrics(self, history: Dict, save_path: str = None):
        """
        Plot training metrics
        
        Args:
            history: Training history dictionary
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        epochs = range(1, len(history['train_losses']) + 1)
        ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics plot saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    def save_results(self, results: Dict, save_path: str):
        """
        Save evaluation results to JSON
        
        Args:
            results: Evaluation results dictionary
            save_path: Path to save results
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Results saved to {save_path}")


def compare_models(results_list: List[Dict], model_names: List[str]) -> None:
    """
    Compare multiple models
    
    Args:
        results_list: List of evaluation results dictionaries
        model_names: List of model names
    """
    comparison = {}
    
    for name, results in zip(model_names, results_list):
        comparison[name] = results['metrics']
        
    # Create comparison plot
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, name in enumerate(model_names):
        values = [comparison[name][m] for m in metrics]
        ax.bar(x + i * width, values, width, label=name)
        
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/model_comparison.png', dpi=300)
    logger.info("Model comparison plot saved")
    plt.close()


if __name__ == "__main__":
    print("Model Evaluator module")
