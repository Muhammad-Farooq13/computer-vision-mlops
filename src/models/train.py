"""
Model Training Module
Handles model training for computer vision tasks
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CVModel(nn.Module):
    """Computer Vision Model Wrapper"""
    
    def __init__(self, model_name: str = "resnet50", num_classes: int = 10, 
                 pretrained: bool = True):
        """
        Initialize CV model
        
        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super(CVModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        if model_name == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
            
        elif model_name == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
            
        elif model_name == "efficientnet_b0":
            self.model = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, num_classes)
            
        elif model_name == "vgg16":
            self.model = models.vgg16(pretrained=pretrained)
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, num_classes)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        logger.info(f"Initialized {model_name} with {num_classes} classes")
        
    def forward(self, x):
        return self.model(x)


class ModelTrainer:
    """Class for training computer vision models"""
    
    def __init__(self, model: nn.Module, device: str = None):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            device: Device to train on (cuda/cpu)
        """
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        logger.info(f"Training on device: {self.device}")
        
    def train_epoch(self, train_loader: DataLoader, criterion, optimizer) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}')
                
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader: DataLoader, criterion) -> Tuple[float, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 10, lr: float = 0.001, 
              save_path: str = "models/saved_models") -> Dict:
        """
        Train model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            lr: Learning rate
            save_path: Path to save model
            
        Returns:
            Training history dictionary
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.1, patience=3)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            logger.info(f'\nEpoch {epoch+1}/{epochs}')
            
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            logger.info(f'Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%')
            
            scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(save_path, f"best_model.pth")
                logger.info(f'Saved best model with validation accuracy: {val_acc:.2f}%')
                
        # Save final model
        self.save_model(save_path, f"final_model.pth")
        
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': best_val_acc
        }
        
        # Save history
        history_path = Path(save_path) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
            
        return history
    
    def save_model(self, save_dir: str, filename: str):
        """
        Save model checkpoint
        
        Args:
            save_dir: Directory to save model
            filename: Filename for the model
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, save_path / filename)
        logger.info(f"Model saved to {save_path / filename}")
    
    def load_model(self, model_path: str):
        """
        Load model checkpoint
        
        Args:
            model_path: Path to model checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {model_path}")


if __name__ == "__main__":
    # Example usage
    model = CVModel(model_name="resnet18", num_classes=10)
    trainer = ModelTrainer(model)
    print(f"Model initialized on {trainer.device}")
