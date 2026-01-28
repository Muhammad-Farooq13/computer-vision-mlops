"""
Visualization Module
Functions for visualizing data, predictions, and results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_images_grid(images: List[np.ndarray], titles: List[str] = None,
                    rows: int = 3, cols: int = 3, figsize: Tuple = (15, 15),
                    save_path: str = None):
    """
    Plot a grid of images
    
    Args:
        images: List of images to plot
        titles: List of titles for each image
        rows: Number of rows in grid
        cols: Number of columns in grid
        figsize: Figure size
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, (ax, img) in enumerate(zip(axes, images)):
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Convert BGR to RGB for matplotlib
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        ax.imshow(img)
        ax.axis('off')
        
        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=10)
            
    # Hide unused subplots
    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Grid plot saved to {save_path}")
    else:
        plt.show()
        
    plt.close()


def plot_image_with_prediction(image: np.ndarray, prediction: Dict,
                               ground_truth: str = None, save_path: str = None):
    """
    Plot image with prediction
    
    Args:
        image: Input image
        prediction: Prediction dictionary
        ground_truth: Ground truth label
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Convert BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    ax.imshow(image)
    ax.axis('off')
    
    # Create title
    title = f"Predicted: {prediction['top_class']} ({prediction['top_probability']:.2%})"
    if ground_truth:
        title += f"\nGround Truth: {ground_truth}"
        
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add top predictions as text
    text_str = "Top Predictions:\n"
    for i, pred in enumerate(prediction['predictions'][:5], 1):
        text_str += f"{i}. {pred['class_name']}: {pred['probability']:.2%}\n"
        
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Prediction plot saved to {save_path}")
    else:
        plt.show()
        
    plt.close()


def plot_class_distribution(labels: List, class_names: List[str] = None,
                           save_path: str = None):
    """
    Plot class distribution
    
    Args:
        labels: List of labels
        class_names: List of class names
        save_path: Path to save the figure
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(12, 6))
    
    if class_names:
        x_labels = [class_names[i] if i < len(class_names) else f"Class {i}" 
                   for i in unique]
    else:
        x_labels = [f"Class {i}" for i in unique]
        
    plt.bar(x_labels, counts, color='steelblue', edgecolor='black')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (label, count) in enumerate(zip(x_labels, counts)):
        plt.text(i, count, str(count), ha='center', va='bottom')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")
    else:
        plt.show()
        
    plt.close()


def plot_training_history(history: Dict, save_path: str = None):
    """
    Plot training history
    
    Args:
        history: Training history dictionary
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Plot losses
    axes[0].plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracies
    axes[1].plot(epochs, history['train_accuracies'], 'b-', 
                label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_accuracies'], 'r-', 
                label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    else:
        plt.show()
        
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                         normalize: bool = False, save_path: str = None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
        
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Count'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
        
    plt.close()


def visualize_augmentations(image: np.ndarray, transform,
                           n_augmentations: int = 9, save_path: str = None):
    """
    Visualize data augmentations
    
    Args:
        image: Original image
        transform: Augmentation transform
        n_augmentations: Number of augmented versions to show
        save_path: Path to save the figure
    """
    rows = int(np.sqrt(n_augmentations))
    cols = int(np.ceil(n_augmentations / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes[:n_augmentations]):
        # Apply augmentation
        augmented = transform(image=image)['image']
        
        # Convert to numpy if tensor
        if hasattr(augmented, 'numpy'):
            augmented = augmented.numpy().transpose(1, 2, 0)
            
        ax.imshow(augmented)
        ax.axis('off')
        ax.set_title(f'Augmentation {idx + 1}', fontsize=10)
        
    # Hide unused subplots
    for idx in range(n_augmentations, len(axes)):
        axes[idx].axis('off')
        
    plt.suptitle('Data Augmentations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Augmentation visualization saved to {save_path}")
    else:
        plt.show()
        
    plt.close()


def create_video_from_images(images: List[np.ndarray], output_path: str,
                            fps: int = 30):
    """
    Create video from list of images
    
    Args:
        images: List of images
        output_path: Path to save video
        fps: Frames per second
    """
    if not images:
        logger.error("No images provided")
        return
        
    height, width = images[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for img in images:
        out.write(img)
        
    out.release()
    logger.info(f"Video saved to {output_path}")


if __name__ == "__main__":
    print("Visualization module")
