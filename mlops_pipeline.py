"""
MLOps Pipeline
Handles continuous integration, deployment, and model monitoring
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import mlflow
import mlflow.pytorch
import torch
from torch.utils.data import DataLoader

from src.data.data_loader import DataLoader as CVDataLoader
from src.data.data_preprocessing import ImagePreprocessor
from src.models.train import CVModel, ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.helpers import set_seed, get_device

logger = setup_logger('mlops_pipeline', 'logs')


class MLOpsPipeline:
    """MLOps Pipeline for model training, evaluation, and deployment"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize MLOps Pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.logger = logger
        
        # Set random seed for reproducibility
        set_seed(42)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.config.get('mlflow.tracking_uri', 'mlruns'))
        mlflow.set_experiment(self.config.get('mlflow.experiment_name', 'cv_model'))
        
        self.logger.info("MLOps Pipeline initialized")
        
    def prepare_data(self) -> tuple:
        """
        Prepare data for training
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        self.logger.info("Preparing data...")
        
        # Load data
        data_dir = self.config.get('data.raw_dir')
        loader = CVDataLoader(data_dir)
        
        # Get dataset statistics
        stats = loader.get_dataset_stats()
        self.logger.info(f"Dataset statistics: {stats}")
        
        # TODO: Implement actual data loading and preprocessing
        # This is a placeholder - you'll need to implement based on your data format
        self.logger.info("Data preparation completed")
        
        return None, None, None
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        Train model with MLflow tracking
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history
        """
        self.logger.info("Starting model training...")
        
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params({
                'model_architecture': self.config.get('model.architecture'),
                'learning_rate': self.config.get('training.learning_rate'),
                'batch_size': self.config.get('data.batch_size'),
                'epochs': self.config.get('training.epochs'),
                'optimizer': self.config.get('training.optimizer')
            })
            
            # Initialize model
            model = CVModel(
                model_name=self.config.get('model.architecture'),
                num_classes=self.config.get('model.num_classes'),
                pretrained=self.config.get('model.pretrained')
            )
            
            # Initialize trainer
            trainer = ModelTrainer(model)
            
            # Train model
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.config.get('training.epochs'),
                lr=self.config.get('training.learning_rate'),
                save_path=self.config.get('paths.model_save_dir')
            )
            
            # Log metrics
            for epoch in range(len(history['train_losses'])):
                mlflow.log_metrics({
                    'train_loss': history['train_losses'][epoch],
                    'val_loss': history['val_losses'][epoch],
                    'train_accuracy': history['train_accuracies'][epoch],
                    'val_accuracy': history['val_accuracies'][epoch]
                }, step=epoch)
            
            # Log best metrics
            mlflow.log_metrics({
                'best_val_accuracy': history['best_val_accuracy']
            })
            
            # Log model
            model_path = Path(self.config.get('paths.model_save_dir')) / 'best_model.pth'
            mlflow.pytorch.log_model(model, "model")
            
            # Log artifacts
            mlflow.log_artifact(str(model_path))
            
            self.logger.info(f"Training completed. Run ID: {run.info.run_id}")
            
        return history
    
    def evaluate_model(self, test_loader: DataLoader, model_path: str) -> Dict:
        """
        Evaluate trained model
        
        Args:
            test_loader: Test data loader
            model_path: Path to trained model
            
        Returns:
            Evaluation results
        """
        self.logger.info("Evaluating model...")
        
        # Load model
        model = CVModel(
            model_name=self.config.get('model.architecture'),
            num_classes=self.config.get('model.num_classes'),
            pretrained=False
        )
        
        # Initialize evaluator
        evaluator = ModelEvaluator(model)
        evaluator.load_model(model_path)
        
        # Evaluate
        results = evaluator.evaluate(test_loader)
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_metrics(results['metrics'])
            
            # Save results
            results_path = Path(self.config.get('paths.results_dir')) / 'evaluation_results.json'
            evaluator.save_results(results, str(results_path))
            mlflow.log_artifact(str(results_path))
        
        self.logger.info(f"Evaluation completed. Accuracy: {results['metrics']['accuracy']:.4f}")
        
        return results
    
    def version_model(self, model_path: str, version: str = None) -> str:
        """
        Version and register model
        
        Args:
            model_path: Path to model checkpoint
            version: Version string (auto-generated if None)
            
        Returns:
            Version identifier
        """
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
            
        # Create versioned directory
        version_dir = Path(self.config.get('paths.model_save_dir')) / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model
        import shutil
        shutil.copy(model_path, version_dir / 'model.pth')
        
        # Save metadata
        metadata = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'architecture': self.config.get('model.architecture'),
            'num_classes': self.config.get('model.num_classes')
        }
        
        with open(version_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
            
        self.logger.info(f"Model versioned: {version}")
        
        return version
    
    def run_pipeline(self):
        """Run complete MLOps pipeline"""
        self.logger.info("=" * 50)
        self.logger.info("Starting MLOps Pipeline")
        self.logger.info("=" * 50)
        
        try:
            # Step 1: Data preparation
            train_loader, val_loader, test_loader = self.prepare_data()
            
            # Step 2: Model training
            if train_loader and val_loader:
                history = self.train_model(train_loader, val_loader)
                
                # Step 3: Model evaluation
                model_path = Path(self.config.get('paths.model_save_dir')) / 'best_model.pth'
                if test_loader and model_path.exists():
                    results = self.evaluate_model(test_loader, str(model_path))
                    
                    # Step 4: Model versioning
                    version = self.version_model(str(model_path))
                    
                    self.logger.info("Pipeline completed successfully!")
                    self.logger.info(f"Model version: {version}")
                    self.logger.info(f"Test accuracy: {results['metrics']['accuracy']:.4f}")
            else:
                self.logger.warning("Data loaders not available. Pipeline incomplete.")
                
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
            
        self.logger.info("=" * 50)
    
    def monitor_model(self, predictions: List, ground_truth: List) -> Dict:
        """
        Monitor model performance in production
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
            
        Returns:
            Monitoring metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        metrics = {
            'accuracy': accuracy_score(ground_truth, predictions),
            'precision': precision_score(ground_truth, predictions, average='weighted', zero_division=0),
            'recall': recall_score(ground_truth, predictions, average='weighted', zero_division=0),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
            
        # Save monitoring data
        monitoring_dir = Path('logs') / 'monitoring'
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        monitoring_file = monitoring_dir / f"monitoring_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Append to monitoring log
        monitoring_data = []
        if monitoring_file.exists():
            with open(monitoring_file, 'r') as f:
                monitoring_data = json.load(f)
                
        monitoring_data.append(metrics)
        
        with open(monitoring_file, 'w') as f:
            json.dump(monitoring_data, f, indent=4)
            
        self.logger.info(f"Model monitoring logged: Accuracy={metrics['accuracy']:.4f}")
        
        return metrics


def main():
    """Main function to run MLOps pipeline"""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='MLOps Pipeline')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'train', 'evaluate', 'monitor'],
                       help='Pipeline mode')
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MLOpsPipeline(config_path=args.config)
    
    # Run pipeline based on mode
    if args.mode == 'full':
        pipeline.run_pipeline()
    elif args.mode == 'train':
        train_loader, val_loader, _ = pipeline.prepare_data()
        if train_loader and val_loader:
            pipeline.train_model(train_loader, val_loader)
    elif args.mode == 'evaluate':
        _, _, test_loader = pipeline.prepare_data()
        model_path = Path(pipeline.config.get('paths.model_save_dir')) / 'best_model.pth'
        if test_loader and model_path.exists():
            pipeline.evaluate_model(test_loader, str(model_path))
    
    logger.info("Pipeline execution completed")


if __name__ == "__main__":
    main()
