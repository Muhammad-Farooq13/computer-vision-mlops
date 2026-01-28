"""
Configuration Module
Handles project configuration and settings
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    """Configuration class for the project"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_default_config()
        
        if config_path:
            self.load_config(config_path)
            
    def _load_default_config(self) -> Dict:
        """Load default configuration"""
        return {
            'data': {
                'raw_dir': 'data/raw',
                'processed_dir': 'data/processed',
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15,
                'image_size': [224, 224],
                'batch_size': 32,
                'num_workers': 4
            },
            'model': {
                'architecture': 'resnet50',
                'pretrained': True,
                'num_classes': 10,
                'dropout': 0.5
            },
            'training': {
                'epochs': 50,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'optimizer': 'adam',
                'scheduler': 'reduce_on_plateau',
                'early_stopping_patience': 10
            },
            'paths': {
                'model_save_dir': 'models/saved_models',
                'log_dir': 'logs',
                'results_dir': 'results'
            },
            'deployment': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': False
            }
        }
    
    def load_config(self, config_path: str):
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file
        """
        path = Path(config_path)
        
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return
            
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'r') as f:
                loaded_config = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                loaded_config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
            
        # Update config with loaded values
        self._update_config(self.config, loaded_config)
        logger.info(f"Configuration loaded from {config_path}")
        
    def _update_config(self, base: Dict, update: Dict):
        """
        Recursively update configuration dictionary
        
        Args:
            base: Base configuration dictionary
            update: Update dictionary
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                self._update_config(base[key], value)
            else:
                base[key] = value
                
    def save_config(self, save_path: str):
        """
        Save configuration to file
        
        Args:
            save_path: Path to save configuration
        """
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=4)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
            
        logger.info(f"Configuration saved to {save_path}")
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key
        
        Args:
            key: Configuration key (supports nested keys with dots)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by key
        
        Args:
            key: Configuration key (supports nested keys with dots)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dictionary syntax"""
        return self.get(key)
        
    def __setitem__(self, key: str, value: Any):
        """Set configuration value using dictionary syntax"""
        self.set(key, value)


# Create global config instance
config = Config()


if __name__ == "__main__":
    # Example usage
    cfg = Config()
    print(f"Default learning rate: {cfg.get('training.learning_rate')}")
    
    # Save default config
    cfg.save_config('config.yaml')
    print("Default configuration saved to config.yaml")
