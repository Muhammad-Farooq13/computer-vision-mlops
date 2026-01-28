"""
Logging Utilities
Custom logging configuration for the project
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logger(name: str = __name__, 
                log_dir: str = 'logs',
                level: int = logging.INFO) -> logging.Logger:
    """
    Setup and configure logger
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler with rotation
    timestamp = datetime.now().strftime('%Y%m%d')
    log_file = log_path / f'{name}_{timestamp}.log'
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""
    
    @property
    def logger(self):
        """Get logger for the class"""
        name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return logging.getLogger(name)


if __name__ == "__main__":
    # Example usage
    logger = setup_logger('test_logger')
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
