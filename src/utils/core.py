"""
Core utilities for the Hospital Length of Stay prediction project.

This module provides utility functions for seeding, device management,
and common operations used throughout the project.
"""

import random
import numpy as np
import torch
from typing import Optional, Union, Dict, Any
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_deterministic_seed(seed: int = 42) -> None:
    """
    Set deterministic seed for all random number generators.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set deterministic seed to {seed}")


def get_device() -> torch.device:
    """
    Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return if denominator is zero
        
    Returns:
        float: The division result or default value
    """
    if abs(denominator) < 1e-8:
        return default
    return numerator / denominator


def format_metrics(metrics: Dict[str, float], precision: int = 3) -> Dict[str, str]:
    """
    Format metrics dictionary with specified precision.
    
    Args:
        metrics: Dictionary of metric names and values
        precision: Number of decimal places
        
    Returns:
        Dict[str, str]: Formatted metrics dictionary
    """
    return {k: f"{v:.{precision}f}" for k, v in metrics.items()}


def validate_config(config: Dict[str, Any], required_keys: list) -> None:
    """
    Validate configuration dictionary has required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: Optional[Any] = None) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score (higher is better)
            model: Model to save weights from
            
        Returns:
            bool: True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and model is not None and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        
        return False


def deidentify_text(text: str) -> str:
    """
    Basic de-identification for demonstration purposes.
    
    Note: This is a simple regex-based approach for demo purposes only.
    For production use, consider using specialized tools like Presidio.
    
    Args:
        text: Text to de-identify
        
    Returns:
        str: De-identified text
    """
    import re
    
    # Replace common PHI patterns (simplified for demo)
    patterns = [
        (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),  # SSN
        (r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]'),  # Phone
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),  # Email
        (r'\b\d{1,2}/\d{1,2}/\d{4}\b', '[DATE]'),  # Date
    ]
    
    deidentified = text
    for pattern, replacement in patterns:
        deidentified = re.sub(pattern, replacement, deidentified)
    
    return deidentified
