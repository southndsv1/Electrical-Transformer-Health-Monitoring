"""
Utility functions for the benchmarking framework.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import random
import logging
from pathlib import Path
from typing import Any, Dict
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import RANDOM_SEED, RESULTS_DIR, LOGGING_CONFIG


def set_seed(seed: int = RANDOM_SEED):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get the best available device (CUDA, MPS, or CPU).

    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def setup_logger(name: str = 'benchmark', log_file: str = None) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name
        log_file: Path to log file (optional)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOGGING_CONFIG['level']))

    # Remove existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(LOGGING_CONFIG['format'])

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is None:
        log_file = LOGGING_CONFIG['log_file']

    # Create directory if it doesn't exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def save_results(results: Dict, filepath: str, format: str = 'json'):
    """
    Save results to file.

    Args:
        results: Results dictionary
        filepath: Path to save file
        format: Format ('json', 'csv', 'pickle')
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format == 'csv':
        if isinstance(results, dict):
            df = pd.DataFrame([results])
        else:
            df = pd.DataFrame(results)
        df.to_csv(filepath, index=False)
    elif format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Results saved to {filepath}")


def load_results(filepath: str, format: str = 'json') -> Dict:
    """
    Load results from file.

    Args:
        filepath: Path to results file
        format: Format ('json', 'csv', 'pickle')

    Returns:
        Results dictionary or DataFrame
    """
    if format == 'json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif format == 'csv':
        return pd.read_csv(filepath)
    elif format == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown format: {format}")


def save_model(model: Any, filepath: str):
    """
    Save model to file.

    Args:
        model: Model object
        filepath: Path to save model
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, torch.nn.Module):
        # Save PyTorch model
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
        }, filepath)
    else:
        # Save sklearn model or other
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)

    print(f"Model saved to {filepath}")


def load_model(filepath: str, model_class: Any = None):
    """
    Load model from file.

    Args:
        filepath: Path to model file
        model_class: Model class for PyTorch models

    Returns:
        Loaded model
    """
    if filepath.endswith('.pt') or filepath.endswith('.pth'):
        # Load PyTorch model
        checkpoint = torch.load(filepath)
        if model_class is not None:
            model = model_class()
            model.load_state_dict(checkpoint['model_state_dict'])
            return model
        else:
            return checkpoint
    else:
        # Load sklearn model or other
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def create_experiment_dir(experiment_name: str, base_dir: Path = None) -> Path:
    """
    Create directory for experiment results.

    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for experiments

    Returns:
        Path to experiment directory
    """
    if base_dir is None:
        base_dir = RESULTS_DIR

    exp_dir = base_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / 'plots').mkdir(exist_ok=True)
    (exp_dir / 'tables').mkdir(exist_ok=True)
    (exp_dir / 'models').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)

    return exp_dir


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_memory(bytes: int) -> str:
    """
    Format memory in bytes to human-readable string.

    Args:
        bytes: Memory in bytes

    Returns:
        Formatted memory string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


class EarlyStopping:
    """
    Early stopping handler for training.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0,
                 mode: str = 'min', verbose: bool = True):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' (whether lower or higher is better)
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, epoch: int, score: float) -> bool:
        """
        Check if should stop training.

        Args:
            epoch: Current epoch
            score: Current validation score

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch}")
                    print(f"Best score: {self.best_score:.6f} at epoch {self.best_epoch}")
                return True

        return False


class ProgressTracker:
    """
    Track progress of experiments with progress bar.
    """

    def __init__(self, total: int, desc: str = "Progress"):
        """
        Initialize progress tracker.

        Args:
            total: Total number of items
            desc: Description for progress bar
        """
        from tqdm import tqdm
        self.pbar = tqdm(total=total, desc=desc)

    def update(self, n: int = 1):
        """Update progress."""
        self.pbar.update(n)

    def close(self):
        """Close progress bar."""
        self.pbar.close()

    def set_description(self, desc: str):
        """Set description."""
        self.pbar.set_description(desc)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in PyTorch model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gpu_memory():
    """
    Get current GPU memory usage.

    Returns:
        Dictionary with memory statistics
    """
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated(),
            'reserved': torch.cuda.memory_reserved(),
            'max_allocated': torch.cuda.max_memory_allocated(),
        }
    else:
        return None


if __name__ == '__main__':
    # Test utilities
    print("Testing utilities...")

    # Test seed setting
    set_seed(42)
    print(f"Random number: {np.random.random()}")

    # Test device
    device = get_device()
    print(f"Device: {device}")

    # Test logger
    logger = setup_logger('test')
    logger.info("Test log message")

    # Test time formatting
    print(f"Time: {format_time(3665)}")

    # Test memory formatting
    print(f"Memory: {format_memory(1024 * 1024 * 100)}")

    # Test early stopping
    early_stop = EarlyStopping(patience=3, verbose=False)
    for epoch in range(10):
        score = 1.0 - epoch * 0.05 if epoch < 5 else 0.75 + epoch * 0.01
        if early_stop(epoch, score):
            print(f"Stopped at epoch {epoch}")
            break

    print("All tests passed!")
