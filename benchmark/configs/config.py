"""
Configuration file for transformer temperature forecasting benchmark.
Contains all hyperparameters and settings for reproducibility.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "benchmark" / "data"
RESULTS_DIR = BASE_DIR / "benchmark" / "results"
MODELS_DIR = BASE_DIR / "benchmark" / "models"

# Data settings
DATA_CONFIG = {
    'datasets': {
        'ETTh1': str(BASE_DIR / 'ETTh1.csv'),
        'ETTh2': str(BASE_DIR / 'ETTh2.csv'),
    },
    'target_column': 'OT',
    'feature_columns': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'],
    'date_column': 'date',
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
}

# Forecast horizons (in hours)
FORECAST_HORIZONS = {
    '24h': 24,
    '48h': 48,
    '1week': 168,  # 7 * 24
    '2weeks': 336,  # 14 * 24
}

# Sequence length for deep learning models (lookback window)
SEQUENCE_LENGTH = 168  # 1 week of hourly data

# Evaluation metrics
METRICS = ['rmse', 'mae', 'mape', 'r2', 'critical_temp_detection']
CRITICAL_TEMP_THRESHOLD = 95.0  # Critical temperature in °C

# Reproducibility
RANDOM_SEED = 42

# Computational settings
DEVICE = 'cuda'  # Will fallback to 'cpu' if CUDA not available
N_JOBS = -1  # For sklearn models (-1 means use all cores)

# Classical Models Hyperparameters
CLASSICAL_MODELS = {
    'arima': {
        'order': (5, 1, 0),  # (p, d, q)
        'seasonal_order': (1, 1, 1, 24),  # (P, D, Q, s) for SARIMA
    },
    'linear_regression': {
        'fit_intercept': True,
        'n_lags': 168,  # Number of lag features
    },
    'ridge': {
        'alpha': 1.0,
        'n_lags': 168,
    },
    'lasso': {
        'alpha': 0.1,
        'n_lags': 168,
    },
    'xgboost': {
        'n_estimators': 500,
        'max_depth': 7,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_SEED,
        'n_jobs': N_JOBS,
        'tree_method': 'hist',
    },
    'lightgbm': {
        'n_estimators': 500,
        'max_depth': 7,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_SEED,
        'n_jobs': N_JOBS,
        'verbose': -1,
    },
    'random_forest': {
        'n_estimators': 300,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_SEED,
        'n_jobs': N_JOBS,
    },
}

# Deep Learning Models Hyperparameters
DEEP_LEARNING_MODELS = {
    'lstm': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'batch_size': 64,
        'epochs': 50,
        'learning_rate': 0.001,
        'patience': 10,  # Early stopping patience
    },
    'bidirectional_lstm': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'batch_size': 64,
        'epochs': 50,
        'learning_rate': 0.001,
        'patience': 10,
    },
    'stacked_lstm': {
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'batch_size': 64,
        'epochs': 50,
        'learning_rate': 0.001,
        'patience': 10,
    },
    'gru': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'batch_size': 64,
        'epochs': 50,
        'learning_rate': 0.001,
        'patience': 10,
    },
    'cnn_lstm': {
        'cnn_channels': [32, 64],
        'kernel_size': 3,
        'lstm_hidden': 128,
        'lstm_layers': 2,
        'dropout': 0.2,
        'batch_size': 64,
        'epochs': 50,
        'learning_rate': 0.001,
        'patience': 10,
    },
    'transformer': {
        'd_model': 128,
        'nhead': 8,
        'num_layers': 3,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'batch_size': 64,
        'epochs': 50,
        'learning_rate': 0.0001,
        'patience': 10,
    },
    'tcn': {
        'num_channels': [64, 64, 64],
        'kernel_size': 3,
        'dropout': 0.2,
        'batch_size': 64,
        'epochs': 50,
        'learning_rate': 0.001,
        'patience': 10,
    },
}

# Advanced Time Series Models
ADVANCED_MODELS = {
    'nbeats': {
        'stack_types': ['trend', 'seasonality'],
        'num_blocks': [3, 3],
        'num_layers': 4,
        'layer_size': 256,
        'batch_size': 64,
        'epochs': 50,
        'learning_rate': 0.001,
        'patience': 10,
    },
    'nhits': {
        'stack_types': ['identity', 'identity', 'identity'],
        'num_blocks': [1, 1, 1],
        'num_layers': 2,
        'layer_size': 512,
        'batch_size': 64,
        'epochs': 50,
        'learning_rate': 0.001,
        'patience': 10,
    },
    'tft': {
        'hidden_size': 64,
        'attention_head_size': 4,
        'dropout': 0.1,
        'hidden_continuous_size': 16,
        'batch_size': 64,
        'max_epochs': 50,
        'learning_rate': 0.001,
        'patience': 10,
    },
    'deepar': {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.1,
        'batch_size': 64,
        'epochs': 50,
        'learning_rate': 0.001,
        'patience': 10,
        'num_samples': 100,  # For probabilistic forecasting
    },
}

# Ensemble settings
ENSEMBLE_CONFIG = {
    'method': 'adaptive',  # 'simple', 'weighted', 'stacking', 'adaptive'
    'sliding_window_size': 24,  # Hours for recent performance evaluation
    'uncertainty_weight': 0.3,  # Weight for uncertainty in adaptive ensemble
    'models_to_ensemble': ['lightgbm', 'lstm', 'tft', 'nbeats'],
}

# Robustness Experiments
ROBUSTNESS_CONFIG = {
    'missing_data_ratios': [0.1, 0.2, 0.3],
    'training_sizes': [0.3, 0.5, 0.7],  # Fraction of training data to use
    'transfer_learning': {
        'source_dataset': 'ETTh2',
        'target_dataset': 'ETTh1',
    },
    'seasonal_analysis': {
        'months': list(range(1, 13)),
    },
}

# Visualization settings
PLOT_CONFIG = {
    'figure_size': (12, 6),
    'dpi': 300,
    'font_size': 12,
    'style': 'seaborn-v0_8-darkgrid',
    'color_palette': 'Set2',
}

# LaTeX table settings
LATEX_CONFIG = {
    'float_format': '%.4f',
    'caption_prefix': 'Transformer Temperature Forecasting:',
    'label_prefix': 'tab:',
}

# Logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': str(RESULTS_DIR / 'logs' / 'benchmark.log'),
}
