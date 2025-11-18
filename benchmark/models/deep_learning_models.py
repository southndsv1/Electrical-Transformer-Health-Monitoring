"""
Deep learning models for time series forecasting.

Includes:
- LSTM (vanilla, bidirectional, stacked)
- GRU
- CNN-LSTM hybrid
- Transformer
- Temporal Convolutional Network (TCN)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
from typing import Tuple, Dict, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import DEEP_LEARNING_MODELS, RANDOM_SEED
from evaluation.metrics import calculate_all_metrics
from utils.helpers import set_seed, get_device, EarlyStopping

set_seed(RANDOM_SEED)


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BaseDeepLearningModel:
    """Base class for deep learning models."""

    def __init__(self, model_name: str, model: nn.Module, config: Dict):
        self.model_name = model_name
        self.model = model
        self.config = config
        self.device = get_device()
        self.model.to(self.device)

        self.training_time = 0
        self.inference_time = 0
        self.train_losses = []
        self.val_losses = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model."""
        print(f"Training {self.model_name}...")
        start_time = time.time()

        # Create datasets and data loaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )

        if X_val is not None:
            val_dataset = TimeSeriesDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False
            )

        # Setup optimizer and loss
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )
        criterion = nn.MSELoss()

        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config['patience'],
            mode='min',
            verbose=True
        )

        # Training loop
        for epoch in range(self.config['epochs']):
            # Train
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            if X_val is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        y_pred = self.model(X_batch)
                        loss = criterion(y_pred, y_batch)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                self.val_losses.append(val_loss)

                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{self.config['epochs']}: "
                          f"Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

                # Early stopping check
                if early_stopping(epoch, val_loss):
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{self.config['epochs']}: "
                          f"Train Loss = {train_loss:.6f}")

        self.training_time = time.time() - start_time
        print(f"  Training completed in {self.training_time:.2f}s")

    def predict(self, X):
        """Make predictions."""
        self.model.eval()

        dataset = TimeSeriesDataset(X, np.zeros((len(X), X.shape[1])))  # Dummy y
        loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False)

        predictions = []
        start_time = time.time()

        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                y_pred = self.model(X_batch)
                predictions.append(y_pred.cpu().numpy())

        self.inference_time = time.time() - start_time

        return np.concatenate(predictions, axis=0)

    def evaluate(self, X, y) -> Dict:
        """Evaluate model."""
        y_pred = self.predict(X)

        # For multi-horizon forecasting, evaluate each horizon
        if len(y.shape) > 1 and y.shape[1] > 1:
            # Average across all horizons for now
            y_mean = y.mean(axis=1)
            y_pred_mean = y_pred.mean(axis=1)
            metrics = calculate_all_metrics(y_mean, y_pred_mean)
        else:
            if len(y.shape) > 1:
                y = y.squeeze()
            if len(y_pred.shape) > 1:
                y_pred = y_pred.squeeze()
            metrics = calculate_all_metrics(y, y_pred)

        metrics['training_time'] = self.training_time
        metrics['inference_time'] = self.inference_time

        return metrics


# ============================================================================
# LSTM Models
# ============================================================================

class LSTMModel(nn.Module):
    """
    Vanilla LSTM for time series forecasting.

    Why relevant:
    - Captures long-term dependencies
    - Handles sequential data naturally
    - Remembers important past information via gates
    - Mitigates vanishing gradient problem

    Architecture:
    - LSTM layers with dropout
    - Fully connected output layer
    """

    def __init__(self, input_size, hidden_size, num_layers, forecast_horizon, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take the last time step
        out = self.fc(lstm_out[:, -1, :])
        return out


class BidirectionalLSTMModel(nn.Module):
    """
    Bidirectional LSTM.

    Why relevant:
    - Processes sequence in both directions
    - Captures future context as well as past
    - Better for pattern recognition in complete sequences
    - Useful when entire sequence is available

    Architecture:
    - Bidirectional LSTM layers
    - Concatenates forward and backward hidden states
    - FC layer for output
    """

    def __init__(self, input_size, hidden_size, num_layers, forecast_horizon, dropout=0.2):
        super(BidirectionalLSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # *2 because bidirectional
        self.fc = nn.Linear(hidden_size * 2, forecast_horizon)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class StackedLSTMModel(nn.Module):
    """
    Stacked LSTM with more layers.

    Why relevant:
    - Deeper architecture for complex patterns
    - Each layer learns different abstraction levels
    - Better for hierarchical temporal features
    - More parameters for complex relationships

    Architecture:
    - Multiple LSTM layers stacked
    - Dropout between layers
    - FC output layer
    """

    def __init__(self, input_size, hidden_size, num_layers, forecast_horizon, dropout=0.2):
        super(StackedLSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


# ============================================================================
# GRU Model
# ============================================================================

class GRUModel(nn.Module):
    """
    GRU (Gated Recurrent Unit).

    Why relevant:
    - Simpler than LSTM (fewer parameters)
    - Faster training than LSTM
    - Often performs comparably to LSTM
    - Better for smaller datasets
    - Less prone to overfitting

    Architecture:
    - GRU layers with dropout
    - FC output layer
    """

    def __init__(self, input_size, hidden_size, num_layers, forecast_horizon, dropout=0.2):
        super(GRUModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out


# ============================================================================
# CNN-LSTM Hybrid
# ============================================================================

class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM Hybrid model.

    Why relevant:
    - CNN extracts local features from time series
    - LSTM captures long-term dependencies
    - Best of both worlds: local patterns + temporal dynamics
    - Reduces input dimensionality for LSTM
    - Often faster than pure LSTM

    Architecture:
    - 1D CNN layers for feature extraction
    - LSTM for temporal modeling
    - FC for output
    """

    def __init__(self, input_size, cnn_channels, kernel_size, lstm_hidden,
                 lstm_layers, forecast_horizon, dropout=0.2):
        super(CNNLSTMModel, self).__init__()

        self.cnn_layers = nn.ModuleList()
        in_channels = 1

        # CNN layers
        for out_channels in cnn_channels:
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout)
                )
            )
            in_channels = out_channels

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        self.fc = nn.Linear(lstm_hidden, forecast_horizon)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # For 1D CNN, need (batch, channels, seq_len)

        # Average across features if multivariate
        if x.shape[-1] > 1:
            x = x.mean(dim=-1, keepdim=True)

        x = x.permute(0, 2, 1)  # (batch, features, seq_len)

        # CNN layers
        for cnn in self.cnn_layers:
            x = cnn(x)

        # Back to (batch, seq_len, features)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])

        return out


# ============================================================================
# Transformer
# ============================================================================

class TransformerModel(nn.Module):
    """
    Transformer for time series forecasting.

    Why relevant:
    - Attention mechanism captures dependencies at any distance
    - Parallelizable (faster training than RNNs)
    - No vanishing gradient issues
    - Excellent for long sequences
    - State-of-the-art in many sequence tasks

    Architecture:
    - Positional encoding
    - Multi-head attention layers
    - Feed-forward networks
    - Layer normalization
    - FC output layer
    """

    def __init__(self, input_size, d_model, nhead, num_layers,
                 dim_feedforward, forecast_horizon, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output layer
        self.fc = nn.Linear(d_model, forecast_horizon)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        seq_len = x.shape[1]

        # Project input
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.pos_encoder[:, :seq_len, :]

        # Transformer
        x = self.transformer(x)

        # Use last time step for prediction
        out = self.fc(x[:, -1, :])

        return out


# ============================================================================
# TCN (Temporal Convolutional Network)
# ============================================================================

class TCNBlock(nn.Module):
    """Temporal convolutional block with residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TCNBlock, self).__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]]  # Causal: remove future
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual
        res = x if self.downsample is None else self.downsample(x)
        res = res[:, :, -out.shape[2]:]  # Match size

        return self.relu(out + res)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network.

    Why relevant:
    - Dilated causal convolutions capture long-range dependencies
    - Parallelizable (faster than RNNs)
    - Flexible receptive field
    - No vanishing gradient
    - Excellent for sequence modeling

    Architecture:
    - Multiple TCN blocks with increasing dilation
    - Residual connections
    - FC output layer
    """

    def __init__(self, input_size, num_channels, kernel_size, forecast_horizon, dropout=0.2):
        super(TCNModel, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers.append(
                TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], forecast_horizon)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # TCN needs (batch, features, seq_len)
        x = x.permute(0, 2, 1)

        # TCN
        x = self.network(x)

        # Use last time step
        x = x[:, :, -1]

        # Output
        out = self.fc(x)

        return out


# ============================================================================
# Factory functions
# ============================================================================

def create_lstm_model(input_size, forecast_horizon, config=None):
    """Create LSTM model."""
    if config is None:
        config = DEEP_LEARNING_MODELS['lstm']

    model = LSTMModel(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        forecast_horizon=forecast_horizon,
        dropout=config['dropout']
    )

    return BaseDeepLearningModel('LSTM', model, config)


def create_bidirectional_lstm_model(input_size, forecast_horizon, config=None):
    """Create Bidirectional LSTM model."""
    if config is None:
        config = DEEP_LEARNING_MODELS['bidirectional_lstm']

    model = BidirectionalLSTMModel(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        forecast_horizon=forecast_horizon,
        dropout=config['dropout']
    )

    return BaseDeepLearningModel('Bidirectional LSTM', model, config)


def create_stacked_lstm_model(input_size, forecast_horizon, config=None):
    """Create Stacked LSTM model."""
    if config is None:
        config = DEEP_LEARNING_MODELS['stacked_lstm']

    model = StackedLSTMModel(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        forecast_horizon=forecast_horizon,
        dropout=config['dropout']
    )

    return BaseDeepLearningModel('Stacked LSTM', model, config)


def create_gru_model(input_size, forecast_horizon, config=None):
    """Create GRU model."""
    if config is None:
        config = DEEP_LEARNING_MODELS['gru']

    model = GRUModel(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        forecast_horizon=forecast_horizon,
        dropout=config['dropout']
    )

    return BaseDeepLearningModel('GRU', model, config)


def create_cnn_lstm_model(input_size, forecast_horizon, config=None):
    """Create CNN-LSTM model."""
    if config is None:
        config = DEEP_LEARNING_MODELS['cnn_lstm']

    model = CNNLSTMModel(
        input_size=input_size,
        cnn_channels=config['cnn_channels'],
        kernel_size=config['kernel_size'],
        lstm_hidden=config['lstm_hidden'],
        lstm_layers=config['lstm_layers'],
        forecast_horizon=forecast_horizon,
        dropout=config['dropout']
    )

    return BaseDeepLearningModel('CNN-LSTM', model, config)


def create_transformer_model(input_size, forecast_horizon, config=None):
    """Create Transformer model."""
    if config is None:
        config = DEEP_LEARNING_MODELS['transformer']

    model = TransformerModel(
        input_size=input_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        forecast_horizon=forecast_horizon,
        dropout=config['dropout']
    )

    return BaseDeepLearningModel('Transformer', model, config)


def create_tcn_model(input_size, forecast_horizon, config=None):
    """Create TCN model."""
    if config is None:
        config = DEEP_LEARNING_MODELS['tcn']

    model = TCNModel(
        input_size=input_size,
        num_channels=config['num_channels'],
        kernel_size=config['kernel_size'],
        forecast_horizon=forecast_horizon,
        dropout=config['dropout']
    )

    return BaseDeepLearningModel('TCN', model, config)


def train_all_deep_learning_models(data_dict: Dict, forecast_horizon: int = 24) -> Tuple[Dict, Dict]:
    """
    Train all deep learning models.

    Args:
        data_dict: Dictionary with sequence data
        forecast_horizon: Forecast horizon in hours

    Returns:
        Tuple of (results_dict, models_dict)
    """
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    input_size = data_dict['n_features']

    results = {}
    models = {}

    print("\n" + "=" * 60)
    print("Training Deep Learning Models")
    print("=" * 60)

    # LSTM variants
    lstm_model = create_lstm_model(input_size, forecast_horizon)
    lstm_model.fit(X_train, y_train, X_val, y_val)
    results['lstm'] = lstm_model.evaluate(X_test, y_test)
    models['lstm'] = lstm_model

    bi_lstm_model = create_bidirectional_lstm_model(input_size, forecast_horizon)
    bi_lstm_model.fit(X_train, y_train, X_val, y_val)
    results['bidirectional_lstm'] = bi_lstm_model.evaluate(X_test, y_test)
    models['bidirectional_lstm'] = bi_lstm_model

    stacked_lstm_model = create_stacked_lstm_model(input_size, forecast_horizon)
    stacked_lstm_model.fit(X_train, y_train, X_val, y_val)
    results['stacked_lstm'] = stacked_lstm_model.evaluate(X_test, y_test)
    models['stacked_lstm'] = stacked_lstm_model

    # GRU
    gru_model = create_gru_model(input_size, forecast_horizon)
    gru_model.fit(X_train, y_train, X_val, y_val)
    results['gru'] = gru_model.evaluate(X_test, y_test)
    models['gru'] = gru_model

    # CNN-LSTM
    cnn_lstm_model = create_cnn_lstm_model(input_size, forecast_horizon)
    cnn_lstm_model.fit(X_train, y_train, X_val, y_val)
    results['cnn_lstm'] = cnn_lstm_model.evaluate(X_test, y_test)
    models['cnn_lstm'] = cnn_lstm_model

    # Transformer
    transformer_model = create_transformer_model(input_size, forecast_horizon)
    transformer_model.fit(X_train, y_train, X_val, y_val)
    results['transformer'] = transformer_model.evaluate(X_test, y_test)
    models['transformer'] = transformer_model

    # TCN
    tcn_model = create_tcn_model(input_size, forecast_horizon)
    tcn_model.fit(X_train, y_train, X_val, y_val)
    results['tcn'] = tcn_model.evaluate(X_test, y_test)
    models['tcn'] = tcn_model

    return results, models


if __name__ == '__main__':
    print("Testing deep learning models...")

    # This would require actual data - skip for now
    print("Deep learning models module loaded successfully!")
