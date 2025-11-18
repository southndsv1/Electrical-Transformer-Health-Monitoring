"""
Physics-Informed Neural Network for Transformer Temperature Prediction

This module implements the PINN architecture that combines:
1. Neural network for temperature prediction
2. Automatic differentiation for computing physics residuals
3. Combined loss function (data + physics)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from physics.transformer_thermal import LearnableTransformerPhysics


class TemperaturePredict

orNN(nn.Module):
    """
    Neural network for temperature prediction.

    Uses smooth activation functions (tanh) for better automatic differentiation.
    Architecture: Input -> Hidden Layers -> Output (oil temperature)
    """

    def __init__(self,
                 input_size: int = 7,  # time + 6 load features
                 hidden_sizes: list = [100, 100, 100, 100],
                 output_size: int = 1,
                 activation: str = 'tanh'):
        """
        Initialize temperature predictor network.

        Args:
            input_size: Number of input features (time + load features)
            hidden_sizes: List of hidden layer sizes
            output_size: Number of outputs (1 for oil temperature)
            activation: Activation function ('tanh', 'relu', 'gelu')
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Choose activation
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation)
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

        # Initialize weights with Xavier/Glorot initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, input_size)

        Returns:
            Predicted temperature of shape (batch, 1)
        """
        return self.network(x)


class TransformerPINN(nn.Module):
    """
    Physics-Informed Neural Network for Transformer Temperature Prediction.

    Combines:
    1. Neural network for temperature prediction
    2. Physics-based constraints from transformer thermal model
    3. Automatic differentiation for computing derivatives
    """

    def __init__(self,
                 input_size: int = 7,
                 hidden_sizes: list = [100, 100, 100, 100],
                 activation: str = 'tanh',
                 learn_physics_params: bool = True,
                 physics_weight_init: float = 0.1):
        """
        Initialize PINN model.

        Args:
            input_size: Number of input features
            hidden_sizes: Hidden layer sizes for NN
            activation: Activation function
            learn_physics_params: Whether to learn physical parameters
            physics_weight_init: Initial weight for physics loss
        """
        super().__init__()

        # Neural network for temperature prediction
        self.nn_predictor = TemperaturePredictorNN(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            activation=activation
        )

        # Learnable physics parameters
        self.physics = LearnableTransformerPhysics(learn_params=learn_physics_params)

        # Physics loss weight (can be adapted during training)
        self.register_buffer('physics_weight', torch.tensor(physics_weight_init))

    def forward(self, x):
        """
        Forward pass - predict temperature.

        Args:
            x: Input features (time, load_features)

        Returns:
            Predicted oil temperature
        """
        return self.nn_predictor(x)

    def compute_time_derivative(self, x, theta_pred, create_graph=True):
        """
        Compute time derivative using automatic differentiation.

        dθ/dt is computed by taking gradient of θ with respect to time.

        Args:
            x: Input tensor with time in first column
            theta_pred: Predicted temperature
            create_graph: Whether to create computation graph (needed for training)

        Returns:
            Time derivative dθ/dt
        """
        # Time is assumed to be first column of input
        time = x[:, 0:1]

        # Compute gradient
        dtheta_dt = torch.autograd.grad(
            outputs=theta_pred,
            inputs=time,
            grad_outputs=torch.ones_like(theta_pred),
            create_graph=create_graph,
            retain_graph=True
        )[0]

        return dtheta_dt

    def physics_loss(self,
                    x: torch.Tensor,
                    theta_pred: torch.Tensor,
                    return_components: bool = False) -> torch.Tensor:
        """
        Calculate physics loss based on transformer thermal equations.

        Args:
            x: Input features (time, load_features)
            theta_pred: Predicted temperature
            return_components: If True, return detailed loss components

        Returns:
            Physics loss (scalar) or dictionary of loss components
        """
        # Compute time derivative
        dtheta_dt_pred = self.compute_time_derivative(x, theta_pred, create_graph=True)

        # Extract load features (assuming columns 1-6 are load features)
        load_features = x[:, 1:7]

        # Calculate physics residual
        residual, K, theta_oil_ss = self.physics(
            theta_pred,
            dtheta_dt_pred,
            load_features
        )

        # Physics loss is MSE of residual (should be zero for perfect physics compliance)
        physics_loss = torch.mean(residual ** 2)

        if return_components:
            return {
                'physics_loss': physics_loss,
                'residual': residual,
                'K': K,
                'theta_ss': theta_oil_ss,
                'dtheta_dt': dtheta_dt_pred
            }

        return physics_loss

    def data_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate data loss (MSE between prediction and measurement).

        Args:
            y_pred: Predicted temperature
            y_true: Actual measured temperature

        Returns:
            Data loss (MSE)
        """
        return F.mse_loss(y_pred, y_true)

    def combined_loss(self,
                     x: torch.Tensor,
                     y_true: torch.Tensor,
                     y_pred: Optional[torch.Tensor] = None,
                     return_components: bool = False) -> torch.Tensor:
        """
        Calculate combined loss = data loss + λ * physics loss.

        Args:
            x: Input features
            y_true: True temperature values
            y_pred: Predicted temperature (computed if None)
            return_components: If True, return loss components separately

        Returns:
            Combined loss or dictionary of components
        """
        if y_pred is None:
            y_pred = self.forward(x)

        # Data loss
        loss_data = self.data_loss(y_pred, y_true)

        # Physics loss
        loss_physics = self.physics_loss(x, y_pred)

        # Combined loss
        loss_total = loss_data + self.physics_weight * loss_physics

        if return_components:
            return {
                'total_loss': loss_total,
                'data_loss': loss_data,
                'physics_loss': loss_physics,
                'physics_weight': self.physics_weight.item()
            }

        return loss_total

    def set_physics_weight(self, weight: float):
        """Update physics loss weight."""
        self.physics_weight = torch.tensor(weight)

    def get_learned_parameters(self) -> Dict:
        """
        Get learned physical parameters.

        Returns:
            Dictionary of learned physics parameters
        """
        return {
            'R': self.physics.R.item(),
            'n': self.physics.n.item(),
            'm': self.physics.m.item(),
            'tau_oil': self.physics.tau_oil.item(),
            'delta_theta_oil_rated': self.physics.delta_theta_oil_rated.item(),
            'delta_theta_hot_rated': self.physics.delta_theta_hot_rated.item(),
        }


class StandardNN(nn.Module):
    """
    Standard neural network without physics constraints.
    Used as baseline for comparison with PINN.
    """

    def __init__(self,
                 input_size: int = 7,
                 hidden_sizes: list = [100, 100, 100, 100],
                 activation: str = 'tanh'):
        """Initialize standard NN."""
        super().__init__()

        self.nn_predictor = TemperaturePredictorNN(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            activation=activation
        )

    def forward(self, x):
        """Forward pass."""
        return self.nn_predictor(x)

    def loss(self, y_pred, y_true):
        """Calculate MSE loss."""
        return F.mse_loss(y_pred, y_true)


class AdaptivePINN(TransformerPINN):
    """
    Adaptive PINN with dynamic physics weight adjustment.

    Features:
    1. Curriculum learning (start simple, add complexity)
    2. Adaptive physics weight based on prediction uncertainty
    3. Dynamic collocation point sampling
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Track training statistics for adaptation
        self.register_buffer('data_loss_ema', torch.tensor(1.0))
        self.register_buffer('physics_loss_ema', torch.tensor(1.0))
        self.ema_decay = 0.9

    def update_physics_weight_adaptive(self, current_epoch: int, max_epochs: int):
        """
        Adaptively update physics weight based on training progress.

        Strategy:
        1. Start with low physics weight (focus on fitting data)
        2. Gradually increase physics weight
        3. Adjust based on relative magnitude of losses

        Args:
            current_epoch: Current training epoch
            max_epochs: Total number of epochs
        """
        # Curriculum learning: increase weight over time
        progress = current_epoch / max_epochs
        base_weight = 0.01 * (1 + 9 * progress)  # 0.01 -> 0.1

        # Adaptive adjustment based on loss balance
        if self.physics_loss_ema > 0:
            loss_ratio = self.data_loss_ema / (self.physics_loss_ema + 1e-8)

            # If physics loss is much larger than data loss, reduce weight
            # If data loss is much larger, increase weight
            adaptive_factor = torch.clamp(loss_ratio, 0.1, 10.0)
            adaptive_weight = base_weight * adaptive_factor
        else:
            adaptive_weight = base_weight

        self.set_physics_weight(adaptive_weight.item())

    def update_loss_ema(self, data_loss: float, physics_loss: float):
        """Update exponential moving average of losses."""
        self.data_loss_ema = self.ema_decay * self.data_loss_ema + (1 - self.ema_decay) * data_loss
        self.physics_loss_ema = self.ema_decay * self.physics_loss_ema + (1 - self.ema_decay) * physics_loss


def create_pinn_model(config: Dict, device: str = 'cpu') -> TransformerPINN:
    """
    Factory function to create PINN model from config.

    Args:
        config: Configuration dictionary
        device: Device to create model on

    Returns:
        PINN model instance
    """
    model = TransformerPINN(
        input_size=config.get('input_size', 7),
        hidden_sizes=config.get('hidden_sizes', [100, 100, 100, 100]),
        activation=config.get('activation', 'tanh'),
        learn_physics_params=config.get('learn_physics_params', True),
        physics_weight_init=config.get('physics_weight_init', 0.01)
    ).to(device)

    return model


def create_standard_nn(config: Dict, device: str = 'cpu') -> StandardNN:
    """
    Factory function to create standard NN for comparison.

    Args:
        config: Configuration dictionary
        device: Device to create model on

    Returns:
        Standard NN model instance
    """
    model = StandardNN(
        input_size=config.get('input_size', 7),
        hidden_sizes=config.get('hidden_sizes', [100, 100, 100, 100]),
        activation=config.get('activation', 'tanh')
    ).to(device)

    return model


if __name__ == '__main__':
    print("Testing PINN Model Architecture")
    print("=" * 60)

    # Create model
    model = TransformerPINN(
        input_size=7,
        hidden_sizes=[50, 50, 50],
        activation='tanh'
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 7, requires_grad=True)
    y_true = torch.randn(batch_size, 1)

    y_pred = model(x)
    print(f"\nPrediction shape: {y_pred.shape}")

    # Test physics loss
    loss_components = model.combined_loss(x, y_true, y_pred, return_components=True)
    print(f"\nLoss components:")
    for key, value in loss_components.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.6f}")
        else:
            print(f"  {key}: {value}")

    # Test learned parameters
    params = model.get_learned_parameters()
    print(f"\nLearned physics parameters:")
    for key, value in params.items():
        print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 60)
    print("PINN model test completed successfully!")
