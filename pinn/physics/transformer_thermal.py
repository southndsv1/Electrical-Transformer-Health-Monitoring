"""
Transformer Thermal Physics based on IEEE C57.91 Standard

This module implements the thermal behavior equations for power transformers
including oil temperature dynamics, hot-spot temperature, and heat transfer.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


class TransformerThermalPhysics:
    """
    IEEE C57.91 compliant transformer thermal model.

    This class encodes the physical laws governing transformer temperature dynamics:
    1. Top oil temperature differential equation
    2. Hot-spot temperature calculation
    3. Heat balance equations
    4. Aging acceleration factors
    """

    def __init__(self,
                 R: float = 5.5,              # Ratio of load loss to no-load loss
                 n: float = 0.9,              # Oil exponent (ONAN cooling)
                 m: float = 0.8,              # Winding exponent
                 tau_oil: float = 180.0,      # Oil time constant (minutes)
                 delta_theta_oil_rated: float = 55.0,  # Rated top-oil temperature rise (K)
                 delta_theta_hot_rated: float = 65.0,  # Rated hot-spot temperature rise (K)
                 ambient_temp: float = 20.0,  # Ambient temperature (°C)
                 ):
        """
        Initialize transformer thermal physics model.

        Args:
            R: Ratio of load loss to no-load loss (typically 5-6)
            n: Oil exponent for ONAN cooling (typically 0.8-0.9)
            m: Winding exponent (typically 0.8-0.9)
            tau_oil: Oil time constant in minutes (typically 150-210)
            delta_theta_oil_rated: Rated top-oil temperature rise in K
            delta_theta_hot_rated: Rated hot-spot temperature rise in K
            ambient_temp: Ambient temperature in °C
        """
        self.R = R
        self.n = n
        self.m = m
        self.tau_oil = tau_oil  # minutes
        self.delta_theta_oil_rated = delta_theta_oil_rated
        self.delta_theta_hot_rated = delta_theta_hot_rated
        self.ambient_temp = ambient_temp

    def calculate_load_factor(self, load_features: torch.Tensor) -> torch.Tensor:
        """
        Calculate load factor K from power load features.

        Load factor represents the ratio of actual load to rated load.
        For ETT dataset, we have multiple load features (HUFL, HULL, etc.)
        We'll use the mean normalized load as K.

        Args:
            load_features: Tensor of shape (batch, 6) containing HUFL, HULL, MUFL, MULL, LUFL, LULL

        Returns:
            Load factor K of shape (batch, 1)
        """
        # Normalize and average the load features
        # In practice, this should be calibrated to actual transformer loading
        K = torch.mean(load_features, dim=-1, keepdim=True)

        # Clamp to reasonable range [0, 2] (0-200% rated load)
        K = torch.clamp(K, min=0.0, max=2.0)

        return K

    def steady_state_oil_temperature(self, K: torch.Tensor, theta_amb: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate steady-state top oil temperature.

        Based on IEEE C57.91:
        θ_oil_ss = θ_amb + Δθ_oil_rated * ((K²*R + 1)/(R + 1))^n

        Args:
            K: Load factor tensor of shape (batch, 1)
            theta_amb: Ambient temperature (optional)

        Returns:
            Steady-state oil temperature
        """
        if theta_amb is None:
            theta_amb = torch.full_like(K, self.ambient_temp)

        # IEEE C57.91 formula
        numerator = K**2 * self.R + 1.0
        denominator = self.R + 1.0

        delta_theta_oil = self.delta_theta_oil_rated * (numerator / denominator) ** self.n

        theta_oil_ss = theta_amb + delta_theta_oil

        return theta_oil_ss

    def oil_temperature_derivative(self, theta_oil: torch.Tensor, theta_oil_ss: torch.Tensor) -> torch.Tensor:
        """
        Calculate the time derivative of oil temperature.

        Differential equation (IEEE C57.91):
        dθ_oil/dt = (1/τ_oil) * (θ_oil_ss - θ_oil)

        This represents first-order thermal dynamics.

        Args:
            theta_oil: Current oil temperature
            theta_oil_ss: Steady-state oil temperature

        Returns:
            Time derivative dθ_oil/dt
        """
        # Convert tau from minutes to hours (assuming time is in hours in dataset)
        tau_oil_hours = self.tau_oil / 60.0

        dtheta_dt = (1.0 / tau_oil_hours) * (theta_oil_ss - theta_oil)

        return dtheta_dt

    def hot_spot_temperature(self, theta_oil: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Calculate hot-spot (winding) temperature.

        IEEE C57.91:
        θ_hot = θ_oil + Δθ_hot_rated * K^(2m)

        Args:
            theta_oil: Top oil temperature
            K: Load factor

        Returns:
            Hot-spot temperature
        """
        delta_theta_hot = self.delta_theta_hot_rated * (K ** (2 * self.m))
        theta_hot = theta_oil + delta_theta_hot

        return theta_hot

    def heat_generation(self, K: torch.Tensor) -> torch.Tensor:
        """
        Calculate heat generation in transformer.

        Q_gen = Q_no_load + Q_load * K²

        Where:
        - Q_no_load: No-load losses (core losses)
        - Q_load: Load losses (copper losses) proportional to I²R

        Args:
            K: Load factor

        Returns:
            Normalized heat generation
        """
        # Normalize to rated load (K=1)
        Q_no_load = 1.0
        Q_load = self.R  # R is ratio of load loss to no-load loss

        Q_total = Q_no_load + Q_load * (K ** 2)

        return Q_total

    def aging_acceleration_factor(self, theta_hot: torch.Tensor,
                                  theta_hot_rated: float = 110.0) -> torch.Tensor:
        """
        Calculate insulation aging acceleration factor.

        Based on Arrhenius equation (IEEE C57.91):
        FAA = exp[15000/383 - 15000/(273 + θ_hot)]

        For θ_hot_rated = 110°C, FAA = 1.0 (normal aging)
        Higher temperatures accelerate aging exponentially.

        Args:
            theta_hot: Hot-spot temperature in °C
            theta_hot_rated: Rated hot-spot temperature (typically 110°C)

        Returns:
            Aging acceleration factor (FAA)
        """
        # Arrhenius equation constants
        B = 15000.0  # Kelvin
        T_ref = 383.0  # Kelvin (110°C + 273)

        # Convert to Kelvin
        T_hot = theta_hot + 273.15

        FAA = torch.exp(B / T_ref - B / T_hot)

        return FAA

    def physics_residual(self,
                        theta_oil_pred: torch.Tensor,
                        dtheta_dt_pred: torch.Tensor,
                        load_features: torch.Tensor,
                        theta_amb: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate physics residual for PINN training.

        Residual = dθ_oil/dt - (1/τ_oil) * (θ_oil_ss - θ_oil)

        For perfect physics compliance, residual should be zero.

        Args:
            theta_oil_pred: Predicted oil temperature
            dtheta_dt_pred: Predicted time derivative (from autograd)
            load_features: Load features for calculating K
            theta_amb: Ambient temperature (optional)

        Returns:
            Physics residual (should be minimized)
        """
        # Calculate load factor
        K = self.calculate_load_factor(load_features)

        # Calculate steady-state temperature
        theta_oil_ss = self.steady_state_oil_temperature(K, theta_amb)

        # Calculate expected derivative from physics
        dtheta_dt_physics = self.oil_temperature_derivative(theta_oil_pred, theta_oil_ss)

        # Residual is difference between predicted and physics-based derivative
        residual = dtheta_dt_pred - dtheta_dt_physics

        return residual, K, theta_oil_ss

    def validate_temperature_limits(self, theta_oil: torch.Tensor,
                                   theta_hot: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Check if temperatures are within acceptable limits.

        Typical limits (IEEE C57.91):
        - Top oil: < 105°C (normal), < 120°C (emergency)
        - Hot-spot: < 110°C (normal), < 140°C (emergency)

        Args:
            theta_oil: Oil temperature
            theta_hot: Hot-spot temperature

        Returns:
            Dictionary with limit violations
        """
        limits = {
            'oil_normal_exceeded': theta_oil > 105.0,
            'oil_emergency_exceeded': theta_oil > 120.0,
            'hot_normal_exceeded': theta_hot > 110.0,
            'hot_emergency_exceeded': theta_hot > 140.0,
        }

        return limits


class LearnableTransformerPhysics(nn.Module):
    """
    Learnable version of transformer physics where parameters can be optimized.

    This allows the PINN to learn the best physical parameters from data
    while still respecting the structure of the physics equations.
    """

    def __init__(self,
                 R_init: float = 5.5,
                 n_init: float = 0.9,
                 m_init: float = 0.8,
                 tau_oil_init: float = 180.0,
                 delta_theta_oil_rated_init: float = 55.0,
                 delta_theta_hot_rated_init: float = 65.0,
                 learn_params: bool = True):
        """
        Initialize learnable physics parameters.

        Args:
            *_init: Initial values for physics parameters
            learn_params: If True, parameters are learnable; if False, they're fixed
        """
        super().__init__()

        # Make parameters learnable or fixed
        if learn_params:
            # Use log-space for positive parameters to ensure they stay positive
            self.log_R = nn.Parameter(torch.log(torch.tensor(R_init)))
            self.logit_n = nn.Parameter(torch.logit(torch.tensor(n_init)))  # Keep in (0,1)
            self.logit_m = nn.Parameter(torch.logit(torch.tensor(m_init)))
            self.log_tau_oil = nn.Parameter(torch.log(torch.tensor(tau_oil_init)))
            self.log_delta_theta_oil = nn.Parameter(torch.log(torch.tensor(delta_theta_oil_rated_init)))
            self.log_delta_theta_hot = nn.Parameter(torch.log(torch.tensor(delta_theta_hot_rated_init)))
        else:
            self.register_buffer('log_R', torch.log(torch.tensor(R_init)))
            self.register_buffer('logit_n', torch.logit(torch.tensor(n_init)))
            self.register_buffer('logit_m', torch.logit(torch.tensor(m_init)))
            self.register_buffer('log_tau_oil', torch.log(torch.tensor(tau_oil_init)))
            self.register_buffer('log_delta_theta_oil', torch.log(torch.tensor(delta_theta_oil_rated_init)))
            self.register_buffer('log_delta_theta_hot', torch.log(torch.tensor(delta_theta_hot_rated_init)))

    @property
    def R(self):
        return torch.exp(self.log_R)

    @property
    def n(self):
        return torch.sigmoid(self.logit_n)

    @property
    def m(self):
        return torch.sigmoid(self.logit_m)

    @property
    def tau_oil(self):
        return torch.exp(self.log_tau_oil)

    @property
    def delta_theta_oil_rated(self):
        return torch.exp(self.log_delta_theta_oil)

    @property
    def delta_theta_hot_rated(self):
        return torch.exp(self.log_delta_theta_hot)

    def get_physics_model(self, ambient_temp: float = 20.0) -> TransformerThermalPhysics:
        """
        Get a TransformerThermalPhysics instance with current parameter values.

        Returns:
            TransformerThermalPhysics with current learned parameters
        """
        return TransformerThermalPhysics(
            R=self.R.item(),
            n=self.n.item(),
            m=self.m.item(),
            tau_oil=self.tau_oil.item(),
            delta_theta_oil_rated=self.delta_theta_oil_rated.item(),
            delta_theta_hot_rated=self.delta_theta_hot_rated.item(),
            ambient_temp=ambient_temp
        )

    def forward(self, theta_oil_pred, dtheta_dt_pred, load_features, theta_amb=None):
        """
        Calculate physics residual with learnable parameters.
        """
        physics = self.get_physics_model()
        return physics.physics_residual(theta_oil_pred, dtheta_dt_pred, load_features, theta_amb)


if __name__ == '__main__':
    # Test the physics model
    print("Testing Transformer Thermal Physics Model")
    print("=" * 60)

    physics = TransformerThermalPhysics()

    # Test with sample data
    batch_size = 10
    load_features = torch.rand(batch_size, 6) * 1.5  # Random loads 0-150%

    K = physics.calculate_load_factor(load_features)
    print(f"\nLoad factors: {K.squeeze()}")

    theta_oil_ss = physics.steady_state_oil_temperature(K)
    print(f"\nSteady-state oil temperatures: {theta_oil_ss.squeeze()}")

    theta_oil_current = torch.ones(batch_size, 1) * 70.0  # Current temp 70°C
    dtheta_dt = physics.oil_temperature_derivative(theta_oil_current, theta_oil_ss)
    print(f"\nTemperature derivatives: {dtheta_dt.squeeze()}")

    theta_hot = physics.hot_spot_temperature(theta_oil_ss, K)
    print(f"\nHot-spot temperatures: {theta_hot.squeeze()}")

    FAA = physics.aging_acceleration_factor(theta_hot)
    print(f"\nAging acceleration factors: {FAA.squeeze()}")

    print("\n" + "=" * 60)
    print("Physics model test completed successfully!")
