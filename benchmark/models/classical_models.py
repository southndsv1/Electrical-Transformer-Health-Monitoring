"""
Classical machine learning models for time series forecasting.

Includes:
- ARIMA and SARIMA (statistical models)
- Linear regression with lag features
- Tree-based models: XGBoost, LightGBM, Random Forest
"""

import numpy as np
import pandas as pd
import time
import warnings
from typing import Tuple, Dict
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
import lightgbm as lgb
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import CLASSICAL_MODELS, RANDOM_SEED
from evaluation.metrics import calculate_all_metrics

warnings.filterwarnings('ignore')


class BaseClassicalModel:
    """
    Base class for classical models.
    """

    def __init__(self, model_name: str):
        """
        Initialize base model.

        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.training_time = 0
        self.inference_time = 0

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        raise NotImplementedError

    def evaluate(self, X, y) -> Dict:
        """
        Evaluate model on test data.

        Args:
            X: Test features
            y: Test targets

        Returns:
            Dictionary with metrics
        """
        start_time = time.time()
        y_pred = self.predict(X)
        self.inference_time = time.time() - start_time

        metrics = calculate_all_metrics(y, y_pred)
        metrics['training_time'] = self.training_time
        metrics['inference_time'] = self.inference_time

        return metrics


class LinearRegressionModel(BaseClassicalModel):
    """
    Linear Regression with lag features.

    Why relevant for time series:
    - Baseline model for comparison
    - Fast training and inference
    - Interpretable coefficients
    - Works well for linear trends

    Hyperparameters:
    - fit_intercept: Whether to calculate the intercept
    """

    def __init__(self, fit_intercept=True):
        super().__init__('Linear Regression')
        self.model = LinearRegression(fit_intercept=fit_intercept)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print(f"Training {self.model_name}...")
        start_time = time.time()

        self.model.fit(X_train, y_train)

        self.training_time = time.time() - start_time
        self.is_fitted = True
        print(f"  Training completed in {self.training_time:.2f}s")

    def predict(self, X):
        return self.model.predict(X)


class RidgeRegressionModel(BaseClassicalModel):
    """
    Ridge Regression (L2 regularization).

    Why relevant for time series:
    - Handles multicollinearity in lag features
    - Prevents overfitting with many features
    - More stable than linear regression

    Hyperparameters:
    - alpha: Regularization strength (higher = more regularization)
    """

    def __init__(self, alpha=1.0):
        super().__init__('Ridge Regression')
        self.model = Ridge(alpha=alpha, random_state=RANDOM_SEED)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print(f"Training {self.model_name}...")
        start_time = time.time()

        self.model.fit(X_train, y_train)

        self.training_time = time.time() - start_time
        self.is_fitted = True
        print(f"  Training completed in {self.training_time:.2f}s")

    def predict(self, X):
        return self.model.predict(X)


class ARIMAModel(BaseClassicalModel):
    """
    ARIMA (AutoRegressive Integrated Moving Average).

    Why relevant for time series:
    - Classic statistical time series model
    - Captures trend and autocorrelation
    - No need for external features
    - Well-understood theoretical properties

    Hyperparameters:
    - order (p, d, q):
      * p: Number of AR (autoregressive) terms
      * d: Degree of differencing
      * q: Number of MA (moving average) terms
    """

    def __init__(self, order=(5, 1, 0)):
        super().__init__('ARIMA')
        self.order = order
        self.model = None
        self.fitted_model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print(f"Training {self.model_name} with order {self.order}...")
        start_time = time.time()

        # ARIMA only uses the target variable
        self.model = ARIMA(y_train, order=self.order)
        self.fitted_model = self.model.fit()

        self.training_time = time.time() - start_time
        self.is_fitted = True
        print(f"  Training completed in {self.training_time:.2f}s")

    def predict(self, X):
        # Predict next n steps
        n_steps = len(X)
        forecast = self.fitted_model.forecast(steps=n_steps)
        return np.array(forecast)


class SARIMAModel(BaseClassicalModel):
    """
    SARIMA (Seasonal ARIMA).

    Why relevant for time series:
    - Extends ARIMA with seasonal components
    - Captures daily/weekly patterns in transformer temperature
    - Handles seasonality explicitly

    Hyperparameters:
    - order (p, d, q): Same as ARIMA
    - seasonal_order (P, D, Q, s):
      * P: Seasonal AR terms
      * D: Seasonal differencing
      * Q: Seasonal MA terms
      * s: Seasonal period (24 for daily, 168 for weekly)
    """

    def __init__(self, order=(5, 1, 0), seasonal_order=(1, 1, 1, 24)):
        super().__init__('SARIMA')
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print(f"Training {self.model_name}...")
        print(f"  Order: {self.order}, Seasonal order: {self.seasonal_order}")
        start_time = time.time()

        self.model = SARIMAX(y_train, order=self.order,
                            seasonal_order=self.seasonal_order)
        self.fitted_model = self.model.fit(disp=False)

        self.training_time = time.time() - start_time
        self.is_fitted = True
        print(f"  Training completed in {self.training_time:.2f}s")

    def predict(self, X):
        n_steps = len(X)
        forecast = self.fitted_model.forecast(steps=n_steps)
        return np.array(forecast)


class XGBoostModel(BaseClassicalModel):
    """
    XGBoost (Extreme Gradient Boosting).

    Why relevant for time series:
    - Excellent performance on tabular data with lag features
    - Handles non-linear relationships
    - Built-in regularization prevents overfitting
    - Fast training with GPU support
    - Robust to outliers and missing data

    Hyperparameters:
    - n_estimators: Number of boosting rounds
    - max_depth: Maximum tree depth (controls model complexity)
    - learning_rate: Step size shrinkage (smaller = more conservative)
    - subsample: Fraction of samples used per tree
    - colsample_bytree: Fraction of features used per tree
    """

    def __init__(self, **kwargs):
        super().__init__('XGBoost')
        self.params = CLASSICAL_MODELS['xgboost'].copy()
        self.params.update(kwargs)
        self.model = xgb.XGBRegressor(**self.params)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print(f"Training {self.model_name}...")
        start_time = time.time()

        if X_val is not None and y_val is not None:
            # Use early stopping with validation set
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

        self.training_time = time.time() - start_time
        self.is_fitted = True
        print(f"  Training completed in {self.training_time:.2f}s")
        print(f"  Best iteration: {self.model.best_iteration if hasattr(self.model, 'best_iteration') else 'N/A'}")

    def predict(self, X):
        return self.model.predict(X)


class LightGBMModel(BaseClassicalModel):
    """
    LightGBM (Light Gradient Boosting Machine).

    Why relevant for time series:
    - Faster training than XGBoost
    - Lower memory usage
    - Handles large datasets efficiently
    - Excellent performance on time series with lag features
    - Leaf-wise tree growth (deeper, more complex trees)

    Hyperparameters:
    - n_estimators: Number of boosting rounds
    - max_depth: Maximum tree depth
    - learning_rate: Boosting learning rate
    - subsample: Fraction of data to be used for each iteration
    - colsample_bytree: Fraction of features per tree
    """

    def __init__(self, **kwargs):
        super().__init__('LightGBM')
        self.params = CLASSICAL_MODELS['lightgbm'].copy()
        self.params.update(kwargs)
        self.model = lgb.LGBMRegressor(**self.params)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print(f"Training {self.model_name}...")
        start_time = time.time()

        if X_val is not None and y_val is not None:
            # Use early stopping with validation set
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
        else:
            self.model.fit(X_train, y_train)

        self.training_time = time.time() - start_time
        self.is_fitted = True
        print(f"  Training completed in {self.training_time:.2f}s")
        print(f"  Best iteration: {self.model.best_iteration_ if hasattr(self.model, 'best_iteration_') else 'N/A'}")

    def predict(self, X):
        return self.model.predict(X)


class RandomForestModel(BaseClassicalModel):
    """
    Random Forest Regressor.

    Why relevant for time series:
    - Ensemble of decision trees (reduces overfitting)
    - Captures non-linear relationships
    - Provides feature importance
    - Robust to outliers
    - Naturally handles missing data
    - Parallelizable (fast with multiple cores)

    Hyperparameters:
    - n_estimators: Number of trees
    - max_depth: Maximum tree depth
    - min_samples_split: Minimum samples to split a node
    - min_samples_leaf: Minimum samples in leaf node
    """

    def __init__(self, **kwargs):
        super().__init__('Random Forest')
        self.params = CLASSICAL_MODELS['random_forest'].copy()
        self.params.update(kwargs)
        self.model = RandomForestRegressor(**self.params)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print(f"Training {self.model_name}...")
        start_time = time.time()

        self.model.fit(X_train, y_train)

        self.training_time = time.time() - start_time
        self.is_fitted = True
        print(f"  Training completed in {self.training_time:.2f}s")

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importance(self):
        """Get feature importance scores."""
        if self.is_fitted:
            return self.model.feature_importances_
        return None


def train_all_classical_models(data_dict: Dict, forecast_horizon: int = 24) -> Dict:
    """
    Train all classical models and return results.

    Args:
        data_dict: Dictionary with training/validation/test data
        forecast_horizon: Forecast horizon in hours

    Returns:
        Dictionary with model results
    """
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']

    results = {}

    # Linear models
    print("\n" + "=" * 60)
    print("Training Linear Models")
    print("=" * 60)

    lr_model = LinearRegressionModel()
    lr_model.fit(X_train, y_train)
    results['linear_regression'] = lr_model.evaluate(X_test, y_test)

    ridge_model = RidgeRegressionModel(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    results['ridge'] = ridge_model.evaluate(X_test, y_test)

    # Statistical models (ARIMA, SARIMA)
    # Note: These are slower and work on target only
    print("\n" + "=" * 60)
    print("Training Statistical Models")
    print("=" * 60)

    try:
        arima_model = ARIMAModel(order=CLASSICAL_MODELS['arima']['order'])
        arima_model.fit(X_train, y_train)
        results['arima'] = arima_model.evaluate(X_test, y_test)
    except Exception as e:
        print(f"ARIMA training failed: {e}")
        results['arima'] = {'error': str(e)}

    # SARIMA can be very slow, skip for large datasets
    if len(y_train) < 5000:
        try:
            sarima_model = SARIMAModel(
                order=CLASSICAL_MODELS['arima']['order'],
                seasonal_order=CLASSICAL_MODELS['arima']['seasonal_order']
            )
            sarima_model.fit(X_train, y_train)
            results['sarima'] = sarima_model.evaluate(X_test, y_test)
        except Exception as e:
            print(f"SARIMA training failed (skipping): {e}")
            results['sarima'] = {'error': str(e)}
    else:
        print("Skipping SARIMA due to large dataset size")

    # Tree-based models
    print("\n" + "=" * 60)
    print("Training Tree-Based Models")
    print("=" * 60)

    xgb_model = XGBoostModel()
    xgb_model.fit(X_train, y_train, X_val, y_val)
    results['xgboost'] = xgb_model.evaluate(X_test, y_test)

    lgb_model = LightGBMModel()
    lgb_model.fit(X_train, y_train, X_val, y_val)
    results['lightgbm'] = lgb_model.evaluate(X_test, y_test)

    rf_model = RandomForestModel()
    rf_model.fit(X_train, y_train)
    results['random_forest'] = rf_model.evaluate(X_test, y_test)

    return results, {
        'linear_regression': lr_model,
        'ridge': ridge_model,
        'xgboost': xgb_model,
        'lightgbm': lgb_model,
        'random_forest': rf_model,
    }


if __name__ == '__main__':
    # Test classical models
    print("Testing classical models...")

    from data.data_loader import get_data_loader

    # Load data
    loader = get_data_loader('ETTh1')
    loader.load_data()
    loader.create_splits()

    # Get data for classical ML models
    data_dict = loader.get_classical_ml_data(forecast_horizon=24, n_lags=168)

    print("\nTraining all classical models...")
    results, models = train_all_classical_models(data_dict)

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    for model_name, metrics in results.items():
        if 'error' in metrics:
            print(f"\n{model_name}: FAILED")
            continue

        print(f"\n{model_name}:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  Training time: {metrics['training_time']:.2f}s")
