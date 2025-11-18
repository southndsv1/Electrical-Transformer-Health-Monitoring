"""
Data loader for ETT (Electricity Transformer Temperature) datasets.
Handles data loading, preprocessing, and split generation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import DATA_CONFIG, SEQUENCE_LENGTH, RANDOM_SEED


class ETTDataLoader:
    """
    Data loader for Electrical Transformer Temperature datasets.

    Features:
    - Loads ETTh1 and ETTh2 datasets
    - Handles missing data
    - Creates train/val/test splits
    - Provides data in multiple formats (sequences, features, etc.)
    - Supports multiple forecast horizons
    """

    def __init__(self, dataset_name: str = 'ETTh1'):
        """
        Initialize data loader.

        Args:
            dataset_name: Name of dataset ('ETTh1' or 'ETTh2')
        """
        self.dataset_name = dataset_name
        self.data_path = DATA_CONFIG['datasets'][dataset_name]
        self.target_col = DATA_CONFIG['target_column']
        self.feature_cols = DATA_CONFIG['feature_columns']
        self.date_col = DATA_CONFIG['date_column']

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess the dataset.

        Returns:
            DataFrame with preprocessed data
        """
        print(f"Loading {self.dataset_name} from {self.data_path}")

        # Load data
        df = pd.read_csv(self.data_path)

        # Convert date to datetime
        df[self.date_col] = pd.to_datetime(df[self.date_col])

        # Sort by date
        df = df.sort_values(self.date_col)

        # Set date as index
        df = df.set_index(self.date_col)

        # Handle missing values
        if df.isnull().any().any():
            print(f"Found {df.isnull().sum().sum()} missing values. Filling with forward fill.")
            df = df.fillna(method='ffill').fillna(method='bfill')

        # Handle negative oil temperatures (physical impossibility)
        if (df[self.target_col] < 0).any():
            neg_count = (df[self.target_col] < 0).sum()
            print(f"Replacing {neg_count} negative temperature values with interpolation")
            df.loc[df[self.target_col] < 0, self.target_col] = np.nan
            df[self.target_col] = df[self.target_col].interpolate(method='linear')

        self.data = df

        print(f"Data loaded successfully:")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  {self.target_col} range: [{df[self.target_col].min():.2f}, {df[self.target_col].max():.2f}]")
        print(f"  Duration: {len(df)} hours ({len(df) / 24:.1f} days)")

        return df

    def create_splits(self, train_ratio: float = None, val_ratio: float = None,
                     test_ratio: float = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits.

        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if self.data is None:
            self.load_data()

        # Use config defaults if not specified
        train_ratio = train_ratio or DATA_CONFIG['train_ratio']
        val_ratio = val_ratio or DATA_CONFIG['val_ratio']
        test_ratio = test_ratio or DATA_CONFIG['test_ratio']

        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

        n = len(self.data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        self.train_data = self.data.iloc[:train_size]
        self.val_data = self.data.iloc[train_size:train_size + val_size]
        self.test_data = self.data.iloc[train_size + val_size:]

        print(f"\nData splits created:")
        print(f"  Train: {len(self.train_data)} samples ({len(self.train_data)/24:.1f} days)")
        print(f"  Val:   {len(self.val_data)} samples ({len(self.val_data)/24:.1f} days)")
        print(f"  Test:  {len(self.test_data)} samples ({len(self.test_data)/24:.1f} days)")

        return self.train_data, self.val_data, self.test_data

    def create_lag_features(self, df: pd.DataFrame, n_lags: int = 168,
                           include_rolling: bool = True) -> pd.DataFrame:
        """
        Create lag features and rolling statistics for classical ML models.

        Args:
            df: Input dataframe
            n_lags: Number of lag features to create
            include_rolling: Whether to include rolling statistics

        Returns:
            DataFrame with lag features
        """
        df_features = df.copy()

        # Add temporal features
        df_features['hour'] = df_features.index.hour
        df_features['day_of_week'] = df_features.index.dayofweek
        df_features['day_of_month'] = df_features.index.day
        df_features['month'] = df_features.index.month

        # Add cyclical encoding of time features
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features.index.hour / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features.index.hour / 24)
        df_features['day_sin'] = np.sin(2 * np.pi * df_features.index.dayofweek / 7)
        df_features['day_cos'] = np.cos(2 * np.pi * df_features.index.dayofweek / 7)
        df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month / 12)

        # Add lag features for target variable
        lag_intervals = [1, 2, 3, 6, 12, 24, 48, 72, 96, 120, 144, 168]
        lag_intervals = [l for l in lag_intervals if l <= n_lags]

        for lag in lag_intervals:
            df_features[f'lag_{lag}'] = df_features[self.target_col].shift(lag)

        # Add lag features for other variables
        for col in self.feature_cols:
            for lag in [1, 24, 168]:
                df_features[f'{col}_lag_{lag}'] = df_features[col].shift(lag)

        if include_rolling:
            # Add rolling statistics
            windows = [6, 12, 24, 48, 168]

            for window in windows:
                # For target variable
                df_features[f'rolling_mean_{window}'] = df_features[self.target_col].rolling(window).mean()
                df_features[f'rolling_std_{window}'] = df_features[self.target_col].rolling(window).std()
                df_features[f'rolling_min_{window}'] = df_features[self.target_col].rolling(window).min()
                df_features[f'rolling_max_{window}'] = df_features[self.target_col].rolling(window).max()

            # Add exponential weighted moving average
            df_features['ewma_24'] = df_features[self.target_col].ewm(span=24).mean()
            df_features['ewma_168'] = df_features[self.target_col].ewm(span=168).mean()

        # Drop rows with NaN values
        df_features = df_features.dropna()

        return df_features

    def create_sequences(self, df: pd.DataFrame, seq_length: int = SEQUENCE_LENGTH,
                        forecast_horizon: int = 24,
                        stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for deep learning models.

        Args:
            df: Input dataframe
            seq_length: Length of input sequence (lookback window)
            forecast_horizon: Number of steps to forecast
            stride: Stride for creating sequences

        Returns:
            Tuple of (X, y) where X is input sequences and y is target sequences
        """
        # Use all columns as features (including target for autoregressive prediction)
        data = df.values

        X, y = [], []

        for i in range(0, len(data) - seq_length - forecast_horizon + 1, stride):
            # Input sequence
            X.append(data[i:i + seq_length])

            # Target: oil temperature for next forecast_horizon steps
            target_idx = df.columns.get_loc(self.target_col)
            y.append(data[i + seq_length:i + seq_length + forecast_horizon, target_idx])

        return np.array(X), np.array(y)

    def get_classical_ml_data(self, forecast_horizon: int = 24, n_lags: int = 168,
                             scale: bool = True) -> Dict:
        """
        Get data formatted for classical ML models (with lag features).

        Args:
            forecast_horizon: Forecast horizon in hours
            n_lags: Number of lag features
            scale: Whether to scale the features

        Returns:
            Dictionary containing train/val/test data
        """
        if self.train_data is None:
            self.create_splits()

        # Create features
        train_features = self.create_lag_features(self.train_data, n_lags=n_lags)
        val_features = self.create_lag_features(self.val_data, n_lags=n_lags)
        test_features = self.create_lag_features(self.test_data, n_lags=n_lags)

        # Separate X and y
        X_train = train_features.drop(self.target_col, axis=1)
        y_train = train_features[self.target_col]

        X_val = val_features.drop(self.target_col, axis=1)
        y_val = val_features[self.target_col]

        X_test = test_features.drop(self.target_col, axis=1)
        y_test = test_features[self.target_col]

        if scale:
            # Fit scaler on training data
            X_train_scaled = self.scaler_X.fit_transform(X_train)
            X_val_scaled = self.scaler_X.transform(X_val)
            X_test_scaled = self.scaler_X.transform(X_test)

            # Convert back to DataFrames
            X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
            X_val = pd.DataFrame(X_val_scaled, index=X_val.index, columns=X_val.columns)
            X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': X_train.columns.tolist(),
        }

    def get_deep_learning_data(self, seq_length: int = SEQUENCE_LENGTH,
                              forecast_horizon: int = 24,
                              scale: bool = True,
                              stride_train: int = 1,
                              stride_val: int = 24) -> Dict:
        """
        Get data formatted for deep learning models (sequences).

        Args:
            seq_length: Length of input sequence
            forecast_horizon: Forecast horizon in hours
            scale: Whether to scale the features
            stride_train: Stride for training sequences (1 for all possible sequences)
            stride_val: Stride for val/test sequences (larger to reduce validation time)

        Returns:
            Dictionary containing train/val/test sequences
        """
        if self.train_data is None:
            self.create_splits()

        # Scale data if requested
        if scale:
            train_scaled = self.train_data.copy()
            val_scaled = self.val_data.copy()
            test_scaled = self.test_data.copy()

            # Fit on train, transform all
            scaler = StandardScaler()
            train_scaled[:] = scaler.fit_transform(train_scaled)
            val_scaled[:] = scaler.transform(val_scaled)
            test_scaled[:] = scaler.transform(test_scaled)

            train_data = train_scaled
            val_data = val_scaled
            test_data = test_scaled

            # Store scaler for inverse transform
            self.scaler_X = scaler
        else:
            train_data = self.train_data
            val_data = self.val_data
            test_data = self.test_data

        # Create sequences
        X_train, y_train = self.create_sequences(train_data, seq_length,
                                                 forecast_horizon, stride=stride_train)
        X_val, y_val = self.create_sequences(val_data, seq_length,
                                             forecast_horizon, stride=stride_val)
        X_test, y_test = self.create_sequences(test_data, seq_length,
                                               forecast_horizon, stride=stride_val)

        print(f"\nSequence data created:")
        print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"  X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        print(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'n_features': X_train.shape[-1],
            'seq_length': seq_length,
            'forecast_horizon': forecast_horizon,
        }

    def inject_missing_data(self, df: pd.DataFrame, missing_ratio: float = 0.1,
                           seed: int = RANDOM_SEED) -> pd.DataFrame:
        """
        Inject missing data randomly for robustness testing.

        Args:
            df: Input dataframe
            missing_ratio: Fraction of data to make missing
            seed: Random seed for reproducibility

        Returns:
            DataFrame with missing data
        """
        np.random.seed(seed)
        df_missing = df.copy()

        # Randomly select positions to make missing
        n_missing = int(len(df) * missing_ratio)
        missing_indices = np.random.choice(len(df), n_missing, replace=False)

        # Make data missing (only for features, not target)
        for col in self.feature_cols:
            df_missing.iloc[missing_indices, df_missing.columns.get_loc(col)] = np.nan

        # Fill missing values with forward fill
        df_missing = df_missing.fillna(method='ffill').fillna(method='bfill')

        return df_missing


def get_data_loader(dataset_name: str = 'ETTh1') -> ETTDataLoader:
    """
    Factory function to get a data loader.

    Args:
        dataset_name: Name of dataset ('ETTh1' or 'ETTh2')

    Returns:
        ETTDataLoader instance
    """
    return ETTDataLoader(dataset_name)


if __name__ == '__main__':
    # Test data loader
    print("Testing data loader...")

    for dataset in ['ETTh1', 'ETTh2']:
        print(f"\n{'='*60}")
        print(f"Testing {dataset}")
        print('='*60)

        loader = get_data_loader(dataset)
        loader.load_data()
        loader.create_splits()

        # Test classical ML data
        print("\nTesting classical ML data format:")
        ml_data = loader.get_classical_ml_data(forecast_horizon=24, n_lags=168)
        print(f"Feature count: {len(ml_data['feature_names'])}")

        # Test deep learning data
        print("\nTesting deep learning data format:")
        dl_data = loader.get_deep_learning_data(seq_length=168, forecast_horizon=24)
