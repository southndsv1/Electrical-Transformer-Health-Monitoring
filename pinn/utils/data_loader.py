"""
Data loader for PINN training on ETT dataset.

Handles:
1. Loading and preprocessing ETTh1/ETTh2 data
2. Normalization for stable training
3. Creating train/val/test splits
4. Generating collocation points for physics loss
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional


class ETTDatasetPINN(Dataset):
    """
    PyTorch Dataset for ETT data with PINN training support.

    Features:
    - Normalized inputs for stable training
    - Time feature for computing derivatives
    - Load features for physics calculations
    """

    def __init__(self,
                 data: pd.DataFrame,
                 normalize: bool = True,
                 scaler_X: Optional[StandardScaler] = None,
                 scaler_y: Optional[StandardScaler] = None):
        """
        Initialize ETT dataset.

        Args:
            data: DataFrame with columns [date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT]
            normalize: Whether to normalize features
            scaler_X: Pre-fitted scaler for features (if None, fit new)
            scaler_y: Pre-fitted scaler for target (if None, fit new)
        """
        super().__init__()

        self.data = data.copy()

        # Extract features
        self.load_columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
        self.target_column = 'OT'

        # Convert date to numerical time (hours from start)
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            time_diff = (self.data['date'] - self.data['date'].min()).dt.total_seconds() / 3600.0
            self.data['time'] = time_diff.values
        else:
            # Create simple time index if no date column
            self.data['time'] = np.arange(len(self.data), dtype=np.float32)

        # Prepare features and target
        self.X_load = self.data[self.load_columns].values.astype(np.float32)
        self.X_time = self.data['time'].values.astype(np.float32).reshape(-1, 1)
        self.y = self.data[self.target_column].values.astype(np.float32).reshape(-1, 1)

        # Normalize
        self.normalize = normalize
        if normalize:
            if scaler_X is None:
                self.scaler_X = StandardScaler()
                self.X_load_norm = self.scaler_X.fit_transform(self.X_load)
            else:
                self.scaler_X = scaler_X
                self.X_load_norm = self.scaler_X.transform(self.X_load)

            # Normalize time to [0, 1] range for better gradients
            self.time_min = self.X_time.min()
            self.time_max = self.X_time.max()
            self.X_time_norm = (self.X_time - self.time_min) / (self.time_max - self.time_min + 1e-8)

            if scaler_y is None:
                self.scaler_y = StandardScaler()
                self.y_norm = self.scaler_y.fit_transform(self.y)
            else:
                self.scaler_y = scaler_y
                self.y_norm = self.scaler_y.transform(self.y)
        else:
            self.X_load_norm = self.X_load
            self.X_time_norm = self.X_time
            self.y_norm = self.y
            self.scaler_X = None
            self.scaler_y = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            Tuple of (X, y) where X = [time, load_features]
        """
        # Combine time and load features
        time = self.X_time_norm[idx]
        load = self.X_load_norm[idx]
        X = np.concatenate([time, load])

        y = self.y_norm[idx]

        return torch.FloatTensor(X), torch.FloatTensor(y)

    def get_all_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all data as tensors.

        Returns:
            Tuple of (X, y) tensors
        """
        X = np.concatenate([self.X_time_norm, self.X_load_norm], axis=1)
        return torch.FloatTensor(X), torch.FloatTensor(self.y_norm)

    def denormalize_predictions(self, y_norm: np.ndarray) -> np.ndarray:
        """
        Convert normalized predictions back to original scale.

        Args:
            y_norm: Normalized predictions

        Returns:
            Denormalized predictions
        """
        if self.normalize and self.scaler_y is not None:
            return self.scaler_y.inverse_transform(y_norm.reshape(-1, 1)).flatten()
        return y_norm.flatten()

    def denormalize_time(self, time_norm: np.ndarray) -> np.ndarray:
        """
        Convert normalized time back to original scale.

        Args:
            time_norm: Normalized time

        Returns:
            Denormalized time
        """
        if self.normalize:
            return time_norm * (self.time_max - self.time_min) + self.time_min
        return time_norm


class CollocationPointSampler:
    """
    Sampler for generating collocation points for physics loss.

    Collocation points are sampled throughout the domain to enforce
    physics constraints everywhere, not just at data points.
    """

    def __init__(self,
                 time_range: Tuple[float, float],
                 load_ranges: Dict[str, Tuple[float, float]],
                 n_points: int = 1000,
                 adaptive: bool = False):
        """
        Initialize collocation point sampler.

        Args:
            time_range: (min, max) for time dimension
            load_ranges: Dictionary of (min, max) for each load feature
            n_points: Number of collocation points to sample
            adaptive: Whether to use adaptive sampling (focus on high-residual regions)
        """
        self.time_range = time_range
        self.load_ranges = load_ranges
        self.n_points = n_points
        self.adaptive = adaptive

        # For adaptive sampling
        self.residual_history = []

    def sample(self, device='cpu') -> torch.Tensor:
        """
        Sample collocation points.

        Returns:
            Tensor of shape (n_points, 7) with [time, load_features]
        """
        # Sample time uniformly
        time = np.random.uniform(
            self.time_range[0],
            self.time_range[1],
            size=(self.n_points, 1)
        ).astype(np.float32)

        # Sample load features uniformly
        load_features = []
        for feature_name in ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']:
            if feature_name in self.load_ranges:
                min_val, max_val = self.load_ranges[feature_name]
                feature = np.random.uniform(min_val, max_val, size=(self.n_points, 1)).astype(np.float32)
            else:
                feature = np.zeros((self.n_points, 1), dtype=np.float32)
            load_features.append(feature)

        load_features = np.concatenate(load_features, axis=1)

        # Combine time and load features
        X_collocation = np.concatenate([time, load_features], axis=1)

        return torch.FloatTensor(X_collocation).to(device)

    def update_adaptive(self, residuals: torch.Tensor, X: torch.Tensor):
        """
        Update adaptive sampling based on physics residuals.

        Args:
            residuals: Physics residuals from recent batch
            X: Input points corresponding to residuals
        """
        if self.adaptive:
            self.residual_history.append({
                'residuals': residuals.detach().cpu().numpy(),
                'X': X.detach().cpu().numpy()
            })


def load_ett_data(dataset_name: str = 'ETTh2',
                  data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load ETT dataset.

    Args:
        dataset_name: 'ETTh1' or 'ETTh2'
        data_path: Custom path to data file (optional)

    Returns:
        DataFrame with ETT data
    """
    if data_path is None:
        # Default paths
        base_path = Path(__file__).parent.parent.parent
        data_path = base_path / f'{dataset_name}.csv'

    df = pd.read_csv(data_path)

    # Handle missing values
    if df.isnull().any().any():
        print(f"Warning: Found {df.isnull().sum().sum()} missing values. Filling with forward fill.")
        df = df.fillna(method='ffill').fillna(method='bfill')

    # Handle negative oil temperatures
    if (df['OT'] < 0).any():
        neg_count = (df['OT'] < 0).sum()
        print(f"Warning: Replacing {neg_count} negative temperature values.")
        df.loc[df['OT'] < 0, 'OT'] = df.loc[df['OT'] >= 0, 'OT'].min()

    return df


def create_pinn_dataloaders(dataset_name: str = 'ETTh2',
                            data_path: Optional[str] = None,
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15,
                            batch_size: int = 64,
                            shuffle: bool = True,
                            n_collocation: int = 1000) -> Dict:
    """
    Create dataloaders for PINN training.

    Args:
        dataset_name: Name of dataset ('ETTh1' or 'ETTh2')
        data_path: Path to data file (optional)
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        batch_size: Batch size for training
        shuffle: Whether to shuffle training data
        n_collocation: Number of collocation points for physics loss

    Returns:
        Dictionary with dataloaders and related info
    """
    # Load data
    df = load_ett_data(dataset_name, data_path)

    print(f"Loaded {dataset_name}: {len(df)} samples")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"OT range: [{df['OT'].min():.2f}, {df['OT'].max():.2f}]")

    # Split data
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    print(f"\nSplits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Create datasets
    train_dataset = ETTDatasetPINN(train_df, normalize=True)
    val_dataset = ETTDatasetPINN(
        val_df,
        normalize=True,
        scaler_X=train_dataset.scaler_X,
        scaler_y=train_dataset.scaler_y
    )
    test_dataset = ETTDatasetPINN(
        test_df,
        normalize=True,
        scaler_X=train_dataset.scaler_X,
        scaler_y=train_dataset.scaler_y
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create collocation point sampler
    time_range = (
        train_dataset.X_time_norm.min(),
        train_dataset.X_time_norm.max()
    )

    load_ranges = {}
    for i, col in enumerate(['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']):
        load_ranges[col] = (
            train_dataset.X_load_norm[:, i].min(),
            train_dataset.X_load_norm[:, i].max()
        )

    collocation_sampler = CollocationPointSampler(
        time_range=time_range,
        load_ranges=load_ranges,
        n_points=n_collocation
    )

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'collocation_sampler': collocation_sampler,
        'scaler_X': train_dataset.scaler_X,
        'scaler_y': train_dataset.scaler_y,
    }


if __name__ == '__main__':
    print("Testing PINN Data Loader")
    print("=" * 60)

    # Create dataloaders
    data_dict = create_pinn_dataloaders(
        dataset_name='ETTh2',
        batch_size=32,
        n_collocation=500
    )

    print("\nDataloader info:")
    print(f"  Train batches: {len(data_dict['train_loader'])}")
    print(f"  Val batches: {len(data_dict['val_loader'])}")
    print(f"  Test batches: {len(data_dict['test_loader'])}")

    # Test batch
    X, y = next(iter(data_dict['train_loader']))
    print(f"\nSample batch:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  X range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  y range: [{y.min():.3f}, {y.max():.3f}]")

    # Test collocation points
    X_colloc = data_dict['collocation_sampler'].sample()
    print(f"\nCollocation points:")
    print(f"  Shape: {X_colloc.shape}")
    print(f"  Time range: [{X_colloc[:, 0].min():.3f}, {X_colloc[:, 0].max():.3f}]")

    print("\n" + "=" * 60)
    print("Data loader test completed successfully!")
