"""
Training utilities for PINN models.

Includes:
1. Training loop with physics loss
2. Curriculum learning
3. Adaptive loss weight adjustment
4. Model evaluation
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from typing import Dict, Optional, Callable
from tqdm import tqdm
import time


class PINNTrainer:
    """
    Trainer for Physics-Informed Neural Networks.

    Features:
    - Combined data + physics loss
    - Adaptive physics weight scheduling
    - Curriculum learning
    - Early stopping
    - Comprehensive logging
    """

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cpu',
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        """
        Initialize PINN trainer.

        Args:
            model: PINN model
            optimizer: Optimizer
            device: Device for training
            scheduler: Learning rate scheduler (optional)
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        # Training history
        self.history = {
            'train_loss': [],
            'train_data_loss': [],
            'train_physics_loss': [],
            'val_loss': [],
            'val_data_loss': [],
            'physics_weight': [],
            'learned_params': [],
            'lr': [],
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0

    def train_epoch(self,
                   train_loader,
                   collocation_sampler=None,
                   use_physics: bool = True,
                   n_collocation: int = 500) -> Dict:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data
            collocation_sampler: Sampler for collocation points
            use_physics: Whether to include physics loss
            n_collocation: Number of collocation points per batch

        Returns:
            Dictionary with epoch statistics
        """
        self.model.train()

        total_loss = 0
        total_data_loss = 0
        total_physics_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc='Training', leave=False)

        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(self.device).requires_grad_(True)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            y_pred = self.model(X_batch)

            # Data loss
            if hasattr(self.model, 'data_loss'):
                data_loss = self.model.data_loss(y_pred, y_batch)
            else:
                data_loss = nn.functional.mse_loss(y_pred, y_batch)

            loss = data_loss

            # Physics loss (if using PINN)
            physics_loss = torch.tensor(0.0).to(self.device)
            if use_physics and hasattr(self.model, 'physics_loss'):
                # Use collocation points for physics loss
                if collocation_sampler is not None:
                    X_colloc = collocation_sampler.sample(device=self.device).requires_grad_(True)
                    physics_loss = self.model.physics_loss(X_colloc, self.model(X_colloc))
                else:
                    physics_loss = self.model.physics_loss(X_batch, y_pred)

                loss = loss + self.model.physics_weight * physics_loss

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track statistics
            total_loss += loss.item()
            total_data_loss += data_loss.item()
            total_physics_loss += physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss
            n_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'data': f'{data_loss.item():.4f}',
                'phys': f'{physics_loss.item():.4f}' if isinstance(physics_loss, torch.Tensor) else '0.0000'
            })

        return {
            'loss': total_loss / n_batches,
            'data_loss': total_data_loss / n_batches,
            'physics_loss': total_physics_loss / n_batches,
        }

    @torch.no_grad()
    def validate(self, val_loader) -> Dict:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Validation metrics
        """
        self.model.eval()

        total_loss = 0
        total_data_loss = 0
        n_batches = 0

        all_preds = []
        all_targets = []

        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            y_pred = self.model(X_batch)

            # Data loss
            if hasattr(self.model, 'data_loss'):
                data_loss = self.model.data_loss(y_pred, y_batch)
            else:
                data_loss = nn.functional.mse_loss(y_pred, y_batch)

            total_loss += data_loss.item()
            total_data_loss += data_loss.item()
            n_batches += 1

            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

        # Concatenate all predictions and targets
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate metrics
        mse = np.mean((all_preds - all_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_preds - all_targets))

        return {
            'loss': total_loss / n_batches,
            'data_loss': total_data_loss / n_batches,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'predictions': all_preds,
            'targets': all_targets,
        }

    def train(self,
             train_loader,
             val_loader,
             n_epochs: int = 100,
             collocation_sampler=None,
             early_stopping_patience: int = 20,
             physics_schedule: str = 'curriculum',
             save_best: bool = True,
             verbose: bool = True) -> Dict:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            collocation_sampler: Sampler for collocation points
            early_stopping_patience: Patience for early stopping
            physics_schedule: Physics loss schedule ('curriculum', 'constant', 'adaptive')
            save_best: Whether to save best model
            verbose: Whether to print progress

        Returns:
            Training history
        """
        print(f"Starting PINN training for {n_epochs} epochs...")
        print(f"Physics schedule: {physics_schedule}")
        print(f"Device: {self.device}")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(n_epochs):
            epoch_start = time.time()

            # Adjust physics weight based on schedule
            if hasattr(self.model, 'set_physics_weight'):
                if physics_schedule == 'curriculum':
                    # Gradually increase physics weight
                    progress = epoch / n_epochs
                    weight = 0.001 * (1 + 99 * progress)  # 0.001 -> 0.1
                    self.model.set_physics_weight(weight)
                elif physics_schedule == 'adaptive' and hasattr(self.model, 'update_physics_weight_adaptive'):
                    self.model.update_physics_weight_adaptive(epoch, n_epochs)

            # Train epoch
            use_physics = True
            if physics_schedule == 'pretrain' and epoch < 20:
                use_physics = False  # Pretrain on data only

            train_metrics = self.train_epoch(
                train_loader,
                collocation_sampler=collocation_sampler,
                use_physics=use_physics
            )

            # Validate
            val_metrics = self.validate(val_loader)

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_data_loss'].append(train_metrics['data_loss'])
            self.history['train_physics_loss'].append(train_metrics['physics_loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_data_loss'].append(val_metrics['data_loss'])

            if hasattr(self.model, 'physics_weight'):
                self.history['physics_weight'].append(self.model.physics_weight.item())

            if hasattr(self.model, 'get_learned_parameters'):
                self.history['learned_params'].append(self.model.get_learned_parameters())

            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)

            # Early stopping and best model saving
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                if save_best:
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Print progress
            if verbose and (epoch % max(1, n_epochs // 20) == 0 or epoch == n_epochs - 1):
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch+1}/{n_epochs} ({epoch_time:.2f}s):")
                print(f"  Train Loss: {train_metrics['loss']:.6f} "
                      f"(Data: {train_metrics['data_loss']:.6f}, "
                      f"Phys: {train_metrics['physics_loss']:.6f})")
                print(f"  Val Loss: {val_metrics['loss']:.6f}, "
                      f"RMSE: {val_metrics['rmse']:.6f}, "
                      f"MAE: {val_metrics['mae']:.6f}")
                if hasattr(self.model, 'physics_weight'):
                    print(f"  Physics Weight: {self.model.physics_weight.item():.6f}")
                print(f"  LR: {current_lr:.6f}")

            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Best validation loss: {self.best_val_loss:.6f}")

        # Restore best model
        if save_best and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Restored best model weights")

        return self.history


def create_optimizer(model: nn.Module,
                     learning_rate: float = 1e-3,
                     weight_decay: float = 1e-5,
                     optimizer_type: str = 'adam') -> torch.optim.Optimizer:
    """
    Create optimizer for PINN training.

    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: L2 regularization
        optimizer_type: Type of optimizer

    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == 'adam':
        return Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_type: str = 'plateau',
                    **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer
        scheduler_type: Type of scheduler
        **kwargs: Additional arguments for scheduler

    Returns:
        Scheduler instance or None
    """
    if scheduler_type.lower() == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            verbose=True
        )
    elif scheduler_type.lower() == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_type.lower() == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


if __name__ == '__main__':
    print("PINN Trainer module loaded successfully!")
