#!/usr/bin/env python3
"""
Main experiment script for Physics-Informed Neural Network (PINN)
for Transformer Oil Temperature Prediction.

This script:
1. Loads ETTh2 data
2. Trains both PINN and Standard NN
3. Compares performance
4. Generates comprehensive visualizations
5. Validates RMSE < 0.15 target
"""

import sys
from pathlib import Path
import argparse
import json
import time
from datetime import datetime

sys.path.append(str(Path(__file__).parent / 'pinn'))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from models.pinn_transformer import create_pinn_model, create_standard_nn
from utils.data_loader import create_pinn_dataloaders
from utils.trainer import PINNTrainer, create_optimizer, create_scheduler

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run PINN experiment for transformer temperature prediction'
    )

    parser.add_argument('--dataset', type=str, default='ETTh2',
                       choices=['ETTh1', 'ETTh2'],
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--hidden-sizes', type=int, nargs='+',
                       default=[100, 100, 100, 100],
                       help='Hidden layer sizes')
    parser.add_argument('--physics-schedule', type=str, default='curriculum',
                       choices=['curriculum', 'constant', 'adaptive', 'pretrain'],
                       help='Physics loss weight schedule')
    parser.add_argument('--n-collocation', type=int, default=1000,
                       help='Number of collocation points')
    parser.add_argument('--output-dir', type=str, default='pinn_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device(device_str: str = 'auto') -> str:
    """Get device for training."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    return device_str


def train_model(model_type: str, model, data_dict, args, device):
    """
    Train a model (PINN or Standard NN).

    Args:
        model_type: 'PINN' or 'Standard NN'
        model: Model instance
        data_dict: Dictionary with data loaders
        args: Command line arguments
        device: Device for training

    Returns:
        Trained model and training history
    """
    print(f"\n{'='*60}")
    print(f"Training {model_type}")
    print(f"{'='*60}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, learning_rate=args.lr, weight_decay=1e-5)
    scheduler = create_scheduler(optimizer, scheduler_type='plateau', patience=15)

    # Create trainer
    trainer = PINNTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )

    # Train
    if model_type == 'PINN':
        history = trainer.train(
            train_loader=data_dict['train_loader'],
            val_loader=data_dict['val_loader'],
            n_epochs=args.epochs,
            collocation_sampler=data_dict['collocation_sampler'],
            early_stopping_patience=30,
            physics_schedule=args.physics_schedule,
            save_best=True,
            verbose=True
        )
    else:
        # Standard NN - no physics loss
        history = trainer.train(
            train_loader=data_dict['train_loader'],
            val_loader=data_dict['val_loader'],
            n_epochs=args.epochs,
            collocation_sampler=None,
            early_stopping_patience=30,
            physics_schedule='none',
            save_best=True,
            verbose=True
        )

    return model, history


@torch.no_grad()
def evaluate_model(model, data_loader, dataset, device, model_name='Model'):
    """
    Evaluate model on test set.

    Returns:
        Dictionary with metrics and predictions
    """
    model.eval()

    all_preds = []
    all_targets = []

    for X_batch, y_batch in data_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(X_batch)

        all_preds.append(y_pred.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

    # Concatenate
    all_preds_norm = np.concatenate(all_preds, axis=0).flatten()
    all_targets_norm = np.concatenate(all_targets, axis=0).flatten()

    # Denormalize
    all_preds = dataset.denormalize_predictions(all_preds_norm)
    all_targets = dataset.denormalize_predictions(all_targets_norm)

    # Calculate metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds - all_targets))
    mape = np.mean(np.abs((all_preds - all_targets) / (all_targets + 1e-8))) * 100
    r2 = 1 - (np.sum((all_targets - all_preds) ** 2) / np.sum((all_targets - all_targets.mean()) ** 2))

    print(f"\n{model_name} Test Results:")
    print(f"  RMSE: {rmse:.6f} °C")
    print(f"  MAE:  {mae:.6f} °C")
    print(f"  MAPE: {mape:.4f} %")
    print(f"  R²:   {r2:.6f}")

    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'predictions': all_preds,
        'targets': all_targets,
        'predictions_norm': all_preds_norm,
        'targets_norm': all_targets_norm,
    }


def plot_results(pinn_results, nn_results, pinn_history, nn_history, output_dir):
    """
    Generate comprehensive visualization plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curves
    axes[0, 0].plot(pinn_history['train_loss'], label='PINN Train', alpha=0.7)
    axes[0, 0].plot(pinn_history['val_loss'], label='PINN Val', alpha=0.7)
    axes[0, 0].plot(nn_history['train_loss'], label='NN Train', alpha=0.7, linestyle='--')
    axes[0, 0].plot(nn_history['val_loss'], label='NN Val', alpha=0.7, linestyle='--')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Physics loss (PINN only)
    if 'train_physics_loss' in pinn_history:
        axes[0, 1].plot(pinn_history['train_physics_loss'], label='Physics Loss', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Physics Loss')
        axes[0, 1].set_title('PINN Physics Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Physics weight evolution
    if 'physics_weight' in pinn_history:
        axes[1, 0].plot(pinn_history['physics_weight'], color='purple')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Physics Weight (λ)')
        axes[1, 0].set_title('Physics Loss Weight Schedule')
        axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(pinn_history['lr'], label='PINN', alpha=0.7)
    axes[1, 1].plot(nn_history['lr'], label='Standard NN', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Predictions vs Actual
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Plot sample (last 500 points for clarity)
    n_plot = min(500, len(pinn_results['targets']))
    indices = np.arange(len(pinn_results['targets']) - n_plot, len(pinn_results['targets']))

    axes[0].plot(indices, pinn_results['targets'][-n_plot:], label='Actual', linewidth=2, alpha=0.8)
    axes[0].plot(indices, pinn_results['predictions'][-n_plot:], label='PINN', linewidth=1.5, alpha=0.8)
    axes[0].plot(indices, nn_results['predictions'][-n_plot:], label='Standard NN', linewidth=1.5, alpha=0.8, linestyle='--')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Oil Temperature (°C)')
    axes[0].set_title('Temperature Predictions (Test Set - Last 500 Points)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Errors
    pinn_errors = pinn_results['predictions'][-n_plot:] - pinn_results['targets'][-n_plot:]
    nn_errors = nn_results['predictions'][-n_plot:] - nn_results['targets'][-n_plot:]

    axes[1].plot(indices, pinn_errors, label=f'PINN (MAE={pinn_results["mae"]:.4f})', alpha=0.7)
    axes[1].plot(indices, nn_errors, label=f'NN (MAE={nn_results["mae"]:.4f})', alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Prediction Error (°C)')
    axes[1].set_title('Prediction Errors')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # PINN
    axes[0].scatter(pinn_results['targets'], pinn_results['predictions'], alpha=0.3, s=10)
    axes[0].plot([pinn_results['targets'].min(), pinn_results['targets'].max()],
                [pinn_results['targets'].min(), pinn_results['targets'].max()],
                'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Temperature (°C)')
    axes[0].set_ylabel('Predicted Temperature (°C)')
    axes[0].set_title(f'PINN: RMSE={pinn_results["rmse"]:.6f}, R²={pinn_results["r2"]:.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Standard NN
    axes[1].scatter(nn_results['targets'], nn_results['predictions'], alpha=0.3, s=10)
    axes[1].plot([nn_results['targets'].min(), nn_results['targets'].max()],
                [nn_results['targets'].min(), nn_results['targets'].max()],
                'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Temperature (°C)')
    axes[1].set_ylabel('Predicted Temperature (°C)')
    axes[1].set_title(f'Standard NN: RMSE={nn_results["rmse"]:.6f}, R²={nn_results["r2"]:.4f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Error distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pinn_errors_all = pinn_results['predictions'] - pinn_results['targets']
    nn_errors_all = nn_results['predictions'] - nn_results['targets']

    axes[0].hist(pinn_errors_all, bins=50, alpha=0.7, label='PINN', density=True, edgecolor='black')
    axes[0].hist(nn_errors_all, bins=50, alpha=0.7, label='Standard NN', density=True, edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Prediction Error (°C)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Q-Q plot
    from scipy import stats
    stats.probplot(pinn_errors_all, dist="norm", plot=axes[1])
    axes[1].set_title('PINN Error Q-Q Plot')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualizations saved to {output_dir}")


def save_results(pinn_model, nn_model, pinn_results, nn_results,
                pinn_history, nn_history, args, output_dir):
    """
    Save all results and models.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    torch.save({
        'model_state_dict': pinn_model.state_dict(),
        'args': vars(args),
    }, output_dir / 'pinn_model.pth')

    torch.save({
        'model_state_dict': nn_model.state_dict(),
        'args': vars(args),
    }, output_dir / 'standard_nn_model.pth')

    # Save metrics
    metrics = {
        'pinn': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in pinn_results.items() if k not in ['predictions', 'targets', 'predictions_norm', 'targets_norm']},
        'standard_nn': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in nn_results.items() if k not in ['predictions', 'targets', 'predictions_norm', 'targets_norm']},
        'improvement': {
            'rmse_reduction': float((nn_results['rmse'] - pinn_results['rmse']) / nn_results['rmse'] * 100),
            'mae_reduction': float((nn_results['mae'] - pinn_results['mae']) / nn_results['mae'] * 100),
        }
    }

    # Add learned parameters if available
    if hasattr(pinn_model, 'get_learned_parameters'):
        metrics['pinn']['learned_parameters'] = pinn_model.get_learned_parameters()

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save predictions
    np.savez(output_dir / 'predictions.npz',
            pinn_pred=pinn_results['predictions'],
            nn_pred=nn_results['predictions'],
            targets=pinn_results['targets'])

    print(f"\nResults saved to {output_dir}")
    print(f"Models: pinn_model.pth, standard_nn_model.pth")
    print(f"Metrics: metrics.json")
    print(f"Predictions: predictions.npz")


def main():
    """Main experiment function."""
    args = parse_args()

    print("="*80)
    print("PHYSICS-INFORMED NEURAL NETWORK (PINN)")
    print("Transformer Oil Temperature Prediction")
    print("="*80)
    print(f"\nExperiment Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*80)

    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device(args.device)
    print(f"\nUsing device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"{args.dataset}_{timestamp}"

    # Load data
    print("\n" + "="*60)
    print("Loading and Preprocessing Data")
    print("="*60)

    data_dict = create_pinn_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        n_collocation=args.n_collocation
    )

    # Create models
    print("\n" + "="*60)
    print("Creating Models")
    print("="*60)

    model_config = {
        'input_size': 7,
        'hidden_sizes': args.hidden_sizes,
        'activation': 'tanh',
        'learn_physics_params': True,
        'physics_weight_init': 0.001,
    }

    pinn_model = create_pinn_model(model_config, device=device)
    print(f"PINN created with {sum(p.numel() for p in pinn_model.parameters())} parameters")

    standard_nn = create_standard_nn(model_config, device=device)
    print(f"Standard NN created with {sum(p.numel() for p in standard_nn.parameters())} parameters")

    # Train PINN
    pinn_model, pinn_history = train_model(
        'PINN', pinn_model, data_dict, args, device
    )

    # Train Standard NN
    standard_nn, nn_history = train_model(
        'Standard NN', standard_nn, data_dict, args, device
    )

    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60)

    pinn_results = evaluate_model(
        pinn_model,
        data_dict['test_loader'],
        data_dict['test_dataset'],
        device,
        model_name='PINN'
    )

    nn_results = evaluate_model(
        standard_nn,
        data_dict['test_loader'],
        data_dict['test_dataset'],
        device,
        model_name='Standard NN'
    )

    # Print comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"\n{'Metric':<15} {'PINN':<12} {'Standard NN':<12} {'Improvement':<12}")
    print("-" * 60)
    print(f"{'RMSE (°C)':<15} {pinn_results['rmse']:<12.6f} {nn_results['rmse']:<12.6f} "
          f"{(nn_results['rmse']-pinn_results['rmse'])/nn_results['rmse']*100:>10.2f}%")
    print(f"{'MAE (°C)':<15} {pinn_results['mae']:<12.6f} {nn_results['mae']:<12.6f} "
          f"{(nn_results['mae']-pinn_results['mae'])/nn_results['mae']*100:>10.2f}%")
    print(f"{'MAPE (%)':<15} {pinn_results['mape']:<12.4f} {nn_results['mape']:<12.4f} "
          f"{(nn_results['mape']-pinn_results['mape'])/nn_results['mape']*100:>10.2f}%")
    print(f"{'R²':<15} {pinn_results['r2']:<12.6f} {nn_results['r2']:<12.6f} "
          f"{(pinn_results['r2']-nn_results['r2']):>11.6f}")

    # Check if target RMSE < 0.15 achieved
    print("\n" + "="*60)
    if pinn_results['rmse'] < 0.15:
        print(f"✅ TARGET ACHIEVED! RMSE = {pinn_results['rmse']:.6f} < 0.15 °C")
    else:
        print(f"⚠️  Target not achieved. RMSE = {pinn_results['rmse']:.6f} °C")
        print(f"   (Target: < 0.15 °C, Gap: {pinn_results['rmse'] - 0.15:.6f} °C)")
    print("="*60)

    # Show learned physics parameters
    if hasattr(pinn_model, 'get_learned_parameters'):
        print("\n" + "="*60)
        print("Learned Physics Parameters (IEEE C57.91)")
        print("="*60)
        params = pinn_model.get_learned_parameters()
        print(f"  R (load/no-load loss ratio):      {params['R']:.4f}  (typical: 5-6)")
        print(f"  n (oil exponent):                  {params['n']:.4f}  (typical: 0.8-0.9)")
        print(f"  m (winding exponent):              {params['m']:.4f}  (typical: 0.8-0.9)")
        print(f"  τ_oil (time constant, min):        {params['tau_oil']:.2f}  (typical: 150-210)")
        print(f"  Δθ_oil_rated (oil temp rise, K):   {params['delta_theta_oil_rated']:.2f}  (typical: 50-60)")
        print(f"  Δθ_hot_rated (hot-spot rise, K):   {params['delta_theta_hot_rated']:.2f}  (typical: 60-70)")

    # Generate visualizations
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)

    plot_results(pinn_results, nn_results, pinn_history, nn_history, output_dir)

    # Save results
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)

    save_results(pinn_model, standard_nn, pinn_results, nn_results,
                pinn_history, nn_history, args, output_dir)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nKey Files:")
    print(f"  - {output_dir}/pinn_model.pth")
    print(f"  - {output_dir}/metrics.json")
    print(f"  - {output_dir}/predictions.npz")
    print(f"  - {output_dir}/training_curves.png")
    print(f"  - {output_dir}/predictions_comparison.png")
    print(f"  - {output_dir}/scatter_comparison.png")
    print(f"  - {output_dir}/error_analysis.png")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
