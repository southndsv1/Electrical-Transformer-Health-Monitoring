"""
Visualization utilities for generating paper-ready plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import PLOT_CONFIG, RESULTS_DIR

# Set style
plt.style.use(PLOT_CONFIG['style'])
sns.set_palette(PLOT_CONFIG['color_palette'])
plt.rcParams['figure.dpi'] = PLOT_CONFIG['dpi']
plt.rcParams['font.size'] = PLOT_CONFIG['font_size']


def plot_model_comparison(results: Dict[str, Dict], metric: str = 'rmse',
                         save_path: Optional[str] = None, title: Optional[str] = None):
    """
    Create bar plot comparing models on a specific metric.

    Args:
        results: Dictionary mapping model names to results dictionaries
        metric: Metric to plot
        save_path: Path to save figure (optional)
        title: Plot title (optional)
    """
    models = list(results.keys())
    values = [results[m][metric] for m in models]

    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figure_size'])

    bars = ax.bar(range(len(models)), values, alpha=0.8)

    # Color bars by value (lower is better for most metrics)
    if metric in ['rmse', 'mae', 'mape']:
        colors = plt.cm.RdYlGn_r(np.array(values) / max(values))
    else:
        colors = plt.cm.RdYlGn(np.array(values) / max(values))

    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel(metric.upper())

    if title is None:
        title = f'Model Comparison - {metric.upper()}'
    ax.set_title(title)

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
    plt.close()


def plot_box_comparison(results: Dict[str, List[float]], metric: str = 'rmse',
                       save_path: Optional[str] = None, title: Optional[str] = None):
    """
    Create box plot showing performance distribution across models.

    Args:
        results: Dictionary mapping model names to lists of metric values
        metric: Metric name
        save_path: Path to save figure (optional)
        title: Plot title (optional)
    """
    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(14, 6))

    box_plot = df.boxplot(ax=ax, rot=45, patch_artist=True)

    # Color boxes
    for patch, color in zip(box_plot.artists, sns.color_palette()):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(metric.upper())
    ax.set_xlabel('Model')

    if title is None:
        title = f'Performance Distribution - {metric.upper()}'
    ax.set_title(title)

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
    plt.close()


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                               dates: Optional[pd.DatetimeIndex] = None,
                               model_name: str = 'Model',
                               save_path: Optional[str] = None,
                               n_points: int = 500):
    """
    Plot predicted vs actual values over time.

    Args:
        y_true: True values
        y_pred: Predicted values
        dates: DatetimeIndex for x-axis (optional)
        model_name: Name of the model
        save_path: Path to save figure (optional)
        n_points: Number of points to plot (for readability)
    """
    # Sample points if too many
    if len(y_true) > n_points:
        indices = np.linspace(0, len(y_true) - 1, n_points, dtype=int)
        y_true = y_true[indices]
        y_pred = y_pred[indices]
        if dates is not None:
            dates = dates[indices]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Time series
    x = dates if dates is not None else np.arange(len(y_true))

    ax1.plot(x, y_true, label='Actual', linewidth=2, alpha=0.8)
    ax1.plot(x, y_pred, label='Predicted', linewidth=2, alpha=0.8)
    ax1.fill_between(x, y_true, y_pred, alpha=0.2, color='red', label='Error')

    ax1.set_xlabel('Time' if dates is not None else 'Sample')
    ax1.set_ylabel('Oil Temperature (°C)')
    ax1.set_title(f'{model_name}: Predictions vs Actual')
    ax1.legend()
    ax1.grid(alpha=0.3)

    if dates is not None:
        ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Scatter plot
    ax2.scatter(y_true, y_pred, alpha=0.5, s=10)

    # Perfect prediction line
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             label='Perfect Prediction')

    ax2.set_xlabel('Actual Temperature (°C)')
    ax2.set_ylabel('Predicted Temperature (°C)')
    ax2.set_title('Predicted vs Actual Scatter')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
    plt.close()


def plot_horizon_performance(horizon_results: Dict[str, Dict[str, float]],
                            metric: str = 'rmse',
                            save_path: Optional[str] = None):
    """
    Plot performance across different forecast horizons.

    Args:
        horizon_results: Dictionary mapping model names to horizon results
        metric: Metric to plot
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for model_name, results in horizon_results.items():
        horizons = sorted([int(h.replace('h', '')) for h in results.keys() if h.startswith('h')])
        values = [results[f'h{h}'][metric] for h in horizons]

        ax.plot(horizons, values, marker='o', linewidth=2, label=model_name)

    ax.set_xlabel('Forecast Horizon (hours)')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'Performance vs Forecast Horizon - {metric.upper()}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
    plt.close()


def plot_computational_efficiency(results: Dict[str, Dict],
                                 accuracy_metric: str = 'rmse',
                                 time_metric: str = 'training_time',
                                 save_path: Optional[str] = None):
    """
    Scatter plot showing accuracy vs computational cost.

    Args:
        results: Dictionary mapping model names to results
        accuracy_metric: Metric for accuracy (y-axis)
        time_metric: Metric for time (x-axis)
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    models = list(results.keys())
    times = [results[m][time_metric] for m in models]
    accuracies = [results[m][accuracy_metric] for m in models]

    # Create scatter plot
    scatter = ax.scatter(times, accuracies, s=200, alpha=0.6, c=range(len(models)),
                        cmap='viridis')

    # Add labels for each point
    for i, model in enumerate(models):
        ax.annotate(model, (times[i], accuracies[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8)

    ax.set_xlabel(f'{time_metric.replace("_", " ").title()} (seconds)')
    ax.set_ylabel(accuracy_metric.upper())
    ax.set_title('Computational Efficiency: Accuracy vs Training Time')
    ax.grid(alpha=0.3)

    # Add pareto frontier
    pareto_indices = []
    sorted_indices = np.argsort(times)
    min_error = float('inf')

    for idx in sorted_indices:
        if accuracies[idx] < min_error:
            min_error = accuracies[idx]
            pareto_indices.append(idx)

    if len(pareto_indices) > 1:
        pareto_times = [times[i] for i in pareto_indices]
        pareto_errors = [accuracies[i] for i in pareto_indices]
        ax.plot(pareto_times, pareto_errors, 'r--', linewidth=2,
               alpha=0.5, label='Pareto Frontier')
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
    plt.close()


def plot_learning_curves(train_losses: List[float], val_losses: List[float],
                        model_name: str = 'Model',
                        save_path: Optional[str] = None):
    """
    Plot training and validation learning curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        model_name: Name of the model
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, label='Validation Loss', linewidth=2)

    # Mark best epoch
    best_epoch = np.argmin(val_losses) + 1
    ax.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5,
              label=f'Best Epoch ({best_epoch})')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{model_name}: Learning Curves')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
    plt.close()


def plot_residuals_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                           model_name: str = 'Model',
                           save_path: Optional[str] = None):
    """
    Create comprehensive residuals analysis plot.

    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        save_path: Path to save figure (optional)
    """
    residuals = y_true - y_pred

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Residuals vs predicted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_pred, residuals, alpha=0.5, s=10)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')
    ax1.grid(alpha=0.3)

    # Histogram of residuals
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Residuals')
    ax2.grid(alpha=0.3)

    # Q-Q plot
    ax3 = fig.add_subplot(gs[1, 0])
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot')
    ax3.grid(alpha=0.3)

    # Residuals over time
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(residuals, alpha=0.5)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax4.axhline(y=np.std(residuals), color='orange', linestyle='--',
               alpha=0.5, label='±1 Std Dev')
    ax4.axhline(y=-np.std(residuals), color='orange', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residuals Over Time')
    ax4.legend()
    ax4.grid(alpha=0.3)

    fig.suptitle(f'{model_name}: Residuals Analysis', fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
    plt.close()


def plot_seasonal_performance(seasonal_results: Dict[int, Dict[str, float]],
                             metric: str = 'rmse',
                             save_path: Optional[str] = None):
    """
    Plot performance across different months/seasons.

    Args:
        seasonal_results: Dictionary mapping months to metrics
        metric: Metric to plot
        save_path: Path to save figure (optional)
    """
    months = sorted(seasonal_results.keys())
    values = [seasonal_results[m][metric] for m in months]

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(range(len(months)), values, alpha=0.8)

    # Color by season
    season_colors = ['#3498db', '#3498db', '#2ecc71', '#2ecc71', '#2ecc71',
                    '#e74c3c', '#e74c3c', '#e74c3c', '#f39c12', '#f39c12',
                    '#3498db', '#3498db']

    for bar, color in zip(bars, [season_colors[m - 1] for m in months]):
        bar.set_color(color)

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([month_names[m - 1] for m in months])
    ax.set_ylabel(metric.upper())
    ax.set_title(f'Seasonal Performance - {metric.upper()}')
    ax.grid(axis='y', alpha=0.3)

    # Add legend for seasons
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Winter'),
        Patch(facecolor='#2ecc71', label='Spring'),
        Patch(facecolor='#e74c3c', label='Summer'),
        Patch(facecolor='#f39c12', label='Fall'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
    plt.close()


def create_results_summary_plot(all_results: Dict[str, Dict[str, any]],
                               save_path: Optional[str] = None):
    """
    Create comprehensive summary plot with multiple subplots.

    Args:
        all_results: Dictionary with all model results
        save_path: Path to save figure (optional)
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    models = list(all_results.keys())

    # RMSE comparison
    ax1 = fig.add_subplot(gs[0, 0])
    rmse_values = [all_results[m]['rmse'] for m in models]
    ax1.barh(models, rmse_values, alpha=0.8)
    ax1.set_xlabel('RMSE')
    ax1.set_title('Root Mean Squared Error')
    ax1.grid(axis='x', alpha=0.3)

    # MAE comparison
    ax2 = fig.add_subplot(gs[0, 1])
    mae_values = [all_results[m]['mae'] for m in models]
    ax2.barh(models, mae_values, alpha=0.8)
    ax2.set_xlabel('MAE')
    ax2.set_title('Mean Absolute Error')
    ax2.grid(axis='x', alpha=0.3)

    # R² comparison
    ax3 = fig.add_subplot(gs[1, 0])
    r2_values = [all_results[m].get('r2', 0) for m in models]
    ax3.barh(models, r2_values, alpha=0.8)
    ax3.set_xlabel('R²')
    ax3.set_title('R² Score')
    ax3.grid(axis='x', alpha=0.3)

    # Training time comparison
    ax4 = fig.add_subplot(gs[1, 1])
    time_values = [all_results[m].get('training_time', 0) for m in models]
    ax4.barh(models, time_values, alpha=0.8)
    ax4.set_xlabel('Time (seconds)')
    ax4.set_title('Training Time')
    ax4.grid(axis='x', alpha=0.3)

    # Critical temperature F1 score
    ax5 = fig.add_subplot(gs[2, 0])
    f1_values = [all_results[m].get('critical_f1', 0) for m in models]
    ax5.barh(models, f1_values, alpha=0.8)
    ax5.set_xlabel('F1 Score')
    ax5.set_title('Critical Temperature Detection F1')
    ax5.grid(axis='x', alpha=0.3)

    # Computational efficiency scatter
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.scatter(time_values, rmse_values, s=200, alpha=0.6)
    for i, model in enumerate(models):
        ax6.annotate(model, (time_values[i], rmse_values[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8)
    ax6.set_xlabel('Training Time (s)')
    ax6.set_ylabel('RMSE')
    ax6.set_title('Computational Efficiency')
    ax6.grid(alpha=0.3)

    fig.suptitle('Comprehensive Model Comparison', fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
    plt.close()


if __name__ == '__main__':
    print("Testing visualization functions...")

    # Generate sample data
    np.random.seed(42)
    models = ['Model_A', 'Model_B', 'Model_C']
    results = {m: {'rmse': np.random.uniform(2, 5),
                   'mae': np.random.uniform(1, 3),
                   'r2': np.random.uniform(0.8, 0.95),
                   'training_time': np.random.uniform(10, 100)}
               for m in models}

    # Test model comparison plot
    plot_model_comparison(results, metric='rmse')

    print("Visualization tests completed!")
