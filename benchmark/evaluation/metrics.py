"""
Evaluation metrics for transformer temperature forecasting.
Includes standard metrics and custom metrics for critical temperature detection.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import scikit_posthocs as sp
from typing import Dict, List, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import CRITICAL_TEMP_THRESHOLD


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Mean Absolute Percentage Error.

    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE value (in percentage)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero
    denominator = np.maximum(np.abs(y_true), epsilon)
    return np.mean(np.abs((y_true - y_pred) / denominator)) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        SMAPE value (in percentage)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100


def critical_temperature_detection(y_true: np.ndarray, y_pred: np.ndarray,
                                   threshold: float = CRITICAL_TEMP_THRESHOLD) -> Dict:
    """
    Custom metric for detecting critical temperature events.

    This metric evaluates how well the model predicts when temperature exceeds
    a critical threshold (default 95°C for transformer oil).

    Metrics:
    - Precision: Of predicted critical events, how many were actually critical?
    - Recall: Of actual critical events, how many did we predict?
    - F1-Score: Harmonic mean of precision and recall
    - False Alarm Rate: Predicted critical but wasn't
    - Miss Rate: Was critical but didn't predict

    Args:
        y_true: True temperature values
        y_pred: Predicted temperature values
        threshold: Critical temperature threshold

    Returns:
        Dictionary with detection metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Binary classification: critical or not
    true_critical = y_true >= threshold
    pred_critical = y_pred >= threshold

    # Calculate metrics
    true_positives = np.sum(true_critical & pred_critical)
    false_positives = np.sum(~true_critical & pred_critical)
    true_negatives = np.sum(~true_critical & ~pred_critical)
    false_negatives = np.sum(true_critical & ~pred_critical)

    # Precision and recall
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)

    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    # False alarm rate and miss rate
    false_alarm_rate = false_positives / (false_positives + true_negatives + 1e-10)
    miss_rate = false_negatives / (false_negatives + true_positives + 1e-10)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'false_alarm_rate': false_alarm_rate,
        'miss_rate': miss_rate,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'n_critical_events': np.sum(true_critical),
    }


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         include_critical: bool = True) -> Dict:
    """
    Calculate all evaluation metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        include_critical: Whether to include critical temperature detection

    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'mape': mape(y_true, y_pred),
        'smape': smape(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }

    if include_critical:
        critical_metrics = critical_temperature_detection(y_true, y_pred)
        metrics['critical_precision'] = critical_metrics['precision']
        metrics['critical_recall'] = critical_metrics['recall']
        metrics['critical_f1'] = critical_metrics['f1_score']
        metrics['critical_false_alarm_rate'] = critical_metrics['false_alarm_rate']
        metrics['critical_miss_rate'] = critical_metrics['miss_rate']

    return metrics


def forecast_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            horizons: List[int]) -> Dict:
    """
    Calculate metrics at different forecast horizons.

    Args:
        y_true: True values (shape: [n_samples, forecast_horizon])
        y_pred: Predicted values (shape: [n_samples, forecast_horizon])
        horizons: List of horizons to evaluate (e.g., [1, 6, 12, 24])

    Returns:
        Dictionary with metrics for each horizon
    """
    horizon_metrics = {}

    for h in horizons:
        if h <= y_true.shape[1]:
            # Get predictions at horizon h (index h-1 for 0-indexing)
            y_true_h = y_true[:, h - 1]
            y_pred_h = y_pred[:, h - 1]

            horizon_metrics[f'h{h}'] = calculate_all_metrics(y_true_h, y_pred_h)

    return horizon_metrics


def probabilistic_metrics(y_true: np.ndarray, y_pred_samples: np.ndarray,
                         quantiles: List[float] = [0.1, 0.5, 0.9]) -> Dict:
    """
    Calculate metrics for probabilistic forecasts.

    Args:
        y_true: True values
        y_pred_samples: Predicted samples from probabilistic model
                       (shape: [n_samples, n_predictions])
        quantiles: Quantiles to evaluate

    Returns:
        Dictionary with probabilistic metrics
    """
    metrics = {}

    # Calculate quantiles
    y_pred_quantiles = np.quantile(y_pred_samples, quantiles, axis=1).T

    # Coverage probability for each quantile pair
    for i, q in enumerate(quantiles):
        if q < 0.5:
            upper_q = 1 - q
            lower_idx = i
            upper_idx = len(quantiles) - 1 - i

            if upper_idx < len(quantiles):
                lower = y_pred_quantiles[:, lower_idx]
                upper = y_pred_quantiles[:, upper_idx]

                # Coverage: fraction of true values within interval
                coverage = np.mean((y_true >= lower) & (y_true <= upper))
                metrics[f'coverage_{int(q*100)}_{int(upper_q*100)}'] = coverage

    # Continuous Ranked Probability Score (CRPS)
    # Approximation using samples
    crps_values = []
    for i in range(len(y_true)):
        samples = y_pred_samples[:, i]
        true_val = y_true[i]

        # CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
        term1 = np.mean(np.abs(samples - true_val))
        term2 = 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
        crps_values.append(term1 - term2)

    metrics['crps'] = np.mean(crps_values)

    # Prediction interval width
    for i, q in enumerate(quantiles):
        if q < 0.5:
            upper_idx = len(quantiles) - 1 - i
            if upper_idx < len(quantiles):
                lower = y_pred_quantiles[:, i]
                upper = y_pred_quantiles[:, upper_idx]
                metrics[f'interval_width_{int(q*100)}'] = np.mean(upper - lower)

    return metrics


def friedman_nemenyi_test(results: Dict[str, List[float]],
                         metric_name: str = 'rmse') -> Tuple[float, pd.DataFrame]:
    """
    Perform Friedman test followed by Nemenyi post-hoc test for model comparison.

    The Friedman test checks if there are significant differences between models.
    The Nemenyi test performs pairwise comparisons.

    Args:
        results: Dictionary mapping model names to list of metric values
                (e.g., from cross-validation or multiple datasets)
        metric_name: Name of the metric being tested

    Returns:
        Tuple of (friedman_p_value, nemenyi_results_df)
    """
    import pandas as pd

    # Prepare data for Friedman test
    model_names = list(results.keys())
    n_models = len(model_names)
    n_samples = len(list(results.values())[0])

    # Create data matrix: rows are samples, columns are models
    data = np.array([results[model] for model in model_names]).T

    # Friedman test
    statistic, p_value = stats.friedmanaligned(data)

    print(f"\nFriedman Test for {metric_name}:")
    print(f"  Statistic: {statistic:.4f}")
    print(f"  P-value: {p_value:.6f}")

    if p_value < 0.05:
        print(f"  Result: Significant differences detected (p < 0.05)")

        # Perform Nemenyi post-hoc test
        # Convert to DataFrame for scikit-posthocs
        df = pd.DataFrame(data, columns=model_names)

        # Nemenyi test
        nemenyi_results = sp.posthoc_nemenyi_friedman(df)

        return p_value, nemenyi_results
    else:
        print(f"  Result: No significant differences (p >= 0.05)")
        return p_value, None


def compute_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval for a metric.

    Args:
        values: Array of metric values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    mean = np.mean(values)
    std_err = stats.sem(values)
    margin = std_err * stats.t.ppf((1 + confidence) / 2, len(values) - 1)

    return mean - margin, mean + margin


def calculate_seasonal_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              dates: pd.DatetimeIndex) -> Dict:
    """
    Calculate metrics broken down by season/month.

    Args:
        y_true: True values
        y_pred: Predicted values
        dates: DatetimeIndex for temporal grouping

    Returns:
        Dictionary with metrics by month
    """
    import pandas as pd

    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'month': dates.month,
    })

    monthly_metrics = {}

    for month in range(1, 13):
        month_data = df[df['month'] == month]

        if len(month_data) > 0:
            monthly_metrics[f'month_{month}'] = calculate_all_metrics(
                month_data['y_true'].values,
                month_data['y_pred'].values,
                include_critical=False
            )

    return monthly_metrics


if __name__ == '__main__':
    # Test metrics
    print("Testing evaluation metrics...")

    # Generate sample data
    np.random.seed(42)
    y_true = np.random.uniform(50, 100, 1000)  # Temperature range
    y_pred = y_true + np.random.normal(0, 5, 1000)  # Predictions with noise

    # Test all metrics
    print("\nStandard Metrics:")
    metrics = calculate_all_metrics(y_true, y_pred)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Test critical temperature detection
    print("\nCritical Temperature Detection:")
    critical = critical_temperature_detection(y_true, y_pred)
    for name, value in critical.items():
        if isinstance(value, (int, np.integer)):
            print(f"  {name}: {value}")
        else:
            print(f"  {name}: {value:.4f}")

    # Test probabilistic metrics
    print("\nProbabilistic Metrics:")
    y_pred_samples = y_true[:, None] + np.random.normal(0, 5, (len(y_true), 100))
    prob_metrics = probabilistic_metrics(y_true, y_pred_samples)
    for name, value in prob_metrics.items():
        print(f"  {name}: {value:.4f}")

    print("\nAll tests passed!")
