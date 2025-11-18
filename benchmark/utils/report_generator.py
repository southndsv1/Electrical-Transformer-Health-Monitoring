"""
Generate comprehensive reports from benchmark results.
Includes LaTeX tables, markdown summaries, and CSV exports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import LATEX_CONFIG, RESULTS_DIR


def create_latex_table(results: Dict[str, Dict], metrics: List[str],
                       caption: str, label: str) -> str:
    """
    Create LaTeX table from results.

    Args:
        results: Dictionary mapping model names to metrics
        metrics: List of metrics to include
        caption: Table caption
        label: Table label

    Returns:
        LaTeX table string
    """
    # Create DataFrame
    data = []
    for model_name, model_results in results.items():
        row = {'Model': model_name}
        for metric in metrics:
            if metric in model_results:
                row[metric.upper()] = model_results[metric]
        data.append(row)

    df = pd.DataFrame(data)

    # Sort by first metric (usually RMSE)
    if len(metrics) > 0 and metrics[0].upper() in df.columns:
        df = df.sort_values(metrics[0].upper())

    # Start LaTeX table
    n_cols = len(df.columns)
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += "\\begin{tabular}{" + "l" + "c" * (n_cols - 1) + "}\n"
    latex += "\\toprule\n"

    # Header
    latex += " & ".join(df.columns) + " \\\\\n"
    latex += "\\midrule\n"

    # Rows
    for _, row in df.iterrows():
        row_str = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, (int, float, np.number)):
                row_str.append(f"{val:.4f}")
            else:
                row_str.append(str(val))
        latex += " & ".join(row_str) + " \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


def create_comparison_table(results: Dict[str, Dict], save_path: Path = None) -> pd.DataFrame:
    """
    Create comprehensive comparison table.

    Args:
        results: Dictionary mapping model names to metrics
        save_path: Path to save CSV (optional)

    Returns:
        DataFrame with comparison
    """
    # Extract all metrics
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())

    # Create DataFrame
    data = []
    for model_name, model_results in results.items():
        row = {'Model': model_name}
        for metric in all_metrics:
            row[metric] = model_results.get(metric, np.nan)
        data.append(row)

    df = pd.DataFrame(data)

    # Order columns
    priority_cols = ['Model', 'rmse', 'mae', 'mape', 'r2', 'training_time', 'inference_time']
    other_cols = [c for c in df.columns if c not in priority_cols]
    ordered_cols = [c for c in priority_cols if c in df.columns] + sorted(other_cols)
    df = df[ordered_cols]

    # Sort by RMSE if available
    if 'rmse' in df.columns:
        df = df.sort_values('rmse')

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Comparison table saved to {save_path}")

    return df


def create_markdown_report(results: Dict[str, Dict], experiment_name: str,
                          save_path: Path = None) -> str:
    """
    Create markdown summary report.

    Args:
        results: Dictionary mapping model names to metrics
        experiment_name: Name of the experiment
        save_path: Path to save report (optional)

    Returns:
        Markdown report string
    """
    report = f"# {experiment_name}\n\n"
    report += f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Summary statistics
    report += "## Summary Statistics\n\n"
    report += f"- Total models evaluated: {len(results)}\n"

    # Find best models for each metric
    metrics_to_check = ['rmse', 'mae', 'r2']
    for metric in metrics_to_check:
        values = {k: v.get(metric, np.nan) for k, v in results.items() if metric in v}
        if values:
            if metric == 'r2':
                best_model = max(values, key=values.get)
                best_value = values[best_model]
            else:
                best_model = min(values, key=values.get)
                best_value = values[best_model]
            report += f"- Best {metric.upper()}: **{best_model}** ({best_value:.4f})\n"

    # Detailed results
    report += "\n## Detailed Results\n\n"

    # Create table
    df = create_comparison_table(results)
    report += df.to_markdown(index=False) + "\n\n"

    # Key findings
    report += "## Key Findings\n\n"

    # Best overall model (based on RMSE)
    if 'rmse' in df.columns:
        best_idx = df['rmse'].idxmin()
        best_model = df.loc[best_idx, 'Model']
        best_rmse = df.loc[best_idx, 'rmse']

        report += f"1. **Best Overall Model**: {best_model}\n"
        report += f"   - Achieved RMSE of {best_rmse:.4f}\n"

        if 'training_time' in df.columns:
            training_time = df.loc[best_idx, 'training_time']
            report += f"   - Training time: {training_time:.2f} seconds\n"

    # Fastest model
    if 'training_time' in df.columns:
        fastest_idx = df['training_time'].idxmin()
        fastest_model = df.loc[fastest_idx, 'Model']
        fastest_time = df.loc[fastest_idx, 'training_time']

        report += f"\n2. **Fastest Training**: {fastest_model}\n"
        report += f"   - Training time: {fastest_time:.2f} seconds\n"

        if 'rmse' in df.columns:
            rmse = df.loc[fastest_idx, 'rmse']
            report += f"   - RMSE: {rmse:.4f}\n"

    # Critical temperature detection
    if 'critical_f1' in df.columns:
        best_critical_idx = df['critical_f1'].idxmax()
        best_critical_model = df.loc[best_critical_idx, 'Model']
        best_critical_f1 = df.loc[best_critical_idx, 'critical_f1']

        report += f"\n3. **Best Critical Temperature Detection**: {best_critical_model}\n"
        report += f"   - F1 Score: {best_critical_f1:.4f}\n"

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Markdown report saved to {save_path}")

    return report


def create_horizon_comparison_table(horizon_results: Dict[str, Dict],
                                   save_path: Path = None) -> pd.DataFrame:
    """
    Create table comparing performance across forecast horizons.

    Args:
        horizon_results: Dictionary mapping horizons to results
        save_path: Path to save CSV (optional)

    Returns:
        DataFrame with horizon comparison
    """
    data = []

    for horizon, models_results in horizon_results.items():
        for model_name, metrics in models_results.items():
            row = {
                'Horizon': horizon,
                'Model': model_name,
            }
            row.update(metrics)
            data.append(row)

    df = pd.DataFrame(data)

    # Pivot table
    if 'rmse' in df.columns:
        pivot = df.pivot_table(
            index='Model',
            columns='Horizon',
            values='rmse',
            aggfunc='mean'
        )

        if save_path:
            pivot.to_csv(save_path)
            print(f"Horizon comparison table saved to {save_path}")

        return pivot

    return df


def generate_all_reports(results: Dict[str, Dict], experiment_name: str,
                        output_dir: Path = None):
    """
    Generate all report formats.

    Args:
        results: Dictionary mapping model names to metrics
        experiment_name: Name of the experiment
        output_dir: Output directory for reports
    """
    if output_dir is None:
        output_dir = RESULTS_DIR / 'tables'

    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV comparison table
    csv_path = output_dir / f'{experiment_name}_comparison.csv'
    df = create_comparison_table(results, csv_path)

    # Markdown report
    md_path = output_dir / f'{experiment_name}_report.md'
    create_markdown_report(results, experiment_name, md_path)

    # LaTeX table
    latex_table = create_latex_table(
        results,
        metrics=['rmse', 'mae', 'r2', 'training_time'],
        caption=f'{LATEX_CONFIG["caption_prefix"]} Model Comparison',
        label=f'{LATEX_CONFIG["label_prefix"]}{experiment_name}'
    )

    latex_path = output_dir / f'{experiment_name}_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to {latex_path}")

    print(f"\nAll reports generated in {output_dir}")

    return df


if __name__ == '__main__':
    # Test report generation
    print("Testing report generator...")

    # Sample results
    results = {
        'Model_A': {'rmse': 2.5, 'mae': 1.8, 'r2': 0.92, 'training_time': 45.2},
        'Model_B': {'rmse': 2.8, 'mae': 2.0, 'r2': 0.90, 'training_time': 120.5},
        'Model_C': {'rmse': 2.3, 'mae': 1.7, 'r2': 0.93, 'training_time': 85.3},
    }

    # Generate reports
    md_report = create_markdown_report(results, 'Test Experiment')
    print(md_report)

    latex_table = create_latex_table(
        results,
        metrics=['rmse', 'mae', 'r2'],
        caption='Test Comparison',
        label='tab:test'
    )
    print("\n" + latex_table)

    print("\nReport generator tests completed!")
