#!/usr/bin/env python3
"""
Complete Benchmarking Framework for Electrical Transformer Health Monitoring

This script runs comprehensive experiments comparing classical machine learning
and deep learning models for transformer oil temperature forecasting.

Author: Claude AI
Date: 2025
"""

import sys
import os
from pathlib import Path
import argparse
import json
import time
from datetime import datetime

# Add benchmark to path
sys.path.append(str(Path(__file__).parent / 'benchmark'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from configs.config import (
    FORECAST_HORIZONS, RANDOM_SEED, RESULTS_DIR,
    CLASSICAL_MODELS, DEEP_LEARNING_MODELS
)
from data.data_loader import get_data_loader
from models.classical_models import train_all_classical_models
from models.deep_learning_models import train_all_deep_learning_models
from utils.helpers import set_seed, setup_logger, save_results, create_experiment_dir
from utils.visualization import (
    plot_model_comparison, plot_predictions_vs_actual,
    plot_computational_efficiency, create_results_summary_plot,
    plot_horizon_performance
)
from utils.report_generator import generate_all_reports


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive transformer temperature forecasting benchmark'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='ETTh1',
        choices=['ETTh1', 'ETTh2', 'both'],
        help='Dataset to use for benchmarking'
    )

    parser.add_argument(
        '--models',
        type=str,
        default='all',
        choices=['all', 'classical', 'deep_learning'],
        help='Which models to train'
    )

    parser.add_argument(
        '--horizon',
        type=int,
        default=24,
        help='Forecast horizon in hours (default: 24)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick benchmark with reduced epochs for deep learning models'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: benchmark/results)'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )

    return parser.parse_args()


def run_benchmark_for_dataset(dataset_name: str, forecast_horizon: int,
                              model_types: str, quick_mode: bool,
                              logger, output_dir: Path):
    """
    Run benchmark for a single dataset.

    Args:
        dataset_name: Name of dataset
        forecast_horizon: Forecast horizon in hours
        model_types: Which models to train ('all', 'classical', 'deep_learning')
        quick_mode: Whether to run in quick mode
        logger: Logger instance
        output_dir: Output directory

    Returns:
        Dictionary with all results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Running Benchmark for {dataset_name}")
    logger.info(f"Forecast Horizon: {forecast_horizon} hours")
    logger.info(f"{'='*80}\n")

    # Load data
    logger.info("Loading and preprocessing data...")
    loader = get_data_loader(dataset_name)
    loader.load_data()
    loader.create_splits()

    all_results = {}
    all_models = {}

    # Classical models
    if model_types in ['all', 'classical']:
        logger.info("\n" + "="*80)
        logger.info("CLASSICAL MODELS")
        logger.info("="*80)

        try:
            ml_data = loader.get_classical_ml_data(
                forecast_horizon=forecast_horizon,
                n_lags=168
            )

            classical_results, classical_models = train_all_classical_models(
                ml_data, forecast_horizon
            )

            all_results.update(classical_results)
            all_models.update(classical_models)

            logger.info("\nClassical models training completed!")

        except Exception as e:
            logger.error(f"Error training classical models: {e}")
            import traceback
            traceback.print_exc()

    # Deep learning models
    if model_types in ['all', 'deep_learning']:
        logger.info("\n" + "="*80)
        logger.info("DEEP LEARNING MODELS")
        logger.info("="*80)

        try:
            # Modify config for quick mode
            if quick_mode:
                logger.info("Running in QUICK MODE (reduced epochs)")
                for model_config in DEEP_LEARNING_MODELS.values():
                    model_config['epochs'] = 10  # Reduce epochs

            dl_data = loader.get_deep_learning_data(
                seq_length=168,
                forecast_horizon=forecast_horizon,
                stride_train=24,  # Sample every 24 hours for speed
                stride_val=24
            )

            dl_results, dl_models = train_all_deep_learning_models(
                dl_data, forecast_horizon
            )

            all_results.update(dl_results)
            all_models.update(dl_models)

            logger.info("\nDeep learning models training completed!")

        except Exception as e:
            logger.error(f"Error training deep learning models: {e}")
            import traceback
            traceback.print_exc()

    return all_results, all_models, loader


def main():
    """Main benchmark execution."""
    args = parse_args()

    # Set random seed
    set_seed(RANDOM_SEED)

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"benchmark_{args.dataset}_{timestamp}"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = RESULTS_DIR / exp_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger('benchmark', output_dir / 'logs' / 'benchmark.log')

    logger.info("="*80)
    logger.info("TRANSFORMER TEMPERATURE FORECASTING BENCHMARK")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Forecast horizon: {args.horizon} hours")
    logger.info(f"Quick mode: {args.quick}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*80)

    # Determine datasets to process
    if args.dataset == 'both':
        datasets = ['ETTh1', 'ETTh2']
    else:
        datasets = [args.dataset]

    # Run benchmarks
    all_experiment_results = {}

    for dataset_name in datasets:
        try:
            results, models, loader = run_benchmark_for_dataset(
                dataset_name=dataset_name,
                forecast_horizon=args.horizon,
                model_types=args.models,
                quick_mode=args.quick,
                logger=logger,
                output_dir=output_dir
            )

            all_experiment_results[dataset_name] = {
                'results': results,
                'models': models,
                'loader': loader
            }

        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate reports and visualizations
    logger.info("\n" + "="*80)
    logger.info("GENERATING REPORTS AND VISUALIZATIONS")
    logger.info("="*80)

    for dataset_name, exp_data in all_experiment_results.items():
        results = exp_data['results']
        loader = exp_data['loader']

        logger.info(f"\nGenerating reports for {dataset_name}...")

        # Save results as JSON
        results_file = output_dir / f'{dataset_name}_results.json'
        save_results(results, results_file, format='json')

        # Generate comprehensive reports
        try:
            df = generate_all_reports(
                results,
                experiment_name=f'{dataset_name}_h{args.horizon}',
                output_dir=output_dir / 'tables'
            )

            logger.info("\nResults Summary:")
            logger.info("\n" + df.to_string())

        except Exception as e:
            logger.error(f"Error generating reports: {e}")

        # Generate visualizations
        if not args.no_plots:
            logger.info("\nGenerating visualizations...")
            plots_dir = output_dir / 'plots' / dataset_name
            plots_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Model comparison plots
                for metric in ['rmse', 'mae', 'r2']:
                    if all(metric in r for r in results.values()):
                        plot_model_comparison(
                            results,
                            metric=metric,
                            save_path=plots_dir / f'comparison_{metric}.png',
                            title=f'{dataset_name}: Model Comparison - {metric.upper()}'
                        )

                # Computational efficiency
                if all('training_time' in r for r in results.values()):
                    plot_computational_efficiency(
                        results,
                        save_path=plots_dir / 'computational_efficiency.png'
                    )

                # Comprehensive summary plot
                create_results_summary_plot(
                    results,
                    save_path=plots_dir / 'summary.png'
                )

                # Prediction plots for best models
                if 'rmse' in list(results.values())[0]:
                    # Find best model
                    best_model = min(results.keys(), key=lambda k: results[k]['rmse'])
                    logger.info(f"\nBest model: {best_model} (RMSE: {results[best_model]['rmse']:.4f})")

                    # Generate prediction plot (would need test data predictions)
                    # This is a placeholder - full implementation would save predictions during training

                logger.info(f"Visualizations saved to {plots_dir}")

            except Exception as e:
                logger.error(f"Error generating visualizations: {e}")
                import traceback
                traceback.print_exc()

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK COMPLETED")
    logger.info("="*80)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Results saved to: {output_dir}")

    logger.info("\n📊 Summary:")
    for dataset_name, exp_data in all_experiment_results.items():
        results = exp_data['results']
        logger.info(f"\n{dataset_name}:")
        logger.info(f"  Models trained: {len(results)}")

        if results:
            best_model = min(results.keys(), key=lambda k: results[k].get('rmse', float('inf')))
            best_rmse = results[best_model].get('rmse', 'N/A')
            logger.info(f"  Best model: {best_model}")
            logger.info(f"  Best RMSE: {best_rmse}")

    logger.info("\n✅ Benchmark framework execution complete!")
    logger.info(f"\nTo view results:")
    logger.info(f"  - Reports: {output_dir / 'tables'}")
    logger.info(f"  - Plots: {output_dir / 'plots'}")
    logger.info(f"  - Logs: {output_dir / 'logs'}")

    print(f"\n\n{'='*80}")
    print("BENCHMARK COMPLETE!")
    print(f"{'='*80}")
    print(f"Results directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
