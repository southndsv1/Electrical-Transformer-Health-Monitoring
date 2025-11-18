#!/usr/bin/env python3
"""
Quick demo script to test the benchmark framework with a subset of models.

This script runs a minimal benchmark with a few models for quick validation.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'benchmark'))

import numpy as np
from configs.config import RANDOM_SEED
from data.data_loader import get_data_loader
from models.classical_models import LinearRegressionModel, LightGBMModel
from utils.helpers import set_seed, setup_logger
from utils.visualization import plot_model_comparison
from utils.report_generator import create_markdown_report

def main():
    print("="*80)
    print("QUICK DEMO: Transformer Temperature Forecasting")
    print("="*80)

    # Set seed
    set_seed(RANDOM_SEED)

    # Load data
    print("\n📊 Loading ETTh1 dataset...")
    loader = get_data_loader('ETTh1')
    loader.load_data()
    loader.create_splits()

    # Get data for classical models
    print("\n🔧 Preparing data...")
    data = loader.get_classical_ml_data(forecast_horizon=24, n_lags=168)

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']

    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")

    # Train models
    results = {}

    print("\n🚀 Training models...")
    print("\n" + "-"*80)

    # Linear Regression
    print("\n1. Linear Regression")
    lr_model = LinearRegressionModel()
    lr_model.fit(X_train, y_train)
    results['Linear Regression'] = lr_model.evaluate(X_test, y_test)

    print(f"   ✓ RMSE: {results['Linear Regression']['rmse']:.4f}")
    print(f"   ✓ MAE: {results['Linear Regression']['mae']:.4f}")
    print(f"   ✓ R²: {results['Linear Regression']['r2']:.4f}")
    print(f"   ✓ Training time: {results['Linear Regression']['training_time']:.2f}s")

    # LightGBM
    print("\n2. LightGBM")
    lgb_model = LightGBMModel()
    lgb_model.fit(X_train, y_train, X_val, y_val)
    results['LightGBM'] = lgb_model.evaluate(X_test, y_test)

    print(f"   ✓ RMSE: {results['LightGBM']['rmse']:.4f}")
    print(f"   ✓ MAE: {results['LightGBM']['mae']:.4f}")
    print(f"   ✓ R²: {results['LightGBM']['r2']:.4f}")
    print(f"   ✓ Training time: {results['LightGBM']['training_time']:.2f}s")

    # Results summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    # Create comparison
    import pandas as pd
    df = pd.DataFrame(results).T
    df = df[['rmse', 'mae', 'r2', 'training_time', 'inference_time']]
    df = df.round(4)

    print("\n" + df.to_string())

    # Best model
    best_model = df['rmse'].idxmin()
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   RMSE: {df.loc[best_model, 'rmse']:.4f}")
    print(f"   Training Time: {df.loc[best_model, 'training_time']:.2f}s")

    # Generate markdown report
    print("\n📝 Generating report...")
    report = create_markdown_report(results, 'Quick Demo')

    output_file = Path('quick_demo_report.md')
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"   ✓ Report saved to {output_file}")

    # Try to generate a simple plot
    try:
        print("\n📊 Generating visualization...")
        plot_model_comparison(
            results,
            metric='rmse',
            save_path='quick_demo_comparison.png',
            title='Model Comparison - RMSE'
        )
        print("   ✓ Plot saved to quick_demo_comparison.png")
    except Exception as e:
        print(f"   ⚠ Could not generate plot: {e}")

    print("\n" + "="*80)
    print("✅ DEMO COMPLETED!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Run full benchmark: python run_complete_benchmark.py --dataset ETTh1")
    print("  2. See BENCHMARK_README.md for detailed documentation")
    print("  3. Customize config in benchmark/configs/config.py")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
