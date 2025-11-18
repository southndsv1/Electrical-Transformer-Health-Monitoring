#!/usr/bin/env python3
"""
Test script for classical models only (no PyTorch required).
This validates the core framework functionality.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'benchmark'))

import numpy as np
from configs.config import RANDOM_SEED
from data.data_loader import get_data_loader
from models.classical_models import (
    LinearRegressionModel,
    RidgeRegressionModel,
    LightGBMModel,
    XGBoostModel,
    RandomForestModel
)

def set_seed_simple(seed):
    """Simple seed setting without torch."""
    import random
    random.seed(seed)
    np.random.seed(seed)

def main():
    print("="*80)
    print("CLASSICAL MODELS TEST (No PyTorch)")
    print("="*80)

    # Set seed
    set_seed_simple(RANDOM_SEED)

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

    print("\n🚀 Training classical models...")
    print("\n" + "-"*80)

    # Linear Regression
    print("\n1. Linear Regression")
    try:
        lr_model = LinearRegressionModel()
        lr_model.fit(X_train, y_train)
        results['Linear Regression'] = lr_model.evaluate(X_test, y_test)
        print(f"   ✓ RMSE: {results['Linear Regression']['rmse']:.4f}")
        print(f"   ✓ MAE: {results['Linear Regression']['mae']:.4f}")
        print(f"   ✓ R²: {results['Linear Regression']['r2']:.4f}")
        print(f"   ✓ Training time: {results['Linear Regression']['training_time']:.2f}s")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Ridge Regression
    print("\n2. Ridge Regression")
    try:
        ridge_model = RidgeRegressionModel(alpha=1.0)
        ridge_model.fit(X_train, y_train)
        results['Ridge'] = ridge_model.evaluate(X_test, y_test)
        print(f"   ✓ RMSE: {results['Ridge']['rmse']:.4f}")
        print(f"   ✓ MAE: {results['Ridge']['mae']:.4f}")
        print(f"   ✓ R²: {results['Ridge']['r2']:.4f}")
        print(f"   ✓ Training time: {results['Ridge']['training_time']:.2f}s")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # LightGBM
    print("\n3. LightGBM")
    try:
        lgb_model = LightGBMModel()
        lgb_model.fit(X_train, y_train, X_val, y_val)
        results['LightGBM'] = lgb_model.evaluate(X_test, y_test)
        print(f"   ✓ RMSE: {results['LightGBM']['rmse']:.4f}")
        print(f"   ✓ MAE: {results['LightGBM']['mae']:.4f}")
        print(f"   ✓ R²: {results['LightGBM']['r2']:.4f}")
        print(f"   ✓ Training time: {results['LightGBM']['training_time']:.2f}s")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # XGBoost
    print("\n4. XGBoost")
    try:
        xgb_model = XGBoostModel()
        xgb_model.fit(X_train, y_train, X_val, y_val)
        results['XGBoost'] = xgb_model.evaluate(X_test, y_test)
        print(f"   ✓ RMSE: {results['XGBoost']['rmse']:.4f}")
        print(f"   ✓ MAE: {results['XGBoost']['mae']:.4f}")
        print(f"   ✓ R²: {results['XGBoost']['r2']:.4f}")
        print(f"   ✓ Training time: {results['XGBoost']['training_time']:.2f}s")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Random Forest
    print("\n5. Random Forest")
    try:
        rf_model = RandomForestModel()
        rf_model.fit(X_train, y_train)
        results['Random Forest'] = rf_model.evaluate(X_test, y_test)
        print(f"   ✓ RMSE: {results['Random Forest']['rmse']:.4f}")
        print(f"   ✓ MAE: {results['Random Forest']['mae']:.4f}")
        print(f"   ✓ R²: {results['Random Forest']['r2']:.4f}")
        print(f"   ✓ Training time: {results['Random Forest']['training_time']:.2f}s")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Results summary
    if results:
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
        print(f"   MAE: {df.loc[best_model, 'mae']:.4f}")
        print(f"   R²: {df.loc[best_model, 'r2']:.4f}")
        print(f"   Training Time: {df.loc[best_model, 'training_time']:.2f}s")

        print("\n" + "="*80)
        print("✅ CLASSICAL MODELS TEST PASSED!")
        print("="*80)
        print(f"\n{len(results)} models successfully trained and evaluated")
        print("\nNext steps:")
        print("  1. Install PyTorch for deep learning models")
        print("  2. Run: python run_complete_benchmark.py --dataset ETTh1 --models all")
        print("="*80 + "\n")
    else:
        print("\n❌ No models were successfully trained")

if __name__ == '__main__':
    main()
