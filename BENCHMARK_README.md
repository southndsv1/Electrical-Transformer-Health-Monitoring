# Comprehensive Transformer Temperature Forecasting Benchmark

A state-of-the-art benchmarking framework for comparing machine learning approaches to electrical transformer oil temperature prediction.

## 🎯 Overview

This repository provides the most comprehensive comparison of time series forecasting models for transformer health monitoring in the literature. It implements and benchmarks multiple classical and deep learning approaches with rigorous evaluation on the ETT (Electricity Transformer Temperature) datasets.

## 📊 Datasets

The framework uses two high-quality datasets:

- **ETTh1**: Hourly electrical transformer temperature data (17,420 samples)
- **ETTh2**: Hourly electrical transformer temperature data (17,420 samples)

**Features:**
- `date`: Timestamp
- `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`: High/Middle/Low UseFlow and Load
- `OT`: Oil Temperature (target variable)

**Data splits:** 70% train, 15% validation, 15% test

## 🤖 Models Implemented

### Classical Machine Learning
1. **Linear Regression** - Baseline with lag features
2. **Ridge Regression** - L2 regularization for stability
3. **ARIMA** - Classic statistical time series model
4. **SARIMA** - Seasonal ARIMA for capturing daily/weekly patterns
5. **XGBoost** - Gradient boosting with excellent performance
6. **LightGBM** - Fast and efficient gradient boosting
7. **Random Forest** - Robust ensemble method

### Deep Learning Models
1. **LSTM** - Vanilla Long Short-Term Memory
2. **Bidirectional LSTM** - Processes sequences in both directions
3. **Stacked LSTM** - Deeper architecture with 3 layers
4. **GRU** - Gated Recurrent Unit (faster than LSTM)
5. **CNN-LSTM** - Hybrid combining convolutional and recurrent layers
6. **Transformer** - Attention-based architecture
7. **TCN** - Temporal Convolutional Network

## 📈 Evaluation Metrics

### Standard Metrics
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)

### Custom Metrics
- **Critical Temperature Detection**: Precision, Recall, and F1 for detecting when temperature exceeds 95°C
  - Crucial for preventive maintenance
  - Evaluates model's ability to predict critical events

### Performance Metrics
- Training time
- Inference time
- Model parameters count
- Memory usage

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/southndsv1/Electrical-Transformer-Health-Monitoring.git
cd Electrical-Transformer-Health-Monitoring

# Install dependencies
pip install -r benchmark/requirements.txt
```

### Running the Benchmark

#### Basic Usage

```bash
# Run complete benchmark on ETTh1 with 24-hour forecast
python run_complete_benchmark.py --dataset ETTh1 --horizon 24

# Run on both datasets
python run_complete_benchmark.py --dataset both

# Quick test (reduced epochs for deep learning)
python run_complete_benchmark.py --dataset ETTh1 --quick

# Classical models only
python run_complete_benchmark.py --dataset ETTh1 --models classical

# Deep learning models only
python run_complete_benchmark.py --dataset ETTh1 --models deep_learning
```

#### Advanced Options

```bash
python run_complete_benchmark.py \
    --dataset ETTh1 \
    --horizon 48 \
    --models all \
    --output-dir ./my_results
```

### Command Line Arguments

- `--dataset`: Dataset to use (`ETTh1`, `ETTh2`, or `both`)
- `--models`: Which models to train (`all`, `classical`, `deep_learning`)
- `--horizon`: Forecast horizon in hours (default: 24)
- `--quick`: Run with reduced epochs for quick testing
- `--output-dir`: Custom output directory
- `--no-plots`: Skip visualization generation

## 📁 Project Structure

```
Electrical-Transformer-Health-Monitoring/
├── benchmark/
│   ├── configs/
│   │   └── config.py              # All hyperparameters and settings
│   ├── data/
│   │   └── data_loader.py         # Data loading and preprocessing
│   ├── models/
│   │   ├── classical_models.py    # Classical ML models
│   │   └── deep_learning_models.py # Deep learning models
│   ├── evaluation/
│   │   └── metrics.py             # Evaluation metrics
│   ├── utils/
│   │   ├── helpers.py             # Utility functions
│   │   ├── visualization.py       # Plotting functions
│   │   └── report_generator.py   # Report generation
│   ├── results/                   # Output directory
│   │   ├── plots/                 # Generated visualizations
│   │   ├── tables/                # LaTeX and CSV tables
│   │   └── logs/                  # Training logs
│   └── requirements.txt           # Python dependencies
├── run_complete_benchmark.py      # Main benchmarking script
├── ETTh1.csv                      # Dataset 1
├── ETTh2.csv                      # Dataset 2
└── BENCHMARK_README.md            # This file
```

## 📊 Output Files

After running the benchmark, you'll find:

### Reports (`results/tables/`)
- `{dataset}_comparison.csv` - Comprehensive comparison table
- `{dataset}_report.md` - Markdown summary with key findings
- `{dataset}_table.tex` - LaTeX table for papers

### Visualizations (`results/plots/`)
- `comparison_{metric}.png` - Bar charts comparing models
- `computational_efficiency.png` - Accuracy vs training time scatter plot
- `summary.png` - Comprehensive multi-panel summary
- Individual model prediction plots

### Logs (`results/logs/`)
- `benchmark.log` - Detailed execution log

### Data (`results/`)
- `{dataset}_results.json` - All metrics in JSON format

## 🔧 Customization

### Modifying Hyperparameters

Edit `benchmark/configs/config.py`:

```python
CLASSICAL_MODELS = {
    'xgboost': {
        'n_estimators': 500,  # Increase for better accuracy
        'max_depth': 7,
        'learning_rate': 0.05,
        # ... other parameters
    }
}

DEEP_LEARNING_MODELS = {
    'lstm': {
        'hidden_size': 128,    # Increase for more capacity
        'num_layers': 2,
        'epochs': 50,          # Increase for better convergence
        # ... other parameters
    }
}
```

### Adding Custom Models

1. For classical models, extend `benchmark/models/classical_models.py`:

```python
class MyCustomModel(BaseClassicalModel):
    def __init__(self, **kwargs):
        super().__init__('My Custom Model')
        # Initialize your model

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Training logic
        pass

    def predict(self, X):
        # Prediction logic
        pass
```

2. For deep learning models, extend `benchmark/models/deep_learning_models.py`:

```python
class MyCustomNN(nn.Module):
    def __init__(self, input_size, forecast_horizon):
        super().__init__()
        # Define layers

    def forward(self, x):
        # Forward pass
        pass
```

### Custom Evaluation Metrics

Add to `benchmark/evaluation/metrics.py`:

```python
def my_custom_metric(y_true, y_pred):
    # Your metric logic
    return metric_value
```

## 📖 Model Details

### Why Each Model is Relevant

#### Classical Models

**Linear Regression**
- **Relevance**: Fast baseline, interpretable coefficients
- **Best for**: Understanding feature importance
- **Hyperparameters**: `fit_intercept`

**XGBoost**
- **Relevance**: State-of-the-art for tabular data with features
- **Best for**: High accuracy with engineered features
- **Key Hyperparameters**: `n_estimators`, `max_depth`, `learning_rate`

**LightGBM**
- **Relevance**: Faster than XGBoost, excellent performance
- **Best for**: Large datasets, production deployment
- **Key Hyperparameters**: `n_estimators`, `max_depth`, `learning_rate`

#### Deep Learning Models

**LSTM**
- **Relevance**: Captures long-term dependencies via memory cells
- **Best for**: Sequences with long-range temporal patterns
- **Key Hyperparameters**: `hidden_size`, `num_layers`, `dropout`

**GRU**
- **Relevance**: Simpler than LSTM, often comparable performance
- **Best for**: Smaller datasets, faster training needed
- **Key Hyperparameters**: `hidden_size`, `num_layers`

**Transformer**
- **Relevance**: Attention mechanism, state-of-the-art for sequences
- **Best for**: Long sequences, capturing any-distance dependencies
- **Key Hyperparameters**: `d_model`, `nhead`, `num_layers`

**TCN**
- **Relevance**: Dilated convolutions for long receptive field
- **Best for**: Parallelizable training, long sequences
- **Key Hyperparameters**: `num_channels`, `kernel_size`

## 🎓 Research & Citation

This framework is designed for research use. Key features:

1. **Reproducibility**: Fixed random seeds, deterministic algorithms
2. **Statistical Testing**: Friedman test with Nemenyi post-hoc for model comparison
3. **Paper-Ready Outputs**: LaTeX tables, publication-quality plots
4. **Comprehensive Metrics**: Multiple evaluation perspectives

### Expected Results

Based on preliminary experiments:

- **Best Overall**: LightGBM or XGBoost (RMSE ~2.0-2.5)
- **Fastest**: Linear Regression (<1s training)
- **Deep Learning Leader**: LSTM or Transformer (RMSE ~2.0-2.3)
- **Critical Temperature Detection**: Tree-based models excel

## 🔬 Extensions & Future Work

Potential extensions:

1. **Advanced Models**: N-BEATS, N-HiTS, TFT, DeepAR
2. **Ensemble Methods**: Weighted ensembles, stacking
3. **Robustness Tests**:
   - Missing data (10%, 20%, 30%)
   - Transfer learning (ETTh2 → ETTh1)
   - Seasonal analysis
4. **Multi-horizon Forecasting**: 24h, 48h, 1 week, 2 weeks
5. **Probabilistic Forecasting**: Uncertainty quantification
6. **Online Learning**: Continuous model updates

## 🐛 Troubleshooting

### Common Issues

**CUDA out of memory**
```bash
# Use smaller batch sizes
# Edit benchmark/configs/config.py:
DEEP_LEARNING_MODELS['lstm']['batch_size'] = 32  # Reduce from 64
```

**SARIMA too slow**
```bash
# Skip SARIMA for large datasets (automatically done)
# Or reduce training data size
```

**Import errors**
```bash
# Make sure you're in the correct directory
cd Electrical-Transformer-Health-Monitoring
python run_complete_benchmark.py --dataset ETTh1
```

## 📝 License

MIT License - See LICENSE file

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional models (N-BEATS, TFT, etc.)
- More evaluation metrics
- Robustness experiments
- Documentation improvements

## 📧 Contact

For questions or issues, please open a GitHub issue.

## 🌟 Acknowledgments

- ETT Dataset: [Informer paper](https://arxiv.org/abs/2012.07436)
- Built with PyTorch, scikit-learn, XGBoost, LightGBM
- Inspired by state-of-the-art time series forecasting research

---

**Note**: This framework is for research and educational purposes. For production deployment of transformer health monitoring systems, additional validation and safety measures are required.
