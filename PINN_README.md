# Physics-Informed Neural Networks (PINNs) for Transformer Temperature Prediction

## 🎯 Goal: Achieve RMSE < 0.15 °C

This implementation combines **data-driven learning** with **physics-based constraints** from IEEE C57.91 transformer thermal standards to achieve superior prediction accuracy.

---

## 🔬 Why PINNs for Transformer Monitoring?

Traditional machine learning methods achieved RMSE ~0.47 °C on this task. PINNs improve this by:

1. **Incorporating Domain Knowledge**: Transformer thermal physics from IEEE C57.91
2. **Better Extrapolation**: Physics constraints prevent unrealistic predictions
3. **Data Efficiency**: Learn from less data by respecting physical laws
4. **Interpretability**: Learned parameters have physical meaning
5. **Robust Predictions**: Physics ensures realistic behavior even in unusual conditions

---

## 📐 Physics Model: IEEE C57.91 Thermal Dynamics

### Top Oil Temperature Dynamics

The differential equation governing oil temperature:

```
dθ_oil/dt = (1/τ_oil) * (θ_oil_ss - θ_oil)
```

Where steady-state oil temperature is:

```
θ_oil_ss = θ_amb + Δθ_oil_rated * ((K²*R + 1)/(R + 1))^n
```

### Hot-Spot Temperature

```
θ_hot = θ_oil + Δθ_hot_rated * K^(2m)
```

### Physical Parameters

| Parameter | Symbol | Typical Range | Meaning |
|-----------|--------|---------------|---------|
| Load Loss Ratio | R | 5-6 | Ratio of load loss to no-load loss |
| Oil Exponent | n | 0.8-0.9 | For ONAN cooling |
| Winding Exponent | m | 0.8-0.9 | Hot-spot gradient exponent |
| Oil Time Constant | τ_oil | 150-210 min | Thermal inertia |
| Rated Oil Rise | Δθ_oil_rated | 50-60 K | At rated load |
| Rated Hot-Spot Rise | Δθ_hot_rated | 60-70 K | At rated load |

---

## 🏗️ Architecture

### PINN Model Structure

```
Input: [time, HUFL, HULL, MUFL, MULL, LUFL, LULL]
   ↓
Neural Network (4 layers × 100 neurons, tanh activation)
   ↓
Predicted Temperature: θ_oil
   ↓
Automatic Differentiation → dθ_oil/dt
   ↓
Physics Residual = dθ_oil/dt - (1/τ_oil)*(θ_oil_ss - θ_oil)
```

### Loss Function

```
Loss_total = Loss_data + λ * Loss_physics

Loss_data = MSE(θ_predicted, θ_measured)
Loss_physics = MSE(physics_residual, 0)
```

Where λ is adaptively adjusted during training.

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies (if not already installed)
pip install torch numpy pandas matplotlib seaborn scikit-learn scipy

# No additional packages needed - uses standard PyTorch
```

### Running the Experiment

```bash
# Basic run (200 epochs, curriculum learning)
python run_pinn_experiment.py --dataset ETTh2

# Custom configuration
python run_pinn_experiment.py \
    --dataset ETTh2 \
    --epochs 300 \
    --batch-size 64 \
    --lr 0.001 \
    --hidden-sizes 100 100 100 100 \
    --physics-schedule curriculum \
    --n-collocation 1000

# Quick test (fewer epochs)
python run_pinn_experiment.py --dataset ETTh2 --epochs 50
```

### Command Line Arguments

- `--dataset`: Dataset to use (`ETTh1` or `ETTh2`)
- `--epochs`: Number of training epochs (default: 200)
- `--batch-size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--hidden-sizes`: Hidden layer sizes (default: `100 100 100 100`)
- `--physics-schedule`: Physics loss weight schedule
  - `curriculum`: Gradually increase from 0.001 to 0.1
  - `constant`: Fixed weight throughout
  - `adaptive`: Adjust based on loss balance
  - `pretrain`: Data-only for 20 epochs, then add physics
- `--n-collocation`: Number of collocation points for physics loss (default: 1000)
- `--output-dir`: Output directory (default: `pinn_results`)
- `--device`: Device (`auto`, `cpu`, or `cuda`)
- `--seed`: Random seed (default: 42)

---

## 📊 Expected Results

### Performance Comparison

| Model | RMSE (°C) | MAE (°C) | R² | Training Time |
|-------|-----------|----------|-----|---------------|
| Linear Regression | 0.474 | 0.353 | 0.972 | 0.03s |
| LightGBM | 0.519 | 0.374 | 0.966 | 1.16s |
| Standard NN | ~0.20 | ~0.15 | ~0.995 | ~60s |
| **PINN (Target)** | **< 0.15** | **< 0.12** | **> 0.997** | **~80s** |

### Why PINN Performs Better

1. **Physics Regularization**: Prevents overfitting by constraining predictions to physically feasible values
2. **Better Derivatives**: Smooth tanh activations + physics loss → accurate temperature gradients
3. **Learned Parameters**: Discovers optimal physical parameters for specific transformer
4. **Extrapolation**: Physics constraints help beyond training data range

---

## 📁 Project Structure

```
pinn/
├── physics/
│   └── transformer_thermal.py     # IEEE C57.91 physics model
├── models/
│   └── pinn_transformer.py        # PINN architecture
├── utils/
│   ├── data_loader.py             # ETT dataset loader
│   └── trainer.py                 # Training loop
└── experiments/                   # (for future extensions)

run_pinn_experiment.py             # Main experiment script
```

---

## 🎨 Output Files

After running, you'll find in `pinn_results/{dataset}_{timestamp}/`:

### Models
- `pinn_model.pth` - Trained PINN model
- `standard_nn_model.pth` - Baseline neural network

### Results
- `metrics.json` - Comprehensive metrics comparison
- `predictions.npz` - All predictions and targets

### Visualizations
- `training_curves.png` - Loss curves, physics weight, learning rate
- `predictions_comparison.png` - Time series predictions
- `scatter_comparison.png` - Predicted vs actual scatter plots
- `error_analysis.png` - Error distributions and Q-Q plot

---

## 🔬 Key Features

### 1. Learnable Physics Parameters

Instead of fixing physics parameters, PINN learns them from data:

```python
# Parameters are optimized during training
R_learned = 5.5 ± 0.5       # Load loss ratio
n_learned = 0.85 ± 0.05     # Oil exponent
m_learned = 0.82 ± 0.05     # Winding exponent
τ_oil_learned = 180 ± 20    # Time constant (minutes)
```

This allows the model to adapt to specific transformer characteristics while maintaining physical structure.

### 2. Curriculum Learning

Physics loss weight increases during training:

```
Epoch 1-20:   λ = 0.001  (focus on fitting data)
Epoch 21-100: λ increases gradually
Epoch 100+:   λ = 0.1    (strong physics enforcement)
```

### 3. Collocation Points

Physics loss is enforced not just at data points, but also at randomly sampled "collocation points" throughout the domain. This ensures physics compliance everywhere.

### 4. Automatic Differentiation

PyTorch's autograd computes exact derivatives:

```python
dθ/dt = autograd.grad(θ, t)  # Exact derivative via backpropagation
```

---

## 📈 Training Process

### Phase 1: Data Pre-training (Epochs 1-20, optional)
- Focus: Fit the temperature measurements
- Physics weight: Low (λ = 0.001)
- Goal: Get model in right ballpark

### Phase 2: Physics Integration (Epochs 21-100)
- Focus: Gradually enforce physics constraints
- Physics weight: Increasing (λ: 0.001 → 0.1)
- Goal: Learn physically consistent patterns

### Phase 3: Fine-tuning (Epochs 100+)
- Focus: Balance data fit and physics compliance
- Physics weight: High (λ = 0.1)
- Goal: Optimal performance

---

## 🎯 Validation

### Metrics Tracked

1. **Data Loss**: MSE on actual measurements
2. **Physics Loss**: MSE of physics residual
3. **RMSE**: Root mean squared error (target: < 0.15 °C)
4. **MAE**: Mean absolute error
5. **MAPE**: Mean absolute percentage error
6. **R²**: Coefficient of determination

### Physics Validation

The learned parameters should match expected ranges:
- If R ∈ [4, 7] ✓ (reasonable load loss ratio)
- If n ∈ [0.7, 1.0] ✓ (valid oil exponent)
- If m ∈ [0.7, 1.0] ✓ (valid winding exponent)
- If τ_oil ∈ [120, 240] ✓ (realistic time constant)

---

## 🔍 Interpreting Results

### Good PINN Performance Indicators

1. ✅ RMSE < 0.15 °C
2. ✅ Physics loss decreases during training
3. ✅ Learned parameters within physical ranges
4. ✅ Predictions smooth and physically plausible
5. ✅ Better extrapolation than standard NN

### Troubleshooting

**Problem**: Physics loss not decreasing
- **Solution**: Increase physics weight gradually, check collocation points

**Problem**: Data loss increasing
- **Solution**: Reduce physics weight, add more training epochs

**Problem**: Unstable training
- **Solution**: Reduce learning rate, use gradient clipping (already included)

**Problem**: Learned parameters unrealistic
- **Solution**: Add parameter bounds, adjust initial values

---

## 🚀 Advanced Usage

### Custom Physics Constraints

Modify `pinn/physics/transformer_thermal.py` to add:

```python
def cooling_system_dynamics(self, load, ambient):
    """Additional constraints for cooling system."""
    if load > 1.0:  # Overload condition
        # Forced cooling kicks in
        cooling_factor = 1.5
    else:
        # Natural cooling
        cooling_factor = 1.0
    return cooling_factor
```

### Multi-Task Learning

Extend to predict both oil and hot-spot temperature:

```python
# In pinn_transformer.py
self.nn_predictor = TemperaturePredictorNN(output_size=2)  # [oil, hotspot]
```

### Transfer Learning

Train on one transformer, fine-tune on another:

```python
# Load pre-trained model
pinn_model.load_state_dict(torch.load('pretrained_pinn.pth'))

# Fine-tune on new data
trainer.train(new_train_loader, epochs=50, lr=1e-4)
```

---

## 📚 References

1. **IEEE C57.91-2011**: Guide for Loading Mineral-Oil-Immersed Transformers and Step-Voltage Regulators
2. **Raissi et al. (2019)**: "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems"
3. **ETT Dataset**: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"

---

## 🤝 Contributing

Areas for improvement:
- Additional physics constraints (aging, moisture, etc.)
- Uncertainty quantification
- Multi-transformer joint modeling
- Real-time adaptation
- Bayesian parameter estimation

---

## 📝 Citation

If you use this code for research, please cite:

```bibtex
@software{pinn_transformer_2025,
  title={Physics-Informed Neural Networks for Transformer Temperature Prediction},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/transformer-pinn}
}
```

---

## ✅ Success Criteria

This PINN implementation is successful if:

1. ✅ **RMSE < 0.15 °C** on test set
2. ✅ **Outperforms standard NN** by significant margin
3. ✅ **Learns realistic physics parameters**
4. ✅ **Shows better extrapolation** than data-only methods
5. ✅ **Maintains physical plausibility** in predictions

---

**Target Achievement**: The goal is to demonstrate that incorporating physics through PINNs can reduce RMSE from ~0.47 (classical ML) to **< 0.15 °C**, making this suitable for critical transformer health monitoring applications where accuracy is paramount.

🎯 **Run the experiment now**: `python run_pinn_experiment.py --dataset ETTh2`
