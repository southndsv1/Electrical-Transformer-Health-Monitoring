# Electrical-Transformer-Health-Monitoring
Oil temperature prediction based on time series forecasting models

# Transformer Oil Temperature Time Series Forecasting

## Overview
This repository contains a time series forecasting model for predicting transformer oil temperature using deep learning frameworks such as [pytorch-forecasting](https://github.com/Lightning-AI/pytorch-forecasting) and [pytorch-lightning](https://github.com/Lightning-AI/lightning). The primary goal is to accurately forecast the oil temperature in power transformer equipment, an essential factor in maintaining safe operations and preventing costly downtime.

## Motivation
Power transformers are crucial components in electrical grids, and unexpected failures can lead to significant economic losses and power outages. By modeling the oil temperature—which is closely tied to transformer health conditions—operators can:

- **Identify Anomalies**: Quickly detect abnormal temperature rises.
- **Schedule Maintenance**: Plan proactive maintenance based on forecasts.
- **Avoid Failures**: Prevent transformer overloads and prolong equipment lifespan.

## Key Features
1. **Data Loading & Preprocessing**:
   - Loads the [ETTh2 dataset](https://github.com/zhouhaoyi/ETDataset) (`ETTh2.csv`) which includes date and oil temperature (`OT`) columns.
   - Converts date columns into datetime format, sorts chronologically, and handles negative values (replacing them with 0.01).
   - Splits the data into training, validation, and test sets.

2. **Feature Engineering**:
   - Creates additional time-related features like hour, day, month, etc., if needed.
   - Applies normalization (e.g., `StandardScaler`) to scale the temperature values.
   - Demonstrates how to engineer new features for improved predictive performance.

3. **Model Implementation**:
   - Leverages [pytorch-forecasting](https://github.com/Lightning-AI/pytorch-forecasting) for building temporal fusion transformer or other advanced deep learning time series models.
   - Integrates [pytorch-lightning](https://github.com/Lightning-AI/lightning) to structure training loops (e.g., callbacks, checkpointing, logging).
   - Includes hyperparameter configuration for model architecture, learning rates, batch sizes, etc.

4. **Training & Validation**:
   - Demonstrates how to train the model using GPU acceleration (if available).
   - Logs metrics such as Loss, Mean Squared Error (MSE), Mean Absolute Error (MAE), and R².
   - Performs hyperparameter tuning by adjusting number of layers, hidden sizes, and dropout.

5. **Evaluation**:
   - Evaluates the trained model on a hold-out test set.
   - Computes the following evaluation metrics:
     - **Mean Squared Error (MSE)**
     - **Mean Absolute Error (MAE)**
     - **R² Score**
   - Generates visual plots for predicted vs. actual temperature, enabling intuitive assessment.

6. **Visualization & Analysis**:
   - Uses Matplotlib to plot the training and validation metrics over epochs.
   - Illustrates time-series line graphs comparing the true temperatures and model forecasts.
   - Analyzes residual patterns to identify possible systematic errors.
   - Also loads an unseen data file ETTh1.csv and predicts the OT for a randomly selected or user provided date.


## Project Structure
```
.
├── Transformer_Oil_Temperature_Time_Series.ipynb  # Main Jupyter notebook
├── ETTh1.csv                                      # Unseen dataset
├── ETTh2.csv                                      # Sample dataset 
├── requirements.txt                               # Python dependencies
└── README.md                                      # This README
```

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/agme2019/Electrical-Transformer-Health-Monitoring.git
   cd Electrical-Transformer-Health-Monitoring
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   If you don't have a requirements file, manually install each package:
   ```bash
   pip install pytorch-forecasting pytorch-lightning matplotlib pandas numpy scikit-learn
   ```

3. **Obtain the Dataset**:
   - Download the `ETTh2.csv` file from the original [ETDataset repository](https://github.com/zhouhaoyi/ETDataset) or any custom dataset with similar structure.
   - Place the data file in the project's root folder.

4. **Run the Notebook**:
   ```bash
   jupyter notebook Transformer_Oil_Temperature_Time_Series.ipynb
   ```
   - Follow the sequential steps in the notebook to load data, train the model, and evaluate performance.

## Usage

### Training the Model
- **Hyperparameters**: Modify hidden sizes, learning rates, dropout rates, and other hyperparameters within the notebook's configuration cells.
- **GPU Support**: If available, configure the notebook to run on GPU for faster training.

### Evaluating the Model
- **Validation Metrics**: Inspect training/validation losses to diagnose underfitting or overfitting.
- **Test Metrics**: Use MSE, MAE, and R² scores to measure final performance on unseen data.

### Visualizing Results
- **Actual vs. Predicted**: Compare side-by-side plots to assess alignment.
- **Residual Analysis**: Investigate systematic biases or patterns in errors over time.

## Results
The final section of the notebook produces:
- **Performance Metrics**: Summaries of MSE, MAE, and R².
- **Plots**: Visual comparisons of predicted vs. actual temperature.
- **Observations**: Suggestions for possible improvements and additional data or features.

## Potential Improvements
1. **Feature Engineering**: Consider integrating exogenous variables (e.g., load, ambient temperature) to enhance model accuracy.
2. **Hyperparameter Tuning**: Implement tools like Optuna or Ray Tune for automated tuning.
3. **Advanced Architectures**: Experiment with architectures like N-BEATS, Informer, or state-of-the-art Transformers.
4. **Ensemble Methods**: Combine multiple models (e.g., linear, tree-based, neural) for robust predictions.

## Contributing
Contributions in the form of pull requests or issues are welcomed. Please include details on bug fixes, new features, or improvements.

## License
Distributed under the [MIT License](LICENSE). Feel free to use the code and dataset references for any purpose.

## References
- [PyTorch Forecasting](https://github.com/Lightning-AI/pytorch-forecasting)
- [PyTorch Lightning](https://github.com/Lightning-AI/lightning)
- [ETDataset Repository](https://github.com/zhouhaoyi/ETDataset)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## Contact
For questions or feedback, please open an issue on GitHub or reach out to the maintainers directly.

---
**Note**: This project is intended for educational and research purposes. Always verify results and consult professional engineering guidelines when deploying models to real-world industrial systems.
