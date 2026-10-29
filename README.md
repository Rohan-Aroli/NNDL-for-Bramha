# End-to-End LSTM Time-Series Forecasting

This project is a beginner-friendly, complete educational guide demonstrating a basic Long Short-Term Memory (LSTM) neural network for time-series forecasting. It is built using Python and TensorFlow.

## Overview
The project uses the classical **Air Passengers dataset** (monthly international airline passenger counts from 1949 to 1960) to demonstrate a complete machine learning workflow:
- Time-series data loading and validation.
- Exploratory data analysis (EDA).
- Data preprocessing (chronological train/validation/test splitting, scaling, sliding-window generation).
- Building and training an LSTM neural network.
- Evaluating the model and comparing it against a simple persistence baseline.
- Generating predictions, evaluating metrics, and creating visualizations.

## Project Objectives
1. Provide a working, end-to-end ML project that runs easily.
2. Teach the core concepts of LSTM time-series forecasting using simple, understandable code.
3. Show how to correctly preprocess time-series data without introducing look-ahead bias (leaking future data).
4. Evaluate forecasting realistically using chronological splitting and common metrics (MAE, MSE, RMSE, MAPE).

## Why LSTM for Sequential Data?
A standard feed-forward neural network assumes that all inputs are independent of each other. However, in time-series data, the sequence and order of the data points matter.

Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN) designed specifically to recognize patterns in sequences of data. LSTMs can maintain internal memory over time, allowing them to "remember" important historical patterns and "forget" irrelevant noise. This makes them highly effective for time-series forecasting tasks.

### What is an LSTM?
An LSTM contains repeating modules with complex internal structures called "cells." Each cell uses mathematical structures known as "gates" (forget gate, input gate, output gate) to regulate the flow of information.
- **Forget Gate:** Decides what information to throw away from the cell state.
- **Input Gate:** Decides which new information to store in the cell state.
- **Output Gate:** Decides what to output based on the current cell state.

## Dataset
This project uses the publicly available **Air Passengers Dataset**.
- **Source:** Automatically downloaded from a reliable public source.
- **Size:** 144 monthly observations.
- **Features:** Month (date) and number of passengers.

*Disclaimer: The Air Passengers dataset is small and historically dated. This project is primarily for learning, and a model trained on such small data may not produce highly accurate real-world forecasts for modern problems.*

## Project Structure
```
lstm-time-series-forecasting/
│
├── run_project.py       # Main entry point to run the entire workflow
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore file
├── LICENSE              # MIT License
│
├── config/
│   └── config.py        # Configuration and hyperparameters
│
├── data/
│   ├── raw/             # Downloaded datasets
│   └── processed/       # Preprocessed data (if saved)
│
├── src/
│   ├── __init__.py
│   ├── dataset.py       # Dataset downloading and validation
│   ├── preprocessing.py # Data scaling and window generation
│   ├── model.py         # LSTM model architecture
│   ├── train.py         # Training loop
│   ├── evaluate.py      # Metric calculations
│   ├── visualize.py     # EDA and result plotting
│   ├── forecast.py      # Future forecasting logic
│   └── utils.py         # Helper functions
│
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py  # Dataset tests
│   ├── test_preprocessing.py # Preprocessing tests
│   └── test_model.py    # Model tests
│
└── outputs/
    ├── figures/         # Generated plots
    ├── models/          # Saved Keras models
    ├── predictions/     # CSVs of predictions
    └── reports/         # JSON evaluation summaries
```
*Note: The directories inside `data/` and `outputs/` are generated at runtime.*

## System Requirements
- OS: Windows, macOS, or Linux.
- Python: 3.9, 3.10, or 3.11.
- No GPU is required (CPU training is supported and completely sufficient for this small model).

## One-Command Execution
You can run the entire workflow with a single command. It will automatically set up a virtual environment (`.venv`), install dependencies, download data, train the model, and generate results.

**On Windows:**
```bash
python run_project.py
```

**On macOS / Linux:**
```bash
python3 run_project.py
```

### What happens when `run_project.py` executes?
1. Detects your Python environment and creates a `.venv` virtual environment if it doesn't exist.
2. Installs required packages from `requirements.txt`.
3. Re-launches the script inside the virtual environment.
4. Creates necessary directories (`data/`, `outputs/`, etc.).
5. Downloads the Air Passengers dataset (and caches it for future runs).
6. Performs Exploratory Data Analysis (EDA) and saves plots to `outputs/figures/`.
7. Preprocesses the data (chronological split, min-max scaling).
8. Generates sliding-window sequences for training.
9. Builds and compiles the LSTM model.
10. Trains the model with Early Stopping to prevent overfitting.
11. Evaluates the model on unseen test data and calculates metrics.
12. Saves predictions, evaluation reports, and plots.

## Understanding the Machine Learning Pipeline

### Data Splitting
Time-series data must be split chronologically, not randomly shuffled. Random splitting would cause "data leakage" by mixing future data into the training set.
- Training: First 70% of observations
- Validation: Next 15%
- Testing: Final 15%

### Lookback Window
The lookback window (default: 12) determines how many past time steps the model uses to predict the next time step. We feed the past 12 months of passenger counts to predict month 13.

### Model Architecture
- **Input Layer:** Shape `(12, 1)` representing 12 time steps, 1 feature.
- **LSTM Layer:** 50 units.
- **Dropout Layer:** Rate 0.2 (prevents overfitting).
- **Dense Output Layer:** 1 unit (predicts the continuous passenger count).

### Evaluation Metrics
- **MAE (Mean Absolute Error):** Average absolute difference between predictions and actual values.
- **MSE (Mean Squared Error):** Average squared difference, penalizes large errors heavily.
- **RMSE (Root Mean Squared Error):** Square root of MSE, interpretable in the same units as the target.
- **MAPE (Mean Absolute Percentage Error):** Average absolute percentage difference.
- **R² (R-squared):** Measures how well the model predictions variance compared to the actual data. *Note: R² can be negative on unseen time-series test data, which simply indicates the model is performing worse than a horizontal line representing the mean.*

### Baseline Comparison
We compare our LSTM against a naive "persistence baseline", which simply predicts that the next value will be exactly the same as the last observed value. If the LSTM cannot beat the baseline, it implies the model is not learning useful complex patterns.

## Generated Output Files
After a successful run, check the `outputs/` directory:
- `outputs/figures/`: EDA plots (historical data, distribution, seasonality) and Result plots (actual vs. predicted, error distribution, scatter).
- `outputs/models/`: Saved `.keras` models.
- `outputs/predictions/`: CSV files containing predictions vs. actual values, and optional future forecasts.
- `outputs/reports/`: JSON files with EDA summaries, training history, and evaluation metrics.

### How to Interpret Prediction Plots
- **Actual vs Predicted:** A line chart showing how closely the model's predictions trace the actual test data.
- **Prediction Errors:** A plot of the residuals over time. Ideally, errors should hover around 0 without clear patterns.
- **Prediction Scatter:** A plot of actual vs. predicted values. Points close to the diagonal line indicate accurate predictions.

## Running Unit Tests
Unit tests ensure that the core logic works correctly. Run them using `pytest` inside the virtual environment:

**Windows:**
```bash
.venv\Scripts\python.exe -m pytest tests -q
```
**macOS / Linux:**
```bash
.venv/bin/python -m pytest tests -q
```

## Common Errors and Troubleshooting
- **Python version incompatible:** Ensure you are using a Python version supported by TensorFlow (generally 3.9 to 3.11). Python 3.12+ might face wheel availability issues.
- **Network errors:** If downloading the dataset fails, the script will output instructions. Check your internet connection.
- **Missing dependencies:** If the environment creation fails, you can manually create a `venv` and run `pip install -r requirements.txt`.

## Limitations and Future Improvements
- The Air Passengers dataset is simple. The model might not generalize to highly volatile or multi-variate time-series.
- Future improvements could include hyperparameter tuning, adding more complex features (e.g., month of the year as a separate input), or comparing with statistical methods like ARIMA.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

The Air Passengers dataset is a classic public domain dataset originally published in *Box, G. E. P., Jenkins, G. M. and Reinsel, G. C. (1976) Time Series Analysis, Forecasting and Control. Third Edition. Holden-Day. Series G.*
