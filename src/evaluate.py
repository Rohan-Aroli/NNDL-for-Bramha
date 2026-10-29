import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils import save_json

def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error safely."""
    # Prevent division by zero
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_model(model, X_test, y_test, scaler, test_dates, baseline_predictions_scaled, config):
    """
    Evaluates the model on test data, calculates metrics, compares with baseline, and saves results.
    """
    # 1. Generate predictions
    predictions_scaled = model.predict(X_test)

    # 2. Inverse transform to original scale
    predictions = scaler.inverse_transform(predictions_scaled).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    baseline_predictions = scaler.inverse_transform(baseline_predictions_scaled.reshape(-1, 1)).flatten()

    # 3. Calculate LSTM Metrics
    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    mape = calculate_mape(actual, predictions)
    r2 = r2_score(actual, predictions)

    # 4. Calculate Baseline Metrics
    base_mae = mean_absolute_error(actual, baseline_predictions)
    base_mse = mean_squared_error(actual, baseline_predictions)
    base_rmse = np.sqrt(base_mse)
    base_mape = calculate_mape(actual, baseline_predictions)
    base_r2 = r2_score(actual, baseline_predictions)

    # 5. Create Report
    report = {
        "dataset_name": "Air Passengers",
        "num_test_samples": len(actual),
        "lookback_window": config.LOOKBACK,
        "lstm_metrics": {
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAPE": float(mape),
            "R2": float(r2)
        },
        "baseline_metrics": {
            "MAE": float(base_mae),
            "MSE": float(base_mse),
            "RMSE": float(base_rmse),
            "MAPE": float(base_mape),
            "R2": float(base_r2)
        }
    }

    save_json(report, config.REPORTS_DIR / "evaluation_metrics.json")

    # Print readable table to terminal
    print("\n" + "="*50)
    print(f"{'Metric':<10} | {'LSTM':<15} | {'Baseline':<15}")
    print("-" * 50)
    print(f"{'MAE':<10} | {mae:<15.4f} | {base_mae:<15.4f}")
    print(f"{'RMSE':<10} | {rmse:<15.4f} | {base_rmse:<15.4f}")
    print(f"{'MAPE %':<10} | {mape:<15.4f} | {base_mape:<15.4f}")
    print(f"{'R-squared':<10} | {r2:<15.4f} | {base_r2:<15.4f}")
    print("="*50 + "\n")

    # 6. Save Predictions to CSV
    errors = actual - predictions
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2

    pred_df = pd.DataFrame({
        "Date": test_dates,
        "Actual": actual,
        "Predicted": predictions,
        "Absolute_Error": abs_errors,
        "Squared_Error": sq_errors,
        "Signed_Residual": errors
    })

    pred_df.to_csv(config.PREDICTIONS_DIR / "test_predictions.csv", index=False)

    return predictions, report
