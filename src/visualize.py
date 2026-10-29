import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from src.utils import save_json

def perform_eda(df, figures_dir, reports_dir):
    """Generates EDA visualizations and saves a summary report."""

    # 1. Historical passenger counts
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Passengers'])
    plt.title('Historical Airline Passengers')
    plt.xlabel('Date')
    plt.ylabel('Number of Passengers')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / 'historical_passengers.png')
    plt.close()

    # 2. Distribution of passenger counts
    plt.figure(figsize=(8, 5))
    plt.hist(df['Passengers'], bins=20, edgecolor='black')
    plt.title('Distribution of Passenger Counts')
    plt.xlabel('Passengers')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(figures_dir / 'passenger_distribution.png')
    plt.close()

    # 3. Monthly seasonal patterns
    df_temp = df.copy()
    df_temp['Month'] = df_temp['Date'].dt.month
    plt.figure(figsize=(10, 5))
    df_temp.boxplot(column='Passengers', by='Month', grid=True)
    plt.title('Monthly Seasonal Patterns')
    plt.suptitle('') # Remove automatic pandas subtitle
    plt.xlabel('Month (1=Jan, 12=Dec)')
    plt.ylabel('Passengers')
    plt.tight_layout()
    plt.savefig(figures_dir / 'monthly_seasonality.png')
    plt.close()

    # EDA Summary
    summary = {
        'num_observations': len(df),
        'date_range': [df['Date'].min().strftime('%Y-%m-%d'), df['Date'].max().strftime('%Y-%m-%d')],
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_dates': int(df['Date'].duplicated().sum()),
        'min_passengers': float(df['Passengers'].min()),
        'max_passengers': float(df['Passengers'].max()),
        'mean_passengers': float(df['Passengers'].mean()),
        'median_passengers': float(df['Passengers'].median()),
        'std_passengers': float(df['Passengers'].std())
    }
    save_json(summary, reports_dir / 'eda_summary.json')


def plot_training_history(history, figures_dir):
    """Plots training and validation loss."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / 'training_loss.png')
    plt.close()


def plot_predictions(dates, actual, predicted, scaler, figures_dir):
    """Plots actual vs predicted, residuals, and scatter plot."""

    actual_unscaled = scaler.inverse_transform(actual.reshape(-1, 1)).flatten()
    predicted_unscaled = scaler.inverse_transform(predicted.reshape(-1, 1)).flatten()

    # 1. Actual vs Predicted
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual_unscaled, label='Actual')
    plt.plot(dates, predicted_unscaled, label='Predicted')
    plt.title('Actual vs Predicted Passenger Counts')
    plt.xlabel('Date')
    plt.ylabel('Passengers')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / 'actual_vs_predicted.png')
    plt.close()

    # 2. Residuals (Prediction Error Plot)
    residuals = actual_unscaled - predicted_unscaled
    plt.figure(figsize=(10, 5))
    plt.plot(dates, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Prediction Errors (Residuals)')
    plt.xlabel('Date')
    plt.ylabel('Error (Actual - Predicted)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / 'prediction_errors.png')
    plt.close()

    # 3. Scatter Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(actual_unscaled, predicted_unscaled, alpha=0.7)

    # Perfect prediction line
    min_val = min(actual_unscaled.min(), predicted_unscaled.min())
    max_val = max(actual_unscaled.max(), predicted_unscaled.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.title('Actual vs Predicted Scatter')
    plt.xlabel('Actual Passengers')
    plt.ylabel('Predicted Passengers')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / 'prediction_scatter.png')
    plt.close()
