import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

def generate_future_forecast(model, df, scaler, config):
    """
    Generates a forecast for future unseen months based on the latest available data.
    """
    # 1. Get the last known observations from the entire dataset
    values = df['Passengers'].values.reshape(-1, 1)
    scaled_values = scaler.transform(values)

    # Start with the last 'lookback' window
    current_input = scaled_values[-config.LOOKBACK:]
    current_input = np.expand_dims(current_input, axis=0) # shape (1, lookback, 1)

    forecast_scaled = []

    # 2. Iteratively predict the next time step, and append it to the window
    for _ in range(config.FORECAST_MONTHS):
        pred = model.predict(current_input, verbose=0) # shape (1, 1)
        forecast_scaled.append(pred[0, 0])

        # Update input: slide window by removing first element, appending new prediction
        # new_element shape must match the sequence feature shape
        new_element = np.array([[[pred[0, 0]]]])
        current_input = np.concatenate((current_input[:, 1:, :], new_element), axis=1)

    # 3. Inverse transform predictions
    forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
    forecast_unscaled = scaler.inverse_transform(forecast_scaled).flatten()

    # 4. Generate future dates
    last_date = df['Date'].iloc[-1]

    # Assuming monthly data
    future_dates = [last_date + relativedelta(months=i+1) for i in range(config.FORECAST_MONTHS)]

    # 5. Save to CSV
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecasted_Passengers": forecast_unscaled
    })

    forecast_path = config.PREDICTIONS_DIR / "future_forecast.csv"
    forecast_df.to_csv(forecast_path, index=False)

    print(f"Generated {config.FORECAST_MONTHS}-month forecast saved to {forecast_path}.")
    return forecast_df
