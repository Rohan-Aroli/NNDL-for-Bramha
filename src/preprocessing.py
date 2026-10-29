import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, lookback):
    """
    Creates sequences of length `lookback` as inputs (X) and the next value as target (y).
    """
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

def load_and_preprocess_data(df, config):
    """
    Splits data chronologically, scales based on train, and creates sliding window sequences.
    """
    values = df['Passengers'].values.reshape(-1, 1)
    dates = df['Date'].values

    n = len(values)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))

    # Chronological Split
    train_data = values[:train_end]
    val_data = values[train_end:val_end]
    test_data = values[val_end:]

    test_dates_full = dates[val_end:]

    # Fit scaler on training data ONLY
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)

    # Transform all splits
    train_scaled = scaler.transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    # Generate sliding windows
    # Note: To predict the first value of val/test sets, we need the preceding `lookback` values.
    # We prepend the last `lookback` values of the previous set.
    val_input = np.concatenate((train_scaled[-config.LOOKBACK:], val_scaled))
    test_input = np.concatenate((val_scaled[-config.LOOKBACK:], test_scaled))

    X_train, y_train = create_sequences(train_scaled, config.LOOKBACK)
    X_val, y_val = create_sequences(val_input, config.LOOKBACK)
    X_test, y_test = create_sequences(test_input, config.LOOKBACK)

    # The dates corresponding to the test targets
    test_dates = test_dates_full

    # Baseline: persistence model (predicts the last known value)
    # The last known value before y_test[i] is X_test[i][-1]
    baseline_predictions = np.array([x[-1][0] for x in X_test])

    return X_train, y_train, X_val, y_val, X_test, y_test, test_dates, scaler, baseline_predictions
