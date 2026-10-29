import pytest
import numpy as np
import pandas as pd
from src.preprocessing import create_sequences, load_and_preprocess_data

class DummyConfig:
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    LOOKBACK = 5

def test_create_sequences():
    data = np.arange(10).reshape(-1, 1)
    X, y = create_sequences(data, lookback=3)

    assert X.shape == (7, 3, 1)
    assert y.shape == (7, 1)

    # First sequence should be [0, 1, 2] predicting 3
    assert np.array_equal(X[0], np.array([[0], [1], [2]]))
    assert y[0] == 3

def test_load_and_preprocess_data():
    config = DummyConfig()

    # 100 rows
    data = {
        'Date': pd.date_range(start='1/1/1949', periods=100, freq='ME'),
        'Passengers': np.arange(1, 101)
    }
    df = pd.DataFrame(data)

    X_train, y_train, X_val, y_val, X_test, y_test, test_dates, scaler, baseline = load_and_preprocess_data(df, config)

    # Total data = 100
    # Train = 70
    # Val = 15
    # Test = 15

    assert len(X_train) == 70 - config.LOOKBACK
    assert len(X_val) == 15
    assert len(X_test) == 15

    assert len(test_dates) == 15
    assert len(baseline) == 15

    # Check max value is scaled properly based on training data
    train_max = scaler.data_max_[0]
    assert train_max == 70.0 # because it scales strictly on training data
