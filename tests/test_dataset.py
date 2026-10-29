import pytest
import pandas as pd
import numpy as np
from src.dataset import validate_dataset

def test_valid_dataset():
    data = {
        'Date': pd.date_range(start='1/1/1949', periods=60, freq='ME'),
        'Passengers': np.random.randint(100, 500, size=60)
    }
    df = pd.DataFrame(data)
    validated_df = validate_dataset(df)
    assert not validated_df.empty
    assert len(validated_df) == 60

def test_missing_columns():
    data = {
        'Date': pd.date_range(start='1/1/1949', periods=60, freq='ME'),
    }
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="Expected 2 columns"):
        validate_dataset(df)

def test_empty_dataset():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_dataset(df)

def test_small_dataset():
    data = {
        'Date': pd.date_range(start='1/1/1949', periods=10, freq='ME'),
        'Passengers': np.random.randint(100, 500, size=10)
    }
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="Dataset too small"):
        validate_dataset(df)

def test_negative_passengers():
    data = {
        'Date': pd.date_range(start='1/1/1949', periods=60, freq='ME'),
        'Passengers': [-5] * 60
    }
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="Dataset contains zero or negative"):
        validate_dataset(df)
