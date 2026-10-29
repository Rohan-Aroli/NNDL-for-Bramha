import pytest
import numpy as np
from src.model import build_lstm_model

class DummyConfig:
    RANDOM_SEED = 42
    LOOKBACK = 12
    LSTM_UNITS = 10
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 0.001

def test_model_architecture():
    config = DummyConfig()
    model = build_lstm_model(config)

    # Model should have 3 layers (Input is not counted in model.layers in newer TF, or it is but it's an InputLayer)
    # Typically: LSTM, Dropout, Dense
    layers = model.layers
    assert len(layers) == 3

    assert layers[0].__class__.__name__ == 'LSTM'
    assert layers[1].__class__.__name__ == 'Dropout'
    assert layers[2].__class__.__name__ == 'Dense'

    # Check output shape
    # Expected output is (None, 1)
    assert model.output_shape == (None, 1)

def test_model_forward_pass():
    config = DummyConfig()
    model = build_lstm_model(config)

    # Create dummy input data: 2 samples, 12 lookback, 1 feature
    dummy_input = np.random.rand(2, 12, 1)

    output = model.predict(dummy_input)
    assert output.shape == (2, 1)
