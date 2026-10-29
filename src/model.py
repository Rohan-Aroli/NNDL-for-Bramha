import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def build_lstm_model(config):
    """
    Builds and compiles a basic LSTM model.
    """
    # Fix random seed for reproducibility (as much as possible)
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    tf.config.experimental.enable_op_determinism()

    model = Sequential([
        Input(shape=(config.LOOKBACK, 1)),
        LSTM(config.LSTM_UNITS, activation='tanh'),
        Dropout(config.DROPOUT_RATE),
        Dense(1) # Linear activation for regression
    ])

    optimizer = Adam(learning_rate=config.LEARNING_RATE)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    model.summary()

    return model
