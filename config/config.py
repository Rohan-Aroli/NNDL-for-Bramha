from pathlib import Path

# Project Roots
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "air_passengers.csv"

# Output Paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Dataset URL
# Using a reliable raw CSV from a known repository.
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"

# Data Splitting
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Preprocessing
LOOKBACK = 12

# Model Configuration
LSTM_UNITS = 50
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001

# Training Configuration
BATCH_SIZE = 16
EPOCHS = 100
PATIENCE = 10
RANDOM_SEED = 42

# Forecast Configuration
FORECAST_MONTHS = 12
