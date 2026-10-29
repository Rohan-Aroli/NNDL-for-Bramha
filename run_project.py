import os
import sys
import subprocess
import venv
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"

def get_venv_python():
    """Returns the path to the Python executable in the virtual environment."""
    if os.name == 'nt':
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"

def create_virtual_environment():
    """Creates a virtual environment if it doesn't exist."""
    if not VENV_DIR.exists():
        print("[2/9] Creating virtual environment...")
        try:
            venv.create(VENV_DIR, with_pip=True)
            print("Virtual environment created successfully.")
        except Exception as e:
            print(f"Error creating virtual environment: {e}")
            sys.exit(1)
    else:
        print("[2/9] Virtual environment already exists, skipping creation.")

def install_dependencies():
    """Installs dependencies from requirements.txt into the virtual environment."""
    print("[3/9] Installing dependencies...")
    venv_python = get_venv_python()
    if not venv_python.exists():
        print(f"Error: Virtual environment Python not found at {venv_python}")
        sys.exit(1)

    try:
        # Upgrade pip first
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True, capture_output=True)
        # Install requirements
        result = subprocess.run([str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)], check=True)
        if result.returncode == 0:
            print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        if e.output:
            print(e.output.decode())
        if e.stderr:
            print(e.stderr.decode())
        sys.exit(1)

def run_workflow_in_venv():
    """Re-launches this script using the virtual environment's Python."""
    venv_python = get_venv_python()
    if not venv_python.exists():
        print(f"Error: Virtual environment Python not found at {venv_python}")
        sys.exit(1)

    # Set an environment variable to prevent infinite loops
    env = os.environ.copy()
    env["VENV_LAUNCHED"] = "1"

    print("Re-launching workflow inside the virtual environment...")
    result = subprocess.run([str(venv_python), str(__file__)], env=env)
    sys.exit(result.returncode)

def create_directories():
    """Creates necessary project directories."""
    directories = [
        "data/raw",
        "data/processed",
        "outputs/figures",
        "outputs/models",
        "outputs/predictions",
        "outputs/reports"
    ]
    for d in directories:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

def main_workflow():
    """The main workflow that runs inside the virtual environment."""
    print("[4/9] Preparing dataset...")
    # Add imports inside the function so they are only evaluated when in the venv
    try:
        import config.config as config
        from src.dataset import download_and_validate_dataset
        from src.visualize import perform_eda, plot_training_history, plot_predictions
        from src.preprocessing import load_and_preprocess_data
        from src.model import build_lstm_model
        from src.train import train_model
        from src.evaluate import evaluate_model
        from src.forecast import generate_future_forecast

        # Ensure directories exist
        create_directories()

        # Step 4: Dataset
        df = download_and_validate_dataset(config.DATA_URL, config.RAW_DATA_PATH)

        # Step 5: EDA
        print("[5/9] Performing exploratory data analysis...")
        perform_eda(df, config.FIGURES_DIR, config.REPORTS_DIR)

        # Step 6: Preprocessing
        print("[6/9] Preprocessing time-series data...")
        X_train, y_train, X_val, y_val, X_test, y_test, test_dates, scaler, baseline_predictions = load_and_preprocess_data(df, config)

        # Step 7: Training
        print("[7/9] Training LSTM model...")
        model = build_lstm_model(config)
        model, history = train_model(model, X_train, y_train, X_val, y_val, config)
        plot_training_history(history, config.FIGURES_DIR)

        # Step 8: Evaluating
        print("[8/9] Evaluating model...")
        predictions, metrics = evaluate_model(model, X_test, y_test, scaler, test_dates, baseline_predictions, config)

        # Step 9: Saving Results and Plotting
        print("[9/9] Saving results and generating prediction plots...")
        plot_predictions(test_dates, y_test, predictions, scaler, config.FIGURES_DIR)

        print("Generating optional future forecast...")
        generate_future_forecast(model, df, scaler, config)

        print("\nWorkflow completed successfully!")

    except ImportError as e:
        print(f"Import error during workflow execution: {e}")
        print("Ensure all modules in 'src/' are correctly implemented.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during workflow execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    if os.environ.get("VENV_LAUNCHED") == "1":
        # We are already in the virtual environment
        main_workflow()
    else:
        # We are in the host environment
        print("[1/9] Checking Python environment...")
        version = sys.version_info
        if not (version.major == 3 and version.minor >= 9):
            print(f"Error: Python 3.9 or newer is required. You are using Python {version.major}.{version.minor}.")
            sys.exit(1)

        create_virtual_environment()
        install_dependencies()
        run_workflow_in_venv()

if __name__ == "__main__":
    main()
