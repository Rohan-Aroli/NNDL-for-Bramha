import os
import pandas as pd
import requests
import io
import sys

def download_and_validate_dataset(url, filepath):
    """
    Downloads the dataset if not cached, validates it, and returns a DataFrame.
    """
    if filepath.exists():
        print(f"Dataset already cached at {filepath}. Loading cached version.")
        try:
            df = pd.read_csv(filepath)
            df = validate_dataset(df)
            return df
        except Exception as e:
            print(f"Cached dataset is invalid or missing ({e}). Re-downloading...")

    print(f"Downloading dataset from {url}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Read the content into pandas
        df = pd.read_csv(io.StringIO(response.text))

        # Validate before saving
        df = validate_dataset(df)

        # Save to cache
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Dataset downloaded and saved to {filepath}.")

        return df
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        print("Please check your internet connection or provide the dataset manually.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing dataset: {e}")
        sys.exit(1)

def validate_dataset(df):
    """
    Validates the Air Passengers dataset.
    Expects 'Month' and 'Passengers' columns (or similar).
    """
    if df.empty:
        raise ValueError("Dataset is empty.")

    # Check for expected columns. The URL we use often has 'Month' and 'Passengers'.
    # We will rename them to standard names for consistency.
    cols = df.columns.tolist()
    if len(cols) != 2:
        raise ValueError(f"Expected 2 columns, found {len(cols)}")

    df.columns = ['Date', 'Passengers']

    # Check types
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception as e:
        raise ValueError(f"Could not parse 'Date' column as datetime: {e}")

    df['Passengers'] = pd.to_numeric(df['Passengers'], errors='coerce')

    # Check for NaNs
    if df['Passengers'].isnull().any() or df['Date'].isnull().any():
        raise ValueError("Dataset contains missing values.")

    # Check for negative passengers
    if (df['Passengers'] <= 0).any():
        raise ValueError("Dataset contains zero or negative passenger counts.")

    # Check duplicates
    if df['Date'].duplicated().any():
        raise ValueError("Dataset contains duplicate dates.")

    # Sort chronologically
    df = df.sort_values(by='Date').reset_index(drop=True)

    if len(df) < 50:
        raise ValueError(f"Dataset too small ({len(df)} rows).")

    return df
