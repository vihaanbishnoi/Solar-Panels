"""Load raw UNISOLAR-like tables from CSV files."""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "timeseries" / "raw"


def load_raw_csv(file_name: str, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Load a CSV from the raw timeseries folder and parse timestamps."""
    file_path = RAW_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Raw file not found: {file_path}")
    return pd.read_csv(file_path, parse_dates=[timestamp_col])

