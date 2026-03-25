"""Feature generation utilities for energy forecasting."""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURE_DIR = BASE_DIR / "data" / "timeseries" / "features"


def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Add cyclic hour/day features."""
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col])
    hour = out[ts_col].dt.hour + out[ts_col].dt.minute / 60.0
    day = out[ts_col].dt.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * day / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * day / 365.25)
    return out


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "generation_kw",
    site_col: str = "site_id",
    lags: tuple[int, ...] = (1, 2, 4, 96),
) -> pd.DataFrame:
    """Add lagged target features (15-min steps)."""
    out = df.copy()
    for lag in lags:
        out[f"{target_col}_lag_{lag}"] = out.groupby(site_col)[target_col].shift(lag)
    return out.dropna().reset_index(drop=True)


def save_features(df: pd.DataFrame, file_name: str = "features.csv") -> Path:
    """Persist features table for model training."""
    out_path = FEATURE_DIR / file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path

