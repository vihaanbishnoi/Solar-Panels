"""Preprocess and align weather/irradiance/generation into one table."""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "timeseries" / "processed"


def align_to_15min(
    generation_df: pd.DataFrame,
    irradiance_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    site_col: str = "site_id",
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """Align all sources on 15-minute buckets for modeling."""
    gen = generation_df.copy()
    irr = irradiance_df.copy()
    wth = weather_df.copy()

    gen[ts_col] = pd.to_datetime(gen[ts_col])
    irr[ts_col] = pd.to_datetime(irr[ts_col])
    wth[ts_col] = pd.to_datetime(wth[ts_col])

    irr = (
        irr.set_index(ts_col)
        .groupby(site_col)
        .resample("15min")
        .mean(numeric_only=True)
        .reset_index()
    )
    wth = (
        wth.set_index(ts_col)
        .groupby(site_col)
        .resample("15min")
        .mean(numeric_only=True)
        .reset_index()
    )

    merged = gen.merge(irr, on=[site_col, ts_col], how="left", suffixes=("", "_irr"))
    merged = merged.merge(wth, on=[site_col, ts_col], how="left", suffixes=("", "_wth"))
    merged = merged.sort_values([site_col, ts_col]).reset_index(drop=True)
    merged = merged.ffill().bfill()
    return merged


def save_processed(df: pd.DataFrame, file_name: str = "aligned_15min.csv") -> Path:
    """Save aligned table for feature engineering."""
    out_path = PROCESSED_DIR / file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path

