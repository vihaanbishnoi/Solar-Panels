"""Residual-based fault flagging for timeseries predictions."""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
FORECAST_PATH = BASE_DIR / "artifacts" / "timeseries" / "forecasts" / "forecast.csv"
OUTPUT_PATH = BASE_DIR / "artifacts" / "timeseries" / "forecasts" / "fault_flags.csv"


def detect_faults(
    df: pd.DataFrame,
    actual_col: str = "generation_kw",
    pred_col: str = "pred_generation_kw",
    residual_sigma: float = 3.0,
) -> pd.DataFrame:
    """Flag points where residual is far below expected production."""
    out = df.copy()
    out["residual"] = out[actual_col] - out[pred_col]
    std = out["residual"].std(ddof=0)
    threshold = -residual_sigma * std
    out["fault_flag"] = out["residual"] < threshold
    return out


def main():
    if not FORECAST_PATH.exists():
        raise FileNotFoundError(f"Forecast file not found: {FORECAST_PATH}")
    df = pd.read_csv(FORECAST_PATH)
    result = detect_faults(df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved fault flags: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

