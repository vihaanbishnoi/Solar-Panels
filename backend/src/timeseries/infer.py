"""Run inference for energy prediction and export forecast CSV."""

import json
from pathlib import Path

import pandas as pd

from backend.src.timeseries.model import PersistenceForecaster

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_PATH = BASE_DIR / "data" / "timeseries" / "features" / "features.csv"
ARTIFACT_PATH = BASE_DIR / "artifacts" / "timeseries" / "checkpoints" / "baseline_model.json"
FORECAST_PATH = BASE_DIR / "artifacts" / "timeseries" / "forecasts" / "forecast.csv"


def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURES_PATH}")
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {ARTIFACT_PATH}")

    meta = json.loads(ARTIFACT_PATH.read_text())
    model = PersistenceForecaster(lag_feature=meta["lag_feature"])

    df = pd.read_csv(FEATURES_PATH)
    preds = model.predict(df)

    output = df.copy()
    output["pred_generation_kw"] = preds
    FORECAST_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(FORECAST_PATH, index=False)
    print(f"Saved forecasts: {FORECAST_PATH}")


if __name__ == "__main__":
    main()

