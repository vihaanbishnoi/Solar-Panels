"""Train baseline timeseries model and save artifact metadata."""

import json
from pathlib import Path

import pandas as pd

from backend.src.timeseries.model import PersistenceForecaster

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_PATH = BASE_DIR / "data" / "timeseries" / "features" / "features.csv"
ARTIFACT_PATH = BASE_DIR / "artifacts" / "timeseries" / "checkpoints" / "baseline_model.json"


def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_PATH}. "
            "Run preprocessing + feature engineering first."
        )

    df = pd.read_csv(FEATURES_PATH)
    model = PersistenceForecaster(lag_feature="generation_kw_lag_1")
    model.fit(df, target_col="generation_kw")

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps({"model_type": "persistence", "lag_feature": model.lag_feature}))
    print(f"Saved baseline model metadata: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()

