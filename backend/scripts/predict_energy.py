"""Generate energy predictions from a prepared feature CSV."""

import argparse
from pathlib import Path

import pandas as pd

from backend.src.timeseries import FEATURE_COLUMNS, forecast_from_frame, load_energy_model


def main():
    parser = argparse.ArgumentParser(
        description="Predict solar energy generation from a prepared feature CSV."
    )
    parser.add_argument("input_csv", type=str, help="Path to input CSV with model features")
    parser.add_argument(
        "--output",
        type=str,
        default="backend/artifacts/timeseries/forecasts/website_predictions.csv",
        help="Where to save the predictions CSV",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output)

    frame = pd.read_csv(input_path)
    missing_columns = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Input CSV is missing required feature columns: "
            + ", ".join(missing_columns)
        )

    model = load_energy_model()
    forecast = forecast_from_frame(model, frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
