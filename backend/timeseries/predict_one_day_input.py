"""Predict one-row solar generation from interactive user input."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backend.timeseries.pipeline import FEATURE_COLUMNS, forecast_from_frame, load_energy_model


def _prompt_float(label: str) -> float:
    while True:
        raw_value = input(f"{label}: ").strip()
        try:
            return float(raw_value)
        except ValueError:
            print("Please enter a valid number.")


def _prompt_int(label: str, *, min_value: int, max_value: int) -> int:
    while True:
        raw_value = input(f"{label} ({min_value}-{max_value}): ").strip()
        try:
            value = int(raw_value)
        except ValueError:
            print("Please enter a valid integer.")
            continue

        if min_value <= value <= max_value:
            return value

        print(f"Value must be between {min_value} and {max_value}.")


def collect_one_row() -> pd.DataFrame:
    print("Enter one-day feature values for a single prediction row.")
    date_label = input("Date label (YYYY-MM-DD, optional): ").strip() or "unknown-date"

    row = {
        "AirTemperature": _prompt_float("AirTemperature"),
        "RelativeHumidity": _prompt_float("RelativeHumidity"),
        "WindSpeed": _prompt_float("WindSpeed"),
        "lag_1": _prompt_float("lag_1 (previous 15-min solar generation)"),
        "lag_4": _prompt_float("lag_4 (previous 1-hour solar generation)"),
        "rolling_mean_4": _prompt_float("rolling_mean_4 (last 4 intervals mean)"),
        "hour": _prompt_int("hour", min_value=0, max_value=23),
        "month": _prompt_int("month", min_value=1, max_value=12),
        "input_date": date_label,
    }

    frame = pd.DataFrame([row])
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive one-row energy prediction from manual feature input."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backend/artifacts/timeseries/forecasts/one_day_user_prediction.csv",
        help="Where to save the single-row prediction CSV.",
    )
    args = parser.parse_args()

    frame = collect_one_row()

    model = load_energy_model()
    forecast = forecast_from_frame(model, frame[FEATURE_COLUMNS])
    forecast.insert(0, "input_date", frame["input_date"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output_path, index=False)

    predicted_value = float(forecast.loc[0, "predicted_solar_generation"])
    print(f"\nPredicted solar generation (one row): {predicted_value:.4f}")
    print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
