"""Prepare aligned + feature-engineered timeseries tables from raw CSVs."""

import argparse

from backend.src.timeseries.feature_engineering import add_lag_features, add_time_features, save_features
from backend.src.timeseries.ingest import load_raw_csv
from backend.src.timeseries.preprocess import align_to_15min, save_processed


def main():
    parser = argparse.ArgumentParser(description="Preprocess UNISOLAR-style data")
    parser.add_argument("--generation", required=True, help="Raw generation CSV file name")
    parser.add_argument("--irradiance", required=True, help="Raw irradiance CSV file name")
    parser.add_argument("--weather", required=True, help="Raw weather CSV file name")
    args = parser.parse_args()

    generation_df = load_raw_csv(args.generation)
    irradiance_df = load_raw_csv(args.irradiance)
    weather_df = load_raw_csv(args.weather)

    aligned = align_to_15min(generation_df, irradiance_df, weather_df)
    save_processed(aligned)

    features = add_time_features(aligned)
    features = add_lag_features(features)
    out_path = save_features(features)
    print(f"Saved features: {out_path}")


if __name__ == "__main__":
    main()

