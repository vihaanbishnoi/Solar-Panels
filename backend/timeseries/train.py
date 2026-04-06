"""Train the energy forecasting model from time-series data."""

from backend.timeseries.pipeline import (
    FORECASTS_DIR,
    build_feature_frame,
    load_raw_data,
    save_energy_model,
    save_forecast_frame,
    save_metrics,
    train_energy_model,
)


def main():
    solar, weather = load_raw_data()
    feature_frame = build_feature_frame(solar, weather)
    artifacts = train_energy_model(feature_frame)

    model_path = save_energy_model(artifacts.model)
    metrics_path = save_metrics(artifacts.metrics)

    prediction_frame = artifacts.test_frame.copy()
    prediction_frame["predicted_solar_generation"] = artifacts.predictions
    forecast_path = save_forecast_frame(
        prediction_frame,
        FORECASTS_DIR / "energy_forecast_test_predictions.csv",
    )

    print("Energy forecasting model trained successfully.")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Predictions saved to: {forecast_path}")
    print(
        "Holdout metrics: "
        f"MAE={artifacts.metrics['mae']:.4f}, "
        f"RMSE={artifacts.metrics['rmse']:.4f}"
    )
    print(
        "Train period: "
        f"{artifacts.train_frame['Timestamp'].min()} -> "
        f"{artifacts.train_frame['Timestamp'].max()}"
    )
    print(
        "Test period: "
        f"{artifacts.test_frame['Timestamp'].min()} -> "
        f"{artifacts.test_frame['Timestamp'].max()}"
    )


if __name__ == "__main__":
    main()
