"""Utilities for solar energy forecasting from time-series data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parents[1]
SOLAR_DATA_PATH = BASE_DIR / "data" / "timeseries" / "raw" / "Solar_Energy_Generation.csv"
WEATHER_DATA_PATH = (
    BASE_DIR / "data" / "timeseries" / "raw" / "Weather_Data_reordered_all.csv"
)
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "timeseries"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FORECASTS_DIR = ARTIFACTS_DIR / "forecasts"
MODEL_PATH = CHECKPOINT_DIR / "energy_forecast_model.joblib"
METRICS_PATH = METRICS_DIR / "energy_forecast_metrics.json"

DEFAULT_SITE_KEY = 1
DEFAULT_CAMPUS_KEY = 2
DEFAULT_TEST_SIZE = 0.2
TIME_FREQUENCY = "15min"
FEATURE_COLUMNS = [
    "AirTemperature",
    "RelativeHumidity",
    "WindSpeed",
    "lag_1",
    "lag_4",
    "rolling_mean_4",
    "hour",
    "month",
]
TARGET_COLUMN = "SolarGeneration"


@dataclass
class ForecastArtifacts:
    """Container for training outputs."""

    model: RandomForestRegressor
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    predictions: np.ndarray
    metrics: dict[str, float]


def load_raw_data(
    solar_path: Path = SOLAR_DATA_PATH,
    weather_path: Path = WEATHER_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw solar generation and weather datasets."""
    solar = pd.read_csv(solar_path)
    weather = pd.read_csv(weather_path)

    solar["Timestamp"] = pd.to_datetime(solar["Timestamp"])
    weather["Timestamp"] = pd.to_datetime(weather["Timestamp"])

    return solar, weather


def _process_solar_day(day_frame: pd.DataFrame) -> pd.DataFrame:
    """Interpolate daytime gaps while keeping nighttime generation at zero."""
    day_frame = day_frame.copy()
    values = day_frame[TARGET_COLUMN].to_numpy()
    valid_idx = np.where(values > 0)[0]

    if len(valid_idx) == 0:
        day_frame[TARGET_COLUMN] = 0
        return day_frame

    first_valid = valid_idx[0]
    last_valid = valid_idx[-1]
    target_idx = day_frame.columns.get_loc(TARGET_COLUMN)

    day_frame.iloc[:first_valid, target_idx] = 0
    day_frame.iloc[last_valid + 1 :, target_idx] = 0

    middle = day_frame.iloc[first_valid : last_valid + 1].copy()
    middle[TARGET_COLUMN] = middle[TARGET_COLUMN].replace(0, np.nan)
    middle[TARGET_COLUMN] = middle[TARGET_COLUMN].interpolate(
        method="linear",
        limit_direction="both",
    )
    day_frame.iloc[first_valid : last_valid + 1] = middle

    return day_frame


def preprocess_solar_data(
    solar: pd.DataFrame,
    site_key: int = DEFAULT_SITE_KEY,
) -> pd.DataFrame:
    """Clean and enrich solar generation data for a single site."""
    site_frame = solar.loc[solar["SiteKey"] == site_key].copy()
    site_frame = site_frame.sort_values("Timestamp").set_index("Timestamp")
    site_frame = site_frame.asfreq(TIME_FREQUENCY)
    site_frame = site_frame.groupby(site_frame.index.date, group_keys=False).apply(
        _process_solar_day
    )

    site_frame[TARGET_COLUMN] = site_frame[TARGET_COLUMN].fillna(0)
    site_frame[["CampusKey", "SiteKey"]] = (
        site_frame[["CampusKey", "SiteKey"]].ffill().bfill()
    )

    site_frame["hour"] = site_frame.index.hour
    site_frame["day"] = site_frame.index.day
    site_frame["month"] = site_frame.index.month
    site_frame["day_of_week"] = site_frame.index.dayofweek

    return site_frame.reset_index().rename(columns={"index": "Timestamp"})


def preprocess_weather_data(
    weather: pd.DataFrame,
    campus_key: int = DEFAULT_CAMPUS_KEY,
) -> pd.DataFrame:
    """Clean weather data and align it to a complete 15-minute timeline."""
    weather_frame = weather.loc[weather["CampusKey"] == campus_key].copy()
    weather_frame = weather_frame.sort_values("Timestamp").set_index("Timestamp")
    weather_frame = weather_frame[~weather_frame.index.duplicated(keep="first")]

    full_time = pd.date_range(
        start=weather_frame.index.min(),
        end=weather_frame.index.max(),
        freq=TIME_FREQUENCY,
    )
    weather_frame = weather_frame.reindex(full_time)
    numeric_columns = weather_frame.select_dtypes(include=["number"]).columns
    weather_frame[numeric_columns] = weather_frame[numeric_columns].interpolate(
        method="time"
    )

    return (
        weather_frame.reset_index()
        .rename(columns={"index": "Timestamp"})
        .drop(columns=["level_0"], errors="ignore")
    )


def build_feature_frame(
    solar: pd.DataFrame,
    weather: pd.DataFrame,
    site_key: int = DEFAULT_SITE_KEY,
    campus_key: int = DEFAULT_CAMPUS_KEY,
) -> pd.DataFrame:
    """Merge cleaned solar and weather data and create model features."""
    clean_solar = preprocess_solar_data(solar, site_key=site_key)
    clean_weather = preprocess_weather_data(weather, campus_key=campus_key)

    data = pd.merge(clean_solar, clean_weather, on="Timestamp", how="inner")
    data["lag_1"] = data[TARGET_COLUMN].shift(1)
    data["lag_4"] = data[TARGET_COLUMN].shift(4)
    data["rolling_mean_4"] = data[TARGET_COLUMN].rolling(4).mean()

    return data.dropna().reset_index(drop=True)


def split_feature_frame(
    data: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the time-series frame without shuffling to preserve chronology."""
    split_index = int(len(data) * (1 - test_size))
    train = data.iloc[:split_index].copy()
    test = data.iloc[split_index:].copy()
    return train, test


def train_energy_model(
    data: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = 42,
    n_estimators: int = 100,
) -> ForecastArtifacts:
    """Train the random forest regressor and evaluate it on the holdout set."""
    train_frame, test_frame = split_feature_frame(data, test_size=test_size)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_frame[FEATURE_COLUMNS], train_frame[TARGET_COLUMN])

    predictions = model.predict(test_frame[FEATURE_COLUMNS])
    metrics = evaluate_predictions(test_frame[TARGET_COLUMN], predictions)

    return ForecastArtifacts(
        model=model,
        train_frame=train_frame,
        test_frame=test_frame,
        predictions=predictions,
        metrics=metrics,
    )


def evaluate_predictions(
    actual: pd.Series,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Compute basic regression metrics."""
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))

    return {
        "mae": float(mae),
        "rmse": rmse,
    }


def save_energy_model(model: RandomForestRegressor, model_path: Path = MODEL_PATH) -> Path:
    """Persist the trained forecasting model."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


def load_energy_model(model_path: Path = MODEL_PATH) -> RandomForestRegressor:
    """Load a persisted forecasting model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Forecast model not found: {model_path}")
    return joblib.load(model_path)


def save_metrics(metrics: dict[str, float], metrics_path: Path = METRICS_PATH) -> Path:
    """Persist evaluation metrics as JSON."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics_path


def save_forecast_frame(
    frame: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Persist forecast rows for downstream use."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def forecast_from_frame(
    model: RandomForestRegressor,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Run energy prediction on a prepared feature frame."""
    forecast = frame.copy()
    forecast["predicted_solar_generation"] = model.predict(frame[FEATURE_COLUMNS])
    return forecast
