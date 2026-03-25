"""Simple baseline forecaster for energy prediction."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class PersistenceForecaster:
    """Baseline forecaster using a lag feature as prediction."""

    lag_feature: str = "generation_kw_lag_1"

    def fit(self, df: pd.DataFrame, target_col: str = "generation_kw") -> "PersistenceForecaster":
        if self.lag_feature not in df.columns:
            raise ValueError(f"Missing lag feature: {self.lag_feature}")
        if target_col not in df.columns:
            raise ValueError(f"Missing target column: {target_col}")
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.lag_feature not in df.columns:
            raise ValueError(f"Missing lag feature: {self.lag_feature}")
        return df[self.lag_feature].astype(float)

