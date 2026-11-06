"""Perturbazioni controllate dei valori previsti per simulazioni what-if."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .forecast import historical_seasonality_indices


def transform_forecast(
    baseline: pd.Series,
    history: pd.Series,
    monthly_trend_pct: float = 0.0,
    seasonality_pct: float = 100.0,
) -> pd.Series:
    """Change relative trend and seasonal amplitude without fitting on future data.

    0 extra trend / 100% seasonality leaves the baseline forecast unchanged.
    """
    if not (-20 < monthly_trend_pct < 20) or not (0 <= seasonality_pct <= 250):
        raise ValueError("Trend o stagionalità non validi")
    indices = historical_seasonality_indices(history)
    multiplier = seasonality_pct / 100.0
    factors = []
    for i, dt in enumerate(baseline.index, start=1):
        season_index = indices.get(dt.month, 1.0)
        adjusted_index = max(0.10, 1 + multiplier * (season_index - 1))
        seasonal_ratio = adjusted_index / max(0.10, season_index)
        trend_ratio = (1.0 + monthly_trend_pct / 100.0) ** i
        factors.append(seasonal_ratio * trend_ratio)
    return pd.Series(np.maximum(0.0, baseline.to_numpy(dtype=float) * factors),
                     index=baseline.index, name="Domanda scenario")
