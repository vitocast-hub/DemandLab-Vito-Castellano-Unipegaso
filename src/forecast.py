"""Metodi previsionali e validazione temporale senza fuga di dati futuri."""
from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = {
    "Media mobile (3 mesi)": "Media aritmetica degli ultimi tre mesi; iterativa nell'orizzonte futuro.",
    "Attenuazione esponenziale": "Livellamento semplice con alpha=0,35; previsioni future costanti.",
    "Regressione lineare": "Retta stimata sui valori storici, estrapolata nel tempo.",
    "Naive stagionale (12 mesi)": "Ripetizione del valore osservato nello stesso mese dell'anno precedente.",
    "Regressione trend + stagionalità": "Minimi quadrati con trend lineare e 11 indicatori mensili.",
}


def _design_matrix(dates: pd.DatetimeIndex, steps: np.ndarray) -> np.ndarray:
    """Intercept, linear trend, dummies Feb-Dec (Jan is omitted)."""
    months = dates.month.to_numpy()
    return np.column_stack(
        [np.ones(len(dates)), steps.astype(float)]
        + [(months == month).astype(float) for month in range(2, 13)]
    )


def forecast_series(history: pd.Series, method: str, horizon: int = 6) -> pd.Series:
    """Forecast future monthly demand; does not read any future observations."""
    if method not in METHODS:
        raise ValueError(f"Metodo sconosciuto: {method}")
    if len(history) < 12 or horizon < 1:
        raise ValueError("Servono almeno 12 mesi e un orizzonte positivo")
    if not isinstance(history.index, pd.DatetimeIndex):
        raise TypeError("Lo storico richiede un indice DatetimeIndex")
    values = history.to_numpy(dtype=float)
    future = pd.date_range(history.index[-1] + pd.offsets.MonthBegin(1),
                           periods=horizon, freq="MS")

    if method == "Media mobile (3 mesi)":
        extended = values.tolist()
        results = []
        for _ in range(horizon):
            next_value = float(np.mean(extended[-3:]))
            results.append(next_value)
            extended.append(next_value)
    elif method == "Attenuazione esponenziale":
        alpha = 0.35
        level = float(values[0])
        for observed in values[1:]:
            level = alpha * float(observed) + (1 - alpha) * level
        results = [level] * horizon
    elif method == "Regressione lineare":
        steps = np.arange(len(values), dtype=float)
        slope, intercept = np.polyfit(steps, values, deg=1)
        results = (intercept + slope * np.arange(len(values), len(values) + horizon)).tolist()
    elif method == "Naive stagionale (12 mesi)":
        extended = values.tolist()
        results = []
        for _ in range(horizon):
            next_value = float(extended[-12])
            results.append(next_value)
            extended.append(next_value)
    else:  # linear trend with additive monthly seasonality
        steps = np.arange(len(values), dtype=float)
        x = _design_matrix(history.index, steps)
        coefs, *_ = np.linalg.lstsq(x, values, rcond=None)
        x_future = _design_matrix(future,
                                  np.arange(len(values), len(values) + horizon))
        results = (x_future @ coefs).tolist()

    return pd.Series(np.maximum(0.0, np.asarray(results, dtype=float)),
                     index=future, name="Previsione")


def error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("Le serie devono avere uguale lunghezza non nulla")
    err = actual - predicted
    abs_err = np.abs(err)
    # symmetric MAPE; zeros contribute zero when actual and forecast both equal 0.
    denom = np.abs(actual) + np.abs(predicted)
    smape = np.mean(np.divide(200 * abs_err, denom,
                             out=np.zeros_like(abs_err), where=denom > 0))
    return {"MAE": float(np.mean(abs_err)),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "sMAPE (%)": float(smape)}


def evaluate_methods(history: pd.Series, holdout: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window, one-step-ahead rolling validation on last N months."""
    if holdout < 1 or len(history) - holdout < 24:
        raise ValueError("Serve un addestramento iniziale di almeno 24 mesi")
    validation_dates = history.index[-holdout:]
    actual = history.iloc[-holdout:].to_numpy(dtype=float)
    records = []
    predicted = pd.DataFrame(index=validation_dates)
    for method in METHODS:
        forecasts = [forecast_series(history.iloc[:t], method, 1).iloc[0]
                     for t in range(len(history) - holdout, len(history))]
        predicted[method] = forecasts
        records.append({"Metodo": method, **error_metrics(actual, np.asarray(forecasts))})
    predicted.insert(0, "Domanda effettiva", actual)
    scores = pd.DataFrame(records).sort_values(["MAE", "RMSE", "Metodo"])
    return scores.reset_index(drop=True), predicted


def historical_seasonality_indices(history: pd.Series) -> dict[int, float]:
    """Approximate monthly indexes after removing a linear trend; global average=1."""
    v = history.to_numpy(dtype=float)
    x = np.arange(len(v), dtype=float)
    slope, intercept = np.polyfit(x, v, deg=1)
    base = np.maximum(1.0, intercept + slope * x)
    ratios = pd.Series(np.clip(v / base, 0.15, 4.0), index=history.index)
    indices = ratios.groupby(ratios.index.month).mean().to_dict()
    normalized_mean = float(np.mean(list(indices.values())))
    return {month: float(np.clip(value / normalized_mean, 0.2, 3.0))
            for month, value in indices.items()}
