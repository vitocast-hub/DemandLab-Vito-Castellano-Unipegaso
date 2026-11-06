"""Dataset didattico: articoli reali come categorie, numeri interamente simulati."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

PRODUCTS = {
    "Scarpe da running": {
        "description": "Domanda sostanzialmente regolare, con oscillazioni moderate.",
        "pattern": "Stabile",
    },
    "Maschere da snorkeling": {
        "description": "Forte domanda durante primavera ed estate nell'emisfero nord.",
        "pattern": "Stagionalità estiva",
    },
    "Giacche termiche": {
        "description": "Domanda concentrata soprattutto nei mesi più freddi.",
        "pattern": "Stagionalità invernale",
    },
    "Tappetini fitness": {
        "description": "Domanda caratterizzata da crescita progressiva.",
        "pattern": "Trend crescente",
    },
}


def generate_dataset(seed: int = 20261105) -> pd.DataFrame:
    """Create 36 monthly observations from Nov 2023 to Oct 2026, reproducibly."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2023-11-01", "2026-10-01", freq="MS")
    output = []
    for month_index, date in enumerate(months):
        m = date.month
        season_summer = {1: .42, 2: .50, 3: .76, 4: 1.05, 5: 1.43,
                         6: 1.85, 7: 2.05, 8: 1.95, 9: 1.28,
                         10: .78, 11: .48, 12: .45}[m]
        season_winter = {1: 1.95, 2: 1.75, 3: 1.32, 4: .85, 5: .55,
                         6: .36, 7: .34, 8: .38, 9: .55,
                         10: .95, 11: 1.56, 12: 1.94}[m]
        values = {
            "Scarpe da running": 540 + .8 * month_index + (14 if m in (3, 9) else 0)
                + rng.normal(0, 25),
            "Maschere da snorkeling": 240 * season_summer + .8 * month_index
                + rng.normal(0, 18),
            "Giacche termiche": 265 * season_winter + .6 * month_index
                + rng.normal(0, 19),
            "Tappetini fitness": 240 + 7.8 * month_index + (18 if m == 1 else 0)
                + rng.normal(0, 19),
        }
        for product, demand in values.items():
            output.append({"mese": date.strftime("%Y-%m-%d"),
                           "articolo": product,
                           "domanda_unita": max(0, int(round(demand))),
                           "natura_dato": "SIMULATO"})
    return pd.DataFrame(output)


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["mese"])
    required = {"mese", "articolo", "domanda_unita", "natura_dato"}
    if not required.issubset(df.columns):
        raise ValueError(f"Mancano colonne: {required - set(df.columns)}")
    if df.duplicated(["mese", "articolo"]).any():
        raise ValueError("Righe duplicate per mese e articolo")
    if df["domanda_unita"].isna().any() or (df["domanda_unita"] < 0).any():
        raise ValueError("Domanda mancante o negativa")
    if not (df["natura_dato"] == "SIMULATO").all():
        raise ValueError("Le origini dei dati vanno dichiarate correttamente")
    for product in PRODUCTS:
        d = df[df["articolo"] == product].sort_values("mese")
        # Calendar months vary in length, so compare monthly period sequences.
        periods = d["mese"].dt.to_period("M")
        expected = pd.period_range(periods.iloc[0], periods.iloc[-1], freq="M") if len(periods) else []
        if len(d) < 36 or len(expected) != len(d) or not periods.reset_index(drop=True).equals(pd.Series(expected)):
            raise ValueError(f"Serie mensile incompleta per {product}")
    return df
