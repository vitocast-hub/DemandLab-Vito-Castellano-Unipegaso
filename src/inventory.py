"""Stima didattica delle quattro componenti di stock e della politica periodica."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
import numpy as np
import pandas as pd

DAYS_PER_MONTH = 30.0  # convenzione esplicita, utilizzata in tutte le formule


@dataclass(frozen=True)
class InventoryResult:
    cyclic: float
    safety: float
    anticipation: float
    pipeline: float
    physical_reference: float
    composite_reference: float
    order_up_to: float
    avg_monthly_demand: float
    cv_monthly: float
    capacity_monthly: float


def calculate_inventory(
    forecast: pd.Series,
    cv_monthly: float,
    lead_time_days: int,
    lead_time_sd_days: float,
    review_days: int,
    service_level: float,
    capacity_monthly: float,
) -> InventoryResult:
    """Calculate expected stock components under stated simplifying assumptions.

    All stocks are quantities (units). ``pipeline`` is in transit and is not
    physically on hand; ``composite_reference`` sums concepts for comparison
    only, it is not an order quantity nor an actual warehouse measurement.
    """
    d = np.asarray(forecast, dtype=float)
    if len(d) == 0 or not np.all(np.isfinite(d)) or np.any(d < 0):
        raise ValueError("Previsioni non valide")
    if cv_monthly < 0 or lead_time_days < 0 or lead_time_sd_days < 0 or review_days <= 0:
        raise ValueError("Parametri di variabilità o tempi non validi")
    if not (0.5 < service_level < 1.0) or capacity_monthly <= 0:
        raise ValueError("Livello di servizio/capacità non validi")

    avg_monthly = float(d.mean())
    mu_daily = avg_monthly / DAYS_PER_MONTH
    # Independent-day approximation so monthly stdev equals cv * monthly mean.
    sigma_daily = cv_monthly * avg_monthly / np.sqrt(DAYS_PER_MONTH)
    protection_days = lead_time_days + review_days
    z = NormalDist().inv_cdf(service_level)

    cyclic = mu_daily * review_days / 2.0
    safety = z * np.sqrt(protection_days * sigma_daily ** 2
                         + (mu_daily * lead_time_sd_days) ** 2)
    pipeline = mu_daily * lead_time_days

    # Initial anticipatory buffer needed when supply each month cannot exceed
    # capacity_monthly and surpluses from earlier months may be stored.
    cumulative_net_requirements = np.cumsum(d - capacity_monthly)
    anticipation = max(0.0, float(np.max(cumulative_net_requirements)))

    physical = cyclic + safety + anticipation
    composite = physical + pipeline
    order_up_to = mu_daily * protection_days + safety + anticipation

    return InventoryResult(
        cyclic=float(cyclic), safety=float(safety), anticipation=float(anticipation),
        pipeline=float(pipeline), physical_reference=float(physical),
        composite_reference=float(composite), order_up_to=float(order_up_to),
        avg_monthly_demand=avg_monthly, cv_monthly=float(cv_monthly),
        capacity_monthly=float(capacity_monthly),
    )
