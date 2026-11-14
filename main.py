"""Logit-space random walk quantile forecaster for polymarket-faa-attribution-02."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any, Final, Sequence, TypedDict


class MarketForecast(TypedDict):
    """Output record for a single Polymarket condition."""

    id: str
    q: list[list[float]]


DEFAULT_INPUT_DIR: Final[str] = "/input"
DEFAULT_OUTPUT_DIR: Final[str] = "/output"
DEFAULT_HORIZON: Final[int] = 240
LOOKBACK_MINUTES: Final[int] = 240
MIN_SAMPLE_LEN: Final[int] = 10
SHRINKAGE_WEIGHT: Final[float] = 600.0
HORIZON_POWER: Final[float] = 0.45
INNER_SCALE: Final[float] = 0.40
OUTER_SCALE: Final[float] = 0.60
TAIL_OFFSET: Final[float] = 0.006
PRICE_CLAMP_EPS: Final[float] = 1e-4
VAR_FLOOR: Final[float] = 1e-12
NORMAL_Z_SCORES: Final[tuple[float, float, float, float, float]] = (
    -1.644854,
    -0.67449,
    0.0,
    0.67449,
    1.644854,
)


def clamp_unit(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Restrict a floating-point value to the closed interval [lo, hi]."""
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


def to_logit(prob: float) -> float:
    """Transform a probability in (0, 1) into log-odds space."""
    bounded = clamp_unit(prob, PRICE_CLAMP_EPS, 1.0 - PRICE_CLAMP_EPS)
    return math.log(bounded / (1.0 - bounded))


def sigmoid(eta: float) -> float:
    """Map a log-odds value back to probability space (0, 1)."""
    return 1.0 / (1.0 + math.exp(-eta))


def estimate_logit_sigma(series: Sequence[float]) -> float:
    """Compute sample standard deviation of up to 240 one-minute logit changes."""
    logit_tail = [to_logit(float(p)) for p in series[-LOOKBACK_MINUTES:]]
    if len(logit_tail) < MIN_SAMPLE_LEN:
        return 0.0
    deltas = [
        logit_tail[i] - logit_tail[i - 1] for i in range(1, len(logit_tail))
    ]
    mean_delta = sum(deltas) / len(deltas)
    sample_var = sum((d - mean_delta) ** 2 for d in deltas) / (len(deltas) - 1)
    return math.sqrt(max(sample_var, VAR_FLOOR))


REVERSION_WEIGHT: float = 0.30
REVERSION_TAU: float = 240.0
REVERSION_LOOKBACK: int = 720
MIX_WEIGHT: float = 0.20
LAPLACE_SCALE: float = 0.15


def laplace_multiplier(tau: float) -> float:
    """Return the standard Laplace quantile multiplier for level tau in (0, 1)."""
    diff = tau - 0.5
    if diff == 0.0:
        return 0.0
    sign = 1.0 if diff > 0.0 else -1.0
    return sign * (-math.log(1.0 - 2.0 * abs(diff)))


def build_market_quantiles(
    prices: Sequence[float],
    case_median_sigma: float,
    horizon_minutes: int = DEFAULT_HORIZON,
) -> list[list[float]]:
    """Build 240 rows of five non-decreasing quantiles in [0, 1] for one market."""
    n_obs = float(len(prices))
    raw_sigma = estimate_logit_sigma(prices)
    shrunk_var = (
        n_obs * raw_sigma * raw_sigma
        + SHRINKAGE_WEIGHT * case_median_sigma * case_median_sigma
    ) / (n_obs + SHRINKAGE_WEIGHT)
    sigma = math.sqrt(shrunk_var)
    eta = to_logit(float(prices[-1]))
    long_tail = [to_logit(float(p)) for p in prices[-REVERSION_LOOKBACK:]]
    long_mean = sum(long_tail) / len(long_tail) if long_tail else eta

    rows: list[list[float]] = []
    taus = (0.05, 0.25, 0.5, 0.75, 0.95)
    for h in range(1, horizon_minutes + 1):
        step_spread = sigma * (h ** HORIZON_POWER)
        pull = (1.0 - math.exp(-h / REVERSION_TAU)) * REVERSION_WEIGHT
        centre = eta + pull * (long_mean - eta)
        row: list[float] = []
        prev_q = 0.0
        for z, tau in zip(NORMAL_Z_SCORES, taus):
            k = INNER_SCALE if abs(z) < 1.0 else OUTER_SCALE
            q = sigmoid(centre + z * step_spread * k)
            if abs(z) >= 1.0:
                q += TAIL_OFFSET if z > 0.0 else -TAIL_OFFSET
            q_laplace = sigmoid(centre + LAPLACE_SCALE * laplace_multiplier(tau))
            q = (1.0 - MIX_WEIGHT) * q + MIX_WEIGHT * q_laplace
            q = clamp_unit(q, 0.0, 1.0)
            if q < prev_q:
                q = prev_q
            row.append(q)
            prev_q = q
        rows.append(row)
    return rows


def forecast_case(
    case_payload: dict[str, Any],
) -> dict[str, list[MarketForecast]]:
    """Produce forecast output dictionary for all markets in a single case."""
    horizon = int(case_payload.get("horizon_minutes", DEFAULT_HORIZON))
    markets: list[dict[str, Any]] = list(case_payload["markets"])
    market_sigmas = [estimate_logit_sigma(m["prices"]) for m in markets]
    median_sigma = float(median(market_sigmas)) if market_sigmas else 0.0

    forecasts: list[MarketForecast] = [
        {
            "id": str(market["id"]),
            "q": build_market_quantiles(
                market["prices"], median_sigma, horizon
            ),
        }
        for market in markets
    ]
    return {"forecasts": forecasts}


def main() -> None:
    """Read every *.in in MAKEFUN_INPUT_DIR in sorted order and write <case>.out."""
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", DEFAULT_INPUT_DIR))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)

    for in_path in sorted(in_dir.glob("*.in")):
        payload: dict[str, Any] = json.loads(in_path.read_text(encoding="utf-8"))
        result = forecast_case(payload)
        out_path = out_dir / f"{in_path.stem}.out"
        out_path.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
