"""Deterministic conditional-Poisson CSEP N, L and R statistics."""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm, poisson


def spatial_counts(event_cells: np.ndarray, cells: int) -> np.ndarray:
    indexes = np.asarray(event_cells, dtype=np.int64)
    if np.any(indexes < 0) or np.any(indexes >= cells):
        raise ValueError("CSEP event cell lies outside forecast support")
    return np.bincount(indexes, minlength=cells).astype(np.int64)


def poisson_log_likelihood(counts: np.ndarray, rates: np.ndarray) -> float:
    observed = np.asarray(counts, dtype=np.int64)
    expected = np.asarray(rates, dtype=np.float64)
    if observed.shape != expected.shape or np.any(observed < 0) or np.any(expected <= 0):
        raise ValueError("invalid CSEP Poisson inputs")
    return float(np.sum(observed * np.log(expected) - expected - gammaln(observed + 1.0)))


def _seed(identity: str, model: str) -> int:
    digest = hashlib.sha256(f"{identity}\n{model}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def model_tests(
    counts: np.ndarray,
    rates: np.ndarray,
    *,
    identity: str,
    model: str,
    simulations: int = 2000,
) -> dict:
    """Return N and L consistency statistics for a gridded Poisson forecast."""

    if simulations < 100:
        raise ValueError("CSEP L-test requires at least 100 simulations")
    observed_count = int(np.sum(counts))
    expected_count = float(np.sum(rates, dtype=np.float64))
    lower = float(poisson.cdf(observed_count, expected_count))
    upper = float(poisson.sf(observed_count - 1, expected_count))
    n_two_sided = min(1.0, 2.0 * min(lower, upper))
    observed_ll = poisson_log_likelihood(counts, rates)
    rng = np.random.default_rng(_seed(identity, model))
    simulated_ll = np.empty(simulations, dtype=np.float64)
    for index in range(simulations):
        draw = rng.poisson(rates)
        simulated_ll[index] = poisson_log_likelihood(draw, rates)
    return {
        "observed_count": observed_count,
        "expected_count": expected_count,
        "n_test_lower_tail": lower,
        "n_test_upper_tail": upper,
        "n_test_two_sided_p": n_two_sided,
        "observed_log_likelihood": observed_ll,
        "l_test_lower_tail_p": float(
            (np.count_nonzero(simulated_ll <= observed_ll) + 1) / (simulations + 1)
        ),
        "l_test_simulated_mean": float(np.mean(simulated_ll)),
        "l_test_simulated_variance": float(np.var(simulated_ll, ddof=1)),
        "simulations": simulations,
    }


def paired_r_test(
    counts: np.ndarray,
    baseline_rates: np.ndarray,
    challenger_rates: np.ndarray,
) -> dict:
    """Paired log-likelihood ratio with an ETAS-null normal reference."""

    baseline = np.asarray(baseline_rates, dtype=np.float64)
    challenger = np.asarray(challenger_rates, dtype=np.float64)
    if baseline.shape != challenger.shape or np.any(baseline <= 0) or np.any(challenger <= 0):
        raise ValueError("invalid paired CSEP forecast rates")
    log_ratio = np.log(challenger / baseline)
    compensator = float(np.sum(challenger - baseline, dtype=np.float64))
    observed = float(np.sum(np.asarray(counts) * log_ratio) - compensator)
    null_mean = float(np.sum(baseline * log_ratio) - compensator)
    null_variance = float(np.sum(baseline * log_ratio * log_ratio))
    z_score = None if null_variance <= 0 else (observed - null_mean) / np.sqrt(null_variance)
    return {
        "observed_log_likelihood_ratio": observed,
        "etas_null_mean": null_mean,
        "etas_null_variance": null_variance,
        "z_score": None if z_score is None else float(z_score),
        "one_sided_p": None if z_score is None else float(norm.sf(z_score)),
        "paired_compensator": compensator,
    }


def csep_daily_report(
    event_cells: np.ndarray,
    baseline_rates: np.ndarray,
    challenger_rates: np.ndarray,
    *,
    identity: str,
    simulations: int = 2000,
) -> dict:
    counts = spatial_counts(event_cells, len(baseline_rates))
    return {
        "method": "conditional_poisson_spatial_csep",
        "magnitude_treatment": "integrated_above_region_threshold",
        "baseline": model_tests(
            counts, baseline_rates, identity=identity, model="etas", simulations=simulations
        ),
        "challenger": model_tests(
            counts, challenger_rates, identity=identity, model="challenger",
            simulations=simulations,
        ),
        "r_test": paired_r_test(counts, baseline_rates, challenger_rates),
    }
