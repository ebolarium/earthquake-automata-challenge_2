"""Frozen multi-expert forecast used by the evidence-gated prospective test."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import erf

from etas_challenge.residual_emergence import bounded_background_mixture


@dataclass(frozen=True, slots=True)
class SeismicFrame:
    center_latitude: float
    center_longitude: float
    along_vector: tuple[float, float]
    cross_vector: tuple[float, float]
    along_scale_km: float
    cross_scale_km: float
    median_gap_seconds: float


@dataclass(frozen=True, slots=True)
class EvidenceGateForecast:
    etas_rates: np.ndarray
    safe_rates: np.ndarray
    fixed_rates: np.ndarray
    gated_rates: np.ndarray
    gate_weight: float
    log_bayes_factor: float
    active_support: np.ndarray


def _gelu(values: np.ndarray) -> np.ndarray:
    return 0.5 * values * (1.0 + erf(values / np.sqrt(2.0)))


def _linear(values: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return values @ weight.T + bias


def _layer_norm(
    values: np.ndarray, weight: np.ndarray, bias: np.ndarray, epsilon: float = 1e-5
) -> np.ndarray:
    mean = np.mean(values, axis=-1, keepdims=True)
    variance = np.mean((values - mean) ** 2, axis=-1, keepdims=True)
    return (values - mean) / np.sqrt(variance + epsilon) * weight + bias


class NumpyFastSlowEnsemble:
    """Inference-only lossless NumPy port of the frozen CH014 PyTorch model."""

    def __init__(self, archive) -> None:
        self.member_count = int(archive["member_count"])
        self.context_events = int(archive["context_events"])
        self.recent_events = int(archive["recent_events"])
        self.slow_group_size = int(archive["slow_group_size"])
        self.members = []
        for index in range(self.member_count):
            prefix = f"member_{index}_"
            self.members.append({
                name: np.asarray(archive[prefix + name], dtype=np.float64)
                for name in (
                    "auxiliary_mean", "auxiliary_scale", "auxiliary_clip",
                    "coordinate_clip", "pair_encoder_0_weight",
                    "pair_encoder_0_bias", "pair_encoder_2_weight",
                    "pair_encoder_2_bias", "scorer_0_weight", "scorer_0_bias",
                    "scorer_1_weight", "scorer_1_bias", "scorer_4_weight",
                    "scorer_4_bias",
                )
            })

    @classmethod
    def load(cls, path: Path) -> "NumpyFastSlowEnsemble":
        with np.load(path, allow_pickle=False) as archive:
            return cls(archive)

    def _transform(self, context: np.ndarray, member: dict) -> np.ndarray:
        values = np.asarray(context, dtype=np.float64).copy()
        values[:, [0, 3, 4]] = np.clip(
            (values[:, [0, 3, 4]] - member["auxiliary_mean"])
            / member["auxiliary_scale"],
            -float(member["auxiliary_clip"]),
            float(member["auxiliary_clip"]),
        )
        values[:, 1:3] = np.clip(
            values[:, 1:3],
            -float(member["coordinate_clip"]),
            float(member["coordinate_clip"]),
        )
        return values

    @staticmethod
    def _encode(events: np.ndarray, candidates: np.ndarray, member: dict) -> np.ndarray:
        relative = candidates[:, None, :] - events[None, :, 1:3]
        radius = np.log1p(np.linalg.norm(relative, axis=-1, keepdims=True))
        marks = events[:, [0, 3, 4, 5]]
        marks = np.broadcast_to(marks[None, :, :], (len(candidates), len(events), 4))
        pairs = np.concatenate((relative, radius, marks), axis=-1)
        hidden = _gelu(_linear(
            pairs, member["pair_encoder_0_weight"], member["pair_encoder_0_bias"]
        ))
        hidden = _gelu(_linear(
            hidden, member["pair_encoder_2_weight"], member["pair_encoder_2_bias"]
        ))
        return np.mean(hidden, axis=1)

    def member_logits(
        self, context: np.ndarray, candidates: np.ndarray, member_index: int,
        *, batch_size: int = 512,
    ) -> np.ndarray:
        if np.asarray(context).shape != (self.context_events, 5):
            raise ValueError("forecast context shape changed")
        if np.asarray(candidates).ndim != 2 or np.asarray(candidates).shape[1] != 2:
            raise ValueError("forecast candidate shape changed")
        member = self.members[member_index]
        values = self._transform(context, member)
        slow_count = self.context_events - self.recent_events
        slow = values[:slow_count].reshape(
            -1, self.slow_group_size, values.shape[1]
        )
        slow = np.concatenate(
            (np.mean(slow, axis=1), np.max(slow[..., 4:5], axis=1)), axis=1
        )
        recent = np.concatenate(
            (values[slow_count:], values[slow_count:, 4:5]), axis=1
        )
        output = []
        for start in range(0, len(candidates), batch_size):
            batch = np.asarray(candidates[start : start + batch_size], dtype=np.float64)
            dynamic = self._encode(recent, batch, member) - self._encode(
                slow, batch, member
            )
            normalized = _layer_norm(
                dynamic, member["scorer_0_weight"], member["scorer_0_bias"]
            )
            hidden = _gelu(_linear(
                normalized, member["scorer_1_weight"], member["scorer_1_bias"]
            ))
            output.append(_linear(
                hidden, member["scorer_4_weight"], member["scorer_4_bias"]
            )[:, 0])
        return np.concatenate(output)

    def logits(
        self, context: np.ndarray, candidates: np.ndarray, *, batch_size: int = 512
    ) -> np.ndarray:
        return np.mean([
            self.member_logits(context, candidates, index, batch_size=batch_size)
            for index in range(self.member_count)
        ], axis=0)


def _local_xy_km(
    latitudes: np.ndarray, longitudes: np.ndarray, frame: SeismicFrame
) -> np.ndarray:
    delta_lon = (np.asarray(longitudes) - frame.center_longitude + 180.0) % 360.0 - 180.0
    scale = np.pi * 6371.0088 / 180.0
    return np.column_stack((
        delta_lon * scale * np.cos(np.radians(frame.center_latitude)),
        (np.asarray(latitudes) - frame.center_latitude) * scale,
    ))


def project_locations(
    frame: SeismicFrame, latitudes: np.ndarray, longitudes: np.ndarray
) -> np.ndarray:
    basis = np.column_stack((frame.along_vector, frame.cross_vector))
    values = _local_xy_km(latitudes, longitudes, frame) @ basis
    values[:, 0] /= frame.along_scale_km
    values[:, 1] /= frame.cross_scale_km
    return values


def build_context(
    frame: SeismicFrame,
    times_ns: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    depths_km: np.ndarray,
    magnitudes: np.ndarray,
    *,
    context_events: int = 256,
    completeness_magnitude: float = 2.5,
    maximum_depth_km: float = 100.0,
) -> np.ndarray:
    arrays = [np.asarray(value) for value in (
        times_ns, latitudes, longitudes, depths_km, magnitudes
    )]
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("catalog arrays disagree")
    if len(arrays[0]) < context_events:
        raise ValueError("insufficient catalog history for neural context")
    start = max(0, len(arrays[0]) - context_events - 1)
    selected = [value[start:] for value in arrays]
    projected = project_locations(frame, selected[1], selected[2])
    gaps = np.diff(selected[0], prepend=selected[0][0]) / 1.0e9
    gaps[0] = frame.median_gap_seconds
    result = np.column_stack((
        np.log1p(np.maximum(gaps, 0.0) / frame.median_gap_seconds),
        projected,
        np.clip(selected[3] / maximum_depth_km, 0.0, 1.0),
        selected[4] - completeness_magnitude,
    ))
    result = result[-context_events:]
    if np.any(result[:, 4] < -1e-8) or not np.all(np.isfinite(result)):
        raise ValueError("invalid neural context")
    return result.astype(np.float32)


def reallocate_rates(etas_rates: np.ndarray, logits: np.ndarray) -> np.ndarray:
    baseline = np.asarray(etas_rates, dtype=np.float64)
    values = np.asarray(logits, dtype=np.float64)
    log_weight = np.log(baseline) + values
    weights = np.exp(log_weight - np.max(log_weight))
    result = np.sum(baseline) * weights / np.sum(weights)
    if np.any(result <= 0) or not np.all(np.isfinite(result)):
        raise ValueError("neural expert produced invalid rates")
    return result


def supported_neighbor_safe_rates(
    etas_rates: np.ndarray,
    background_rates: np.ndarray,
    ch008_score: np.ndarray,
    daily_background_roots: np.ndarray,
    daily_root_days: np.ndarray,
    issue_day: int,
    grid,
    *,
    background_mixture_fraction: float,
    acceleration_support_power: float,
    acceleration_additive_mix: float,
    residual_scale: float,
) -> np.ndarray:
    days = np.asarray(daily_root_days, dtype=np.int64)
    roots = np.asarray(daily_background_roots, dtype=np.float64)
    if roots.ndim != 2 or roots.shape[0] != len(days):
        raise ValueError("daily root history disagrees")
    def rolling(window: int) -> np.ndarray:
        return np.sum(roots[(days >= issue_day - window) & (days < issue_day)], axis=0)
    roll7 = rolling(7)
    roll30 = rolling(30)
    expected_neighbor = 30.0 * grid.neighbor_sum(background_rates)
    neighbor = np.maximum(
        np.log((grid.neighbor_sum(roll30) + 1.0) / (expected_neighbor + 1.0)), 0.0
    )
    acceleration = np.maximum(
        np.log(((roll7 + 1.0) / 8.0) / ((roll30 + 1.0) / 31.0)), 0.0
    )
    support = np.power(neighbor, 1.0 - acceleration_support_power) * np.power(
        acceleration, acceleration_support_power
    )
    residual = (1.0 - acceleration_additive_mix) * support + (
        acceleration_additive_mix * acceleration
    )
    adjusted = bounded_background_mixture(
        background_rates,
        np.asarray(ch008_score) + residual_scale * residual,
        background_mixture_fraction,
        1.0,
        4.0,
    )
    result = np.asarray(etas_rates) + adjusted - np.asarray(background_rates)
    if np.any(result <= 0) or not np.isclose(np.sum(result), np.sum(etas_rates), atol=1e-10):
        raise ValueError("safe forecast violates positivity or count conservation")
    return result


def support_restricted_mix(
    safe_rates: np.ndarray,
    expert_rates: np.ndarray,
    etas_rates: np.ndarray,
    *,
    active_quantile: float = 0.99,
    mixture_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    safe = np.asarray(safe_rates, dtype=np.float64)
    expert = np.asarray(expert_rates, dtype=np.float64)
    active = np.asarray(etas_rates) >= np.quantile(etas_rates, active_quantile)
    safe_mass = float(np.sum(safe[active]))
    expert_mass = float(np.sum(expert[active]))
    conditional = safe.copy()
    conditional[active] = expert[active] * safe_mass / expert_mass
    result = safe + mixture_fraction * (conditional - safe)
    if not np.array_equal(result[~active], safe[~active]):
        raise ValueError("inactive support changed")
    if not np.isclose(np.sum(result), np.sum(safe), atol=1e-10):
        raise ValueError("fixed expert changed daily count")
    return result, active


def evidence_hurdle_weight(log_bayes_factor: float, alpha: float = 0.05) -> float:
    excess = float(log_bayes_factor) - float(np.log(1.0 / alpha))
    return float(np.tanh(excess / 2.0)) if excess > 0.0 else 0.0


def assemble_forecast(
    etas_rates: np.ndarray,
    safe_rates: np.ndarray,
    neural_rates: np.ndarray,
    log_bayes_factor: float,
) -> EvidenceGateForecast:
    fixed, active = support_restricted_mix(safe_rates, neural_rates, etas_rates)
    weight = evidence_hurdle_weight(log_bayes_factor)
    gated = np.asarray(safe_rates) + weight * (fixed - np.asarray(safe_rates))
    return EvidenceGateForecast(
        np.asarray(etas_rates), np.asarray(safe_rates), fixed, gated,
        weight, float(log_bayes_factor), active,
    )
