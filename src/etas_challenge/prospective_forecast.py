"""Deterministic forecast artifacts for the prospective ETAS challenge."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile

import numpy as np

from etas_challenge.prospective_daily import california_background_forecast
from etas_challenge.prospective_daily import regional_background_forecast
from etas_challenge.prospective_etas import california_daily_etas_grid
from etas_challenge.prospective_replay import CH008State
from etas_challenge.training_matrix import write_deterministic_npz
from etas_challenge.evidence_gate_forecast import NumpyFastSlowEnsemble
from etas_challenge.evidence_gate_forecast import SeismicFrame
from etas_challenge.evidence_gate_forecast import assemble_forecast
from etas_challenge.evidence_gate_forecast import build_context
from etas_challenge.evidence_gate_forecast import project_locations
from etas_challenge.evidence_gate_forecast import reallocate_rates
from etas_challenge.evidence_gate_forecast import supported_neighbor_safe_rates


@dataclass(frozen=True, slots=True)
class ForecastArtifactPair:
    baseline_arrays: dict[str, np.ndarray]
    challenger_arrays: dict[str, np.ndarray]
    summary: dict


def canonical_json_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "forecast.npz"
        write_deterministic_npz(path, arrays)
        return path.read_bytes()


def payload_identity(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def forecast_run_id(protocol_id: str, region_id: str, target_start: datetime) -> str:
    value = f"{protocol_id}\n{region_id}\n{target_start.astimezone(timezone.utc).isoformat()}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def forecast_window(state_as_of: datetime) -> tuple[datetime, datetime]:
    boundary = state_as_of.astimezone(timezone.utc)
    if any((boundary.hour, boundary.minute, boundary.second, boundary.microsecond)):
        raise ValueError("state boundary must be UTC midnight")
    target_start = boundary + timedelta(days=1)
    return target_start, target_start + timedelta(days=1)


def enforce_publication_deadline(
    issue_time: datetime,
    state_as_of: datetime,
    *,
    deadline_minutes: int,
    minimum_lead_time_minutes: int,
) -> tuple[datetime, datetime]:
    issue = issue_time.astimezone(timezone.utc)
    boundary = state_as_of.astimezone(timezone.utc)
    target_start, target_end = forecast_window(boundary)
    deadline = boundary + timedelta(minutes=deadline_minutes)
    if issue < boundary:
        raise ValueError("forecast cannot be issued before its state boundary")
    if issue > deadline:
        raise ValueError("forecast publication deadline has passed")
    lead = (target_start - issue).total_seconds() / 60.0
    if lead < minimum_lead_time_minutes:
        raise ValueError("forecast minimum lead time is not satisfied")
    return target_start, target_end


def state_from_archive(source) -> CH008State:
    return CH008State(
        np.asarray(source["ch008_age"], dtype=float),
        np.asarray(source["ch008_exposure"], dtype=float),
        np.asarray(source["ch008_roots"], dtype=float),
    )


def build_regional_artifacts(
    source,
    *,
    grid,
    etas_model: dict,
    parent_model: dict,
    ch008_model: dict,
) -> ForecastArtifactPair:
    state = state_from_archive(source)
    direct_background = 10.0 ** etas_model["parameters"]["log10_mu"] * grid.areas_km2
    forecast = regional_background_forecast(
        state,
        background_mass=direct_background,
        transition=grid.transition,
        parent_parameters=parent_model["parameters"],
        ch008_parameters=ch008_model["parameters"],
    )
    common = {
        "cell_areas_km2": np.asarray(grid.areas_km2),
    }
    if hasattr(grid, "latent_keys"):
        common["latent_keys"] = np.asarray(grid.latent_keys)
    else:
        common["longitude_edges"] = np.asarray(grid.longitude_edges)
        common["latitude_edges"] = np.asarray(grid.latitude_edges)
    baseline = {**common, "direct_background_mass": forecast.baseline_mass}
    challenger = {
        **common,
        "direct_background_mass": forecast.challenger_mass,
        "ch008_score": forecast.score,
    }
    baseline_total = float(np.sum(forecast.baseline_mass, dtype=np.float64))
    challenger_total = float(np.sum(forecast.challenger_mass, dtype=np.float64))
    if not np.isclose(baseline_total, challenger_total, rtol=1e-12, atol=1e-12):
        raise ValueError("regional CH-008 changed direct background mass")
    return ForecastArtifactPair(
        baseline,
        challenger,
        {
            "artifact_semantics": "pre_target_latent_direct_background_mass",
            "cells": len(forecast.baseline_mass),
            "baseline_direct_background_mass": baseline_total,
            "challenger_direct_background_mass": challenger_total,
            "paired_compensator_gain": 0.0,
        },
    )


def build_california_artifacts(
    source,
    *,
    forecast_start: datetime,
    context,
    etas_model: dict,
    parent_model: dict,
    ch008_model: dict,
    simulation_reference: dict,
    simulations: int,
    random_seed: int,
) -> ForecastArtifactPair:
    state = state_from_archive(source)
    baseline = california_daily_etas_grid(
        issue_time=np.datetime64(forecast_start.replace(tzinfo=None), "ns"),
        history_origin_time_ns=np.asarray(source["origin_time_ns"]),
        history_latitudes=np.asarray(source["latitudes"]),
        history_longitudes=np.asarray(source["longitudes"]),
        history_magnitudes=np.asarray(source["magnitudes"]),
        grid=context.grid,
        background_rates=context.baseline_background_grid,
        polygon_lat_lon=np.asarray(simulation_reference["polygon_lat_lon"]),
        area_km2=simulation_reference["area_km2"],
        beta=etas_model["beta"],
        magnitude_reference=etas_model["magnitude_reference"],
        magnitude_bin_width=simulation_reference["delta_m"],
        parameters=etas_model["parameters"],
        simulations=simulations,
        random_seed=random_seed,
        earth_radius_km=simulation_reference["earth_radius_km"],
        max_events_per_catalog=simulation_reference["max_events_per_catalog"],
    )
    background = california_background_forecast(
        state,
        background_grid=context.baseline_background_grid,
        transitions=list(context.transitions),
        grid_geometries=list(context.grid_geometries),
        parent_parameter_values=np.asarray(
            [
                parent_model["parameters"][name]
                for name in (
                    "full_reset_magnitude",
                    "magnitude_exponent",
                    "bpt_aperiodicity",
                    "graph_neighborhood_mix",
                    "minimum_branch_consensus",
                    "background_mixture_fraction",
                    "renewal_sensitivity",
                )
            ]
        ),
        ch008_parameters=ch008_model["parameters"],
        maximum_log_tilt=context.maximum_log_tilt,
    )
    challenger_rates = baseline.rates + background.challenger_mass - background.baseline_mass
    if np.any(~np.isfinite(challenger_rates)) or np.any(challenger_rates <= 0):
        raise ValueError("California CH-008 grid contains invalid rates")
    baseline_total = float(np.sum(baseline.rates, dtype=np.float64))
    challenger_total = float(np.sum(challenger_rates, dtype=np.float64))
    if not np.isclose(baseline_total, challenger_total, rtol=1e-12, atol=1e-12):
        raise ValueError("California CH-008 changed expected event count")
    common = {
        "origin_units": np.asarray(context.grid.origin_units),
        "coordinate_units_per_degree": np.asarray(context.grid.units_per_degree),
    }
    return ForecastArtifactPair(
        {
            **common,
            "daily_rates": baseline.rates,
            "direct_background_rates": background.baseline_mass,
        },
        {
            **common,
            "daily_rates": challenger_rates,
            "direct_background_rates": background.challenger_mass,
            "ch008_score": background.score,
        },
        {
            "artifact_semantics": "one_day_expected_count_per_relm_cell",
            "cells": len(baseline.rates),
            "baseline_expected_count": baseline_total,
            "challenger_expected_count": challenger_total,
            "paired_compensator_gain": 0.0,
            "simulations": baseline.simulations,
            "sampled_nonbackground_events": baseline.sampled_nonbackground_events,
            "sampled_nonbackground_inside": baseline.sampled_nonbackground_inside,
        },
    )


def build_california_evidence_gate_artifacts(
    source,
    *,
    forecast_start: datetime,
    context,
    etas_model: dict,
    parent_model: dict,
    ch008_model: dict,
    simulation_reference: dict,
    simulations: int,
    random_seed: int,
    evidence_model: dict,
    ensemble: NumpyFastSlowEnsemble,
) -> ForecastArtifactPair:
    """Build ETAS, safe, fixed, and gated grids from one causal state."""

    pair = build_california_artifacts(
        source,
        forecast_start=forecast_start,
        context=context,
        etas_model=etas_model,
        parent_model=parent_model,
        ch008_model=ch008_model,
        simulation_reference=simulation_reference,
        simulations=simulations,
        random_seed=random_seed,
    )
    etas_rates = pair.baseline_arrays["daily_rates"]
    safe_config = evidence_model["safe_expert"]
    causal_boundary = forecast_start - timedelta(days=1)
    issue_day = int(
        np.datetime64(causal_boundary.astimezone(timezone.utc).replace(tzinfo=None), "D")
        .astype(np.int64)
    )
    safe_rates = supported_neighbor_safe_rates(
        etas_rates,
        pair.baseline_arrays["direct_background_rates"],
        pair.challenger_arrays["ch008_score"],
        np.asarray(source["background_root_values"]),
        np.asarray(source["background_root_days"]),
        issue_day,
        context.grid,
        background_mixture_fraction=ch008_model["parameters"]["background_mixture_fraction"],
        acceleration_support_power=safe_config["acceleration_support_power"],
        acceleration_additive_mix=safe_config["acceleration_additive_mix"],
        residual_scale=safe_config["residual_scale"],
    )
    frame_config = evidence_model["neural_expert"]["frame"]
    frame = SeismicFrame(
        center_latitude=frame_config["center_latitude"],
        center_longitude=frame_config["center_longitude"],
        along_vector=tuple(frame_config["along_vector"]),
        cross_vector=tuple(frame_config["cross_vector"]),
        along_scale_km=frame_config["along_scale_km"],
        cross_scale_km=frame_config["cross_scale_km"],
        median_gap_seconds=frame_config["median_gap_seconds"],
    )
    history_context = build_context(
        frame,
        np.asarray(source["origin_time_ns"]),
        np.asarray(source["latitudes"]),
        np.asarray(source["longitudes"]),
        np.asarray(source["depths_km"]),
        np.asarray(source["magnitudes"]),
        context_events=int(evidence_model["neural_expert"]["context_events"]),
        completeness_magnitude=frame_config["completeness_magnitude"],
        maximum_depth_km=frame_config["maximum_depth_km"],
    )
    centers = (context.grid.origin_units.astype(np.float64) + 0.5) / context.grid.units_per_degree
    candidates = project_locations(frame, centers[:, 1], centers[:, 0])
    neural_rates = reallocate_rates(
        etas_rates,
        ensemble.logits(
            history_context,
            candidates,
            batch_size=int(evidence_model["neural_expert"]["candidate_batch_size"]),
        ),
    )
    gate = assemble_forecast(
        etas_rates,
        safe_rates,
        neural_rates,
        float(np.asarray(source["gate_log_bayes_factor"])),
    )
    expected = float(np.sum(etas_rates, dtype=np.float64))
    for name, values in (
        ("safe", gate.safe_rates), ("fixed", gate.fixed_rates),
        ("gated", gate.gated_rates), ("neural", neural_rates),
    ):
        if not np.isclose(np.sum(values, dtype=np.float64), expected, rtol=1e-12, atol=1e-10):
            raise ValueError(f"{name} forecast changed expected event count")
    return ForecastArtifactPair(
        pair.baseline_arrays,
        {
            **{key: value for key, value in pair.challenger_arrays.items() if key != "daily_rates"},
            "daily_rates": gate.gated_rates,
            "safe_daily_rates": gate.safe_rates,
            "fixed_daily_rates": gate.fixed_rates,
            "neural_daily_rates": neural_rates,
            "active_support": gate.active_support.astype(np.uint8),
            "gate_weight": np.asarray(gate.gate_weight),
            "gate_log_bayes_factor": np.asarray(gate.log_bayes_factor),
        },
        {
            **pair.summary,
            "model_family": "causal_evidence_gated_spatial_forecast",
            "baseline_expected_count": expected,
            "safe_expected_count": float(np.sum(gate.safe_rates)),
            "fixed_expected_count": float(np.sum(gate.fixed_rates)),
            "gated_expected_count": float(np.sum(gate.gated_rates)),
            "gate_weight": gate.gate_weight,
            "gate_log_bayes_factor": gate.log_bayes_factor,
            "gate_qualified": bool(gate.gate_weight > 0.0),
            "active_support_cells": int(np.count_nonzero(gate.active_support)),
        },
    )
