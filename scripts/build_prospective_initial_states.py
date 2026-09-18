#!/usr/bin/env python3
"""Build and persist the three-region prospective model checkpoint bundle."""

from __future__ import annotations

import argparse
from datetime import timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etas_challenge.etas_native import event_rates  # noqa: E402
from etas_challenge.fern_ch008 import regional_grid  # noqa: E402
from etas_challenge.masked_grid import masked_grid  # noqa: E402
from etas_challenge.object_storage import ObjectStorageConfig  # noqa: E402
from etas_challenge.object_storage import object_key, put_verified_bytes  # noqa: E402
from etas_challenge.prospective_bootstrap import auxiliary_start, utc_timestamp  # noqa: E402
from etas_challenge.prospective_bootstrap import validate_contiguous_windows  # noqa: E402
from etas_challenge.prospective_protocol import validate_protocol  # noqa: E402
from etas_challenge.prospective_protocol import configured_protocol_path  # noqa: E402
from etas_challenge.prospective_context import load_california_runtime_context  # noqa: E402
from etas_challenge.prospective_etas import california_daily_etas_grid  # noqa: E402
from etas_challenge.prospective_state import load_bootstrap_catalog  # noqa: E402
from etas_challenge.prospective_replay import replay_regional_ch008_state  # noqa: E402
from etas_challenge.prospective_state import catalog_history_sha256  # noqa: E402
from etas_challenge.prospective_state import model_state_id  # noqa: E402
from etas_challenge.prospective_state import selected_bootstrap_snapshots  # noqa: E402
from etas_challenge.training_matrix import sha256_file, write_deterministic_npz  # noqa: E402


PROTOCOL_PATH = configured_protocol_path(ROOT)
RUNTIME_PATH = ROOT / "configs/prospective/daily-runtime-v1.json"
PARENT_MODEL_PATH = ROOT / "models/ch004-marked-renewal-v1.json"
CH008_MODEL_PATH = ROOT / "models/ch008-boundary-sensitivity-v1.json"
CALIFORNIA_SEED = ROOT / "data/production/california-ch008-seed-20260819-v1.npz"
CALIFORNIA_SEED_MANIFEST = ROOT / "data/manifests/california-ch008-seed-20260819-v1.json"
REPLAY_MODULE_PATH = ROOT / "src/etas_challenge/prospective_replay.py"
EPOCH_NS = np.datetime64("1970-01-01T00:00:00", "ns")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", type=utc_timestamp, required=True)
    parser.add_argument("--catalog-cutoff", type=utc_timestamp, required=True)
    parser.add_argument("--region", action="append", dest="regions")
    return parser.parse_args()


def day_number(value) -> int:
    timestamp = np.datetime64(value.astimezone(timezone.utc).replace(tzinfo=None), "D")
    return int((timestamp - EPOCH_NS.astype("datetime64[D]")).astype(int))


def regional_geometry(region: dict):
    geometry = region["geometry"]
    if region["region_id"] == "new-zealand-csep":
        origins = np.load(ROOT / geometry["path"], allow_pickle=False)["origins"]
        return masked_grid(
            origins, geometry["spacing_degrees"], geometry["latent_spacing_degrees"]
        )
    return regional_grid(
        tuple(geometry["longitude"]),
        tuple(geometry["latitude"]),
        geometry["spacing_degrees"],
    )


def common_arrays(catalog) -> dict[str, np.ndarray]:
    return {
        "event_ids": catalog.event_ids,
        "origin_time_ns": catalog.origin_time_ns,
        "latitudes": catalog.latitudes,
        "longitudes": catalog.longitudes,
        "depths_km": catalog.depths_km,
        "magnitudes": catalog.magnitudes,
    }


def california_root_history(catalog, as_of, etas_model: dict, runtime: dict):
    context = load_california_runtime_context(ROOT)
    reference = json.loads(
        (ROOT / runtime["california_etas_grid"]["simulation_reference"]).read_text(
            encoding="utf-8"
        )
    )
    start = as_of - timedelta(days=90)
    days = np.arange(day_number(start), day_number(as_of), dtype=np.int64)
    event_days = catalog.origin_time_ns // (86_400 * 1_000_000_000)
    values = np.zeros((len(days), context.grid.num_cells), dtype=np.float64)
    for row, number in enumerate(days):
        selected = np.flatnonzero(event_days == number)
        if not len(selected):
            continue
        issue = EPOCH_NS + np.timedelta64(int(number), "D")
        baseline = california_daily_etas_grid(
            issue_time=issue,
            history_origin_time_ns=catalog.origin_time_ns,
            history_latitudes=catalog.latitudes,
            history_longitudes=catalog.longitudes,
            history_magnitudes=catalog.magnitudes,
            grid=context.grid,
            background_rates=context.baseline_background_grid,
            polygon_lat_lon=np.asarray(reference["polygon_lat_lon"]),
            area_km2=reference["area_km2"],
            beta=etas_model["beta"],
            magnitude_reference=etas_model["magnitude_reference"],
            magnitude_bin_width=reference["delta_m"],
            parameters=etas_model["parameters"],
            simulations=runtime["california_etas_grid"]["simulations"],
            random_seed=runtime["california_etas_grid"]["random_seed"],
            earth_radius_km=reference["earth_radius_km"],
            max_events_per_catalog=reference["max_events_per_catalog"],
        )
        cells = context.grid.cell_indexes(
            catalog.longitudes[selected], catalog.latitudes[selected]
        )
        probabilities = np.clip(
            context.baseline_background_grid[cells] / baseline.rates[cells], 0.0, 1.0
        )
        np.add.at(values[row], cells, probabilities)
    return days, values


def california_arrays(
    catalog, as_of, etas_model: dict, runtime: dict, evidence_gate: bool
) -> tuple[dict[str, np.ndarray], dict]:
    manifest = json.loads(CALIFORNIA_SEED_MANIFEST.read_text(encoding="utf-8"))
    if manifest["as_of"] != as_of.strftime("%Y-%m-%dT%H:%M:%SZ"):
        raise ValueError("California initial state must use the locked retrospective boundary")
    if sha256_file(CALIFORNIA_SEED) != manifest["output"]["sha256"]:
        raise ValueError("California CH-008 seed hash changed")
    with np.load(CALIFORNIA_SEED, allow_pickle=False) as seed:
        arrays = {
            **common_arrays(catalog),
            "ch008_age": seed["age"].copy(),
            "ch008_exposure": seed["exposure"].copy(),
            "ch008_roots": seed["roots"].copy(),
        }
    if evidence_gate:
        root_days, root_values = california_root_history(
            catalog, as_of, etas_model, runtime
        )
        arrays.update({
            "background_root_days": root_days,
            "background_root_values": root_values,
            "gate_log_bayes_factor": np.asarray(0.0, dtype=np.float64),
        })
    return arrays, {
        "ch008_state_source": manifest["state_id"],
        "ch008_state_source_sha256": manifest["output"]["sha256"],
        "etas_history_source": "prospective_bootstrap_before_as_of",
    }


def regional_arrays(region: dict, catalog, as_of, etas_model: dict, parent: dict, ch008: dict):
    grid = regional_geometry(region)
    times_days = catalog.origin_time_ns.astype(np.float64) / (86_400 * 1_000_000_000)
    rates = event_rates(
        times_days,
        catalog.latitudes,
        catalog.longitudes,
        catalog.magnitudes,
        magnitude_reference=etas_model["magnitude_reference"],
        parameters=etas_model["parameters"],
    )
    mu = 10.0 ** etas_model["parameters"]["log10_mu"]
    probabilities = np.divide(mu, rates, out=np.zeros_like(rates), where=rates > 0)
    cells = grid.cells(catalog.latitudes, catalog.longitudes)
    background_mass = mu * grid.areas_km2
    start = auxiliary_start(region, ROOT)
    state = replay_regional_ch008_state(
        event_days=(catalog.origin_time_ns // (86_400 * 1_000_000_000)).astype(np.int64),
        event_cells=cells,
        event_magnitudes=catalog.magnitudes,
        event_background_probabilities=probabilities,
        background_mass=background_mass,
        beta=etas_model["beta"],
        magnitude_reference=etas_model["magnitude_reference"],
        issue_day_start=day_number(start),
        issue_day_end_exclusive=day_number(as_of),
        renewal_scale=region["renewal_exposure_scale"],
        parent_parameters=parent["parameters"],
        ch008_parameters=ch008["parameters"],
    )
    arrays = {
        **common_arrays(catalog),
        "event_etas_rates": rates,
        "event_background_probabilities": probabilities,
        "event_cells": cells,
        "ch008_age": state.age,
        "ch008_exposure": state.exposure,
        "ch008_roots": state.roots,
    }
    return arrays, {
        "ch008_state_source": "full_native_bootstrap_replay_before_as_of",
        "etas_history_source": "prospective_bootstrap_before_as_of",
    }


def deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.npz"
        write_deterministic_npz(path, arrays)
        return path.read_bytes()


def persist_state(connection, record: dict) -> None:
    connection.execute(
        """
        INSERT INTO prospective.model_states
            (state_id, protocol_id, region_id, as_of, source_catalog_cutoff,
             source_snapshot_ids, baseline_model_id, challenger_model_id,
             artifact_key, artifact_sha256, artifact_bytes, manifest_key,
             manifest_sha256)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (protocol_id, region_id, as_of) DO NOTHING
        """,
        (
            record["state_id"], record["protocol_id"], record["region_id"],
            record["as_of"], record["catalog_cutoff"], record["snapshot_ids"],
            record["baseline_model_id"], record["challenger_model_id"],
            record["artifact_key"], record["artifact_sha256"], record["artifact_bytes"],
            record["manifest_key"], record["manifest_sha256"],
        ),
    )
    row = connection.execute(
        """
        SELECT state_id, artifact_sha256, manifest_sha256
        FROM prospective.model_states
        WHERE protocol_id = %s AND region_id = %s AND as_of = %s
        """,
        (record["protocol_id"], record["region_id"], record["as_of"]),
    ).fetchone()
    if row != (record["state_id"], record["artifact_sha256"], record["manifest_sha256"]):
        raise RuntimeError("stored prospective model state disagrees")


def main() -> int:
    args = parse_args()
    if args.as_of > args.catalog_cutoff:
        raise SystemExit("--as-of cannot exceed --catalog-cutoff")
    if any((args.as_of.hour, args.as_of.minute, args.as_of.second, args.as_of.microsecond)):
        raise SystemExit("--as-of must be a UTC midnight boundary")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")
    import psycopg

    protocol = validate_protocol(PROTOCOL_PATH, ROOT)
    runtime = json.loads(RUNTIME_PATH.read_text(encoding="utf-8"))
    selected = set(args.regions or [region["region_id"] for region in protocol["regions"]])
    known = {region["region_id"] for region in protocol["regions"]}
    if not selected <= known:
        raise SystemExit(f"unknown regions: {', '.join(sorted(selected - known))}")
    parent = json.loads(PARENT_MODEL_PATH.read_text(encoding="utf-8"))
    ch008 = json.loads(CH008_MODEL_PATH.read_text(encoding="utf-8"))
    ch008_sha = sha256_file(CH008_MODEL_PATH)
    challenger_sha = (
        sha256_file(ROOT / protocol["challenger"]["model_path"])
        if protocol.get("forecast_family") == "causal_evidence_gate" else ch008_sha
    )
    storage = ObjectStorageConfig.from_environment()
    client = storage.client()
    results = []
    with psycopg.connect(database_url) as connection:
        for region in protocol["regions"]:
            if region["region_id"] not in selected:
                continue
            start = auxiliary_start(region, ROOT)
            snapshots = selected_bootstrap_snapshots(
                connection, region["region_id"], start, args.catalog_cutoff
            )
            validate_contiguous_windows(
                [(item["start"], item["cutoff"]) for item in snapshots],
                start,
                args.catalog_cutoff,
            )
            snapshot_ids = [item["snapshot_id"] for item in snapshots]
            catalog = load_bootstrap_catalog(connection, snapshot_ids, args.as_of)
            catalog_sha = catalog_history_sha256(catalog)
            etas_path = ROOT / region["etas_model_path"]
            etas_model = json.loads(etas_path.read_text(encoding="utf-8"))
            etas_sha = sha256_file(etas_path)
            state_id = model_state_id(
                protocol["protocol_id"], region["region_id"], args.as_of,
                args.catalog_cutoff, catalog_sha, etas_sha, challenger_sha,
            )
            if region["region_id"] == "california-relm":
                arrays, method = california_arrays(
                    catalog, args.as_of, etas_model, runtime,
                    protocol.get("forecast_family") == "causal_evidence_gate",
                )
            else:
                arrays, method = regional_arrays(
                    region, catalog, args.as_of, etas_model, parent, ch008
                )
            arrays.update(
                {
                    "as_of": np.asarray(args.as_of.isoformat()),
                    "catalog_cutoff": np.asarray(args.catalog_cutoff.isoformat()),
                    "etas_model_sha256": np.asarray(etas_sha),
                    "ch008_model_sha256": np.asarray(ch008_sha),
                }
            )
            artifact = deterministic_npz_bytes(arrays)
            artifact_sha = hashlib.sha256(artifact).hexdigest()
            stem = args.as_of.strftime("%Y%m%dT%H%M%SZ")
            artifact_key = object_key(
                storage, f"dry-run/states/{region['region_id']}/{stem}-{artifact_sha}.npz"
            )
            put_verified_bytes(storage, artifact_key, artifact, "application/octet-stream", client)
            manifest = {
                "schema_version": 1,
                "state_id": state_id,
                "protocol_id": protocol["protocol_id"],
                "region_id": region["region_id"],
                "as_of": args.as_of.isoformat(),
                "catalog_cutoff": args.catalog_cutoff.isoformat(),
                "history_predicate": "origin_time < as_of",
                "snapshot_ids": snapshot_ids,
                "catalog_history_sha256": catalog_sha,
                "events": len(catalog.event_ids),
                "baseline_model_id": region["baseline_model_id"],
                "baseline_model_sha256": etas_sha,
                "challenger_model_id": region["challenger_model_id"],
                "challenger_model_sha256": challenger_sha,
                "state_builder_sha256": sha256_file(Path(__file__)),
                "state_replay_module_sha256": sha256_file(REPLAY_MODULE_PATH),
                "artifact_key": artifact_key,
                "artifact_sha256": artifact_sha,
                "artifact_bytes": len(artifact),
                "method": method,
            }
            manifest_bytes = json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
            manifest_key = object_key(
                storage, f"dry-run/states/{region['region_id']}/{stem}-{state_id}.json"
            )
            put_verified_bytes(storage, manifest_key, manifest_bytes, "application/json", client)
            record = {
                **manifest,
                "manifest_key": manifest_key,
                "manifest_sha256": manifest_sha,
            }
            persist_state(connection, record)
            connection.commit()
            results.append(
                {
                    "region_id": region["region_id"],
                    "state_id": state_id,
                    "events": len(catalog.event_ids),
                    "artifact_bytes": len(artifact),
                    "artifact_sha256": artifact_sha,
                }
            )
            print(json.dumps({"status": "built", **results[-1]}, sort_keys=True), flush=True)
    print(json.dumps({"status": "ok", "as_of": args.as_of.isoformat(), "regions": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
