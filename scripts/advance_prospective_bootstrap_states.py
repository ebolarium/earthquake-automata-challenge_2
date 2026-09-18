#!/usr/bin/env python3
"""Causally advance initial prospective states within the verified bootstrap."""

from __future__ import annotations

import argparse
from datetime import timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etas_challenge.fern_ch008 import regional_grid  # noqa: E402
from etas_challenge.masked_grid import masked_grid  # noqa: E402
from etas_challenge.object_storage import ObjectStorageConfig  # noqa: E402
from etas_challenge.object_storage import object_key, put_verified_bytes  # noqa: E402
from etas_challenge.prospective_bootstrap import auxiliary_start, utc_timestamp  # noqa: E402
from etas_challenge.prospective_bootstrap import validate_contiguous_windows  # noqa: E402
from etas_challenge.prospective_context import load_california_runtime_context  # noqa: E402
from etas_challenge.prospective_daily import advance_california_day  # noqa: E402
from etas_challenge.prospective_daily import advance_regional_day  # noqa: E402
from etas_challenge.prospective_daily import incremental_event_rates  # noqa: E402
from etas_challenge.prospective_etas import california_daily_etas_grid  # noqa: E402
from etas_challenge.prospective_protocol import validate_protocol  # noqa: E402
from etas_challenge.prospective_protocol import configured_protocol_path  # noqa: E402
from etas_challenge.prospective_protocol import artifact_lane  # noqa: E402
from etas_challenge.prospective_replay import CH008State  # noqa: E402
from etas_challenge.prospective_state import catalog_history_sha256  # noqa: E402
from etas_challenge.prospective_state import load_bootstrap_catalog  # noqa: E402
from etas_challenge.prospective_state import model_state_id  # noqa: E402
from etas_challenge.prospective_state import selected_bootstrap_snapshots  # noqa: E402
from etas_challenge.prospective_state import filter_catalog  # noqa: E402
from etas_challenge.training_matrix import sha256_file, write_deterministic_npz  # noqa: E402


PROTOCOL_PATH = configured_protocol_path(ROOT)
RUNTIME_PATH = ROOT / "configs/prospective/daily-runtime-v1.json"
PARENT_MODEL_PATH = ROOT / "models/ch004-marked-renewal-v1.json"
CH008_MODEL_PATH = ROOT / "models/ch008-boundary-sensitivity-v1.json"
REPLAY_MODULE_PATH = ROOT / "src/etas_challenge/prospective_replay.py"
DAY_NS = 86_400 * 1_000_000_000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-as-of", type=utc_timestamp, required=True)
    parser.add_argument("--to-as-of", type=utc_timestamp, required=True)
    parser.add_argument("--catalog-cutoff", type=utc_timestamp, required=True)
    parser.add_argument("--region", action="append", dest="regions")
    parser.add_argument("--california-simulations", type=int)
    return parser.parse_args()


def require_midnight(value, name: str) -> None:
    if any((value.hour, value.minute, value.second, value.microsecond)):
        raise SystemExit(f"{name} must be a UTC midnight boundary")


def read_verified_object(client, storage, key: str, checksum: str, byte_count: int) -> bytes:
    metadata = client.head_object(Bucket=storage.bucket, Key=key)
    if (
        int(metadata["ContentLength"]) != byte_count
        or metadata.get("Metadata", {}).get("sha256") != checksum
    ):
        raise ValueError("source state S3 metadata disagrees")
    payload = client.get_object(Bucket=storage.bucket, Key=key)["Body"].read()
    if hashlib.sha256(payload).hexdigest() != checksum:
        raise ValueError("source state S3 checksum disagrees")
    return payload


def load_source_state(connection, client, storage, protocol_id: str, region_id: str, as_of):
    row = connection.execute(
        """
        SELECT state_id, artifact_key, artifact_sha256, artifact_bytes
        FROM prospective.model_states
        WHERE protocol_id = %s AND region_id = %s AND as_of = %s
        """,
        (protocol_id, region_id, as_of),
    ).fetchone()
    if row is None:
        raise ValueError("source model state is missing")
    payload = read_verified_object(client, storage, row[1], row[2], int(row[3]))
    return row[0], payload


def regional_geometry(region: dict):
    geometry = region["geometry"]
    if region["region_id"] == "new-zealand-csep":
        with np.load(ROOT / geometry["path"], allow_pickle=False) as archive:
            origins = archive["origins"].copy()
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


def verify_source_prefix(source, catalog, from_as_of) -> int:
    count = int(np.searchsorted(catalog.origin_time_ns, int(from_as_of.timestamp() * 1e9)))
    names = (
        "event_ids", "origin_time_ns", "latitudes", "longitudes", "depths_km", "magnitudes"
    )
    for name in names:
        if not np.array_equal(np.asarray(source[name]), getattr(catalog, name)[:count]):
            raise ValueError(f"source state catalog prefix disagrees: {name}")
    return count


def advance_regional(
    source,
    catalog,
    prefix_count: int,
    region: dict,
    etas_model: dict,
    parent: dict,
    ch008: dict,
    from_as_of,
    to_as_of,
):
    grid = regional_geometry(region)
    background = 10.0 ** etas_model["parameters"]["log10_mu"] * grid.areas_km2
    new_slice = slice(prefix_count, len(catalog.event_ids))
    new_rates = incremental_event_rates(
        catalog.origin_time_ns[:prefix_count] / DAY_NS,
        catalog.latitudes[:prefix_count],
        catalog.longitudes[:prefix_count],
        catalog.magnitudes[:prefix_count],
        catalog.origin_time_ns[new_slice] / DAY_NS,
        catalog.latitudes[new_slice],
        catalog.longitudes[new_slice],
        catalog.magnitudes[new_slice],
        magnitude_reference=etas_model["magnitude_reference"],
        parameters=etas_model["parameters"],
    )
    rates = np.concatenate((np.asarray(source["event_etas_rates"]), new_rates))
    mu = 10.0 ** etas_model["parameters"]["log10_mu"]
    probabilities = np.concatenate(
        (
            np.asarray(source["event_background_probabilities"]),
            np.divide(mu, new_rates, out=np.zeros_like(new_rates), where=new_rates > 0),
        )
    )
    cells = np.concatenate(
        (
            np.asarray(source["event_cells"]),
            grid.cells(catalog.latitudes[new_slice], catalog.longitudes[new_slice]),
        )
    ).astype(np.int32)
    state = CH008State(
        np.asarray(source["ch008_age"]).copy(),
        np.asarray(source["ch008_exposure"]).copy(),
        np.asarray(source["ch008_roots"]).copy(),
    )
    event_days = catalog.origin_time_ns // DAY_NS
    day = from_as_of
    while day < to_as_of:
        number = int(day.timestamp() // 86400)
        selected = np.flatnonzero(event_days == number)
        state = advance_regional_day(
            state,
            event_cells=cells[selected],
            event_magnitudes=catalog.magnitudes[selected],
            event_background_probabilities=probabilities[selected],
            background_mass=background,
            beta=etas_model["beta"],
            magnitude_reference=etas_model["magnitude_reference"],
            renewal_scale=region["renewal_exposure_scale"],
            parent_parameters=parent["parameters"],
            ch008_parameters=ch008["parameters"],
        )
        day += timedelta(days=1)
    arrays = {
        **common_arrays(catalog),
        "event_etas_rates": rates,
        "event_background_probabilities": probabilities,
        "event_cells": cells,
        "ch008_age": state.age,
        "ch008_exposure": state.exposure,
        "ch008_roots": state.roots,
    }
    if "gate_log_bayes_factor" in source:
        arrays["gate_log_bayes_factor"] = np.asarray(
            source["gate_log_bayes_factor"], dtype=np.float64
        )
    return arrays


def advance_california(
    source,
    catalog,
    region: dict,
    etas_model: dict,
    parent: dict,
    ch008: dict,
    runtime: dict,
    from_as_of,
    to_as_of,
    simulations: int,
):
    context = load_california_runtime_context(ROOT)
    reference = json.loads(
        (ROOT / runtime["california_etas_grid"]["simulation_reference"]).read_text(
            encoding="utf-8"
        )
    )
    state = CH008State(
        np.asarray(source["ch008_age"]).copy(),
        np.asarray(source["ch008_exposure"]).copy(),
        np.asarray(source["ch008_roots"]).copy(),
    )
    evidence_gate = "background_root_days" in source
    root_days = (
        list(np.asarray(source["background_root_days"], dtype=np.int64))
        if evidence_gate else []
    )
    root_values = (
        [row.copy() for row in np.asarray(source["background_root_values"], dtype=np.float64)]
        if evidence_gate else []
    )
    event_days = catalog.origin_time_ns // DAY_NS
    day = from_as_of
    while day < to_as_of:
        number = int(day.timestamp() // 86400)
        selected = np.flatnonzero(event_days == number)
        if len(selected):
            issue = np.datetime64(day.astimezone(timezone.utc).replace(tzinfo=None), "ns")
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
                simulations=simulations,
                random_seed=runtime["california_etas_grid"]["random_seed"],
                earth_radius_km=reference["earth_radius_km"],
                max_events_per_catalog=reference["max_events_per_catalog"],
            )
            cells = context.grid.cell_indexes(
                catalog.longitudes[selected], catalog.latitudes[selected]
            )
            probabilities = np.clip(
                context.baseline_background_grid[cells] / baseline.rates[cells],
                0.0,
                1.0,
            )
            geometries = context.event_geometries(
                catalog.latitudes[selected], catalog.longitudes[selected]
            )
        else:
            probabilities = np.array([])
            geometries = tuple(
                type(geometry)(
                    np.empty((0, geometry.section_indexes.shape[1]), dtype=np.int32),
                    np.empty((0, geometry.probabilities.shape[1]), dtype=float),
                )
                for geometry in context.grid_geometries
            )
        if evidence_gate:
            today_roots = np.zeros(context.grid.num_cells, dtype=np.float64)
            if len(selected):
                np.add.at(today_roots, cells, probabilities)
            root_days.append(number)
            root_values.append(today_roots)
        state = advance_california_day(
            state,
            event_geometries=list(geometries),
            event_magnitudes=catalog.magnitudes[selected],
            event_background_probabilities=probabilities,
            expected_section_background=context.expected_section_background,
            beta=context.beta,
            magnitude_reference=context.magnitude_reference,
            parent_parameters=parent["parameters"],
            ch008_parameters=ch008["parameters"],
        )
        day += timedelta(days=1)
    arrays = {
        **common_arrays(catalog),
        "ch008_age": state.age,
        "ch008_exposure": state.exposure,
        "ch008_roots": state.roots,
    }
    if evidence_gate:
        keep = np.asarray(root_days) >= int(to_as_of.timestamp() // 86400) - 90
        arrays.update({
            "background_root_days": np.asarray(root_days, dtype=np.int64)[keep],
            "background_root_values": np.asarray(root_values, dtype=np.float64)[keep],
            "gate_log_bayes_factor": np.asarray(source["gate_log_bayes_factor"], dtype=np.float64),
        })
    return arrays


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
    stored = connection.execute(
        """
        SELECT state_id, artifact_sha256, manifest_sha256
        FROM prospective.model_states
        WHERE protocol_id = %s AND region_id = %s AND as_of = %s
        """,
        (record["protocol_id"], record["region_id"], record["as_of"]),
    ).fetchone()
    if stored != (record["state_id"], record["artifact_sha256"], record["manifest_sha256"]):
        raise RuntimeError("stored advanced model state disagrees")


def main() -> int:
    args = parse_args()
    require_midnight(args.from_as_of, "--from-as-of")
    require_midnight(args.to_as_of, "--to-as-of")
    if not args.from_as_of < args.to_as_of <= args.catalog_cutoff:
        raise SystemExit("require from-as-of < to-as-of <= catalog-cutoff")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")
    import psycopg

    protocol = validate_protocol(PROTOCOL_PATH, ROOT)
    lane = artifact_lane(protocol)
    runtime = json.loads(RUNTIME_PATH.read_text(encoding="utf-8"))
    selected = set(args.regions or [region["region_id"] for region in protocol["regions"]])
    known = {region["region_id"] for region in protocol["regions"]}
    if not selected <= known:
        raise SystemExit(f"unknown regions: {', '.join(sorted(selected - known))}")
    simulations = args.california_simulations or runtime["california_etas_grid"]["simulations"]
    if simulations != runtime["california_etas_grid"]["simulations"]:
        raise SystemExit("California bootstrap advance requires the locked simulation count")
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
            source_state_id, source_payload = load_source_state(
                connection, client, storage, protocol["protocol_id"],
                region["region_id"], args.from_as_of,
            )
            start = auxiliary_start(region, ROOT)
            snapshots = selected_bootstrap_snapshots(
                connection, protocol["protocol_id"], region["region_id"], start,
                args.catalog_cutoff,
            )
            validate_contiguous_windows(
                [(item["start"], item["cutoff"]) for item in snapshots],
                start,
                args.catalog_cutoff,
            )
            snapshot_ids = [item["snapshot_id"] for item in snapshots]
            full_catalog = load_bootstrap_catalog(connection, snapshot_ids, args.to_as_of)
            catalog = filter_catalog(full_catalog, region["minimum_magnitude"])
            with np.load(io.BytesIO(source_payload), allow_pickle=False) as source:
                prefix_count = verify_source_prefix(source, catalog, args.from_as_of)
                etas_path = ROOT / region["etas_model_path"]
                etas_model = json.loads(etas_path.read_text(encoding="utf-8"))
                if region["region_id"] == "california-relm":
                    arrays = advance_california(
                        source, catalog, region, etas_model, parent, ch008, runtime,
                        args.from_as_of, args.to_as_of, simulations,
                    )
                else:
                    arrays = advance_regional(
                        source, catalog, prefix_count, region, etas_model, parent,
                        ch008, args.from_as_of, args.to_as_of,
                    )
                    if region.get("catalog_minimum_magnitude", region["minimum_magnitude"]) < region["minimum_magnitude"]:
                        arrays.update({
                            "context_event_ids": full_catalog.event_ids,
                            "context_origin_time_ns": full_catalog.origin_time_ns,
                            "context_latitudes": full_catalog.latitudes,
                            "context_longitudes": full_catalog.longitudes,
                            "context_depths_km": full_catalog.depths_km,
                            "context_magnitudes": full_catalog.magnitudes,
                        })
            etas_sha = sha256_file(etas_path)
            catalog_sha = catalog_history_sha256(catalog)
            state_id = model_state_id(
                protocol["protocol_id"], region["region_id"], args.to_as_of,
                args.catalog_cutoff, catalog_sha, etas_sha, challenger_sha,
            )
            arrays.update(
                {
                    "as_of": np.asarray(args.to_as_of.isoformat()),
                    "catalog_cutoff": np.asarray(args.catalog_cutoff.isoformat()),
                    "etas_model_sha256": np.asarray(etas_sha),
                    "ch008_model_sha256": np.asarray(ch008_sha),
                }
            )
            artifact = deterministic_npz_bytes(arrays)
            artifact_sha = hashlib.sha256(artifact).hexdigest()
            stem = args.to_as_of.strftime("%Y%m%dT%H%M%SZ")
            artifact_key = object_key(
                storage, f"{lane}/states/{region['region_id']}/{stem}-{artifact_sha}.npz"
            )
            put_verified_bytes(storage, artifact_key, artifact, "application/octet-stream", client)
            manifest = {
                "schema_version": 1,
                "state_id": state_id,
                "protocol_id": protocol["protocol_id"],
                "region_id": region["region_id"],
                "as_of": args.to_as_of.isoformat(),
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
                "method": {
                    "state_builder": "scripts/advance_prospective_bootstrap_states.py",
                    "state_source": "causal_verified_bootstrap_advance",
                    "parent_state_id": source_state_id,
                    "days_advanced": (args.to_as_of - args.from_as_of).days,
                    "california_simulations_per_event_day": simulations,
                },
            }
            manifest_bytes = json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
            manifest_key = object_key(
                storage, f"{lane}/states/{region['region_id']}/{stem}-{state_id}.json"
            )
            put_verified_bytes(storage, manifest_key, manifest_bytes, "application/json", client)
            record = {**manifest, "manifest_key": manifest_key, "manifest_sha256": manifest_sha}
            persist_state(connection, record)
            connection.commit()
            result = {
                "region_id": region["region_id"],
                "state_id": state_id,
                "events": len(catalog.event_ids),
                "new_events": len(catalog.event_ids) - prefix_count,
                "artifact_bytes": len(artifact),
                "artifact_sha256": artifact_sha,
            }
            results.append(result)
            print(json.dumps({"status": "advanced", **result}, sort_keys=True), flush=True)
    print(json.dumps({"status": "ok", "as_of": args.to_as_of.isoformat(), "regions": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
