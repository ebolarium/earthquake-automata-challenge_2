#!/usr/bin/env python3
"""Advance yesterday's checkpoint from the latest rolling catalog snapshot."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from advance_prospective_bootstrap_states import advance_california  # noqa: E402
from advance_prospective_bootstrap_states import advance_regional  # noqa: E402
from advance_prospective_bootstrap_states import deterministic_npz_bytes  # noqa: E402
from advance_prospective_bootstrap_states import persist_state  # noqa: E402
from advance_prospective_bootstrap_states import read_verified_object  # noqa: E402
from advance_prospective_bootstrap_states import verify_source_prefix  # noqa: E402
from etas_challenge.object_storage import ObjectStorageConfig  # noqa: E402
from etas_challenge.object_storage import object_key, put_verified_bytes  # noqa: E402
from etas_challenge.prospective_bootstrap import utc_timestamp  # noqa: E402
from etas_challenge.prospective_protocol import validate_protocol  # noqa: E402
from etas_challenge.prospective_protocol import configured_protocol_path  # noqa: E402
from etas_challenge.prospective_protocol import artifact_lane  # noqa: E402
from etas_challenge.prospective_state import BootstrapCatalog  # noqa: E402
from etas_challenge.prospective_state import catalog_history_sha256  # noqa: E402
from etas_challenge.prospective_state import model_state_id  # noqa: E402
from etas_challenge.prospective_context import load_california_runtime_context  # noqa: E402
from etas_challenge.fern_ch008 import regional_grid  # noqa: E402
from etas_challenge.masked_grid import masked_grid  # noqa: E402
from etas_challenge.training_matrix import sha256_file  # noqa: E402


PROTOCOL_PATH = configured_protocol_path(ROOT)
RUNTIME_PATH = ROOT / "configs/prospective/daily-runtime-v1.json"
PARENT_MODEL_PATH = ROOT / "models/ch004-marked-renewal-v1.json"
CH008_MODEL_PATH = ROOT / "models/ch008-boundary-sensitivity-v1.json"
REPLAY_MODULE_PATH = ROOT / "src/etas_challenge/prospective_replay.py"
ADVANCE_RUNTIME_PATH = ROOT / "scripts/advance_prospective_bootstrap_states.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=utc_timestamp)
    parser.add_argument("--region", action="append", dest="regions")
    return parser.parse_args()


def load_parent(connection, client, storage, protocol_id: str, region_id: str, as_of):
    row = connection.execute(
        """
        SELECT state_id, artifact_key, artifact_sha256, artifact_bytes,
               source_snapshot_ids
        FROM prospective.model_states
        WHERE protocol_id = %s AND region_id = %s AND as_of = %s
        """,
        (protocol_id, region_id, as_of),
    ).fetchone()
    if row is None:
        raise ValueError("previous daily model state is missing")
    payload = read_verified_object(client, storage, row[1], row[2], int(row[3]))
    return row[0], payload, list(row[4])


def append_completed_day(
    connection, source, protocol_id: str, region_id: str, start, end, cutoff,
    minimum_magnitude: float | None = None,
):
    row = connection.execute(
        """
        SELECT snapshot_id
        FROM prospective.catalog_snapshots
        WHERE protocol_id = %s AND region_id = %s
          AND collection_kind = 'rolling'
          AND source_start_at <= %s
          AND source_cutoff_at >= %s
          AND source_cutoff_at <= %s
        ORDER BY source_cutoff_at DESC, captured_at DESC, snapshot_id DESC
        LIMIT 1
        """,
        (protocol_id, region_id, start, end, cutoff),
    ).fetchone()
    if row is None:
        raise ValueError("no rolling snapshot covers the completed UTC day")
    snapshot_id = int(row[0])
    rows = connection.execute(
        """
        SELECT source_event_id, origin_time, latitude, longitude, depth_km, magnitude
        FROM prospective.catalog_event_versions
        WHERE snapshot_id = %s AND origin_time >= %s AND origin_time < %s
        ORDER BY origin_time, source_event_id
        """,
        (snapshot_id, start, end),
    ).fetchall()
    if minimum_magnitude is not None:
        rows = [item for item in rows if float(item[5]) >= minimum_magnitude]
    old_ids = np.asarray(source["event_ids"])
    new_ids = np.asarray([item[0] for item in rows]) if rows else old_ids[:0]
    if len(np.intersect1d(old_ids, new_ids)) or len(np.unique(new_ids)) != len(new_ids):
        raise ValueError("rolling event identities disagree with parent state")
    old_times = np.asarray(source["origin_time_ns"], dtype=np.int64)
    new_times = np.asarray(
        [int(item[1].timestamp() * 1_000_000_000) for item in rows], dtype=np.int64
    )
    if len(old_times) and len(new_times) and new_times[0] < old_times[-1]:
        raise ValueError("rolling events precede parent history")

    def joined(name: str, index: int) -> np.ndarray:
        return np.concatenate(
            (np.asarray(source[name]), np.asarray([item[index] for item in rows], dtype=float))
        )

    return BootstrapCatalog(
        (snapshot_id,),
        np.concatenate((old_ids, new_ids)),
        np.concatenate((old_times, new_times)),
        joined("latitudes", 2),
        joined("longitudes", 3),
        joined("depths_km", 4),
        joined("magnitudes", 5),
    ), snapshot_id


def append_feature_context(connection, source, snapshot_id: int, start, end) -> dict:
    rows = connection.execute(
        """
        SELECT source_event_id, origin_time, latitude, longitude, depth_km, magnitude
        FROM prospective.catalog_event_versions
        WHERE snapshot_id = %s AND origin_time >= %s AND origin_time < %s
        ORDER BY origin_time, source_event_id
        """,
        (snapshot_id, start, end),
    ).fetchall()
    names = (
        "context_event_ids", "context_origin_time_ns", "context_latitudes",
        "context_longitudes", "context_depths_km", "context_magnitudes",
    )
    old_ids = np.asarray(source[names[0]])
    new_ids = np.asarray([row[0] for row in rows]) if rows else old_ids[:0]
    if len(np.intersect1d(old_ids, new_ids)) or len(np.unique(new_ids)) != len(new_ids):
        raise ValueError("feature-context event identities disagree")
    return {
        names[0]: np.concatenate((old_ids, new_ids)),
        names[1]: np.concatenate((
            np.asarray(source[names[1]], dtype=np.int64),
            np.asarray([int(row[1].timestamp() * 1e9) for row in rows], dtype=np.int64),
        )),
        **{
            name: np.concatenate((
                np.asarray(source[name]), np.asarray([row[index] for row in rows], dtype=float)
            ))
            for name, index in zip(names[2:], (2, 3, 4, 5))
        },
    }


def existing_daily_state(connection, protocol_id: str, region_id: str, as_of):
    return connection.execute(
        """
        SELECT state_id, artifact_sha256
        FROM prospective.model_states
        WHERE protocol_id = %s AND region_id = %s AND as_of = %s
        """,
        (protocol_id, region_id, as_of),
    ).fetchone()


def completed_shadow_gain(
    connection, client, storage, protocol_id: str, region_id: str,
    target_start, catalog, region: dict,
) -> float:
    """Score fixed versus safe from the already-published completed-day grid."""

    row = connection.execute(
        """
        SELECT a.object_key, a.content_sha256, a.byte_count
        FROM prospective.forecast_runs r
        JOIN prospective.forecast_artifacts a ON a.run_id = r.run_id
        JOIN prospective.model_states s ON s.state_id = r.state_id
        WHERE r.protocol_id = %s AND r.region_id = %s
          AND r.target_start = %s AND r.status = 'published'
          AND a.model_id = s.challenger_model_id AND a.artifact_kind = 'grid'
        """,
        (protocol_id, region_id, target_start),
    ).fetchone()
    if row is None:
        return 0.0
    payload = read_verified_object(client, storage, row[0], row[1], int(row[2]))
    with np.load(io.BytesIO(payload), allow_pickle=False) as grid_archive:
        if "safe_daily_rates" not in grid_archive.files:
            return 0.0
        safe = grid_archive["safe_daily_rates"]
        fixed = grid_archive["fixed_daily_rates"]
    event_days = catalog.origin_time_ns // (86_400 * 1_000_000_000)
    number = int(target_start.timestamp() // 86400)
    selected = np.flatnonzero(event_days == number)
    if not len(selected):
        return 0.0
    if region_id == "california-relm":
        grid = load_california_runtime_context(ROOT).grid
        cells = grid.cell_indexes(catalog.longitudes[selected], catalog.latitudes[selected])
    elif region_id == "new-zealand-csep":
        geometry = region["geometry"]
        with np.load(ROOT / geometry["path"], allow_pickle=False) as archive:
            grid = masked_grid(
                archive["origins"], geometry["spacing_degrees"],
                geometry["latent_spacing_degrees"],
            )
        cells = grid.cells(catalog.latitudes[selected], catalog.longitudes[selected])
    else:
        geometry = region["geometry"]
        grid = regional_grid(
            tuple(geometry["longitude"]), tuple(geometry["latitude"]),
            geometry["spacing_degrees"],
        )
        cells = grid.cells(catalog.latitudes[selected], catalog.longitudes[selected])
    if np.any(cells < 0):
        raise ValueError("completed event lies outside the forecast grid")
    return float(np.sum(np.log(fixed[cells] / safe[cells])))


def main() -> int:
    args = parse_args()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")
    import psycopg

    cutoff = args.cutoff or datetime.now(timezone.utc)
    to_as_of = cutoff.replace(hour=0, minute=0, second=0, microsecond=0)
    from_as_of = to_as_of - timedelta(days=1)
    protocol = validate_protocol(PROTOCOL_PATH, ROOT)
    lane = artifact_lane(protocol)
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
    simulations = runtime["california_etas_grid"]["simulations"]
    storage = ObjectStorageConfig.from_environment()
    client = storage.client()
    results = []
    with psycopg.connect(database_url) as connection:
        for region in protocol["regions"]:
            region_id = region["region_id"]
            if region_id not in selected:
                continue
            existing = existing_daily_state(
                connection, protocol["protocol_id"], region_id, to_as_of
            )
            if existing is not None:
                result = {
                    "region_id": region_id,
                    "state_id": existing[0],
                    "artifact_sha256": existing[1],
                }
                results.append(result)
                print(json.dumps({"status": "already_advanced", **result}, sort_keys=True), flush=True)
                continue
            parent_id, parent_payload, parent_snapshot_ids = load_parent(
                connection, client, storage, protocol["protocol_id"], region_id,
                from_as_of,
            )
            with np.load(io.BytesIO(parent_payload), allow_pickle=False) as source:
                catalog, rolling_snapshot_id = append_completed_day(
                    connection, source, protocol["protocol_id"], region_id,
                    from_as_of, to_as_of, cutoff,
                    minimum_magnitude=region["minimum_magnitude"],
                )
                prefix_count = verify_source_prefix(source, catalog, from_as_of)
                etas_path = ROOT / region["etas_model_path"]
                etas_model = json.loads(etas_path.read_text(encoding="utf-8"))
                if region_id == "california-relm":
                    arrays = advance_california(
                        source, catalog, region, etas_model, parent, ch008, runtime,
                        from_as_of, to_as_of, simulations,
                    )
                    if protocol.get("forecast_family") == "causal_evidence_gate":
                        arrays["gate_log_bayes_factor"] = np.asarray(
                            float(np.asarray(source["gate_log_bayes_factor"]))
                            + completed_shadow_gain(
                                connection, client, storage, protocol["protocol_id"],
                                region_id, from_as_of, catalog, region,
                            ),
                            dtype=np.float64,
                        )
                else:
                    arrays = advance_regional(
                        source, catalog, prefix_count, region, etas_model, parent,
                        ch008, from_as_of, to_as_of,
                    )
                    if protocol.get("forecast_family") == "causal_evidence_gate":
                        arrays["gate_log_bayes_factor"] = np.asarray(
                            float(np.asarray(source["gate_log_bayes_factor"]))
                            + completed_shadow_gain(
                                connection, client, storage, protocol["protocol_id"],
                                region_id, from_as_of, catalog, region,
                            ),
                            dtype=np.float64,
                        )
                    if "context_event_ids" in source:
                        arrays.update(append_feature_context(
                            connection, source, rolling_snapshot_id, from_as_of, to_as_of
                        ))
            snapshot_ids = list(dict.fromkeys(
                [*parent_snapshot_ids, rolling_snapshot_id]
            ))
            etas_sha = sha256_file(etas_path)
            catalog_sha = catalog_history_sha256(catalog)
            state_id = model_state_id(
                protocol["protocol_id"], region_id, to_as_of, cutoff,
                catalog_sha, etas_sha, challenger_sha,
            )
            arrays.update({
                "as_of": np.asarray(to_as_of.isoformat()),
                "catalog_cutoff": np.asarray(cutoff.isoformat()),
                "etas_model_sha256": np.asarray(etas_sha),
                "ch008_model_sha256": np.asarray(ch008_sha),
            })
            artifact = deterministic_npz_bytes(arrays)
            artifact_sha = hashlib.sha256(artifact).hexdigest()
            stem = to_as_of.strftime("%Y%m%dT%H%M%SZ")
            artifact_key = object_key(
                storage, f"{lane}/states/{region_id}/{stem}-{artifact_sha}.npz"
            )
            put_verified_bytes(
                storage, artifact_key, artifact, "application/octet-stream", client
            )
            manifest = {
                "schema_version": 1,
                "state_id": state_id,
                "protocol_id": protocol["protocol_id"],
                "region_id": region_id,
                "as_of": to_as_of.isoformat(),
                "catalog_cutoff": cutoff.isoformat(),
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
                    "state_builder": "scripts/advance_prospective_daily_states.py",
                    "advance_runtime_sha256": sha256_file(ADVANCE_RUNTIME_PATH),
                    "state_source": "causal_rolling_snapshot_advance",
                    "parent_state_id": parent_id,
                    "days_advanced": 1,
                    "rolling_snapshot_id": rolling_snapshot_id,
                    "california_simulations_per_event_day": simulations,
                },
            }
            manifest_bytes = json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
            manifest_key = object_key(
                storage, f"{lane}/states/{region_id}/{stem}-{state_id}.json"
            )
            put_verified_bytes(
                storage, manifest_key, manifest_bytes, "application/json", client
            )
            persist_state(connection, {
                **manifest,
                "manifest_key": manifest_key,
                "manifest_sha256": manifest_sha,
            })
            connection.commit()
            result = {
                "region_id": region_id,
                "state_id": state_id,
                "events": len(catalog.event_ids),
                "new_events": len(catalog.event_ids) - prefix_count,
                "artifact_sha256": artifact_sha,
            }
            results.append(result)
            print(json.dumps({"status": "advanced", **result}, sort_keys=True), flush=True)
    print(json.dumps({"status": "ok", "as_of": to_as_of.isoformat(), "regions": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
