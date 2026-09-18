#!/usr/bin/env python3
"""Publish one deadline-safe daily ETAS and CH-008 forecast per region."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etas_challenge.fern_ch008 import regional_grid  # noqa: E402
from etas_challenge.masked_grid import masked_grid  # noqa: E402
from etas_challenge.object_storage import ObjectStorageConfig  # noqa: E402
from etas_challenge.object_storage import object_key, put_verified_bytes  # noqa: E402
from etas_challenge.prospective_bootstrap import utc_timestamp  # noqa: E402
from etas_challenge.prospective_context import load_california_runtime_context  # noqa: E402
from etas_challenge.prospective_forecast import build_california_artifacts  # noqa: E402
from etas_challenge.prospective_forecast import build_california_evidence_gate_artifacts  # noqa: E402
from etas_challenge.prospective_forecast import build_regional_artifacts  # noqa: E402
from etas_challenge.prospective_forecast import build_regional_evidence_gate_artifacts  # noqa: E402
from etas_challenge.prospective_forecast import canonical_json_bytes  # noqa: E402
from etas_challenge.prospective_forecast import deterministic_npz_bytes  # noqa: E402
from etas_challenge.prospective_forecast import enforce_publication_deadline  # noqa: E402
from etas_challenge.prospective_forecast import forecast_run_id  # noqa: E402
from etas_challenge.prospective_forecast import payload_identity  # noqa: E402
from etas_challenge.prospective_protocol import artifact_lane  # noqa: E402
from etas_challenge.prospective_protocol import configured_protocol_path  # noqa: E402
from etas_challenge.prospective_protocol import validate_protocol  # noqa: E402
from etas_challenge.training_matrix import sha256_file  # noqa: E402
from etas_challenge.evidence_gate_forecast import NumpyFastSlowEnsemble  # noqa: E402


PROTOCOL_PATH = configured_protocol_path(ROOT)
RUNTIME_PATH = ROOT / "configs/prospective/daily-runtime-v1.json"
PARENT_MODEL_PATH = ROOT / "models/ch004-marked-renewal-v1.json"
CH008_MODEL_PATH = ROOT / "models/ch008-boundary-sensitivity-v1.json"
FORECAST_MODULE_PATH = ROOT / "src/etas_challenge/prospective_forecast.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue-time", type=utc_timestamp)
    parser.add_argument("--state-as-of", type=utc_timestamp)
    parser.add_argument("--region", action="append", dest="regions")
    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def schedule_minutes(value: str) -> int:
    hour, minute, second = (int(part) for part in value.split(":"))
    if second or not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError("runtime schedule must use whole UTC minutes")
    return hour * 60 + minute


def read_verified_object(client, storage, key: str, checksum: str, byte_count: int) -> bytes:
    metadata = client.head_object(Bucket=storage.bucket, Key=key)
    if (
        int(metadata["ContentLength"]) != byte_count
        or metadata.get("Metadata", {}).get("sha256") != checksum
    ):
        raise ValueError("state S3 metadata disagrees")
    payload = client.get_object(Bucket=storage.bucket, Key=key)["Body"].read()
    if hashlib.sha256(payload).hexdigest() != checksum:
        raise ValueError("state S3 checksum disagrees")
    return payload


def load_state(connection, client, storage, protocol_id: str, region_id: str, as_of):
    row = connection.execute(
        """
        SELECT state_id, artifact_key, artifact_sha256, artifact_bytes,
               baseline_model_id, challenger_model_id
        FROM prospective.model_states
        WHERE protocol_id = %s AND region_id = %s AND as_of = %s
        """,
        (protocol_id, region_id, as_of),
    ).fetchone()
    if row is None:
        raise ValueError("forecast state is missing")
    payload = read_verified_object(client, storage, row[1], row[2], int(row[3]))
    return {
        "state_id": row[0],
        "payload": payload,
        "baseline_model_id": row[4],
        "challenger_model_id": row[5],
    }


def latest_input_snapshot(
    connection, protocol_id: str, region_id: str, state_as_of, issue_time
) -> int:
    row = connection.execute(
        """
        SELECT snapshot_id
        FROM prospective.catalog_snapshots
        WHERE protocol_id = %s AND region_id = %s
          AND collection_kind = 'rolling'
          AND source_cutoff_at >= %s
          AND source_cutoff_at <= %s
          AND captured_at <= %s
        ORDER BY source_cutoff_at DESC, captured_at DESC, snapshot_id DESC
        LIMIT 1
        """,
        (protocol_id, region_id, state_as_of, issue_time, issue_time),
    ).fetchone()
    if row is None:
        raise ValueError("no catalog snapshot was captured before issue time")
    return int(row[0])


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


def artifact_record(storage, client, suffix: str, payload: bytes, content_type: str) -> dict:
    checksum = payload_identity(payload)
    extension = "json" if content_type == "application/json" else "npz"
    key = object_key(storage, f"{suffix}-{checksum}.{extension}")
    put_verified_bytes(storage, key, payload, content_type, client)
    return {"object_key": key, "content_sha256": checksum, "byte_count": len(payload)}


def persist_run(connection, run: dict, artifacts: list[dict]) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (run["run_id"],))
        cursor.execute(
            """
            INSERT INTO prospective.forecast_runs
                (run_id, protocol_id, region_id, input_snapshot_id, state_id,
                 issue_time, target_start, target_end, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'running')
            ON CONFLICT (protocol_id, region_id, target_start) DO NOTHING
            """,
            (
                run["run_id"], run["protocol_id"], run["region_id"],
                run["input_snapshot_id"], run["state_id"], run["issue_time"],
                run["target_start"], run["target_end"],
            ),
        )
        cursor.execute(
            """
            SELECT run_id, input_snapshot_id, state_id, issue_time, target_end, status
            FROM prospective.forecast_runs
            WHERE protocol_id = %s AND region_id = %s AND target_start = %s
            """,
            (run["protocol_id"], run["region_id"], run["target_start"]),
        )
        stored = cursor.fetchone()
        expected = (
            run["run_id"], run["input_snapshot_id"], run["state_id"],
            run["issue_time"], run["target_end"],
        )
        if stored is None or tuple(stored[:5]) != expected:
            raise RuntimeError("stored forecast run disagrees")
        if stored[5] == "published":
            cursor.execute(
                "SELECT count(*) FROM prospective.forecast_artifacts WHERE run_id = %s",
                (run["run_id"],),
            )
            if cursor.fetchone()[0] != len(artifacts):
                raise RuntimeError("published forecast artifact count disagrees")
            return
        for artifact in artifacts:
            cursor.execute(
                """
                INSERT INTO prospective.forecast_artifacts
                    (run_id, model_id, artifact_kind, object_key, content_sha256,
                     byte_count, uploaded_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, model_id, artifact_kind) DO NOTHING
                """,
                (
                    run["run_id"], artifact["model_id"], artifact["artifact_kind"],
                    artifact["object_key"], artifact["content_sha256"],
                    artifact["byte_count"], run["issue_time"],
                ),
            )
            cursor.execute(
                """
                SELECT object_key, content_sha256, byte_count
                FROM prospective.forecast_artifacts
                WHERE run_id = %s AND model_id = %s AND artifact_kind = %s
                """,
                (run["run_id"], artifact["model_id"], artifact["artifact_kind"]),
            )
            if cursor.fetchone() != (
                artifact["object_key"], artifact["content_sha256"], artifact["byte_count"]
            ):
                raise RuntimeError("stored forecast artifact disagrees")
        cursor.execute(
            """
            UPDATE prospective.forecast_runs
            SET status = 'published', finished_at = %s, error_code = NULL
            WHERE run_id = %s
            """,
            (run["published_at"], run["run_id"]),
        )


def published_run(connection, protocol_id: str, region_id: str, target_start) -> dict | None:
    row = connection.execute(
        """
        SELECT run_id, state_id,
               (SELECT count(*) FROM prospective.forecast_artifacts a
                WHERE a.run_id = r.run_id)
        FROM prospective.forecast_runs r
        WHERE protocol_id = %s AND region_id = %s AND target_start = %s
          AND status = 'published'
        """,
        (protocol_id, region_id, target_start),
    ).fetchone()
    if row is None:
        return None
    if row[2] != 6:
        raise RuntimeError("published forecast artifact count disagrees")
    return {"run_id": row[0], "state_id": row[1]}


def main() -> int:
    args = parse_args()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")
    import psycopg

    issue_time = args.issue_time or utc_now()
    state_as_of = args.state_as_of or issue_time.replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    protocol = validate_protocol(PROTOCOL_PATH, ROOT)
    lane = artifact_lane(protocol)
    runtime = json.loads(RUNTIME_PATH.read_text(encoding="utf-8"))
    deadline = schedule_minutes(runtime["issue_schedule"]["forecast_deadline_utc"])
    target_start, target_end = enforce_publication_deadline(
        issue_time,
        state_as_of,
        deadline_minutes=deadline,
        minimum_lead_time_minutes=runtime["issue_schedule"]["minimum_lead_time_minutes"],
    )
    selected = set(args.regions or [region["region_id"] for region in protocol["regions"]])
    known = {region["region_id"] for region in protocol["regions"]}
    if not selected <= known:
        raise SystemExit(f"unknown regions: {', '.join(sorted(selected - known))}")

    parent = json.loads(PARENT_MODEL_PATH.read_text(encoding="utf-8"))
    ch008 = json.loads(CH008_MODEL_PATH.read_text(encoding="utf-8"))
    evidence_model = None
    ensembles = {}
    if protocol.get("forecast_family") == "causal_evidence_gate":
        evidence_path = ROOT / protocol["challenger"]["model_path"]
        evidence_model = json.loads(evidence_path.read_text(encoding="utf-8"))
        region_configs = evidence_model.get("regions")
        if region_configs is None:
            region_configs = {"california-relm": evidence_model["neural_expert"]}
        for region_id, region_config in region_configs.items():
            weights_path = ROOT / region_config["weights_path"]
            if sha256_file(weights_path) != region_config["weights_sha256"]:
                raise ValueError(f"frozen neural weights changed: {region_id}")
            ensembles[region_id] = NumpyFastSlowEnsemble.load(weights_path)
    simulation_reference = json.loads(
        (ROOT / runtime["california_etas_grid"]["simulation_reference"]).read_text(
            encoding="utf-8"
        )
    )
    storage = ObjectStorageConfig.from_environment()
    client = storage.client()
    results = []
    with psycopg.connect(database_url) as connection:
        for region in protocol["regions"]:
            region_id = region["region_id"]
            if region_id not in selected:
                continue
            state = load_state(
                connection, client, storage, protocol["protocol_id"], region_id, state_as_of
            )
            existing = published_run(
                connection, protocol["protocol_id"], region_id, target_start
            )
            if existing is not None:
                if existing["state_id"] != state["state_id"]:
                    raise RuntimeError("published forecast state disagrees")
                result = {
                    "region_id": region_id,
                    "run_id": existing["run_id"],
                    "state_id": existing["state_id"],
                    "target_start": target_start.isoformat(),
                    "artifacts": 6,
                }
                results.append(result)
                print(json.dumps({"status": "already_published", **result}, sort_keys=True), flush=True)
                continue
            snapshot_id = latest_input_snapshot(
                connection, protocol["protocol_id"], region_id, state_as_of, issue_time
            )
            etas_model = json.loads((ROOT / region["etas_model_path"]).read_text(encoding="utf-8"))
            with np.load(io.BytesIO(state["payload"]), allow_pickle=False) as source:
                expected_etas_sha = sha256_file(ROOT / region["etas_model_path"])
                if str(np.asarray(source["as_of"]).item()) != state_as_of.isoformat():
                    raise ValueError("forecast state boundary disagrees")
                if str(np.asarray(source["etas_model_sha256"]).item()) != expected_etas_sha:
                    raise ValueError("forecast state ETAS model hash disagrees")
                if str(np.asarray(source["ch008_model_sha256"]).item()) != sha256_file(CH008_MODEL_PATH):
                    raise ValueError("forecast state CH-008 model hash disagrees")
                if (
                    region_id == "california-relm"
                    and protocol.get("forecast_family") == "causal_evidence_gate"
                ):
                    pair = build_california_evidence_gate_artifacts(
                        source,
                        forecast_start=target_start,
                        context=load_california_runtime_context(ROOT),
                        etas_model=etas_model,
                        parent_model=parent,
                        ch008_model=ch008,
                        simulation_reference=simulation_reference,
                        simulations=runtime["california_etas_grid"]["simulations"],
                        random_seed=runtime["california_etas_grid"]["random_seed"],
                        evidence_model=evidence_model,
                        ensemble=ensembles[region_id],
                    )
                elif region_id == "california-relm":
                    pair = build_california_artifacts(
                        source,
                        forecast_start=target_start,
                        context=load_california_runtime_context(ROOT),
                        etas_model=etas_model,
                        parent_model=parent,
                        ch008_model=ch008,
                        simulation_reference=simulation_reference,
                        simulations=runtime["california_etas_grid"]["simulations"],
                        random_seed=runtime["california_etas_grid"]["random_seed"],
                    )
                elif protocol.get("forecast_family") == "causal_evidence_gate":
                    grid = regional_geometry(region)
                    pair = build_regional_evidence_gate_artifacts(
                        source,
                        forecast_start=target_start,
                        grid=grid,
                        geometry=region["geometry"],
                        etas_model=etas_model,
                        parent_model=parent,
                        ch008_model=ch008,
                        evidence_region=evidence_model["regions"][region_id],
                        ensemble=ensembles[region_id],
                    )
                else:
                    pair = build_regional_artifacts(
                        source,
                        grid=regional_geometry(region),
                        etas_model=etas_model,
                        parent_model=parent,
                        ch008_model=ch008,
                    )
            run_id = forecast_run_id(protocol["protocol_id"], region_id, target_start)
            stem = f"{lane}/forecasts/{target_start.date()}/{region_id}/{run_id}"
            model_payloads = {
                state["baseline_model_id"]: deterministic_npz_bytes(pair.baseline_arrays),
                state["challenger_model_id"]: deterministic_npz_bytes(pair.challenger_arrays),
            }
            artifacts = []
            grid_records = {}
            for model_id, payload in model_payloads.items():
                record = artifact_record(
                    storage, client, f"{stem}/{model_id}/grid", payload,
                    "application/octet-stream",
                )
                grid_records[model_id] = record
                artifacts.append({**record, "model_id": model_id, "artifact_kind": "grid"})
            for model_id in model_payloads:
                summary_payload = canonical_json_bytes({
                    **pair.summary,
                    "schema_version": 1,
                    "protocol_id": protocol["protocol_id"],
                    "region_id": region_id,
                    "model_id": model_id,
                    "target_start": target_start.isoformat(),
                    "target_end": target_end.isoformat(),
                })
                summary_record = artifact_record(
                    storage, client, f"{stem}/{model_id}/summary", summary_payload,
                    "application/json",
                )
                artifacts.append({**summary_record, "model_id": model_id, "artifact_kind": "summary"})
                manifest_payload = canonical_json_bytes({
                    "schema_version": 1,
                    "run_id": run_id,
                    "protocol_id": protocol["protocol_id"],
                    "runtime_id": runtime["runtime_id"],
                    "runtime_sha256": sha256_file(RUNTIME_PATH),
                    "forecast_builder_sha256": sha256_file(Path(__file__)),
                    "forecast_module_sha256": sha256_file(FORECAST_MODULE_PATH),
                    "region_id": region_id,
                    "model_id": model_id,
                    "state_id": state["state_id"],
                    "input_snapshot_id": snapshot_id,
                    "issue_time": issue_time.isoformat(),
                    "target_start": target_start.isoformat(),
                    "target_end": target_end.isoformat(),
                    "history_predicate": "origin_time < state_as_of",
                    "state_as_of": state_as_of.isoformat(),
                    "grid_artifact": grid_records[model_id],
                    "summary_artifact": summary_record,
                })
                manifest_record = artifact_record(
                    storage, client, f"{stem}/{model_id}/manifest", manifest_payload,
                    "application/json",
                )
                artifacts.append({**manifest_record, "model_id": model_id, "artifact_kind": "manifest"})
            published_at = utc_now()
            enforce_publication_deadline(
                published_at,
                state_as_of,
                deadline_minutes=deadline,
                minimum_lead_time_minutes=runtime["issue_schedule"]["minimum_lead_time_minutes"],
            )
            run = {
                "run_id": run_id,
                "protocol_id": protocol["protocol_id"],
                "region_id": region_id,
                "input_snapshot_id": snapshot_id,
                "state_id": state["state_id"],
                "issue_time": issue_time,
                "published_at": published_at,
                "target_start": target_start,
                "target_end": target_end,
            }
            persist_run(connection, run, artifacts)
            connection.commit()
            result = {
                "region_id": region_id,
                "run_id": run_id,
                "state_id": state["state_id"],
                "target_start": target_start.isoformat(),
                "artifacts": len(artifacts),
            }
            results.append(result)
            print(json.dumps({"status": "published", **result}, sort_keys=True), flush=True)
    print(json.dumps({"status": "ok", "issue_time": issue_time.isoformat(), "regions": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
