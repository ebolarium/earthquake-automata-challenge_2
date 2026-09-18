#!/usr/bin/env python3
"""Verify persisted initial ETAS and CH-008 state checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etas_challenge.masked_grid import masked_grid  # noqa: E402
from etas_challenge.object_storage import ObjectStorageConfig  # noqa: E402
from etas_challenge.prospective_bootstrap import utc_timestamp  # noqa: E402
from etas_challenge.prospective_protocol import validate_protocol  # noqa: E402
from etas_challenge.prospective_protocol import configured_protocol_path  # noqa: E402
from etas_challenge.prospective_state import model_state_id  # noqa: E402
from etas_challenge.prospective_state import validate_state_artifact  # noqa: E402
from etas_challenge.training_matrix import sha256_file  # noqa: E402


PROTOCOL_PATH = configured_protocol_path(ROOT)
CH008_MODEL_PATH = ROOT / "models/ch008-boundary-sensitivity-v1.json"
STATE_BUILDER_PATH = ROOT / "scripts/build_prospective_initial_states.py"
REPLAY_MODULE_PATH = ROOT / "src/etas_challenge/prospective_replay.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", type=utc_timestamp, required=True)
    parser.add_argument("--region", action="append", dest="regions")
    return parser.parse_args()


def state_shape(region: dict) -> tuple[int, ...]:
    if region["region_id"] == "california-relm":
        return (8, 350)
    geometry = region["geometry"]
    if region["region_id"] == "new-zealand-csep":
        with np.load(ROOT / geometry["path"], allow_pickle=False) as archive:
            grid = masked_grid(
                archive["origins"],
                geometry["spacing_degrees"],
                geometry["latent_spacing_degrees"],
            )
        return (len(grid.areas_km2),)
    return (int(geometry["cells"]),)


def read_verified_object(client, storage, key: str, expected_sha: str, expected_bytes=None) -> bytes:
    metadata = client.head_object(Bucket=storage.bucket, Key=key)
    if metadata.get("Metadata", {}).get("sha256") != expected_sha:
        raise ValueError(f"S3 checksum metadata disagrees: {key}")
    if expected_bytes is not None and int(metadata["ContentLength"]) != expected_bytes:
        raise ValueError(f"S3 byte count disagrees: {key}")
    payload = client.get_object(Bucket=storage.bucket, Key=key)["Body"].read()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError(f"S3 object checksum disagrees: {key}")
    return payload


def verify_region(connection, client, storage, protocol: dict, region: dict, as_of) -> dict:
    row = connection.execute(
        """
        SELECT state_id, source_catalog_cutoff, source_snapshot_ids,
               baseline_model_id, challenger_model_id, artifact_key,
               artifact_sha256, artifact_bytes, manifest_key, manifest_sha256
        FROM prospective.model_states
        WHERE protocol_id = %s AND region_id = %s AND as_of = %s
        """,
        (protocol["protocol_id"], region["region_id"], as_of),
    ).fetchone()
    if row is None:
        raise ValueError("initial model state is missing")
    (
        stored_state_id, catalog_cutoff, snapshot_ids, baseline_model_id,
        challenger_model_id, artifact_key, artifact_sha, artifact_bytes,
        manifest_key, manifest_sha,
    ) = row
    manifest_bytes = read_verified_object(client, storage, manifest_key, manifest_sha)
    manifest = json.loads(manifest_bytes)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if canonical != manifest_bytes:
        raise ValueError("state manifest is not canonical JSON")
    expected_manifest = {
        "state_id": stored_state_id,
        "protocol_id": protocol["protocol_id"],
        "region_id": region["region_id"],
        "as_of": as_of.isoformat(),
        "catalog_cutoff": catalog_cutoff.isoformat(),
        "snapshot_ids": list(snapshot_ids),
        "baseline_model_id": baseline_model_id,
        "challenger_model_id": challenger_model_id,
        "artifact_key": artifact_key,
        "artifact_sha256": artifact_sha,
        "artifact_bytes": artifact_bytes,
    }
    for name, expected in expected_manifest.items():
        if manifest.get(name) != expected:
            raise ValueError(f"state manifest disagrees with database: {name}")
    if baseline_model_id != region["baseline_model_id"] or challenger_model_id != region["challenger_model_id"]:
        raise ValueError("state model IDs disagree with protocol")
    etas_sha = sha256_file(ROOT / region["etas_model_path"])
    ch008_sha = sha256_file(CH008_MODEL_PATH)
    evidence_gate = protocol.get("forecast_family") == "causal_evidence_gate"
    challenger_sha = (
        sha256_file(ROOT / protocol["challenger"]["model_path"])
        if evidence_gate else ch008_sha
    )
    if (
        manifest["baseline_model_sha256"] != etas_sha
        or manifest["challenger_model_sha256"] != challenger_sha
    ):
        raise ValueError("state model hashes disagree with locked files")
    builder_name = manifest.get("method", {}).get(
        "state_builder", "scripts/build_prospective_initial_states.py"
    )
    admitted_builders = {
        "scripts/build_prospective_initial_states.py",
        "scripts/advance_prospective_bootstrap_states.py",
        "scripts/advance_prospective_daily_states.py",
    }
    if builder_name not in admitted_builders:
        raise ValueError("state builder is not admitted")
    if manifest["state_builder_sha256"] != sha256_file(ROOT / builder_name):
        raise ValueError("state builder hash disagrees")
    if manifest["state_replay_module_sha256"] != sha256_file(REPLAY_MODULE_PATH):
        raise ValueError("state replay hash disagrees")
    expected_state_id = model_state_id(
        protocol["protocol_id"], region["region_id"], as_of, catalog_cutoff,
        manifest["catalog_history_sha256"], etas_sha, challenger_sha,
    )
    if expected_state_id != stored_state_id:
        raise ValueError("state identity disagrees")
    artifact = read_verified_object(
        client, storage, artifact_key, artifact_sha, int(artifact_bytes)
    )
    with np.load(io.BytesIO(artifact), allow_pickle=False) as arrays:
        summary = validate_state_artifact(
            arrays,
            manifest,
            expected_state_shape=state_shape(region),
            regional=region["region_id"] != "california-relm",
            evidence_gate=evidence_gate,
            incumbent_model_sha256=ch008_sha if evidence_gate else None,
        )
    return {
        "region_id": region["region_id"],
        "state_id": stored_state_id,
        "artifact_sha256": artifact_sha,
        **summary,
    }


def main() -> int:
    args = parse_args()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")
    import psycopg

    protocol = validate_protocol(PROTOCOL_PATH, ROOT)
    selected = set(args.regions or [region["region_id"] for region in protocol["regions"]])
    known = {region["region_id"] for region in protocol["regions"]}
    if not selected <= known:
        raise SystemExit(f"unknown regions: {', '.join(sorted(selected - known))}")
    storage = ObjectStorageConfig.from_environment()
    client = storage.client()
    results = []
    failures = []
    with psycopg.connect(database_url) as connection:
        for region in protocol["regions"]:
            if region["region_id"] not in selected:
                continue
            try:
                results.append(
                    verify_region(connection, client, storage, protocol, region, args.as_of)
                )
            except Exception as error:
                failures.append(
                    {
                        "region_id": region["region_id"],
                        "error": type(error).__name__,
                        "message": str(error)[:500],
                    }
                )
    print(
        json.dumps(
            {
                "status": "ok" if not failures else "failed",
                "as_of": args.as_of.isoformat(),
                "regions": results,
                "failures": failures,
            },
            sort_keys=True,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
