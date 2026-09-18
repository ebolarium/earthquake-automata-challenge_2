#!/usr/bin/env python3
"""Reuse the verified California causal state without replaying its catalog."""

from __future__ import annotations

import argparse
from datetime import timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from advance_prospective_bootstrap_states import persist_state  # noqa: E402
from etas_challenge.object_storage import ObjectStorageConfig  # noqa: E402
from etas_challenge.object_storage import object_key, put_verified_bytes  # noqa: E402
from etas_challenge.prospective_bootstrap import utc_timestamp  # noqa: E402
from etas_challenge.prospective_protocol import configured_protocol_path  # noqa: E402
from etas_challenge.prospective_protocol import validate_protocol  # noqa: E402
from etas_challenge.prospective_state import model_state_id  # noqa: E402
from etas_challenge.training_matrix import sha256_file  # noqa: E402


SOURCE_PROTOCOL_ID = "evidence-gate-california-dry-run-v1"
REGION_ID = "california-relm"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True, type=utc_timestamp)
    return parser.parse_args()


def verified_object(client, storage, key, checksum, byte_count=None) -> bytes:
    payload = client.get_object(Bucket=storage.bucket, Key=key)["Body"].read()
    if byte_count is not None and len(payload) != int(byte_count):
        raise RuntimeError("source object byte count disagrees")
    if hashlib.sha256(payload).hexdigest() != checksum:
        raise RuntimeError("source object checksum disagrees")
    return payload


def main() -> int:
    args = parse_args()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")
    protocol = validate_protocol(configured_protocol_path(ROOT), ROOT)
    region = next(item for item in protocol["regions"] if item["region_id"] == REGION_ID)
    challenger_sha = sha256_file(ROOT / protocol["challenger"]["model_path"])
    storage = ObjectStorageConfig.from_environment()
    client = storage.client()

    import psycopg

    with psycopg.connect(database_url) as connection:
        row = connection.execute(
            """
            SELECT state_id, source_catalog_cutoff, source_snapshot_ids,
                   artifact_key, artifact_sha256, artifact_bytes,
                   manifest_key, manifest_sha256
            FROM prospective.model_states
            WHERE protocol_id = %s AND region_id = %s AND as_of = %s
            """,
            (SOURCE_PROTOCOL_ID, REGION_ID, args.as_of),
        ).fetchone()
        if row is None:
            raise RuntimeError("verified California source state is missing")
        artifact = verified_object(client, storage, row[3], row[4], row[5])
        source_manifest = json.loads(
            verified_object(client, storage, row[6], row[7]).decode("utf-8")
        )
        etas_sha = sha256_file(ROOT / region["etas_model_path"])
        if source_manifest["baseline_model_sha256"] != etas_sha:
            raise RuntimeError("California source ETAS hash disagrees")
        state_id = model_state_id(
            protocol["protocol_id"], REGION_ID, args.as_of, row[1],
            source_manifest["catalog_history_sha256"], etas_sha, challenger_sha,
        )
        stem = args.as_of.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        artifact_key = object_key(
            storage, f"prospective/states/{REGION_ID}/{stem}-{row[4]}.npz"
        )
        put_verified_bytes(
            storage, artifact_key, artifact, "application/octet-stream", client
        )
        manifest = {
            **source_manifest,
            "schema_version": 2,
            "state_id": state_id,
            "protocol_id": protocol["protocol_id"],
            "challenger_model_id": region["challenger_model_id"],
            "challenger_model_sha256": challenger_sha,
            "state_builder_sha256": sha256_file(Path(__file__)),
            "artifact_key": artifact_key,
            "method": {
                "state_builder": "scripts/transfer_california_state_to_multiregion.py",
                "state_source": "verified_byte_identical_california_transfer",
                "source_protocol_id": SOURCE_PROTOCOL_ID,
                "source_state_id": row[0],
            },
        }
        manifest_bytes = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        manifest_key = object_key(
            storage, f"prospective/states/{REGION_ID}/{stem}-{state_id}.json"
        )
        put_verified_bytes(
            storage, manifest_key, manifest_bytes, "application/json", client
        )
        persist_state(connection, {
            "state_id": state_id,
            "protocol_id": protocol["protocol_id"],
            "region_id": REGION_ID,
            "as_of": args.as_of,
            "catalog_cutoff": row[1],
            "snapshot_ids": list(row[2]),
            "baseline_model_id": region["baseline_model_id"],
            "challenger_model_id": region["challenger_model_id"],
            "artifact_key": artifact_key,
            "artifact_sha256": row[4],
            "artifact_bytes": int(row[5]),
            "manifest_key": manifest_key,
            "manifest_sha256": manifest_sha,
        })
    print(json.dumps({
        "status": "ok", "region_id": REGION_ID, "as_of": args.as_of.isoformat(),
        "state_id": state_id, "artifact_sha256": row[4],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
