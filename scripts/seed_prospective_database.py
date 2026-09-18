#!/usr/bin/env python3
"""Idempotently seed locked models, protocol, and regions."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etas_challenge.prospective_seed import build_seed_records  # noqa: E402
from etas_challenge.prospective_protocol import configured_protocol_path  # noqa: E402


PROTOCOL_PATHS = (configured_protocol_path(ROOT),)


def verify_row(cursor, query: str, parameters: tuple, expected: tuple, label: str) -> None:
    cursor.execute(query, parameters)
    actual = cursor.fetchone()
    if actual != expected:
        raise RuntimeError(f"locked database record disagrees: {label}")


def main() -> int:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")
    import psycopg

    record_sets = [build_seed_records(path, ROOT) for path in PROTOCOL_PATHS]
    frozen_at = datetime.now(timezone.utc)
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("etas-challenge-seed",))
            for model in record_sets[0]["models"]:
                values = (
                    model["model_id"], model["role"], model["source_commit"],
                    model["model_sha256"], model["runtime_sha256"],
                    json.dumps(model["parameters"]), frozen_at,
                )
                cursor.execute(
                    """
                    INSERT INTO prospective.model_versions
                        (model_id, role, source_commit, model_sha256, runtime_sha256, parameters, frozen_at)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                    ON CONFLICT (model_id) DO NOTHING
                    """,
                    values,
                )
                verify_row(
                    cursor,
                    """SELECT role, source_commit, model_sha256, runtime_sha256, parameters
                       FROM prospective.model_versions WHERE model_id = %s""",
                    (model["model_id"],),
                    (model["role"], model["source_commit"], model["model_sha256"], model["runtime_sha256"], model["parameters"]),
                    model["model_id"],
                )
            for records in record_sets:
                protocol = records["protocol"]
                cursor.execute(
                    """
                    INSERT INTO prospective.protocols
                        (protocol_id, status, config, config_sha256, planned_start,
                         planned_days, minimum_events)
                    VALUES (%s, 'draft', %s::jsonb, %s, NULL, %s, %s)
                    ON CONFLICT (protocol_id) DO NOTHING
                    """,
                    (
                        protocol["protocol_id"], json.dumps(protocol),
                        records["protocol_sha256"], protocol["duration_days"],
                        int(protocol.get("minimum_events", 1)),
                    ),
                )
                verify_row(
                    cursor,
                    """SELECT config_sha256, planned_days, minimum_events
                       FROM prospective.protocols WHERE protocol_id = %s""",
                    (protocol["protocol_id"],),
                    (
                        records["protocol_sha256"], protocol["duration_days"],
                        int(protocol.get("minimum_events", 1)),
                    ),
                    protocol["protocol_id"],
                )
                for region in records["regions"]:
                    cursor.execute(
                        """
                        INSERT INTO prospective.regions
                            (region_id, protocol_id, name, catalog_source,
                             catalog_endpoint, geometry, minimum_magnitude,
                             minimum_depth_km, maximum_depth_km, c_region,
                             config_sha256)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s)
                        ON CONFLICT (protocol_id, region_id) DO NOTHING
                        """,
                        (
                            region["region_id"], protocol["protocol_id"],
                            region["name"], region["catalog_source"],
                            region["catalog_endpoint"], json.dumps(region["geometry"]),
                            region["minimum_magnitude"], region["minimum_depth_km"],
                            region["maximum_depth_km"], region["c_region"],
                            region["config_sha256"],
                        ),
                    )
                    verify_row(
                        cursor,
                        """SELECT config_sha256 FROM prospective.regions
                           WHERE protocol_id = %s AND region_id = %s""",
                        (protocol["protocol_id"], region["region_id"]),
                        (region["config_sha256"],),
                        f"{protocol['protocol_id']}:{region['region_id']}",
                    )
    print(json.dumps({
        "status": "ok",
        "protocols": [records["protocol"]["protocol_id"] for records in record_sets],
        "models": len(record_sets[0]["models"]),
        "regions": sum(len(records["regions"]) for records in record_sets),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
