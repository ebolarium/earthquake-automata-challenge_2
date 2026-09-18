#!/usr/bin/env python3
"""Import restartable historical FDSN catalog windows for prospective models."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etas_challenge.object_storage import ObjectStorageConfig  # noqa: E402
from etas_challenge.object_storage import object_key, put_verified_bytes  # noqa: E402
from etas_challenge.prospective_bootstrap import auxiliary_start  # noqa: E402
from etas_challenge.prospective_bootstrap import calendar_year_windows, utc_timestamp  # noqa: E402
from etas_challenge.prospective_catalog import CatalogResultLimitError  # noqa: E402
from etas_challenge.prospective_catalog import fetch_snapshot  # noqa: E402
from etas_challenge.prospective_persistence import existing_window, persist_snapshot  # noqa: E402
from etas_challenge.prospective_protocol import validate_protocol  # noqa: E402
from etas_challenge.prospective_protocol import configured_protocol_path  # noqa: E402


PROTOCOL_PATH = configured_protocol_path(ROOT)
MINIMUM_SPLIT = timedelta(hours=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=utc_timestamp, required=True)
    parser.add_argument("--region", action="append", dest="regions")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=float(os.environ.get("BOOTSTRAP_REQUEST_DELAY_SECONDS", "0.2")),
    )
    return parser.parse_args()


def stored_window(
    database_url: str, protocol_id: str, region_id: str, start: datetime, cutoff: datetime
):
    import psycopg

    with psycopg.connect(database_url) as connection:
        return existing_window(
            connection, protocol_id, region_id, start, cutoff, "bootstrap"
        )


def store_snapshot(
    database_url: str, protocol_id: str, snapshot, artifact_key: str
) -> int:
    import psycopg

    with psycopg.connect(database_url) as connection:
        return persist_snapshot(
            connection,
            protocol_id,
            snapshot,
            artifact_key,
            datetime.now(timezone.utc),
            collection_kind="bootstrap",
        )


def record_incident(database_url: str, protocol_id: str, region_id: str, error: Exception) -> None:
    import psycopg

    with psycopg.connect(database_url) as connection:
        connection.execute(
            """
            INSERT INTO prospective.incidents
                (protocol_id, region_id, severity, incident_type, message, details, occurred_at)
            VALUES (%s, %s, 'critical', 'catalog_bootstrap_failed', %s, %s::jsonb, %s)
            """,
            (
                protocol_id,
                region_id,
                str(error)[:1000],
                json.dumps({"error_type": type(error).__name__}),
                datetime.now(timezone.utc),
            ),
        )


def artifact_suffix(snapshot) -> str:
    start = snapshot.start.strftime("%Y%m%dT%H%M%S%fZ")
    cutoff = snapshot.cutoff.strftime("%Y%m%dT%H%M%S%fZ")
    return (
        f"dry-run/bootstrap/catalogs/{snapshot.region_id}/"
        f"{start}-{cutoff}-{snapshot.content_sha256}.txt"
    )


def collect_window(
    database_url: str,
    storage: ObjectStorageConfig,
    client,
    protocol_id: str,
    region: dict,
    start: datetime,
    cutoff: datetime,
    refresh: bool,
    delay_seconds: float,
) -> dict[str, int]:
    existing = None if refresh else stored_window(
        database_url, protocol_id, region["region_id"], start, cutoff
    )
    if existing is not None:
        snapshot_id, event_count = existing
        print(
            json.dumps(
                {
                    "status": "skipped",
                    "region_id": region["region_id"],
                    "start": start.isoformat(),
                    "cutoff": cutoff.isoformat(),
                    "snapshot_id": snapshot_id,
                    "events": event_count,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return {"collected": 0, "skipped": 1, "events": event_count}
    try:
        snapshot = fetch_snapshot(region, ROOT, start, cutoff)
    except CatalogResultLimitError:
        if cutoff - start <= MINIMUM_SPLIT:
            raise RuntimeError("FDSN result limit exceeded inside the minimum one-hour window")
        midpoint = start + (cutoff - start) / 2
        left = collect_window(
            database_url, storage, client, protocol_id, region, start, midpoint,
            refresh, delay_seconds,
        )
        right = collect_window(
            database_url, storage, client, protocol_id, region, midpoint, cutoff,
            refresh, delay_seconds,
        )
        return {key: left[key] + right[key] for key in left}
    key = object_key(storage, artifact_suffix(snapshot))
    put_verified_bytes(storage, key, snapshot.raw_payload, "text/plain; charset=utf-8", client)
    snapshot_id = store_snapshot(database_url, protocol_id, snapshot, key)
    print(
        json.dumps(
            {
                "status": "collected",
                "region_id": region["region_id"],
                "start": start.isoformat(),
                "cutoff": cutoff.isoformat(),
                "snapshot_id": snapshot_id,
                "events": len(snapshot.events),
                "sha256": snapshot.content_sha256,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if delay_seconds:
        time.sleep(delay_seconds)
    return {"collected": 1, "skipped": 0, "events": len(snapshot.events)}


def main() -> int:
    args = parse_args()
    if args.request_delay_seconds < 0:
        raise SystemExit("--request-delay-seconds cannot be negative")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")
    protocol = validate_protocol(PROTOCOL_PATH, ROOT)
    selected = set(args.regions or [region["region_id"] for region in protocol["regions"]])
    known = {region["region_id"] for region in protocol["regions"]}
    if not selected <= known:
        raise SystemExit(f"unknown regions: {', '.join(sorted(selected - known))}")
    storage = ObjectStorageConfig.from_environment()
    if protocol["artifact_contract"]["s3_prefix"] != f"{storage.prefix}/dry-run":
        raise ValueError("configured S3 prefix disagrees with dry-run protocol")
    client = storage.client()
    summaries = []
    failures = []
    for region in protocol["regions"]:
        if region["region_id"] not in selected:
            continue
        start = auxiliary_start(region, ROOT)
        totals = {"collected": 0, "skipped": 0, "events": 0}
        try:
            for window_start, window_end in calendar_year_windows(start, args.cutoff):
                result = collect_window(
                    database_url,
                    storage,
                    client,
                    protocol["protocol_id"],
                    region,
                    window_start,
                    window_end,
                    args.refresh,
                    args.request_delay_seconds,
                )
                totals = {key: totals[key] + result[key] for key in totals}
            summaries.append({"region_id": region["region_id"], **totals})
        except Exception as error:
            failures.append(
                {
                    "region_id": region["region_id"],
                    "error": type(error).__name__,
                    "message": str(error)[:500],
                }
            )
            record_incident(database_url, protocol["protocol_id"], region["region_id"], error)
    print(
        json.dumps(
            {
                "status": "ok" if not failures else "failed",
                "cutoff": args.cutoff.isoformat(),
                "regions": summaries,
                "failures": failures,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
