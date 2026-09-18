#!/usr/bin/env python3
"""Score completed prospective ETAS and CH-008 target days."""

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

from etas_challenge.fern_ch008 import regional_grid  # noqa: E402
from etas_challenge.masked_grid import masked_grid  # noqa: E402
from etas_challenge.object_storage import ObjectStorageConfig  # noqa: E402
from etas_challenge.prospective_bootstrap import utc_timestamp  # noqa: E402
from etas_challenge.prospective_protocol import configured_protocol_path  # noqa: E402
from etas_challenge.prospective_protocol import validate_protocol  # noqa: E402
from etas_challenge.prospective_scoring import score_california_grid  # noqa: E402
from etas_challenge.prospective_scoring import score_regional_events  # noqa: E402
from etas_challenge.training_matrix import GridDefinition  # noqa: E402
from etas_challenge.training_matrix import sha256_file  # noqa: E402


PROTOCOL_PATH = configured_protocol_path(ROOT)
SCORING_MODULE_PATH = ROOT / "src/etas_challenge/prospective_scoring.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", type=utc_timestamp)
    parser.add_argument(
        "--revision", choices=("provisional", "final"), action="append",
        dest="revisions",
    )
    parser.add_argument("--region", action="append", dest="regions")
    return parser.parse_args()


def read_verified_object(client, storage, key: str, checksum: str, byte_count: int) -> bytes:
    metadata = client.head_object(Bucket=storage.bucket, Key=key)
    if (
        int(metadata["ContentLength"]) != int(byte_count)
        or metadata.get("Metadata", {}).get("sha256") != checksum
    ):
        raise ValueError("scoring input S3 metadata disagrees")
    payload = client.get_object(Bucket=storage.bucket, Key=key)["Body"].read()
    if hashlib.sha256(payload).hexdigest() != checksum:
        raise ValueError("scoring input S3 checksum disagrees")
    return payload


def due_runs(connection, protocol_id: str, as_of, region_ids: set[str]):
    rows = connection.execute(
        """
        SELECT run_id, region_id, state_id, target_start, target_end
        FROM prospective.forecast_runs
        WHERE protocol_id = %s
          AND status = 'published'
          AND target_end <= %s
          AND region_id = ANY(%s)
        ORDER BY target_start, region_id
        """,
        (protocol_id, as_of, list(region_ids)),
    ).fetchall()
    return [
        {
            "run_id": row[0],
            "region_id": row[1],
            "state_id": row[2],
            "target_start": row[3],
            "target_end": row[4],
        }
        for row in rows
    ]


def score_exists(connection, protocol_id: str, region: dict, run: dict, revision: str) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM prospective.daily_scores
        WHERE protocol_id = %s AND region_id = %s AND target_date = %s
          AND baseline_model_id = %s AND challenger_model_id = %s
          AND score_revision = %s
        """,
        (
            protocol_id, region["region_id"], run["target_start"].date(),
            region["baseline_model_id"], region["challenger_model_id"], revision,
        ),
    ).fetchone()
    return row is not None


def scoring_snapshot(
    connection, protocol_id: str, run: dict, revision: str, as_of, delay_days: int
):
    threshold = run["target_end"] + (
        timedelta(days=delay_days) if revision == "final" else timedelta(0)
    )
    if threshold > as_of:
        return None
    row = connection.execute(
        """
        SELECT snapshot_id, content_sha256, source_cutoff_at
        FROM prospective.catalog_snapshots
        WHERE protocol_id = %s AND region_id = %s
          AND collection_kind = 'rolling'
          AND source_start_at <= %s
          AND source_cutoff_at >= %s
          AND source_cutoff_at <= %s
          AND captured_at <= %s
        ORDER BY source_cutoff_at, captured_at, snapshot_id
        LIMIT 1
        """,
        (
            protocol_id, run["region_id"], run["target_start"], threshold, as_of, as_of,
        ),
    ).fetchone()
    if row is None:
        raise ValueError(f"no {revision} catalog snapshot covers target day")
    return {
        "snapshot_id": int(row[0]),
        "content_sha256": row[1],
        "source_cutoff_at": row[2],
    }


def target_events(connection, snapshot_id: int, start, end) -> dict[str, np.ndarray]:
    rows = connection.execute(
        """
        SELECT source_event_id, origin_time, latitude, longitude, depth_km, magnitude
        FROM prospective.catalog_event_versions
        WHERE snapshot_id = %s AND origin_time >= %s AND origin_time < %s
        ORDER BY origin_time, source_event_id
        """,
        (snapshot_id, start, end),
    ).fetchall()
    event_ids = np.asarray([row[0] for row in rows])
    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError("scoring snapshot contains duplicate event IDs")
    return {
        "event_ids": event_ids,
        "origin_time_ns": np.asarray(
            [int(row[1].timestamp() * 1_000_000_000) for row in rows],
            dtype=np.int64,
        ),
        "origin_times": [row[1].isoformat() for row in rows],
        "latitudes": np.asarray([row[2] for row in rows], dtype=float),
        "longitudes": np.asarray([row[3] for row in rows], dtype=float),
        "depths_km": np.asarray([row[4] for row in rows], dtype=float),
        "magnitudes": np.asarray([row[5] for row in rows], dtype=float),
    }


def forecast_grids(connection, client, storage, run_id: str, region: dict):
    rows = connection.execute(
        """
        SELECT model_id, object_key, content_sha256, byte_count
        FROM prospective.forecast_artifacts
        WHERE run_id = %s AND artifact_kind = 'grid'
          AND model_id = ANY(%s)
        """,
        (run_id, [region["baseline_model_id"], region["challenger_model_id"]]),
    ).fetchall()
    if len(rows) != 2:
        raise ValueError("forecast grid artifacts are incomplete")
    result = {}
    for model_id, key, checksum, byte_count in rows:
        payload = read_verified_object(client, storage, key, checksum, byte_count)
        result[model_id] = {
            "payload": payload,
            "object_key": key,
            "content_sha256": checksum,
            "byte_count": int(byte_count),
        }
    return result


def state_artifact(connection, client, storage, state_id: str):
    row = connection.execute(
        """
        SELECT artifact_key, artifact_sha256, artifact_bytes, as_of
        FROM prospective.model_states WHERE state_id = %s
        """,
        (state_id,),
    ).fetchone()
    if row is None:
        raise ValueError("forecast state is missing during scoring")
    return read_verified_object(client, storage, row[0], row[1], row[2]), row[3]


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


def compute_score(connection, client, storage, region: dict, run: dict, snapshot: dict):
    events = target_events(
        connection, snapshot["snapshot_id"], run["target_start"], run["target_end"]
    )
    artifacts = forecast_grids(connection, client, storage, run["run_id"], region)
    baseline_record = artifacts[region["baseline_model_id"]]
    challenger_record = artifacts[region["challenger_model_id"]]
    auxiliary = None
    with np.load(io.BytesIO(baseline_record["payload"]), allow_pickle=False) as baseline:
        with np.load(io.BytesIO(challenger_record["payload"]), allow_pickle=False) as challenger:
            if region["region_id"] == "california-relm":
                grid = GridDefinition.load(ROOT / region["geometry"]["path"])
                cells = grid.cell_indexes(events["longitudes"], events["latitudes"])
                score = score_california_grid(
                    cells, baseline["daily_rates"], challenger["daily_rates"]
                )
                if "safe_daily_rates" in challenger.files:
                    rates = {
                        "etas": np.asarray(baseline["daily_rates"])[cells],
                        "safe": np.asarray(challenger["safe_daily_rates"])[cells],
                        "fixed": np.asarray(challenger["fixed_daily_rates"])[cells],
                        "gated": np.asarray(challenger["daily_rates"])[cells],
                    }
                    def comparison(numerator: str, denominator: str) -> dict:
                        gains = np.log(rates[numerator] / rates[denominator])
                        return {
                            "events": int(len(gains)),
                            "total_gain": float(np.sum(gains)),
                            "mean_igpe": None if not len(gains) else float(np.mean(gains)),
                            "event_gains": gains.tolist(),
                        }
                    auxiliary = {
                        "model_event_rates": {
                            name: values.tolist() for name, values in rates.items()
                        },
                        "comparisons": {
                            "safe_vs_etas": comparison("safe", "etas"),
                            "fixed_vs_etas": comparison("fixed", "etas"),
                            "gated_vs_etas": comparison("gated", "etas"),
                            "fixed_vs_safe": comparison("fixed", "safe"),
                            "gated_vs_safe": comparison("gated", "safe"),
                        },
                        "gate_weight": float(np.asarray(challenger["gate_weight"])),
                        "gate_log_bayes_factor": float(
                            np.asarray(challenger["gate_log_bayes_factor"])
                        ),
                        "active_support_cells": int(
                            np.count_nonzero(challenger["active_support"])
                        ),
                    }
            else:
                grid = regional_geometry(region)
                cells = grid.cells(events["latitudes"], events["longitudes"])
                state_payload, state_as_of = state_artifact(
                    connection, client, storage, run["state_id"]
                )
                if state_as_of != run["target_start"] - timedelta(days=1):
                    raise ValueError("forecast state lead-day boundary disagrees")
                etas_model = json.loads(
                    (ROOT / region["etas_model_path"]).read_text(encoding="utf-8")
                )
                with np.load(io.BytesIO(state_payload), allow_pickle=False) as state:
                    score = score_regional_events(
                        history_origin_time_ns=state["origin_time_ns"],
                        history_latitudes=state["latitudes"],
                        history_longitudes=state["longitudes"],
                        history_magnitudes=state["magnitudes"],
                        event_origin_time_ns=events["origin_time_ns"],
                        event_latitudes=events["latitudes"],
                        event_longitudes=events["longitudes"],
                        event_magnitudes=events["magnitudes"],
                        event_cells=cells,
                        cell_areas_km2=baseline["cell_areas_km2"],
                        baseline_background_mass=baseline["direct_background_mass"],
                        challenger_background_mass=challenger["direct_background_mass"],
                        magnitude_reference=etas_model["magnitude_reference"],
                        etas_parameters=etas_model["parameters"],
                    )
    summary = score.summary()
    metrics = {
        "schema_version": 1,
        "run_id": run["run_id"],
        "target_start": run["target_start"].isoformat(),
        "target_end": run["target_end"].isoformat(),
        "catalog_snapshot_sha256": snapshot["content_sha256"],
        "catalog_source_cutoff": snapshot["source_cutoff_at"].isoformat(),
        "baseline_grid_sha256": baseline_record["content_sha256"],
        "challenger_grid_sha256": challenger_record["content_sha256"],
        "scoring_method": (
            "published_daily_grid_event_log_rate_ratio"
            if region["region_id"] == "california-relm"
            else "frozen_history_sequential_etas_plus_published_background_delta"
        ),
        "event_ids": events["event_ids"].tolist(),
        "event_origin_times": events["origin_times"],
        "event_latitudes": events["latitudes"].tolist(),
        "event_longitudes": events["longitudes"].tolist(),
        "event_depths_km": events["depths_km"].tolist(),
        "event_magnitudes": events["magnitudes"].tolist(),
        "baseline_event_rates": score.baseline_rates.tolist(),
        "challenger_event_rates": score.challenger_rates.tolist(),
        "event_log_likelihood_gains": score.event_gains.tolist(),
        **summary,
    }
    if auxiliary is not None:
        metrics["multi_model"] = auxiliary
    return summary, metrics


def persist_score(connection, protocol_id: str, region: dict, run: dict, snapshot: dict, revision: str, summary: dict, metrics: dict):
    key = f"{run['run_id']}:{revision}"
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (key,))
        cursor.execute(
            """
            INSERT INTO prospective.daily_scores
                (protocol_id, region_id, target_date, baseline_model_id,
                 challenger_model_id, catalog_snapshot_id, score_revision,
                 event_count, total_log_likelihood_gain, mean_igpe, metrics, run_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT DO NOTHING
            """,
            (
                protocol_id, region["region_id"], run["target_start"].date(),
                region["baseline_model_id"], region["challenger_model_id"],
                snapshot["snapshot_id"], revision, summary["event_count"],
                summary["total_log_likelihood_gain"], summary["mean_igpe"],
                json.dumps(metrics, sort_keys=True, separators=(",", ":")),
                run["run_id"],
            ),
        )
        cursor.execute(
            """
            SELECT run_id, catalog_snapshot_id, event_count,
                   total_log_likelihood_gain, mean_igpe, metrics
            FROM prospective.daily_scores
            WHERE protocol_id = %s AND region_id = %s AND target_date = %s
              AND baseline_model_id = %s AND challenger_model_id = %s
              AND score_revision = %s
            """,
            (
                protocol_id, region["region_id"], run["target_start"].date(),
                region["baseline_model_id"], region["challenger_model_id"], revision,
            ),
        )
        stored = cursor.fetchone()
        expected = (
            run["run_id"], snapshot["snapshot_id"], summary["event_count"],
            summary["total_log_likelihood_gain"], summary["mean_igpe"], metrics,
        )
        if stored != expected:
            raise RuntimeError("stored prospective score disagrees")


def record_incident(
    connection, protocol_id: str, run: dict, revision: str, error: Exception
) -> None:
    details = {
        "region": run["region_id"],
        "issue_date": (run["target_start"].date() - timedelta(days=1)).isoformat(),
        "failure_class": "scoring_service_outage",
        "cause": f"{type(error).__name__}: {error}"[:1000],
        "attempt_count": 1,
        "affected_event_count": None,
        "resolution": "defer_scoring",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "affected_date_range": [
            run["target_start"].date().isoformat(),
            (run["target_end"].date() - timedelta(days=1)).isoformat(),
        ],
        "score_revision": revision,
    }
    existing = connection.execute(
        """
        SELECT incident_id, COALESCE((details->>'attempt_count')::integer, 0)
        FROM prospective.incidents
        WHERE protocol_id = %s AND region_id = %s AND run_id = %s
          AND incident_type = 'scoring_service_outage' AND resolved_at IS NULL
          AND details->>'score_revision' = %s
        ORDER BY occurred_at DESC LIMIT 1
        """,
        (protocol_id, run["region_id"], run["run_id"], revision),
    ).fetchone()
    if existing is None:
        connection.execute(
            """
            INSERT INTO prospective.incidents
                (protocol_id, region_id, run_id, severity, incident_type,
                 message, details, occurred_at)
            VALUES (%s, %s, %s, 'warning', 'scoring_service_outage',
                    %s, %s::jsonb, %s)
            """,
            (
                protocol_id, run["region_id"], run["run_id"], str(error)[:1000],
                json.dumps(details), datetime.now(timezone.utc),
            ),
        )
    else:
        details["attempt_count"] = int(existing[1]) + 1
        details["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        connection.execute(
            """
            UPDATE prospective.incidents
            SET message = %s, details = %s::jsonb, occurred_at = %s
            WHERE incident_id = %s
            """,
            (
                str(error)[:1000], json.dumps(details),
                datetime.now(timezone.utc), existing[0],
            ),
        )
    connection.execute(
        """
        UPDATE prospective.region_day_operations
        SET scoring_status = 'deferred', failed_stage = 'scoring',
            scoring_attempt_count = scoring_attempt_count + 1,
            failure_cause = %s, updated_at = now()
        WHERE protocol_id = %s AND region_id = %s AND target_date = %s
          AND publication_status = 'published'
        """,
        (
            details["cause"], protocol_id, run["region_id"],
            run["target_start"].date(),
        ),
    )


def record_scoring_recovery(
    connection, protocol_id: str, run: dict, revision: str, event_count: int
) -> None:
    connection.execute(
        """
        UPDATE prospective.incidents
        SET resolved_at = now(),
            details = jsonb_set(
                jsonb_set(details, '{resolution}', '"scored_after_recovery"'),
                '{affected_event_count}', to_jsonb(%s::integer)
            )
        WHERE protocol_id = %s AND region_id = %s AND run_id = %s
          AND incident_type = 'scoring_service_outage' AND resolved_at IS NULL
          AND details->>'score_revision' = %s
        """,
        (event_count, protocol_id, run["region_id"], run["run_id"], revision),
    )
    connection.execute(
        """
        UPDATE prospective.region_day_operations operation
        SET scoring_status = CASE WHEN EXISTS (
                SELECT 1 FROM prospective.incidents incident
                WHERE incident.protocol_id = %s
                  AND incident.region_id = %s
                  AND incident.run_id = %s
                  AND incident.incident_type = 'scoring_service_outage'
                  AND incident.resolved_at IS NULL
            ) THEN 'deferred' ELSE 'scored' END,
            failed_stage = CASE WHEN EXISTS (
                SELECT 1 FROM prospective.incidents incident
                WHERE incident.protocol_id = %s
                  AND incident.region_id = %s
                  AND incident.run_id = %s
                  AND incident.incident_type = 'scoring_service_outage'
                  AND incident.resolved_at IS NULL
            ) THEN 'scoring' ELSE NULL END,
            failure_cause = NULL, affected_event_count = %s, updated_at = now()
        WHERE operation.protocol_id = %s AND operation.region_id = %s
          AND operation.target_date = %s AND operation.publication_status = 'published'
        """,
        (
            protocol_id, run["region_id"], run["run_id"],
            protocol_id, run["region_id"], run["run_id"], event_count,
            protocol_id, run["region_id"], run["target_start"].date(),
        ),
    )


def main() -> int:
    args = parse_args()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")
    import psycopg

    as_of = args.as_of or datetime.now(timezone.utc)
    protocol = validate_protocol(PROTOCOL_PATH, ROOT)
    regions = {region["region_id"]: region for region in protocol["regions"]}
    selected = set(args.regions or regions)
    if not selected <= set(regions):
        raise SystemExit(f"unknown regions: {', '.join(sorted(selected - set(regions)))}")
    revisions = args.revisions or ["provisional", "final"]
    delay_days = protocol["catalog_revision_contract"]["settled_score_delay_days"]
    storage = ObjectStorageConfig.from_environment()
    client = storage.client()
    results = []
    failures = []
    with psycopg.connect(database_url) as connection:
        for run in due_runs(connection, protocol["protocol_id"], as_of, selected):
            region = regions[run["region_id"]]
            for revision in revisions:
                if score_exists(connection, protocol["protocol_id"], region, run, revision):
                    continue
                try:
                    snapshot = scoring_snapshot(
                        connection, protocol["protocol_id"], run, revision,
                        as_of, delay_days,
                    )
                    if snapshot is None:
                        continue
                    summary, metrics = compute_score(
                        connection, client, storage, region, run, snapshot
                    )
                    metrics.update({
                        "score_revision": revision,
                        "scoring_script_sha256": sha256_file(Path(__file__)),
                        "scoring_module_sha256": sha256_file(SCORING_MODULE_PATH),
                    })
                    persist_score(
                        connection, protocol["protocol_id"], region, run, snapshot,
                        revision, summary, metrics,
                    )
                    record_scoring_recovery(
                        connection, protocol["protocol_id"], run, revision,
                        summary["event_count"],
                    )
                    connection.commit()
                    result = {
                        "region_id": run["region_id"],
                        "target_date": run["target_start"].date().isoformat(),
                        "revision": revision,
                        **summary,
                    }
                    results.append(result)
                    print(json.dumps({"status": "scored", **result}, sort_keys=True), flush=True)
                except Exception as error:
                    connection.rollback()
                    record_incident(
                        connection, protocol["protocol_id"], run, revision, error
                    )
                    connection.commit()
                    failures.append({
                        "region_id": run["region_id"],
                        "target_date": run["target_start"].date().isoformat(),
                        "revision": revision,
                        "error": type(error).__name__,
                        "message": str(error)[:500],
                    })
    print(json.dumps({
        "status": "ok" if not failures else "failed",
        "as_of": as_of.isoformat(),
        "scores": results,
        "failures": failures,
    }, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
