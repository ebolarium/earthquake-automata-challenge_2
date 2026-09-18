#!/usr/bin/env python3
"""Run the policy-guarded prospective pipeline with isolated regional retries."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etas_challenge.prospective_bootstrap import utc_timestamp  # noqa: E402
from etas_challenge.prospective_downtime import evaluate_invalidation  # noqa: E402
from etas_challenge.prospective_downtime import publication_deadline  # noqa: E402
from etas_challenge.prospective_downtime import run_with_retries  # noqa: E402
from etas_challenge.prospective_downtime import validate_downtime_policy  # noqa: E402
from etas_challenge.prospective_protocol import artifact_lane  # noqa: E402
from etas_challenge.prospective_protocol import validate_protocol  # noqa: E402
from etas_challenge.prospective_runtime import DRY_RUN_PROTOCOL_PATH  # noqa: E402
from etas_challenge.prospective_runtime import PROSPECTIVE_PROTOCOL_PATH  # noqa: E402
from etas_challenge.object_storage import ObjectStorageConfig  # noqa: E402
from etas_challenge.object_storage import object_key, put_verified_bytes  # noqa: E402


POLICY_PATH = ROOT / "configs/challenge/ch008-downtime-policy.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue-time", type=utc_timestamp)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--region", action="append", dest="regions")
    return parser.parse_args()


def run_command(arguments: list[str], protocol_path: Path) -> None:
    environment = os.environ.copy()
    environment["PROSPECTIVE_PROTOCOL_PATH"] = str(protocol_path)
    completed = subprocess.run(
        arguments, cwd=ROOT, env=environment, text=True, capture_output=True
    )
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.returncode:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(message[-1000:] or f"command exited {completed.returncode}")


def ensure_region_day(connection, protocol_id, region_id, issue_time, target_date):
    connection.execute(
        """
        INSERT INTO prospective.region_day_operations
            (protocol_id, region_id, issue_date, target_date, logical_issue_time)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (protocol_id, region_id, issue_date) DO NOTHING
        """,
        (protocol_id, region_id, issue_time.date(), target_date, issue_time),
    )
    row = connection.execute(
        """
        SELECT logical_issue_time, publication_status
        FROM prospective.region_day_operations
        WHERE protocol_id = %s AND region_id = %s AND issue_date = %s
        """,
        (protocol_id, region_id, issue_time.date()),
    ).fetchone()
    if row is None:
        raise RuntimeError("region-day operation is missing")
    return row[0], row[1]


def ensure_publication_issue_time(
    database_url: str, protocol_id: str, region_id: str, issue_date
) -> datetime:
    import psycopg

    with psycopg.connect(database_url) as connection:
        value = datetime.now(timezone.utc)
        connection.execute(
            """
            UPDATE prospective.region_day_operations
            SET publication_issue_time = COALESCE(publication_issue_time, %s),
                updated_at = now()
            WHERE protocol_id = %s AND region_id = %s AND issue_date = %s
            """,
            (value, protocol_id, region_id, issue_date),
        )
        row = connection.execute(
            """
            SELECT publication_issue_time
            FROM prospective.region_day_operations
            WHERE protocol_id = %s AND region_id = %s AND issue_date = %s
            """,
            (protocol_id, region_id, issue_date),
        ).fetchone()
    if row is None or row[0] is None:
        raise RuntimeError("publication issue time was not persisted")
    return row[0]


def forecast_is_published(
    database_url: str, protocol_id: str, region_id: str, target_date
) -> bool:
    import psycopg

    target_start = datetime.combine(target_date, datetime.min.time(), timezone.utc)
    with psycopg.connect(database_url) as connection:
        row = connection.execute(
            """
            SELECT 1 FROM prospective.forecast_runs
            WHERE protocol_id = %s AND region_id = %s AND target_start = %s
              AND status = 'published'
            """,
            (protocol_id, region_id, target_start),
        ).fetchone()
    return row is not None


def record_terminal_state(
    database_url: str,
    protocol_id: str,
    region_id: str,
    issue_time: datetime,
    target_date,
    *,
    publication_status: str,
    scoring_status: str,
    failed_stage: str | None,
    publication_attempts: int,
    scoring_attempts: int,
    cause: str | None,
    failed_stage_attempts: int | None = None,
) -> None:
    import psycopg

    with psycopg.connect(database_url) as connection:
        stored_issue_time, _ = ensure_region_day(
            connection, protocol_id, region_id, issue_time, target_date
        )
        if stored_issue_time != issue_time:
            raise RuntimeError("region-day logical issue time disagrees")
        connection.execute(
            """
            UPDATE prospective.region_day_operations
            SET publication_status = %s, scoring_status = %s, failed_stage = %s,
                publication_attempt_count = GREATEST(publication_attempt_count, %s),
                scoring_attempt_count = GREATEST(scoring_attempt_count, %s),
                failure_cause = %s, updated_at = now()
            WHERE protocol_id = %s AND region_id = %s AND issue_date = %s
            """,
            (
                publication_status, scoring_status, failed_stage,
                publication_attempts, scoring_attempts, cause,
                protocol_id, region_id, issue_time.date(),
            ),
        )
        if publication_status == "missed":
            details = {
                "region": region_id,
                "issue_date": issue_time.date().isoformat(),
                "failure_class": "publication_missed",
                "cause": cause,
                "attempt_count": failed_stage_attempts or publication_attempts,
                "pipeline_attempt_count": publication_attempts,
                "affected_event_count": None,
                "resolution": "excluded_from_N",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "affected_date_range": [target_date.isoformat(), target_date.isoformat()],
            }
            connection.execute(
                """
                INSERT INTO prospective.incidents
                    (protocol_id, region_id, severity, incident_type, message,
                     details, occurred_at)
                SELECT %s, %s, 'critical', 'publication_missed', %s, %s::jsonb, %s
                WHERE NOT EXISTS (
                    SELECT 1 FROM prospective.incidents
                    WHERE protocol_id = %s AND region_id = %s
                      AND incident_type = 'publication_missed'
                      AND details->>'issue_date' = %s
                )
                """,
                (
                    protocol_id, region_id, cause or "forecast publication missed",
                    json.dumps(details), datetime.now(timezone.utc),
                    protocol_id, region_id, issue_time.date().isoformat(),
                ),
            )
        elif publication_status == "published":
            connection.execute(
                """
                UPDATE prospective.incidents
                SET resolved_at = now(),
                    details = jsonb_set(
                        details, '{operational_recovery}',
                        to_jsonb(%s::text)
                    )
                WHERE protocol_id = %s AND region_id = %s
                  AND incident_type = 'publication_missed'
                  AND resolved_at IS NULL
                """,
                (
                    f"next_successful_publication:{target_date.isoformat()}",
                    protocol_id, region_id,
                ),
            )


def refresh_invalidation(database_url: str, protocol_id: str, region_id: str, policy: dict):
    import psycopg

    with psycopg.connect(database_url) as connection:
        rows = connection.execute(
            """
            SELECT issue_date FROM prospective.region_day_operations
            WHERE protocol_id = %s AND region_id = %s
              AND publication_status = 'missed'
            ORDER BY issue_date
            """,
            (protocol_id, region_id),
        ).fetchall()
        result = evaluate_invalidation([row[0] for row in rows], policy)
        connection.execute(
            """
            INSERT INTO prospective.region_operational_status
                (protocol_id, region_id, primary_eligible, missed_region_days,
                 consecutive_missed_days, invalidated_at, invalidation_reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (protocol_id, region_id) DO UPDATE
            SET primary_eligible = prospective.region_operational_status.primary_eligible
                                   AND EXCLUDED.primary_eligible,
                missed_region_days = EXCLUDED.missed_region_days,
                consecutive_missed_days = EXCLUDED.consecutive_missed_days,
                invalidated_at = COALESCE(
                    prospective.region_operational_status.invalidated_at,
                    EXCLUDED.invalidated_at
                ),
                invalidation_reason = COALESCE(
                    prospective.region_operational_status.invalidation_reason,
                    EXCLUDED.invalidation_reason
                ),
                updated_at = now()
            """,
            (
                protocol_id, region_id, not result.invalid, result.missed_days,
                result.consecutive_missed_days,
                datetime.now(timezone.utc) if result.invalid else None,
                result.reason,
            ),
        )
    return result


def reconcile_missed_event_counts(database_url: str, protocol_id: str) -> int:
    import psycopg

    reconciled = 0
    with psycopg.connect(database_url) as connection:
        rows = connection.execute(
            """
            SELECT region_id, issue_date, target_date
            FROM prospective.region_day_operations
            WHERE protocol_id = %s AND publication_status = 'missed'
              AND affected_event_count IS NULL
              AND target_date < (now() AT TIME ZONE 'UTC')::date
            ORDER BY target_date, region_id
            """,
            (protocol_id,),
        ).fetchall()
        for region_id, issue_date, target_date in rows:
            target_start = datetime.combine(target_date, datetime.min.time(), timezone.utc)
            target_end = target_start + timedelta(days=1)
            snapshot = connection.execute(
                """
                SELECT snapshot_id FROM prospective.catalog_snapshots
                WHERE protocol_id = %s AND region_id = %s
                  AND collection_kind = 'rolling'
                  AND source_start_at <= %s AND source_cutoff_at >= %s
                ORDER BY source_cutoff_at, captured_at, snapshot_id LIMIT 1
                """,
                (protocol_id, region_id, target_start, target_end),
            ).fetchone()
            if snapshot is None:
                continue
            event_count = int(connection.execute(
                """
                SELECT count(*) FROM prospective.catalog_event_versions
                WHERE snapshot_id = %s AND origin_time >= %s AND origin_time < %s
                """,
                (snapshot[0], target_start, target_end),
            ).fetchone()[0])
            connection.execute(
                """
                UPDATE prospective.region_day_operations
                SET affected_event_count = %s, updated_at = now()
                WHERE protocol_id = %s AND region_id = %s AND issue_date = %s
                """,
                (event_count, protocol_id, region_id, issue_date),
            )
            connection.execute(
                """
                UPDATE prospective.incidents
                SET details = jsonb_set(
                    details, '{affected_event_count}', to_jsonb(%s::integer)
                )
                WHERE protocol_id = %s AND region_id = %s
                  AND incident_type = 'publication_missed'
                  AND details->>'issue_date' = %s
                """,
                (event_count, protocol_id, region_id, issue_date.isoformat()),
            )
            reconciled += 1
    return reconciled


def export_incident_log(
    database_url: str, protocol_id: str, issue_time: datetime, lane: str
) -> dict:
    import psycopg

    with psycopg.connect(database_url) as connection:
        rows = connection.execute(
            """
            SELECT incident_id, region_id, run_id, severity, incident_type,
                   message, details, occurred_at, resolved_at
            FROM prospective.incidents
            WHERE protocol_id = %s ORDER BY incident_id
            """,
            (protocol_id,),
        ).fetchall()
    payload = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "exported_at": issue_time.isoformat(),
        "incidents": [
            {
                "incident_id": int(row[0]),
                "region_id": row[1],
                "run_id": row[2],
                "severity": row[3],
                "incident_type": row[4],
                "message": row[5],
                "details": row[6],
                "occurred_at": row[7].isoformat(),
                "resolved_at": None if row[8] is None else row[8].isoformat(),
            }
            for row in rows
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    checksum = hashlib.sha256(encoded).hexdigest()
    storage = ObjectStorageConfig.from_environment()
    key = object_key(
        storage,
        f"{lane}/incidents/{issue_time.strftime('%Y/%m/%d')}/{checksum}.json",
    )
    put_verified_bytes(storage, key, encoded, "application/json", storage.client())
    return {"object_key": key, "sha256": checksum, "incidents": len(rows)}


def run_scoring_cycle(region_id: str, retry: dict, protocol_path: Path):
    command = [
        sys.executable, "scripts/score_prospective_forecasts.py", "--region", region_id,
    ]
    started = datetime.now(timezone.utc)
    deadline = started + timedelta(
        seconds=sum(int(value) for value in retry["backoff_seconds"]) + 60
    )
    return run_with_retries(
        lambda: run_command(command, protocol_path),
        max_attempts=int(retry["max_attempts"]),
        backoff_seconds=retry["backoff_seconds"],
        deadline=deadline,
        now=lambda: datetime.now(timezone.utc),
        sleep=time.sleep,
    )


def execute_region(
    region_id: str,
    issue_time: datetime,
    deadline: datetime,
    policy: dict,
    database_url: str,
    protocol_id: str,
    lookback_days: int,
    protocol_path: Path,
) -> dict:
    python = sys.executable
    target_date = (issue_time.date() + timedelta(days=1))
    retry = policy["retry"]
    attempts = {"publication": 0, "scoring": 0}
    import psycopg

    with psycopg.connect(database_url) as connection:
        issue_time, operation_status = ensure_region_day(
            connection, protocol_id, region_id, issue_time, target_date
        )
    if operation_status == "missed":
        invalidation = refresh_invalidation(
            database_url, protocol_id, region_id, policy
        )
        return {
            "region_id": region_id,
            "status": "publication_missed",
            "failed_stage": "previously_finalized",
            "attempts": attempts,
            "primary_eligible": not invalidation.invalid,
        }
    timestamp = issue_time.isoformat()
    prepublication = [
        (
            "catalog_collection",
            [python, "scripts/collect_prospective_catalogs.py", "--lookback-days",
             str(lookback_days), "--cutoff", timestamp, "--region", region_id],
        ),
        (
            "state_advance",
            [python, "scripts/advance_prospective_daily_states.py", "--cutoff",
             timestamp, "--region", region_id],
        ),
    ]
    already_published = forecast_is_published(
        database_url, protocol_id, region_id, target_date
    )
    if not already_published:
        for stage, command in prepublication:
            result = run_with_retries(
                lambda command=command: run_command(command, protocol_path),
                max_attempts=int(retry["max_attempts"]),
                backoff_seconds=retry["backoff_seconds"],
                deadline=deadline,
                now=lambda: datetime.now(timezone.utc),
                sleep=time.sleep,
            )
            attempts["publication"] += result.attempts
            if not result.succeeded:
                record_terminal_state(
                    database_url, protocol_id, region_id, issue_time, target_date,
                    publication_status="missed", scoring_status="pending",
                    failed_stage=stage, publication_attempts=attempts["publication"],
                    scoring_attempts=0,
                    cause=result.error or "publication deadline reached",
                    failed_stage_attempts=result.attempts,
                )
                invalidation = refresh_invalidation(
                    database_url, protocol_id, region_id, policy
                )
                score = run_scoring_cycle(region_id, retry, protocol_path)
                attempts["scoring"] = score.attempts
                return {
                    "region_id": region_id,
                    "status": "publication_missed",
                    "failed_stage": stage,
                    "attempts": attempts,
                    "primary_eligible": not invalidation.invalid,
                }

        publication_issue_time = ensure_publication_issue_time(
            database_url, protocol_id, region_id, issue_time.date()
        )
        publication_command = [
            python, "scripts/publish_prospective_forecasts.py", "--issue-time",
            publication_issue_time.isoformat(), "--region", region_id,
        ]
        publication = run_with_retries(
            lambda: run_command(publication_command, protocol_path),
            max_attempts=int(retry["max_attempts"]),
            backoff_seconds=retry["backoff_seconds"],
            deadline=deadline,
            now=lambda: datetime.now(timezone.utc),
            sleep=time.sleep,
        )
        attempts["publication"] += publication.attempts
        if not publication.succeeded:
            record_terminal_state(
                database_url, protocol_id, region_id, issue_time, target_date,
                publication_status="missed", scoring_status="pending",
                failed_stage="forecast_publication",
                publication_attempts=attempts["publication"], scoring_attempts=0,
                cause=publication.error or "publication deadline reached",
                failed_stage_attempts=publication.attempts,
            )
            invalidation = refresh_invalidation(
                database_url, protocol_id, region_id, policy
            )
            score = run_scoring_cycle(region_id, retry, protocol_path)
            attempts["scoring"] = score.attempts
            return {
                "region_id": region_id,
                "status": "publication_missed",
                "failed_stage": "forecast_publication",
                "attempts": attempts,
                "primary_eligible": not invalidation.invalid,
            }

    score = run_scoring_cycle(region_id, retry, protocol_path)
    attempts["scoring"] = score.attempts
    record_terminal_state(
        database_url, protocol_id, region_id, issue_time, target_date,
        publication_status="published", scoring_status="pending",
        failed_stage=None,
        publication_attempts=attempts["publication"],
        scoring_attempts=0, cause=None,
    )
    invalidation = refresh_invalidation(database_url, protocol_id, region_id, policy)
    return {
        "region_id": region_id,
        "status": "published" if score.succeeded else "published_scoring_deferred",
        "attempts": attempts,
        "primary_eligible": not invalidation.invalid,
    }


def run_protocol(
    args, database_url: str, issue_time: datetime, relative_protocol_path: Path
) -> int:
    protocol_path = ROOT / relative_protocol_path
    protocol = validate_protocol(protocol_path, ROOT)
    if protocol["mode"] == "dry_run":
        activate_dry_run(database_url, protocol["protocol_id"], issue_time)
    policy = validate_downtime_policy(POLICY_PATH)
    known = [region["region_id"] for region in protocol["regions"]]
    selected = args.regions or known
    if not set(selected) <= set(known):
        raise SystemExit("unknown prospective region")
    deadline = publication_deadline(issue_time, policy)
    results = []
    with ThreadPoolExecutor(max_workers=len(selected)) as executor:
        futures = {
            executor.submit(
                execute_region, region_id, issue_time, deadline, policy,
                database_url, protocol["protocol_id"], args.lookback_days,
                relative_protocol_path,
            ): region_id
            for region_id in selected
        }
        for future in as_completed(futures):
            region_id = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                results.append({
                    "region_id": region_id,
                    "status": "launch_guard_error",
                    "error": f"{type(error).__name__}: {error}"[:1000],
                })
    reconciled = reconcile_missed_event_counts(
        database_url, protocol["protocol_id"]
    )
    incident_export = export_incident_log(
        database_url, protocol["protocol_id"], issue_time, artifact_lane(protocol)
    )
    results.sort(key=lambda item: item["region_id"])
    failed = any(
        item["status"] in {"publication_missed", "launch_guard_error"}
        for item in results
    )
    if not failed and protocol["mode"] == "prospective" and set(selected) == set(known):
        activate_prospective(database_url, protocol["protocol_id"], issue_time)
    print(json.dumps({
        "status": "failed" if failed else "ok",
        "protocol_id": protocol["protocol_id"],
        "logical_issue_time": issue_time.isoformat(),
        "deadline": deadline.isoformat(),
        "missed_event_counts_reconciled": reconciled,
        "incident_export": incident_export,
        "regions": results,
    }, sort_keys=True))
    return 1 if failed else 0


def activate_dry_run(
    database_url: str, protocol_id: str, issue_time: datetime
) -> None:
    """Atomically record the first real-time issue as the dry-run boundary."""

    import psycopg

    issue = issue_time.astimezone(timezone.utc)
    first_target = issue.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        days=1
    )
    with psycopg.connect(database_url) as connection:
        connection.execute(
            "SELECT pg_advisory_xact_lock(hashtext(%s))",
            (f"{protocol_id}:activation",),
        )
        row = connection.execute(
            "SELECT status, planned_start FROM prospective.protocols WHERE protocol_id = %s",
            (protocol_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("dry-run protocol was not seeded")
        if row[0] == "draft":
            connection.execute(
                """
                UPDATE prospective.protocols
                SET status = 'dry_run', planned_start = %s, activated_at = %s
                WHERE protocol_id = %s AND status = 'draft'
                """,
                (first_target, issue, protocol_id),
            )
        elif row[0] == "dry_run":
            if row[1] is None:
                raise RuntimeError("active dry-run protocol has no planned start")
        else:
            raise RuntimeError(f"dry-run protocol cannot run from status {row[0]}")


def activate_prospective(database_url: str, protocol_id: str, issue_time: datetime) -> None:
    """Start the 365-day clock only after the first complete four-region issue."""

    import psycopg

    issue = issue_time.astimezone(timezone.utc)
    first_target = issue.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        days=1
    )
    with psycopg.connect(database_url) as connection:
        connection.execute(
            "SELECT pg_advisory_xact_lock(hashtext(%s))", (f"{protocol_id}:activation",)
        )
        row = connection.execute(
            "SELECT status, planned_start FROM prospective.protocols WHERE protocol_id = %s",
            (protocol_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("prospective protocol was not seeded")
        if row[0] == "active":
            return
        if row[0] != "draft":
            raise RuntimeError(f"prospective protocol cannot activate from status {row[0]}")
        complete_issue = connection.execute(
            """
            SELECT target_start
            FROM prospective.forecast_runs
            WHERE protocol_id = %s AND status = 'published'
            GROUP BY target_start
            HAVING count(DISTINCT region_id) = 4
            ORDER BY target_start
            LIMIT 1
            """,
            (protocol_id,),
        ).fetchone()
        if complete_issue is None or complete_issue[0] > first_target:
            raise RuntimeError("formal activation requires one atomic four-region issue")
        first_target = complete_issue[0]
        connection.execute(
            """
            UPDATE prospective.protocols
            SET status = 'completed', completed_at = %s
            WHERE status IN ('active', 'dry_run') AND protocol_id <> %s
            """,
            (issue, protocol_id),
        )
        connection.execute(
            """
            UPDATE prospective.protocols
            SET status = 'active', planned_start = %s, activated_at = %s
            WHERE protocol_id = %s AND status = 'draft'
            """,
            (first_target, issue, protocol_id),
        )


def main() -> int:
    args = parse_args()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")
    issue_time = (args.issue_time or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return run_protocol(args, database_url, issue_time, PROSPECTIVE_PROTOCOL_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
