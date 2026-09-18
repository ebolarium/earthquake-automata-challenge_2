"""Read-only PostgreSQL projection for the prospective dashboard."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
import math

from scipy.stats import norm, poisson


def _iso(value):
    return None if value is None else value.isoformat()


def _score_summary(rows: list[dict], revision: str) -> dict:
    selected = [row for row in rows if row["revision"] == revision]
    events = sum(row["event_count"] for row in selected)
    total = sum(row["total_gain"] for row in selected)
    mean = None if not events else total / events
    return {
        "days": len({row["target_date"] for row in selected}),
        "events": events,
        "total_gain": total,
        "mean_igpe": mean,
        "relative_factor": None if mean is None else math.exp(mean),
    }


def _multi_model_summary(rows: list[dict], revision: str) -> dict:
    selected = [
        row for row in rows
        if row["revision"] == revision and row.get("multi_model") is not None
    ]
    names = (
        "safe_vs_etas", "fixed_vs_etas", "gated_vs_etas",
        "fixed_vs_safe", "gated_vs_safe",
    )
    comparisons = {}
    for name in names:
        events = sum(row["multi_model"]["comparisons"][name]["events"] for row in selected)
        total = sum(row["multi_model"]["comparisons"][name]["total_gain"] for row in selected)
        comparisons[name] = {
            "events": events,
            "total_gain": total,
            "mean_igpe": None if not events else total / events,
            "relative_factor": None if not events else math.exp(total / events),
        }
    latest = max(selected, key=lambda row: row["target_date"], default=None)
    return {
        "comparisons": comparisons,
        "latest_gate": None if latest is None else {
            "target_date": latest["target_date"],
            "weight": latest["multi_model"]["gate_weight"],
            "log_bayes_factor": latest["multi_model"]["gate_log_bayes_factor"],
            "active_support_cells": latest["multi_model"]["active_support_cells"],
        },
    }


def _csep_summary(rows: list[dict], revision: str) -> dict | None:
    selected = [
        row for row in rows if row["revision"] == revision and row.get("csep")
    ]
    if not selected:
        return None
    output = {"method": "conditional_poisson_spatial_csep", "days": len(selected)}
    for model in ("baseline", "challenger"):
        reports = [row["csep"][model] for row in selected]
        observed_count = sum(item["observed_count"] for item in reports)
        expected_count = sum(item["expected_count"] for item in reports)
        lower = float(poisson.cdf(observed_count, expected_count))
        upper = float(poisson.sf(observed_count - 1, expected_count))
        observed_ll = sum(item["observed_log_likelihood"] for item in reports)
        simulated_mean = sum(item["l_test_simulated_mean"] for item in reports)
        simulated_variance = sum(item["l_test_simulated_variance"] for item in reports)
        output[model] = {
            "observed_count": observed_count,
            "expected_count": expected_count,
            "n_test_two_sided_p": min(1.0, 2.0 * min(lower, upper)),
            "observed_log_likelihood": observed_ll,
            "l_test_lower_tail_p": (
                None if simulated_variance <= 0 else float(
                    norm.cdf((observed_ll - simulated_mean) / math.sqrt(simulated_variance))
                )
            ),
        }
    r_reports = [row["csep"]["r_test"] for row in selected]
    observed = sum(item["observed_log_likelihood_ratio"] for item in r_reports)
    null_mean = sum(item["etas_null_mean"] for item in r_reports)
    null_variance = sum(item["etas_null_variance"] for item in r_reports)
    z_score = None if null_variance <= 0 else (observed - null_mean) / math.sqrt(null_variance)
    output["r_test"] = {
        "observed_log_likelihood_ratio": observed,
        "z_score": z_score,
        "one_sided_p": None if z_score is None else float(norm.sf(z_score)),
    }
    return output


def _dry_run_progress(
    rows: list[dict],
    region_ids: list[str],
    planned_days: int,
    *,
    schedule_start=None,
    generated_date=None,
    missed_target_dates=None,
) -> dict:
    expected = set(region_ids)

    def complete_days(revision: str) -> int:
        coverage = {}
        for row in rows:
            if row["revision"] == revision:
                target_date = row["target_date"]
                if isinstance(target_date, str):
                    target_date = datetime.fromisoformat(target_date).date()
                if schedule_start is not None and not (
                    schedule_start
                    <= target_date
                    < schedule_start + timedelta(days=planned_days)
                ):
                    continue
                coverage.setdefault(target_date, set()).add(row["region_id"])
        return sum(regions >= expected for regions in coverage.values()) if expected else 0

    provisional_days = complete_days("provisional")
    final_days = complete_days("final")
    if schedule_start is not None and generated_date is not None:
        elapsed_days = min(
            planned_days, max(0, (generated_date - schedule_start).days)
        )
        calendar_end_exclusive = schedule_start + timedelta(days=planned_days)
        missed_days = len({
            value for value in (missed_target_dates or [])
            if schedule_start <= value < calendar_end_exclusive
            and value < generated_date
        })
        unscored_completed_days = max(0, elapsed_days - provisional_days)
        if elapsed_days >= planned_days:
            phase = (
                "complete"
                if final_days >= provisional_days
                and unscored_completed_days <= missed_days
                else "settling"
            )
        elif elapsed_days:
            phase = "running"
        else:
            phase = "awaiting_scores"
        calendar_start = schedule_start.isoformat()
        calendar_end = calendar_end_exclusive.isoformat()
    else:
        elapsed_days = min(provisional_days, planned_days)
        missed_days = 0
        unscored_completed_days = max(0, elapsed_days - provisional_days)
        if final_days >= planned_days:
            phase = "complete"
        elif provisional_days >= planned_days:
            phase = "settling"
        elif provisional_days:
            phase = "running"
        else:
            phase = "awaiting_scores"
        calendar_start = None
        calendar_end = None
    return {
        "phase": phase,
        "planned_days": planned_days,
        "calendar_start": calendar_start,
        "calendar_end_exclusive": calendar_end,
        "calendar_days_elapsed": elapsed_days,
        "calendar_days_remaining": max(0, planned_days - elapsed_days),
        "successful_scored_days": provisional_days,
        "missed_calendar_days": missed_days,
        "unscored_completed_days": unscored_completed_days,
        "provisional_days": provisional_days,
        "final_days": final_days,
    }


def build_dashboard(connection, protocol_id: str, now=None) -> dict:
    generated_at = now or datetime.now(timezone.utc)
    protocol_row = connection.execute(
        """
        SELECT status, config, planned_days, minimum_events
        FROM prospective.protocols WHERE protocol_id = %s
        """,
        (protocol_id,),
    ).fetchone()
    if protocol_row is None:
        raise ValueError("prospective dashboard protocol is missing")
    config = protocol_row[1]
    region_rows = connection.execute(
        """
        SELECT region_id, name, catalog_source, minimum_magnitude,
               minimum_depth_km, maximum_depth_km
        FROM prospective.regions
        WHERE protocol_id = %s
        ORDER BY region_id
        """,
        (protocol_id,),
    ).fetchall()
    run_rows = connection.execute(
        """
        SELECT DISTINCT ON (region_id)
               region_id, run_id, issue_time, target_start, target_end,
               status, finished_at, state_id,
               (SELECT count(*) FROM prospective.forecast_artifacts a
                WHERE a.run_id = r.run_id)
        FROM prospective.forecast_runs r
        WHERE protocol_id = %s
        ORDER BY region_id, target_start DESC
        """,
        (protocol_id,),
    ).fetchall()
    latest_runs = {
        row[0]: {
            "run_id": row[1],
            "issue_time": _iso(row[2]),
            "target_start": _iso(row[3]),
            "target_end": _iso(row[4]),
            "status": row[5],
            "finished_at": _iso(row[6]),
            "state_id": row[7],
            "artifacts": int(row[8]),
        }
        for row in run_rows
    }
    catalog_rows = connection.execute(
        """
        SELECT DISTINCT ON (region_id)
               region_id, source_cutoff_at, event_count, content_sha256
        FROM prospective.catalog_snapshots
        WHERE protocol_id = %s AND region_id = ANY(%s)
          AND collection_kind = 'rolling'
        ORDER BY region_id, source_cutoff_at DESC, captured_at DESC
        """,
        (protocol_id, [row[0] for row in region_rows]),
    ).fetchall()
    latest_catalogs = {
        row[0]: {
            "cutoff": _iso(row[1]),
            "window_events": int(row[2]),
            "sha256": row[3],
        }
        for row in catalog_rows
    }
    score_rows = connection.execute(
        """
        SELECT region_id, target_date, score_revision, event_count,
               total_log_likelihood_gain, mean_igpe, computed_at, metrics
        FROM prospective.daily_scores
        WHERE protocol_id = %s
        ORDER BY target_date, region_id, score_revision
        """,
        (protocol_id,),
    ).fetchall()
    scores = [
        {
            "region_id": row[0],
            "target_date": row[1].isoformat(),
            "revision": row[2],
            "event_count": int(row[3]),
            "total_gain": float(row[4]),
            "mean_igpe": None if row[5] is None else float(row[5]),
            "computed_at": _iso(row[6]),
            "multi_model": (
                row[7].get("multi_model") if len(row) > 7 and row[7] else None
            ),
            "csep": row[7].get("csep") if len(row) > 7 and row[7] else None,
        }
        for row in score_rows
    ]
    incident_row = connection.execute(
        """
        SELECT count(*) FILTER (
                   WHERE resolved_at IS NULL
                     AND severity IN ('warning', 'critical')
               ),
               max(occurred_at),
               max(occurred_at) FILTER (
                   WHERE resolved_at IS NULL
                     AND severity IN ('warning', 'critical')
               )
        FROM prospective.incidents
        WHERE protocol_id = %s
        """,
        (protocol_id,),
    ).fetchone()
    operational_rows = connection.execute(
        """
        SELECT region_id, primary_eligible, missed_region_days,
               consecutive_missed_days, invalidated_at, invalidation_reason
        FROM prospective.region_operational_status
        WHERE protocol_id = %s
        """,
        (protocol_id,),
    ).fetchall()
    operation_rows = connection.execute(
        """
        SELECT region_id, issue_date, target_date, publication_status
        FROM prospective.region_day_operations
        WHERE protocol_id = %s
        ORDER BY region_id, issue_date
        """,
        (protocol_id,),
    ).fetchall()
    operation_history = {}
    for row in operation_rows:
        operation_history.setdefault(row[0], []).append({
            "issue_date": row[1],
            "target_date": row[2],
            "publication_status": row[3],
        })
    observed_target_dates = [
        item["target_date"]
        for records in operation_history.values()
        for item in records
    ] + [datetime.fromisoformat(item["target_date"]).date() for item in scores]
    schedule_start = min(observed_target_dates, default=None)
    schedule_end = (
        None if schedule_start is None
        else schedule_start + timedelta(days=int(protocol_row[2]))
    )
    scoped_scores = scores
    if config["mode"] == "dry_run" and schedule_end is not None:
        scoped_scores = [
            item for item in scores
            if datetime.fromisoformat(item["target_date"]).date() < schedule_end
        ]

    def current_missed_streak(region_id):
        streak = 0
        for record in reversed(operation_history.get(region_id, [])):
            if record["publication_status"] == "missed":
                streak += 1
            elif record["publication_status"] == "published":
                break
        return streak

    operational = {
        row[0]: {
            "primary_eligible": bool(row[1]),
            "missed_region_days": int(row[2]),
            "consecutive_missed_days": current_missed_streak(row[0]),
            "current_consecutive_missed_days": current_missed_streak(row[0]),
            "longest_consecutive_missed_days": int(row[3]),
            "invalidated_at": _iso(row[4]),
            "invalidation_reason": row[5],
        }
        for row in operational_rows
    }
    regions = []
    for row in region_rows:
        region_id = row[0]
        region_scores = [
            score for score in scoped_scores if score["region_id"] == region_id
        ]
        regions.append({
            "region_id": region_id,
            "name": row[1],
            "catalog_source": row[2],
            "minimum_magnitude": float(row[3]),
            "minimum_depth_km": None if row[4] is None else float(row[4]),
            "maximum_depth_km": None if row[5] is None else float(row[5]),
            "latest_forecast": latest_runs.get(region_id),
            "latest_catalog": latest_catalogs.get(region_id),
            "provisional": _score_summary(region_scores, "provisional"),
            "final": _score_summary(region_scores, "final"),
            "model_comparisons": {
                "provisional": _multi_model_summary(region_scores, "provisional"),
                "final": _multi_model_summary(region_scores, "final"),
            },
            "csep": {
                "provisional": _csep_summary(region_scores, "provisional"),
                "final": _csep_summary(region_scores, "final"),
            },
            "operations": operational.get(region_id, {
                "primary_eligible": True,
                "missed_region_days": 0,
                "consecutive_missed_days": 0,
                "current_consecutive_missed_days": 0,
                "longest_consecutive_missed_days": 0,
                "invalidated_at": None,
                "invalidation_reason": None,
            }),
        })
    latest_targets = {
        item["latest_forecast"]["target_start"]
        for item in regions
        if item["latest_forecast"] is not None
    }
    published_regions = sum(
        item["latest_forecast"] is not None
        and item["latest_forecast"]["status"] == "published"
        and item["latest_forecast"]["artifacts"] == 6
        for item in regions
    )
    open_incidents = int(incident_row[0])
    issue_boundary = datetime.combine(
        generated_at.astimezone(timezone.utc).date(), time.min, timezone.utc
    )
    expected_minimum_target = issue_boundary + (
        timedelta(days=1)
        if generated_at.astimezone(timezone.utc) > issue_boundary + timedelta(minutes=15)
        else timedelta(0)
    )
    forecast_current = all(
        item["latest_forecast"] is not None
        and datetime.fromisoformat(item["latest_forecast"]["target_start"])
        >= expected_minimum_target
        for item in regions
    )
    pooled_primary_eligible = all(
        item["operations"]["primary_eligible"] for item in regions
    )
    pipeline_status = (
        "ok"
        if published_regions == len(regions)
        and len(latest_targets) == 1
        and forecast_current
        and open_incidents == 0
        and pooled_primary_eligible
        else "attention"
    )
    planned_days = int(protocol_row[2])
    progress = _dry_run_progress(
        scoped_scores,
        [row[0] for row in region_rows],
        planned_days,
        schedule_start=schedule_start,
        generated_date=generated_at.astimezone(timezone.utc).date(),
        missed_target_dates=[
            item["target_date"]
            for records in operation_history.values()
            for item in records
            if item["publication_status"] == "missed"
        ],
    )
    return {
        "schema_version": 4,
        "generated_at": generated_at.isoformat(),
        "pipeline_status": pipeline_status,
        "protocol": {
            "protocol_id": protocol_id,
            "database_status": protocol_row[0],
            "mode": config["mode"],
            "counts_toward_prospective_claim": config["counts_toward_prospective_claim"],
            "planned_days": planned_days,
            "minimum_events": int(protocol_row[3]),
            "settled_score_delay_days": config["catalog_revision_contract"]["settled_score_delay_days"],
        },
        "latest_target_start": next(iter(latest_targets)) if len(latest_targets) == 1 else None,
        "expected_minimum_target_start": expected_minimum_target.isoformat(),
        "forecast_current": forecast_current,
        "published_regions": published_regions,
        "open_incidents": open_incidents,
        "last_incident_at": _iso(incident_row[1]),
        "last_open_incident_at": _iso(incident_row[2]),
        "pooled_primary_claim_status": (
            "eligible" if pooled_primary_eligible else "inconclusive"
        ),
        "test_progress": progress,
        "dry_run": progress if config["mode"] == "dry_run" else None,
        "post_window_score_rows_excluded": len(scores) - len(scoped_scores),
        "provisional": _score_summary(scoped_scores, "provisional"),
        "final": _score_summary(scoped_scores, "final"),
        "model_comparisons": {
            "provisional": _multi_model_summary(scoped_scores, "provisional"),
            "final": _multi_model_summary(scoped_scores, "final"),
        },
        "csep": {
            "provisional": _csep_summary(scoped_scores, "provisional"),
            "final": _csep_summary(scoped_scores, "final"),
        },
        "parameter_reporting": config.get("parameter_reporting"),
        "research_question": config.get("research_question"),
        "public_title": config.get("public_title"),
        "regions": regions,
        "daily_scores": scoped_scores,
    }


def read_dashboard(database_url: str, protocol_id: str) -> dict:
    import psycopg

    with psycopg.connect(database_url, connect_timeout=5) as connection:
        return build_dashboard(connection, protocol_id)
