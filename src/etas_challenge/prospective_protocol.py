"""Validation for operational prospective protocol files."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


EXPECTED_REGIONS = {"california-relm", "new-zealand-csep", "chile-subduction"}
DEFAULT_PROTOCOL_PATH = "configs/prospective/evidence-gate-california-dry-run-v1.json"


def configured_protocol_path(root: Path) -> Path:
    relative = os.environ.get("PROSPECTIVE_PROTOCOL_PATH", DEFAULT_PROTOCOL_PATH)
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("prospective protocol path is outside project root")
    return path


def artifact_lane(protocol: dict) -> str:
    lane = protocol["artifact_contract"].get(
        "lane", "dry-run" if protocol.get("mode") == "dry_run" else None
    )
    if lane not in {"dry-run", "prospective"}:
        raise ValueError("invalid prospective artifact lane")
    return lane


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_protocol(path: Path, root: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    mode = protocol.get("mode")
    duration = protocol.get("duration_days")
    counts = protocol.get("counts_toward_prospective_claim")
    if mode == "dry_run":
        if duration != 14 or counts is not False:
            raise ValueError("expected a non-claim 14-day dry-run protocol")
    elif mode == "prospective":
        if duration != 365 or counts is not True:
            raise ValueError("expected a claim-bearing 365-day prospective protocol")
        primary = protocol.get("primary_claim", {})
        if (
            primary.get("minimum_pooled_events") != 500
            or primary.get("score_revision") != "final"
            or primary.get("mean_igpe_must_exceed") != 0.0
            or primary.get("bootstrap_30_day_lower_must_exceed") != 0.0
            or primary.get("bootstrap_90_day_lower_must_exceed") != 0.0
        ):
            raise ValueError("prospective primary claim is not fully frozen")
    else:
        raise ValueError("unknown prospective protocol mode")
    expected_lane = "dry-run" if mode == "dry_run" else "prospective"
    if artifact_lane(protocol) != expected_lane:
        raise ValueError("artifact lane disagrees with protocol mode")
    if protocol["issue_contract"].get("backfill_permitted") is not False:
        raise ValueError("forecast backfill must be prohibited")

    regions = protocol.get("regions", [])
    region_ids = [region.get("region_id") for region in regions]
    expected_regions = (
        {"california-relm"}
        if protocol.get("forecast_family") == "causal_evidence_gate"
        else EXPECTED_REGIONS
    )
    if len(region_ids) != len(set(region_ids)) or set(region_ids) != expected_regions:
        if protocol.get("forecast_family") == "causal_evidence_gate":
            raise ValueError("evidence-gate protocol must contain California only")
        raise ValueError("protocol must contain exactly the three admitted regions")

    if protocol.get("forecast_family") == "causal_evidence_gate":
        locked_files = [
            (protocol["challenger"]["model_path"], protocol["challenger"]["model_sha256"]),
            (protocol["challenger"]["runtime_path"], protocol["challenger"]["runtime_sha256"]),
            (
                protocol["challenger"]["forecast_builder_path"],
                protocol["challenger"]["forecast_builder_sha256"],
            ),
            (protocol["challenger"]["weights_path"], protocol["challenger"]["weights_sha256"]),
            (protocol["baseline_runtime"]["path"], protocol["baseline_runtime"]["sha256"]),
        ]
    else:
        locked_files = [
            (protocol["challenger"]["model_path"], protocol["challenger"]["model_sha256"]),
            (
                protocol["challenger"]["california_runtime_path"],
                protocol["challenger"]["california_runtime_sha256"],
            ),
            (
                protocol["challenger"]["normalized_runtime_path"],
                protocol["challenger"]["normalized_runtime_sha256"],
            ),
            (protocol["baseline_runtime"]["path"], protocol["baseline_runtime"]["sha256"]),
        ]
    for region in regions:
        locked_files.append((region["etas_model_path"], region["etas_model_sha256"]))
        geometry = region["geometry"]
        if "path" in geometry:
            locked_files.append((geometry["path"], geometry["sha256"]))
        if "protocol_path" in geometry:
            locked_files.append((geometry["protocol_path"], geometry["protocol_sha256"]))

    for relative, expected in locked_files:
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ValueError(f"locked file changed: {relative}")
    return protocol
