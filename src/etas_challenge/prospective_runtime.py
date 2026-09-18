"""Runtime protocol selection for the dry run and formal test."""

from __future__ import annotations

from pathlib import Path


DRY_RUN_PROTOCOL_ID = "evidence-gate-california-dry-run-v1"
PROSPECTIVE_PROTOCOL_ID = "evidence-gate-california-prospective-v1"
DRY_RUN_PROTOCOL_PATH = Path("configs/prospective/evidence-gate-california-dry-run-v1.json")
PROSPECTIVE_PROTOCOL_PATH = Path("configs/prospective/evidence-gate-california-prospective-v1.json")
LEGACY_PROTOCOL_PATHS = {
    "ch008-three-region-dry-run-v1": Path("configs/prospective/three-region-dry-run-v1.json"),
    "ch008-three-region-prospective-v1": Path("configs/prospective/ch008-three-region-prospective-v1.json"),
}


def protocol_path(protocol_id: str) -> Path:
    if protocol_id == DRY_RUN_PROTOCOL_ID:
        return DRY_RUN_PROTOCOL_PATH
    if protocol_id == PROSPECTIVE_PROTOCOL_ID:
        return PROSPECTIVE_PROTOCOL_PATH
    if protocol_id in LEGACY_PROTOCOL_PATHS:
        return LEGACY_PROTOCOL_PATHS[protocol_id]
    raise ValueError(f"unknown runtime protocol: {protocol_id}")


def active_protocol_id(connection) -> str:
    row = connection.execute(
        """
        SELECT protocol_id
        FROM prospective.protocols
        WHERE status = 'active'
        ORDER BY activated_at DESC NULLS LAST, created_at DESC
        LIMIT 1
        """
    ).fetchone()
    return DRY_RUN_PROTOCOL_ID if row is None else row[0]


def read_active_protocol_id(database_url: str) -> str:
    import psycopg

    with psycopg.connect(database_url, connect_timeout=5) as connection:
        return active_protocol_id(connection)
