"""Read-only projection of published forecast grids for the public dashboard."""

from __future__ import annotations

import hashlib
import io
import json
import threading

import numpy as np

from etas_challenge.object_storage import ObjectStorageConfig


def _verified_object(client, storage, record: dict) -> bytes:
    response = client.get_object(Bucket=storage.bucket, Key=record["object_key"])
    payload = response["Body"].read()
    if len(payload) != record["byte_count"]:
        raise RuntimeError("forecast map artifact byte count disagrees")
    if hashlib.sha256(payload).hexdigest() != record["content_sha256"]:
        raise RuntimeError("forecast map artifact checksum disagrees")
    return payload


def _finite_values(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(result) or np.any(~np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"invalid {name} forecast map values")
    return result


def _forecast_cells(baseline_payload: bytes, challenger_payload: bytes) -> dict:
    with np.load(io.BytesIO(baseline_payload), allow_pickle=False) as baseline, np.load(
        io.BytesIO(challenger_payload), allow_pickle=False
    ) as challenger:
        if "daily_rates" in baseline.files and "daily_rates" in challenger.files:
            value_key = "daily_rates"
            semantics = "one_day_expected_count_per_cell"
        elif (
            "direct_background_mass" in baseline.files
            and "direct_background_mass" in challenger.files
        ):
            value_key = "direct_background_mass"
            semantics = "pre_target_direct_background_mass_per_cell"
        else:
            raise ValueError("unsupported forecast map artifact")

        etas = _finite_values(baseline[value_key], "ETAS")
        ch008 = _finite_values(challenger[value_key], "CH-008")
        if etas.shape != ch008.shape:
            raise ValueError("forecast map layers disagree")

        if "origin_units" in baseline.files:
            units = float(np.asarray(baseline["coordinate_units_per_degree"]).item())
            origins = np.asarray(baseline["origin_units"], dtype=np.float64) / units
            spacing = 1.0 / units
        elif "longitude_edges" in baseline.files:
            longitude_edges = np.asarray(baseline["longitude_edges"], dtype=np.float64)
            latitude_edges = np.asarray(baseline["latitude_edges"], dtype=np.float64)
            longitude, latitude = np.meshgrid(longitude_edges[:-1], latitude_edges[:-1])
            origins = np.column_stack((longitude.reshape(-1), latitude.reshape(-1)))
            spacing = float(longitude_edges[1] - longitude_edges[0])
        elif "latent_keys" in baseline.files:
            keys = np.asarray(baseline["latent_keys"], dtype=np.float64)
            if len(keys) != len(etas):
                raise ValueError("forecast map latent geometry disagrees")
            spacing = 0.5
            origins = keys * spacing
        else:
            raise ValueError("unsupported forecast map geometry")

        extra_layers = {}
        if value_key == "daily_rates" and "safe_daily_rates" in challenger.files:
            extra_layers = {
                "safe": _finite_values(challenger["safe_daily_rates"], "safe"),
                "fixed": _finite_values(challenger["fixed_daily_rates"], "fixed"),
                "gated": ch008,
            }
            if any(values.shape != etas.shape for values in extra_layers.values()):
                raise ValueError("multi-model forecast map layers disagree")

    if origins.shape != (len(etas), 2) or np.any(~np.isfinite(origins)):
        raise ValueError("forecast map geometry disagrees")

    ratio = np.zeros_like(etas)
    valid = (etas > 0) & (ch008 > 0)
    ratio[valid] = np.log(ch008[valid] / etas[valid])
    layers = {"etas": etas.tolist(), "ch008": ch008.tolist(), "log_ratio": ratio.tolist()}
    summary = {
        "cells": len(etas),
        "etas_total": float(np.sum(etas, dtype=np.float64)),
        "ch008_total": float(np.sum(ch008, dtype=np.float64)),
    }
    if extra_layers:
        layers.update({name: values.tolist() for name, values in extra_layers.items()})
        for name, numerator, denominator in (
            ("safe_etas_log_ratio", extra_layers["safe"], etas),
            ("fixed_safe_log_ratio", extra_layers["fixed"], extra_layers["safe"]),
            ("gated_etas_log_ratio", extra_layers["gated"], etas),
        ):
            values = np.zeros_like(etas)
            valid = (numerator > 0) & (denominator > 0)
            values[valid] = np.log(numerator[valid] / denominator[valid])
            layers[name] = values.tolist()
        summary.update({
            "safe_total": float(np.sum(extra_layers["safe"])),
            "fixed_total": float(np.sum(extra_layers["fixed"])),
            "gated_total": float(np.sum(extra_layers["gated"])),
        })
    return {
        "semantics": semantics,
        "spacing_degrees": spacing,
        "origins": np.round(origins, 6).tolist(),
        "layers": layers,
        "summary": summary,
    }


def load_latest_forecast_map(connection, client, storage, protocol_id: str, region_id: str) -> dict:
    protocol_row = connection.execute(
        "SELECT config FROM prospective.protocols WHERE protocol_id = %s",
        (protocol_id,),
    ).fetchone()
    if protocol_row is None:
        raise LookupError("protocol not found")
    config = protocol_row[0]
    if isinstance(config, str):
        config = json.loads(config)
    region = next(
        (item for item in config["regions"] if item["region_id"] == region_id), None
    )
    if region is None:
        raise LookupError("region not found")

    run = connection.execute(
        """
        SELECT run_id, issue_time, target_start, target_end, finished_at
        FROM prospective.forecast_runs
        WHERE protocol_id = %s AND region_id = %s AND status = 'published'
        ORDER BY target_start DESC
        LIMIT 1
        """,
        (protocol_id, region_id),
    ).fetchone()
    if run is None:
        raise LookupError("published forecast not found")
    artifacts = connection.execute(
        """
        SELECT model_id, object_key, content_sha256, byte_count
        FROM prospective.forecast_artifacts
        WHERE run_id = %s AND artifact_kind = 'grid'
          AND model_id = ANY(%s)
        """,
        (run[0], [region["baseline_model_id"], region["challenger_model_id"]]),
    ).fetchall()
    records = {
        row[0]: {
            "object_key": row[1],
            "content_sha256": row[2],
            "byte_count": int(row[3]),
        }
        for row in artifacts
    }
    required = (region["baseline_model_id"], region["challenger_model_id"])
    if any(model_id not in records for model_id in required):
        raise RuntimeError("published forecast grid artifacts are incomplete")
    cells = _forecast_cells(
        _verified_object(client, storage, records[required[0]]),
        _verified_object(client, storage, records[required[1]]),
    )
    return {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "region_id": region_id,
        "region_name": region["name"],
        "run_id": run[0],
        "issue_time": run[1].isoformat(),
        "target_start": run[2].isoformat(),
        "target_end": run[3].isoformat(),
        "published_at": run[4].isoformat(),
        **cells,
    }


class ForecastMapReader:
    """Cache immutable map projections while still checking for a newer run."""

    def __init__(self, database_url: str, protocol_id: str, storage=None, client=None):
        self.database_url = database_url
        self.protocol_id = protocol_id
        self.storage = storage or ObjectStorageConfig.from_environment()
        self.client = client or self.storage.client()
        self._cache: dict[str, tuple[str, dict]] = {}
        self._lock = threading.Lock()

    def __call__(self, region_id: str) -> dict:
        import psycopg

        with psycopg.connect(self.database_url, connect_timeout=5) as connection:
            latest = connection.execute(
                """
                SELECT run_id FROM prospective.forecast_runs
                WHERE protocol_id = %s AND region_id = %s AND status = 'published'
                ORDER BY target_start DESC LIMIT 1
                """,
                (self.protocol_id, region_id),
            ).fetchone()
            if latest is None:
                raise LookupError("published forecast not found")
            with self._lock:
                cached = self._cache.get(region_id)
                if cached is not None and cached[0] == latest[0]:
                    return cached[1]
            result = load_latest_forecast_map(
                connection, self.client, self.storage, self.protocol_id, region_id
            )
        with self._lock:
            self._cache[region_id] = (result["run_id"], result)
        return result
