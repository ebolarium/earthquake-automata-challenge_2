"""Canonical bootstrap catalog reads and prospective model-state identities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib

import numpy as np


@dataclass(frozen=True, slots=True)
class BootstrapCatalog:
    snapshot_ids: tuple[int, ...]
    event_ids: np.ndarray
    origin_time_ns: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    depths_km: np.ndarray
    magnitudes: np.ndarray


def filter_catalog(catalog: BootstrapCatalog, minimum_magnitude: float) -> BootstrapCatalog:
    selected = np.asarray(catalog.magnitudes) >= float(minimum_magnitude)
    return BootstrapCatalog(
        catalog.snapshot_ids,
        catalog.event_ids[selected], catalog.origin_time_ns[selected],
        catalog.latitudes[selected], catalog.longitudes[selected],
        catalog.depths_km[selected], catalog.magnitudes[selected],
    )


def selected_bootstrap_snapshots(
    connection, protocol_id: str, region_id: str, start, cutoff
) -> list[dict]:
    rows = connection.execute(
        """
        SELECT DISTINCT ON (source_start_at, source_cutoff_at)
               snapshot_id, source_start_at, source_cutoff_at, event_count,
               artifact_key, content_sha256
        FROM prospective.catalog_snapshots
        WHERE protocol_id = %s AND region_id = %s
          AND collection_kind = 'bootstrap'
          AND source_start_at >= %s
          AND source_cutoff_at <= %s
        ORDER BY source_start_at, source_cutoff_at, captured_at DESC, snapshot_id DESC
        """,
        (protocol_id, region_id, start, cutoff),
    ).fetchall()
    return [
        {
            "snapshot_id": row[0],
            "start": row[1],
            "cutoff": row[2],
            "event_count": row[3],
            "artifact_key": row[4],
            "content_sha256": row[5],
        }
        for row in rows
    ]


def load_bootstrap_catalog(connection, snapshot_ids: list[int], as_of: datetime) -> BootstrapCatalog:
    if not snapshot_ids:
        raise ValueError("bootstrap catalog requires snapshot IDs")
    rows = connection.execute(
        """
        SELECT source_event_id, origin_time, latitude, longitude, depth_km, magnitude
        FROM prospective.catalog_event_versions
        WHERE snapshot_id = ANY(%s) AND origin_time < %s
        ORDER BY origin_time, source_event_id
        """,
        (snapshot_ids, as_of),
    ).fetchall()
    if not rows:
        raise ValueError("bootstrap catalog contains no events before state boundary")
    event_ids = np.asarray([row[0] for row in rows])
    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError("bootstrap state catalog contains duplicate event IDs")
    origin_time_ns = np.asarray(
        [int(row[1].timestamp() * 1_000_000_000) for row in rows], dtype=np.int64
    )
    return BootstrapCatalog(
        tuple(snapshot_ids),
        event_ids,
        origin_time_ns,
        np.asarray([row[2] for row in rows], dtype=np.float64),
        np.asarray([row[3] for row in rows], dtype=np.float64),
        np.asarray([row[4] for row in rows], dtype=np.float64),
        np.asarray([row[5] for row in rows], dtype=np.float64),
    )


def model_state_id(
    protocol_id: str,
    region_id: str,
    as_of: datetime,
    catalog_cutoff: datetime,
    catalog_sha256: str,
    baseline_model_sha256: str,
    challenger_model_sha256: str,
) -> str:
    value = "\n".join(
        (
            protocol_id,
            region_id,
            as_of.isoformat(),
            catalog_cutoff.isoformat(),
            catalog_sha256,
            baseline_model_sha256,
            challenger_model_sha256,
        )
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def catalog_history_sha256(catalog: BootstrapCatalog) -> str:
    """Hash only the canonical event history admitted before the state boundary."""

    digest = hashlib.sha256(b"prospective-catalog-history-v1\0")
    for event_id in catalog.event_ids:
        encoded = str(event_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    for name, values, dtype in (
        ("origin_time_ns", catalog.origin_time_ns, "<i8"),
        ("latitudes", catalog.latitudes, "<f8"),
        ("longitudes", catalog.longitudes, "<f8"),
        ("depths_km", catalog.depths_km, "<f8"),
        ("magnitudes", catalog.magnitudes, "<f8"),
    ):
        array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
        digest.update(name.encode("ascii") + b"\0")
        digest.update(len(array).to_bytes(8, "big"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_state_artifact(
    arrays,
    manifest: dict,
    *,
    expected_state_shape: tuple[int, ...],
    regional: bool,
    evidence_gate: bool = False,
    incumbent_model_sha256: str | None = None,
) -> dict:
    """Validate one downloaded prospective state against its manifest."""

    common = {
        "event_ids", "origin_time_ns", "latitudes", "longitudes", "depths_km",
        "magnitudes", "ch008_age", "ch008_exposure", "ch008_roots", "as_of",
        "catalog_cutoff", "etas_model_sha256", "ch008_model_sha256",
    }
    regional_names = {
        "event_etas_rates", "event_background_probabilities", "event_cells"
    }
    evidence_names = (
        {"gate_log_bayes_factor"}
        if regional else
        {"background_root_days", "background_root_values", "gate_log_bayes_factor"}
    )
    context_names = {
        "context_event_ids", "context_origin_time_ns", "context_latitudes",
        "context_longitudes", "context_depths_km", "context_magnitudes",
    }
    present_context = context_names & set(arrays.files)
    if present_context and present_context != context_names:
        raise ValueError("state feature-context arrays are incomplete")
    expected_names = (
        common
        | (regional_names if regional else set())
        | (evidence_names if evidence_gate else set())
        | present_context
    )
    if set(arrays.files) != expected_names:
        raise ValueError("state artifact arrays disagree with region contract")

    def scalar(name: str) -> str:
        value = np.asarray(arrays[name])
        if value.shape != ():
            raise ValueError(f"state scalar is not scalar: {name}")
        return str(value.item())

    if scalar("as_of") != manifest["as_of"]:
        raise ValueError("state as_of disagrees with manifest")
    if scalar("catalog_cutoff") != manifest["catalog_cutoff"]:
        raise ValueError("state catalog cutoff disagrees with manifest")
    if scalar("etas_model_sha256") != manifest["baseline_model_sha256"]:
        raise ValueError("state ETAS model hash disagrees with manifest")
    expected_incumbent = (
        manifest["challenger_model_sha256"]
        if incumbent_model_sha256 is None else incumbent_model_sha256
    )
    if scalar("ch008_model_sha256") != expected_incumbent:
        raise ValueError("state CH-008 model hash disagrees with manifest")

    if evidence_gate:
        gate_evidence = np.asarray(arrays["gate_log_bayes_factor"])
        if gate_evidence.shape != () or not np.isfinite(float(gate_evidence)):
            raise ValueError("invalid evidence-gate state arrays")
        if not regional:
            root_days = np.asarray(arrays["background_root_days"])
            root_values = np.asarray(arrays["background_root_values"])
            if (
                root_days.ndim != 1
                or root_values.ndim != 2
                or root_values.shape[0] != len(root_days)
                or not len(root_days)
                or np.any(np.diff(root_days.astype(np.int64)) <= 0)
                or np.any(~np.isfinite(root_values))
                or np.any(root_values < 0)
            ):
                raise ValueError("invalid evidence-gate root history")

    event_ids = np.asarray(arrays["event_ids"])
    event_count = len(event_ids)
    event_arrays = {
        name: np.asarray(arrays[name])
        for name in (
            "origin_time_ns", "latitudes", "longitudes", "depths_km", "magnitudes"
        )
    }
    if manifest["events"] != event_count or any(
        value.shape != (event_count,) for value in event_arrays.values()
    ):
        raise ValueError("state event arrays disagree")
    if event_ids.shape != (event_count,) or len(np.unique(event_ids)) != event_count:
        raise ValueError("state event IDs are not unique")
    if present_context:
        context_count = len(np.asarray(arrays["context_event_ids"]))
        for name in context_names - {"context_event_ids"}:
            if np.asarray(arrays[name]).shape != (context_count,):
                raise ValueError("state feature-context arrays disagree")
        if context_count < event_count:
            raise ValueError("feature context cannot contain fewer events than target history")
    origin_time_ns = event_arrays["origin_time_ns"].astype(np.int64, copy=False)
    if np.any(np.diff(origin_time_ns) < 0):
        raise ValueError("state events are not time ordered")
    as_of = datetime.fromisoformat(manifest["as_of"])
    as_of_ns = int(np.datetime64(as_of.replace(tzinfo=None), "ns").astype(np.int64))
    if event_count and int(origin_time_ns[-1]) >= as_of_ns:
        raise ValueError("state contains an event at or after as_of")
    numeric = np.concatenate(
        [event_arrays[name].astype(np.float64, copy=False) for name in event_arrays]
    )
    if np.any(~np.isfinite(numeric)):
        raise ValueError("state event arrays contain non-finite values")
    if np.any(np.abs(event_arrays["latitudes"]) > 90) or np.any(
        np.abs(event_arrays["longitudes"]) > 180
    ):
        raise ValueError("state event coordinates are invalid")

    catalog = BootstrapCatalog(
        tuple(manifest["snapshot_ids"]),
        event_ids,
        origin_time_ns,
        event_arrays["latitudes"],
        event_arrays["longitudes"],
        event_arrays["depths_km"],
        event_arrays["magnitudes"],
    )
    if catalog_history_sha256(catalog) != manifest["catalog_history_sha256"]:
        raise ValueError("state catalog history hash disagrees")

    for name in ("ch008_age", "ch008_exposure", "ch008_roots"):
        value = np.asarray(arrays[name])
        if value.shape != expected_state_shape or np.any(~np.isfinite(value)) or np.any(value < 0):
            raise ValueError(f"invalid CH-008 state array: {name}")
    if regional:
        rates = np.asarray(arrays["event_etas_rates"])
        probabilities = np.asarray(arrays["event_background_probabilities"])
        cells = np.asarray(arrays["event_cells"])
        if any(value.shape != (event_count,) for value in (rates, probabilities, cells)):
            raise ValueError("regional ETAS event arrays disagree")
        if (
            np.any(~np.isfinite(rates))
            or np.any(rates <= 0)
            or np.any(~np.isfinite(probabilities))
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or np.any(cells < 0)
            or np.any(cells >= expected_state_shape[0])
        ):
            raise ValueError("regional ETAS event state is invalid")
    return {
        "events": event_count,
        "state_shape": list(expected_state_shape),
        "first_event_ns": None if not event_count else int(origin_time_ns[0]),
        "last_event_ns": None if not event_count else int(origin_time_ns[-1]),
    }
