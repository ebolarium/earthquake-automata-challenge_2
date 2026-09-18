"""Regional FDSN requests and filtering for prospective catalog snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from etas_challenge.geonet_catalog import GeoNetEvent, parse_fdsn_text
from etas_challenge.masked_grid import masked_grid
from etas_challenge.training_matrix import GridDefinition


@dataclass(frozen=True)
class CatalogSnapshot:
    region_id: str
    start: datetime
    cutoff: datetime
    request_url: str
    request_parameters: dict[str, str]
    raw_payload: bytes
    events: tuple[GeoNetEvent, ...]

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.raw_payload).hexdigest()



class CatalogResultLimitError(RuntimeError):
    """Raised when an FDSN service requires a smaller query window."""


def _bounds(region: dict, root: Path) -> tuple[float, float, float, float]:
    geometry = region["geometry"]
    if region["region_id"] == "california-relm":
        grid = GridDefinition.load(root / geometry["path"])
        scale = grid.units_per_degree
        return (
            float(np.min(grid.origin_units[:, 0]) / scale),
            float((np.max(grid.origin_units[:, 0]) + 1) / scale),
            float(np.min(grid.origin_units[:, 1]) / scale),
            float((np.max(grid.origin_units[:, 1]) + 1) / scale),
        )
    if region["region_id"] == "new-zealand-csep":
        archive = np.load(root / geometry["path"], allow_pickle=False)
        origins = archive["origins"]
        spacing = geometry["spacing_degrees"]
        return (
            float(np.min(origins[:, 0])),
            float(np.max(origins[:, 0]) + spacing),
            float(np.min(origins[:, 1])),
            float(np.max(origins[:, 1]) + spacing),
        )
    return (
        float(geometry["longitude"][0]),
        float(geometry["longitude"][1]),
        float(geometry["latitude"][0]),
        float(geometry["latitude"][1]),
    )


def request_parameters(region: dict, root: Path, start: datetime, cutoff: datetime) -> dict[str, str]:
    lon_min, lon_max, lat_min, lat_max = _bounds(region, root)
    parameters = {
        "format": "text",
        "starttime": start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "endtime": cutoff.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "minlongitude": str(lon_min),
        "maxlongitude": str(lon_max),
        "minlatitude": str(lat_min),
        "maxlatitude": str(lat_max),
        "minmagnitude": str(region.get("catalog_minimum_magnitude", region["minimum_magnitude"])),
        "eventtype": "earthquake",
        "orderby": "time-asc",
    }
    if region["minimum_depth_km"] is not None:
        parameters["mindepth"] = str(region["minimum_depth_km"])
    if region["maximum_depth_km_exclusive"] is not None:
        parameters["maxdepth"] = str(region["maximum_depth_km_exclusive"] - 1e-9)
    return parameters


def _inside(region: dict, root: Path, events: list[GeoNetEvent]) -> np.ndarray:
    latitudes = np.asarray([event.latitude for event in events])
    longitudes = np.asarray([event.longitude for event in events])
    if region["region_id"] == "california-relm":
        grid = GridDefinition.load(root / region["geometry"]["path"])
        return grid.cell_indexes(longitudes, latitudes) >= 0
    if region["region_id"] == "new-zealand-csep":
        geometry = region["geometry"]
        origins = np.load(root / geometry["path"], allow_pickle=False)["origins"]
        grid = masked_grid(origins, geometry["spacing_degrees"], geometry["latent_spacing_degrees"])
        return grid.contains(latitudes, longitudes)
    geometry = region["geometry"]
    return (
        (longitudes >= geometry["longitude"][0])
        & (longitudes < geometry["longitude"][1])
        & (latitudes >= geometry["latitude"][0])
        & (latitudes < geometry["latitude"][1])
    )


def parse_and_filter(region: dict, root: Path, raw_payload: bytes, start: datetime, cutoff: datetime) -> tuple[GeoNetEvent, ...]:
    # FDSN services return HTTP 204 with an empty body when no events match.
    if not raw_payload.strip():
        return ()
    parsed = parse_fdsn_text(raw_payload.decode("utf-8").splitlines())
    inside = _inside(region, root, parsed) if parsed else np.asarray([], dtype=bool)
    accepted = []
    seen = set()
    for event, is_inside in zip(parsed, inside):
        minimum_depth = region["minimum_depth_km"]
        maximum_depth = region["maximum_depth_km_exclusive"]
        if (
            event.event_id not in seen
            and start <= event.time_utc < cutoff
            and event.magnitude >= region.get(
                "catalog_minimum_magnitude", region["minimum_magnitude"]
            )
            and (minimum_depth is None or event.depth_km >= minimum_depth)
            and (maximum_depth is None or event.depth_km < maximum_depth)
            and event.event_type.lower() == "earthquake"
            and bool(is_inside)
        ):
            accepted.append(event)
            seen.add(event.event_id)
    return tuple(accepted)


def fetch_snapshot(region: dict, root: Path, start: datetime, cutoff: datetime, timeout: int = 120) -> CatalogSnapshot:
    parameters = request_parameters(region, root, start, cutoff)
    url = f"{region['catalog_endpoint']}?{urlencode(parameters)}"
    request = Request(url, headers={"User-Agent": "earthquake-automata-challenge/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        normalized = body.lower()
        limit_markers = ("search limit", "exceeds", "too many", "maximum number")
        if error.code == 400 and any(marker in normalized for marker in limit_markers):
            raise CatalogResultLimitError(body[:1000]) from error
        raise
    events = parse_and_filter(region, root, payload, start, cutoff)
    return CatalogSnapshot(region["region_id"], start, cutoff, region["catalog_endpoint"], parameters, payload, events)


def event_payload(event: GeoNetEvent) -> dict:
    payload = asdict(event)
    payload["time_utc"] = event.time_utc.isoformat()
    return payload
