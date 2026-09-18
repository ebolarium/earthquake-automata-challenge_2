"""Build deterministic database seed records from the dry-run protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from etas_challenge.prospective_protocol import sha256_file
from etas_challenge.prospective_protocol import validate_protocol


ETAS_SOURCE_COMMIT = "51e0c8e419197df3f88349035a682b90fbd4dfb5"
CH008_SOURCE_COMMIT = "15f762ba6087b6639830555490f073c456e48bf0"
EVIDENCE_GATE_SOURCE_COMMIT = "377c54d9d02b020d6355f6aa11e7863e5c26e84a"


def canonical_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_seed_records(protocol_path: Path, root: Path) -> dict:
    protocol = validate_protocol(protocol_path, root)
    challenger_path = root / protocol["challenger"]["model_path"]
    challenger_parameters = json.loads(challenger_path.read_text(encoding="utf-8"))
    models = []
    for region in protocol["regions"]:
        model_path = root / region["etas_model_path"]
        models.append(
            {
                "model_id": region["baseline_model_id"],
                "role": "baseline",
                "source_commit": ETAS_SOURCE_COMMIT,
                "model_sha256": sha256_file(model_path),
                "runtime_sha256": protocol["baseline_runtime"]["sha256"],
                "parameters": json.loads(model_path.read_text(encoding="utf-8")),
            }
        )
    if protocol.get("forecast_family") == "causal_evidence_gate":
        models.append(
            {
                "model_id": protocol["challenger"]["model_id"],
                "role": "challenger",
                "source_commit": EVIDENCE_GATE_SOURCE_COMMIT,
                "model_sha256": sha256_file(challenger_path),
                "runtime_sha256": protocol["challenger"]["runtime_sha256"],
                "parameters": challenger_parameters,
            }
        )
    else:
        models.extend([
            {
                "model_id": protocol["challenger"]["california_model_id"],
                "role": "challenger",
                "source_commit": CH008_SOURCE_COMMIT,
                "model_sha256": sha256_file(challenger_path),
                "runtime_sha256": protocol["challenger"]["california_runtime_sha256"],
                "parameters": challenger_parameters,
            },
            {
                "model_id": protocol["challenger"]["normalized_model_id"],
                "role": "challenger",
                "source_commit": CH008_SOURCE_COMMIT,
                "model_sha256": sha256_file(challenger_path),
                "runtime_sha256": protocol["challenger"]["normalized_runtime_sha256"],
                "parameters": challenger_parameters,
            },
        ])
    regions = []
    for region in protocol["regions"]:
        regions.append(
            {
                "region_id": region["region_id"],
                "name": region["name"],
                "catalog_source": region["catalog_source"],
                "catalog_endpoint": region["catalog_endpoint"],
                "geometry": region["geometry"],
                "minimum_magnitude": region["minimum_magnitude"],
                "minimum_depth_km": region["minimum_depth_km"],
                "maximum_depth_km": region["maximum_depth_km_exclusive"],
                "c_region": region["renewal_exposure_scale"],
                "config_sha256": canonical_sha256(region),
            }
        )
    return {
        "protocol": protocol,
        "protocol_sha256": sha256_file(protocol_path),
        "models": models,
        "regions": regions,
    }
