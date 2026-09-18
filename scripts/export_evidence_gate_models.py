#!/usr/bin/env python3
"""Export the frozen California CH014 ensemble to deterministic NumPy weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etas_challenge.training_matrix import sha256_file, write_deterministic_npz  # noqa: E402


DEFAULT_SOURCES = (
    "artifacts/ch014-chronology-regularized-loro-v1/california-relative-fast-slow.pt",
    "artifacts/ch014-chronology-seed-14003-v1/california-relative-fast-slow.pt",
    "artifacts/ch014-chronology-seed-14004-v1/california-relative-fast-slow.pt",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "models/evidence-gate/california-ch014-ensemble-v1.npz",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
    except ImportError as error:
        raise SystemExit("export requires torch in the one-time development environment") from error

    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "member_count": np.asarray(len(DEFAULT_SOURCES), dtype=np.int64),
        "context_events": np.asarray(256, dtype=np.int64),
        "recent_events": np.asarray(32, dtype=np.int64),
        "slow_group_size": np.asarray(8, dtype=np.int64),
    }
    sources = []
    for member, relative in enumerate(DEFAULT_SOURCES):
        path = args.source_root / relative
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        sources.append({"path": relative, "sha256": sha256_file(path)})
        prefix = f"member_{member}_"
        arrays[prefix + "auxiliary_mean"] = np.asarray(checkpoint["auxiliary_mean"])
        arrays[prefix + "auxiliary_scale"] = np.asarray(checkpoint["auxiliary_scale"])
        arrays[prefix + "auxiliary_clip"] = np.asarray(
            checkpoint["auxiliary_normalization_clip"], dtype=np.float32
        )
        arrays[prefix + "coordinate_clip"] = np.asarray(
            checkpoint["coordinate_clip"], dtype=np.float32
        )
        for name, tensor in checkpoint["state_dict"].items():
            arrays[prefix + name.replace(".", "_")] = tensor.cpu().numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_npz(args.output, arrays)
    manifest = {
        "schema_version": 1,
        "model_id": "california-ch014-logit-ensemble-v1",
        "output": str(args.output.relative_to(ROOT)),
        "output_sha256": sha256_file(args.output),
        "sources": sources,
        "conversion": "lossless tensor-to-NumPy export; inference equivalence tested separately",
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
