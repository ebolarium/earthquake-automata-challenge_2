#!/usr/bin/env python3
"""Export each frozen leave-one-region-out ensemble to NumPy weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etas_challenge.training_matrix import sha256_file, write_deterministic_npz  # noqa: E402


SEEDS = (
    (14002, "ch014-chronology-regularized-loro-v1"),
    (14003, "ch014-chronology-seed-14003-v1"),
    (14004, "ch014-chronology-seed-14004-v1"),
)
REGIONS = {
    "california": "california",
    "new-zealand": "new_zealand",
    "chile": "chile",
    "japan-c": "japan_c",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--region", choices=tuple(REGIONS), action="append")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "models/evidence-gate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
    except ImportError as error:
        raise SystemExit("export requires torch in the one-time development environment") from error

    manifests = []
    for public_region in args.region or tuple(REGIONS):
        checkpoint_region = REGIONS[public_region]
        arrays: dict[str, np.ndarray] = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "member_count": np.asarray(len(SEEDS), dtype=np.int64),
            "context_events": np.asarray(256, dtype=np.int64),
            "recent_events": np.asarray(32, dtype=np.int64),
            "slow_group_size": np.asarray(8, dtype=np.int64),
        }
        sources = []
        for member, (seed, directory) in enumerate(SEEDS):
            relative = f"artifacts/{directory}/{checkpoint_region}-relative-fast-slow.pt"
            path = args.source_root / relative
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("heldout_region") != checkpoint_region:
                raise ValueError(f"checkpoint is not LORO for {checkpoint_region}: {relative}")
            sources.append({"seed": seed, "path": relative, "sha256": sha256_file(path)})
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

        output = args.output_directory / f"{public_region}-causal-spatial-ensemble-v1.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        write_deterministic_npz(output, arrays)
        manifest = {
            "schema_version": 1,
            "model_id": f"{public_region}-causal-spatial-ensemble-v1",
            "training_contract": "leave-one-region-out; target region excluded",
            "output": str(output.relative_to(ROOT)),
            "output_sha256": sha256_file(output),
            "sources": sources,
            "conversion": "lossless tensor-to-NumPy export; inference equivalence tested separately",
        }
        output.with_suffix(".json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        manifests.append(manifest)
    print(json.dumps(manifests, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
