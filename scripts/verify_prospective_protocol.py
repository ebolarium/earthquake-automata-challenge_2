#!/usr/bin/env python3
"""Verify the dry-run and formal protocols and all locked files."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etas_challenge.prospective_protocol import sha256_file  # noqa: E402
from etas_challenge.prospective_protocol import validate_protocol  # noqa: E402
from etas_challenge.prospective_protocol import configured_protocol_path  # noqa: E402
from etas_challenge.prospective_downtime import validate_downtime_policy  # noqa: E402


PROTOCOLS = (configured_protocol_path(ROOT),)
POLICY = ROOT / "configs/challenge/ch008-downtime-policy.json"


def main() -> int:
    protocols = [validate_protocol(path, ROOT) for path in PROTOCOLS]
    policy = validate_downtime_policy(POLICY)
    print(
        json.dumps(
            {
                "status": "ok",
                "protocols": [
                    {
                        "protocol_id": protocol["protocol_id"],
                        "protocol_sha256": sha256_file(path),
                        "mode": protocol["mode"],
                    }
                    for path, protocol in zip(PROTOCOLS, protocols)
                ],
                "downtime_policy_sha256": sha256_file(POLICY),
                "downtime_policy_status": policy["status"],
                "regions": [
                    region["region_id"] for region in protocols[0]["regions"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
