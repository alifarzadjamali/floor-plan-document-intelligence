"""Verify the published frozen model before running inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/segmentation/phase2_dev/checkpoint_manifest.json"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = args.manifest.parent / manifest["checkpoint"]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = {
        "checkpoint": str(checkpoint),
        "expected_sha256": manifest["sha256"],
        "actual_sha256": digest,
        "valid": digest == manifest["sha256"],
    }
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit("Checkpoint checksum mismatch")


if __name__ == "__main__":
    main()
