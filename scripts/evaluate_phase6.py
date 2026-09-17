"""Write a reproducible Phase 6 capability report without hiding optional backends."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from floorplan_di.evaluation.calibration import confidence_band, expected_calibration_error
from floorplan_di.ocr.paddle_engine import paddle_available


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/phase6/report.json"))
    args = parser.parse_args()
    checkpoint = Path("results/segmentation/phase2_dev/checkpoints/best.pt")
    scores = [0.1, 0.4, 0.7, 0.9]
    correct = [False, False, True, True]
    report = {
        "phase": 6,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "published": True,
        },
        "confidence_calibration": {
            "status": "implemented",
            "probe_ece": expected_calibration_error(scores, correct),
            "bands": [confidence_band(score) for score in scores],
            "note": (
                "Probe values verify the implementation; no fabricated model calibration score "
                "is reported."
            ),
        },
        "paddleocr": {
            "status": "available" if paddle_available() else "optional_not_installed",
            "comparison_command": "python scripts/compare_ocr.py --input <plan.png>",
        },
        "geometry": {
            "status": "implemented",
            "features": ["polygon validity metadata", "wall PCA centerline approximation"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
