"""Optional PaddleOCR 3 adapter with a stable project-level result shape."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import numpy as np


def paddle_available() -> bool:
    """Return whether PaddleOCR and its runtime are installed."""
    return find_spec("paddleocr") is not None and find_spec("paddle") is not None


def recognise_page(image_rgb: np.ndarray, language: str = "en") -> list[dict[str, object]]:
    """Run PaddleOCR lazily and normalise its prediction objects.

    Paddle is intentionally optional because its native runtime is platform and
    accelerator dependent.  The import occurs only when this function is used.
    """
    if not paddle_available():
        raise RuntimeError(
            "PaddleOCR is not installed. Install the optional backend with "
            "uv pip install -e '.[paddle]' and install a matching PaddlePaddle runtime."
        )
    from paddleocr import PaddleOCR  # type: ignore[import-not-found]

    engine = PaddleOCR(
        lang=language,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    rows: list[dict[str, object]] = []
    for prediction in engine.predict(image_rgb):
        payload: Any = prediction.json if hasattr(prediction, "json") else prediction
        if callable(payload):
            payload = payload()
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)
        rows.extend(_normalise_payload(payload))
    return rows


def _normalise_payload(payload: Any) -> list[dict[str, object]]:
    """Handle PaddleOCR's JSON result and the older list-style result."""
    if isinstance(payload, dict):
        payload = payload.get("res", payload)
        texts = payload.get("rec_texts", [])
        scores = payload.get("rec_scores", [])
        boxes = payload.get("rec_polys", payload.get("rec_boxes", []))
        return [
            {"text": str(text), "confidence": float(score), "polygon": np.asarray(box).tolist()}
            for text, score, box in zip(texts, scores, boxes)
        ]
    rows = []
    for item in payload if isinstance(payload, list) else []:
        if len(item) >= 2:
            polygon, recognised = item[0], item[1]
            rows.append(
                {
                    "text": str(recognised[0]),
                    "confidence": float(recognised[1]),
                    "polygon": np.asarray(polygon).tolist(),
                }
            )
    return rows
