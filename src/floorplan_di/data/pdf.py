"""Safe, explicit rasterisation for single-page floor-plan PDF inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_plan_page(path: Path, page_number: int = 1, dpi: int = 200) -> np.ndarray:
    """Load an RGB image or rasterise one 1-indexed PDF page to RGB.

    PDF support deliberately handles one selected page per prediction.  This
    keeps output coordinates tied to a single raster page and avoids implying
    document-level interpretation across pages.
    """
    if page_number < 1:
        raise ValueError("page_number must be at least 1")
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg"}:
        if page_number != 1:
            raise ValueError("Raster images contain only page 1")
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))
    if suffix != ".pdf":
        raise ValueError("Input must be a PNG, JPG, JPEG, or PDF file")
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - exercised in installation environments
        raise RuntimeError(
            "PDF support requires PyMuPDF; reinstall the project dependencies"
        ) from exc

    with pymupdf.open(path) as document:
        if page_number > document.page_count:
            raise ValueError(f"PDF has {document.page_count} pages; requested page {page_number}")
        page = document.load_page(page_number - 1)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False)
        pixels = np.frombuffer(pixmap.samples, dtype=np.uint8)
        return pixels.reshape(pixmap.height, pixmap.width, 3).copy()
