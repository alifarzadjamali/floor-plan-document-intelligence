from pathlib import Path

import numpy as np
from PIL import Image

from floorplan_di.data.pdf import load_plan_page


def test_load_plan_page_reads_raster_image(tmp_path: Path) -> None:
    image_path = tmp_path / "plan.png"
    Image.fromarray(np.full((8, 12, 3), 127, dtype=np.uint8)).save(image_path)
    image = load_plan_page(image_path)
    assert image.shape == (8, 12, 3)


def test_load_plan_page_rejects_non_first_raster_page(tmp_path: Path) -> None:
    image_path = tmp_path / "plan.png"
    Image.new("RGB", (4, 4)).save(image_path)
    try:
        load_plan_page(image_path, page_number=2)
    except ValueError as exc:
        assert "only page 1" in str(exc)
    else:
        raise AssertionError("Expected a page-number error")


def test_load_plan_page_rasterizes_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "plan.pdf"
    Image.new("RGB", (20, 10), "white").save(pdf_path)
    image = load_plan_page(pdf_path, dpi=72)
    assert image.shape == (10, 20, 3)
