from pathlib import Path

import numpy as np

from floorplan_di.constants import PlanClass
from floorplan_di.data.masks import parse_points, rasterize_annotation

SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">
  <g class="Space Bedroom"><polygon points="1,1 18,1 18,18 1,18 "/></g>
  <g class="Space Outdoor"><polygon points="0,0 2,0 2,2 0,2 "/></g>
  <g id="Wall"><polygon points="8,1 11,1 11,18 8,18 "/></g>
  <g id="Railing"><polygon points="1,3 18,3 18,4 1,4 "/></g>
  <g id="Door"><polygon points="8,8 11,8 11,11 8,11 "/></g>
  <g id="Window"><polygon points="8,2 11,2 11,4 8,4 "/></g>
</svg>
"""


def test_parse_points_supports_svg_spacing_and_decimals() -> None:
    points = parse_points("1.2,2.8  4,5\n6, 7")
    assert points.tolist() == [[1, 3], [4, 5], [6, 7]]


def test_mapping_and_precedence(tmp_path: Path) -> None:
    path = tmp_path / "model.svg"
    path.write_text(SVG, encoding="utf-8")
    result = rasterize_annotation(path, 20, 20)

    assert result.object_counts == {"room": 1, "wall": 1, "door": 1, "window": 1}
    assert result.ignored_counts == {"outdoor_space": 1, "railing": 1}
    assert result.mask[5, 5] == PlanClass.ROOM
    assert result.mask[6, 9] == PlanClass.WALL
    assert result.mask[9, 9] == PlanClass.DOOR
    assert result.mask[3, 9] == PlanClass.WINDOW
    assert result.mask.dtype == np.uint8
