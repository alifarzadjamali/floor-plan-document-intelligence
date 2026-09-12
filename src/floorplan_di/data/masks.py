from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from defusedxml import ElementTree

from floorplan_di.constants import PlanClass

_POINT_RE = re.compile(
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*,\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


@dataclass(frozen=True)
class AnnotationResult:
    mask: np.ndarray
    object_counts: dict[str, int]
    source_polygons: dict[str, list[np.ndarray]]
    ignored_counts: dict[str, int]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _direct_polygon(group: ElementTree.Element) -> ElementTree.Element | None:
    return next((child for child in group if _local_name(child.tag) == "polygon"), None)


def parse_points(text: str) -> np.ndarray:
    points = [(float(x), float(y)) for x, y in _POINT_RE.findall(text)]
    if len(points) < 3:
        raise ValueError(f"Polygon has fewer than three valid points: {text[:80]!r}")
    return np.rint(np.asarray(points)).astype(np.int32)


def _category(group: ElementTree.Element) -> tuple[str | None, str | None]:
    source_id = group.attrib.get("id", "")
    source_classes = group.attrib.get("class", "").split()
    if source_id == "Wall":
        return "wall", None
    if source_id == "Door":
        return "door", None
    if source_id == "Window":
        return "window", None
    if "Space" in source_classes:
        try:
            subtype = source_classes[source_classes.index("Space") + 1]
        except (ValueError, IndexError):
            subtype = "Unknown"
        return (None, "outdoor_space") if subtype == "Outdoor" else ("room", None)
    if source_id == "Railing":
        return None, "railing"
    return None, None


def rasterize_annotation(svg_path: Path, height: int, width: int) -> AnnotationResult:
    """Rasterise the verified task ontology in image pixel coordinates.

    CubiCasa's official parser uses polygon coordinates directly against the
    dimensions of F1_scaled.png. Drawing low-priority classes first makes overlap
    behaviour explicit and independent of source XML order.
    """
    root = ElementTree.parse(svg_path).getroot()
    polygons: dict[str, list[np.ndarray]] = {
        "room": [],
        "wall": [],
        "door": [],
        "window": [],
    }
    ignored_counts = {"outdoor_space": 0, "railing": 0}
    parsing_errors: list[str] = []

    for group in (element for element in root.iter() if _local_name(element.tag) == "g"):
        category, ignored = _category(group)
        if ignored:
            ignored_counts[ignored] += 1
        if category is None:
            continue
        polygon = _direct_polygon(group)
        if polygon is None:
            parsing_errors.append(f"{category}: missing direct polygon")
            continue
        try:
            points = parse_points(polygon.attrib.get("points", ""))
        except ValueError as exc:
            parsing_errors.append(f"{category}: {exc}")
            continue
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        polygons[category].append(points)

    if parsing_errors:
        preview = "; ".join(parsing_errors[:5])
        raise ValueError(f"Could not parse {len(parsing_errors)} target polygons: {preview}")

    mask = np.zeros((height, width), dtype=np.uint8)
    for category, class_id in (
        ("room", PlanClass.ROOM),
        ("wall", PlanClass.WALL),
        ("door", PlanClass.DOOR),
        ("window", PlanClass.WINDOW),
    ):
        if polygons[category]:
            cv2.fillPoly(mask, polygons[category], int(class_id))

    return AnnotationResult(
        mask=mask,
        object_counts={name: len(items) for name, items in polygons.items()},
        source_polygons=polygons,
        ignored_counts=ignored_counts,
    )
