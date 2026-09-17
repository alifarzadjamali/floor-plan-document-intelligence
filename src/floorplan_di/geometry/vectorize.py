from __future__ import annotations

import math

import cv2
import numpy as np

from floorplan_di.constants import PlanClass
from floorplan_di.evaluation.calibration import confidence_band


def _points(contour: np.ndarray) -> list[list[int]]:
    return [[int(x), int(y)] for x, y in contour.reshape(-1, 2)]


def _component_confidence(
    probability: np.ndarray | None, component: np.ndarray, class_id: int
) -> float | None:
    """Return the mean class probability inside one connected component."""
    if probability is None:
        return None
    values = probability[class_id][component.astype(bool)]
    return round(float(values.mean()), 4) if values.size else None


def _orientation(component: np.ndarray) -> float | None:
    """Estimate a component's main axis in degrees when it has enough pixels."""
    ys, xs = np.where(component)
    if len(xs) < 2:
        return None
    covariance = np.cov(np.column_stack((xs, ys)), rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    vector = vectors[:, int(np.argmax(values))]
    return round(float(math.degrees(math.atan2(vector[1], vector[0])) % 180), 1)


def _centerline(component: np.ndarray) -> list[list[int]] | None:
    """Return a PCA major-axis segment for a wall component."""
    ys, xs = np.where(component)
    if len(xs) < 2:
        return None
    points = np.column_stack((xs, ys)).astype(float)
    centre = points.mean(axis=0)
    _, _, vectors = np.linalg.svd(points - centre, full_matrices=False)
    axis = vectors[0]
    projections = (points - centre) @ axis
    start = centre + axis * projections.min()
    end = centre + axis * projections.max()
    return [[int(round(start[0])), int(round(start[1]))], [int(round(end[0])), int(round(end[1]))]]


def _objects(
    mask: np.ndarray,
    probability: np.ndarray | None,
    class_id: int,
    prefix: str,
    min_area: int,
    simplify_fraction: float,
) -> list[dict[str, object]]:
    binary = (mask == class_id).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    objects: list[dict[str, object]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == label
        contours, _ = cv2.findContours(
            component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        epsilon = simplify_fraction * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        x, y, width, height, _ = stats[label]
        centre_x, centre_y = centroids[label]
        confidence = _component_confidence(probability, component, class_id)
        contour_area = float(cv2.contourArea(contour))
        objects.append(
            {
                "id": f"{prefix}{len(objects) + 1:02d}",
                "polygon": _points(polygon),
                "bbox": [int(x), int(y), int(width), int(height)],
                "centroid": [round(float(centre_x), 1), round(float(centre_y), 1)],
                "area_pixels": area,
                "polygon_area_pixels": round(contour_area, 1),
                "geometry_valid": bool(len(polygon) >= 3 and contour_area > 0.0),
                "confidence": confidence,
                "confidence_band": confidence_band(confidence) if confidence is not None else None,
            }
        )
        if class_id == PlanClass.WALL:
            objects[-1]["centerline"] = _centerline(component)
        if class_id in (PlanClass.DOOR, PlanClass.WINDOW):
            objects[-1]["orientation_degrees"] = _orientation(component)
    return objects


def vectorize_mask(
    mask: np.ndarray,
    probability: np.ndarray | None = None,
    room_min_area: int = 250,
    wall_min_area: int = 80,
    opening_min_area: int = 20,
    simplify_fraction: float = 0.008,
) -> dict[str, list[dict[str, object]]]:
    """Extract approximate polygons from a class-id mask.

    Coordinates stay in the input image's pixel coordinate system. Walls are
    returned as polygons, and small components are discarded.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2-D class-id array")
    if probability is not None and probability.shape[1:] != mask.shape:
        raise ValueError("probability must have shape (classes, height, width)")
    return {
        "rooms": _objects(mask, probability, PlanClass.ROOM, "R", room_min_area, simplify_fraction),
        "walls": _objects(mask, probability, PlanClass.WALL, "W", wall_min_area, simplify_fraction),
        "doors": _objects(
            mask, probability, PlanClass.DOOR, "D", opening_min_area, simplify_fraction
        ),
        "windows": _objects(
            mask, probability, PlanClass.WINDOW, "N", opening_min_area, simplify_fraction
        ),
    }
