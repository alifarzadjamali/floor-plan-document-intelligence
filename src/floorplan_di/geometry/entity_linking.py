from __future__ import annotations

import cv2
import numpy as np


def _distance_to_polygon(point: tuple[float, float], polygon: list[list[int]]) -> float:
    contour = np.asarray(polygon, dtype=np.float32).reshape((-1, 1, 2))
    return float(cv2.pointPolygonTest(contour, point, True))


def link_text_to_rooms(
    tokens: list[dict[str, object]], rooms: list[dict[str, object]], nearest_threshold: float = 30.0
) -> list[dict[str, object]]:
    """Assign text-box centres only when inside, or conservatively near, a room."""
    links: list[dict[str, object]] = []
    for token in tokens:
        x, y, width, height = token["bbox"]  # type: ignore[misc]
        centre = (float(x + width / 2), float(y + height / 2))
        containing: tuple[dict[str, object], float] | None = None
        nearest: tuple[dict[str, object], float] | None = None
        for room in rooms:
            signed_distance = _distance_to_polygon(centre, room["polygon"])  # type: ignore[arg-type]
            if signed_distance >= 0 and (
                containing is None or signed_distance > containing[1]
            ):
                containing = (room, signed_distance)
            if nearest is None or signed_distance > nearest[1]:
                nearest = (room, signed_distance)

        if containing is not None:
            room, distance = containing
            method = "contains"
        elif nearest is not None:
            room, signed_distance = nearest
            distance = abs(signed_distance)
            method = "nearest"
            if distance > nearest_threshold:
                continue
        else:
            continue
        links.append(
            {
                "text_id": token["id"],
                "room_id": room["id"],
                "method": method,
                "distance_pixels": round(distance, 2),
            }
        )
    return links
