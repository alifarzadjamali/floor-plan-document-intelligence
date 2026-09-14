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
        candidates = [(room, _distance_to_polygon(centre, room["polygon"])) for room in rooms]  # type: ignore[arg-type]
        inside = [(room, distance) for room, distance in candidates if distance >= 0]
        if inside:
            room, distance = max(inside, key=lambda item: item[1])
            method = "contains"
        elif candidates:
            room, signed_distance = max(candidates, key=lambda item: item[1])
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
