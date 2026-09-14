from __future__ import annotations

import cv2
import numpy as np


def _distance(point: tuple[float, float], polygon: list[list[int]]) -> float:
    contour = np.asarray(polygon, dtype=np.float32).reshape((-1, 1, 2))
    return abs(float(cv2.pointPolygonTest(contour, point, True)))


def link_doors_to_rooms(
    doors: list[dict[str, object]], rooms: list[dict[str, object]], threshold: float = 45.0
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Infer a room edge only when a door is near two distinct room boundaries."""
    door_links: list[dict[str, object]] = []
    graph: list[dict[str, object]] = []
    for door in doors:
        x, y = door["centroid"]  # type: ignore[misc]
        closest = sorted(
            ((_distance((x, y), room["polygon"]), room) for room in rooms), key=lambda item: item[0]
        )  # type: ignore[arg-type]
        linked = [room for distance, room in closest[:2] if distance <= threshold]
        door_links.append(
            {
                "door_id": door["id"],
                "room_ids": [room["id"] for room in linked],
                "resolved": len(linked) == 2,
            }
        )
        if len(linked) == 2:
            graph.append({"door_id": door["id"], "rooms": [linked[0]["id"], linked[1]["id"]]})
    return door_links, graph
