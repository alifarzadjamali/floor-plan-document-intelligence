from __future__ import annotations

from heapq import nsmallest

import cv2
import numpy as np


def _distance(point: tuple[float, float], contour: np.ndarray) -> float:
    return abs(float(cv2.pointPolygonTest(contour, point, True)))


def link_doors_to_rooms(
    doors: list[dict[str, object]], rooms: list[dict[str, object]], threshold: float = 45.0
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Infer a room edge only when a door is near two distinct room boundaries."""
    door_links: list[dict[str, object]] = []
    graph: list[dict[str, object]] = []
    room_contours = [
        (room, np.asarray(room["polygon"], dtype=np.float32).reshape((-1, 1, 2)))
        for room in rooms
    ]
    for door in doors:
        x, y = door["centroid"]  # type: ignore[misc]
        closest = nsmallest(
            2,
            ((_distance((x, y), contour), room) for room, contour in room_contours),
            key=lambda item: item[0],
        )
        linked = [room for distance, room in closest if distance <= threshold]
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
