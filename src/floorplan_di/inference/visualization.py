from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from floorplan_di.visualization import overlay_mask


def structured_overlay(
    image: np.ndarray, mask: np.ndarray, document: dict[str, object]
) -> np.ndarray:
    canvas = Image.fromarray(overlay_mask(image, mask, alpha=0.35))
    draw = ImageDraw.Draw(canvas)
    colours = {"rooms": "deepskyblue", "walls": "black", "doors": "orange", "windows": "limegreen"}
    for category, colour in colours.items():
        for item in document[category]:  # type: ignore[index]
            points = [tuple(point) for point in item["polygon"]]
            if len(points) >= 2:
                draw.line(points + [points[0]], fill=colour, width=2)
            draw.text(tuple(item["centroid"]), item["id"], fill=colour)
    for token in document["text_entities"]:  # type: ignore[index]
        x, y, width, height = token["bbox"]
        draw.rectangle((x, y, x + width, y + height), outline="red", width=2)
        draw.text((x, max(0, y - 12)), token["text"], fill="red")
    return np.asarray(canvas)
