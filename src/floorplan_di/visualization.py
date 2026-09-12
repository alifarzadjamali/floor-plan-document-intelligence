from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from floorplan_di.constants import CLASS_COLOURS_RGB


def colourize_mask(mask: np.ndarray) -> np.ndarray:
    palette = np.asarray(CLASS_COLOURS_RGB, dtype=np.uint8)
    return palette[mask]


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    colour = colourize_mask(mask)
    active = mask > 0
    result = image_rgb.copy()
    result[active] = np.round(
        (1 - alpha) * image_rgb[active].astype(np.float32)
        + alpha * colour[active].astype(np.float32)
    ).astype(np.uint8)
    return result


def source_annotation_image(
    image_rgb: np.ndarray, source_polygons: dict[str, list[np.ndarray]]
) -> np.ndarray:
    canvas = image_rgb.copy()
    for class_id, category in enumerate(("room", "wall", "door", "window"), start=1):
        colour = CLASS_COLOURS_RGB[class_id]
        for polygon in source_polygons[category]:
            cv2.polylines(canvas, [polygon], True, colour, max(2, min(canvas.shape[:2]) // 400))
    return canvas


def save_contact_sheet(
    panels: list[tuple[str, np.ndarray]], output_path: Path, max_panel_width: int = 700
) -> None:
    prepared: list[tuple[str, Image.Image]] = []
    for title, array in panels:
        image = Image.fromarray(array)
        if image.width > max_panel_width:
            height = round(image.height * max_panel_width / image.width)
            image = image.resize((max_panel_width, height), Image.Resampling.LANCZOS)
        prepared.append((title, image))

    margin, header = 16, 38
    width = sum(image.width for _, image in prepared) + margin * (len(prepared) + 1)
    height = max(image.height for _, image in prepared) + header + 2 * margin
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    x = margin
    for title, image in prepared:
        draw.text((x, margin), title, fill="black")
        sheet.paste(image, (x, margin + header))
        x += image.width + margin
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, optimize=True)
