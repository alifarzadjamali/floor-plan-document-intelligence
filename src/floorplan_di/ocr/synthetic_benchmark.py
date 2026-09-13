from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from floorplan_di.ocr.entities import classify_entity

LABELS = (
    "KITCHEN",
    "BEDROOM",
    "LIVING ROOM",
    "BATHROOM",
    "WC",
    "HALL",
    "STORE",
    "UTILITY",
    "GARAGE",
    "3.20 m",
    "2.45 m",
    "4200",
    "900",
    "1200",
    "FFL +0.150",
    "A-102",
)
FONT_NAMES = ("arial.ttf", "bahnschrift.ttf", "calibri.ttf", "times.ttf")


def _font_paths() -> list[Path]:
    root = Path(r"C:\Windows\Fonts")
    available = [root / name for name in FONT_NAMES if (root / name).is_file()]
    if len(available) < 2:
        raise FileNotFoundError("The benchmark requires at least two installed TrueType fonts")
    return available


def render_crop(
    text: str,
    font_path: Path,
    font_size: int,
    orientation: int,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> Image.Image:
    background = int(rng.integers(235, 256))
    image = Image.new("RGB", (width, height), (background, background, min(255, background + 2)))
    draw = ImageDraw.Draw(image)
    for _ in range(int(rng.integers(2, 6))):
        colour = int(rng.integers(165, 210))
        if rng.random() < 0.5:
            y = int(rng.integers(5, height - 5))
            draw.line((0, y, width, y + int(rng.integers(-2, 3))), fill=(colour,) * 3, width=1)
        else:
            x = int(rng.integers(5, width - 5))
            draw.line((x, 0, x + int(rng.integers(-2, 3)), height), fill=(colour,) * 3, width=1)
    font = ImageFont.truetype(str(font_path), font_size)
    box = draw.textbbox((0, 0), text, font=font)
    x = max(12, (width - (box[2] - box[0])) // 2 + int(rng.integers(-12, 13)))
    y = max(8, (height - (box[3] - box[1])) // 2 + int(rng.integers(-10, 11)))
    draw.text((x, y), text, font=font, fill=(int(rng.integers(0, 55)),) * 3)
    if rng.random() < 0.6:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.0, 0.7))))
    array = np.asarray(image).astype(np.float32)
    array += rng.normal(0, rng.uniform(0, 4), size=array.shape)
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(rng.integers(76, 96)))
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    if orientation:
        image = image.rotate(orientation, expand=True, fillcolor=(255, 255, 255))
    return image


def generate_benchmark(config: dict[str, object], output: Path) -> dict[str, int]:
    benchmark = config["benchmark"]
    fonts = _font_paths()
    counts: dict[str, int] = {}
    for split, offset in (("train", 0), ("val", 10_000), ("test", 20_000)):
        count = int(
            benchmark[
                {"train": "train_samples", "val": "validation_samples", "test": "test_samples"}[
                    split
                ]
            ]
        )
        rng = np.random.default_rng(int(config["seed"]) + offset)
        records = []
        for index in range(count):
            text = str(rng.choice(LABELS))
            orientation = int(rng.choice(benchmark["orientations"]))
            crop = render_crop(
                text,
                fonts[int(rng.integers(len(fonts)))],
                int(rng.choice(benchmark["font_sizes"])),
                orientation,
                int(benchmark["canvas_width"]),
                int(benchmark["canvas_height"]),
                rng,
            )
            image_path = output / "images" / split / f"{index:04d}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(image_path, quality=95)
            records.append(
                {
                    "id": f"{split}_{index:04d}",
                    "image": image_path.as_posix(),
                    "text": text,
                    "entity_type": classify_entity(text),
                    "orientation": orientation,
                }
            )
        manifest = output / "manifests" / f"{split}.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        counts[split] = count
    return counts
