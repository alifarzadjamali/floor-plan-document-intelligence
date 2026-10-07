from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class CubiCasaSample:
    sample_id: str
    directory: Path

    @property
    def image_path(self) -> Path:
        return self.directory / "F1_scaled.png"

    @property
    def original_image_path(self) -> Path:
        return self.directory / "F1_original.png"

    @property
    def annotation_path(self) -> Path:
        return self.directory / "model.svg"

    def image_size(self) -> tuple[int, int]:
        with Image.open(self.image_path) as image:
            return image.size


def load_official_split(root: Path, split: str) -> list[CubiCasaSample]:
    """Load one official split without changing its order or membership."""
    split_path = root / f"{split}.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"Official split file not found: {split_path}")

    samples: list[CubiCasaSample] = []
    resolved_root = root.resolve()
    for line_number, raw_line in enumerate(
        split_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        relative = raw_line.strip().replace("\\", "/")
        if not relative:
            continue
        relative_path = Path(relative)
        directory = (root / relative_path).resolve()
        if relative_path.is_absolute() or not directory.is_relative_to(resolved_root):
            raise ValueError(
                f"Official split contains an unsafe path on line {line_number}: {relative!r}"
            )
        samples.append(CubiCasaSample(relative, directory))
    if not samples:
        raise ValueError(f"Official split is empty: {split_path}")
    return samples


class CubiCasaDataset:
    def __init__(self, root: Path, split: str) -> None:
        self.root = root
        self.split = split
        self.samples = load_official_split(root, split)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[CubiCasaSample]:
        return iter(self.samples)

    def __getitem__(self, index: int) -> CubiCasaSample:
        return self.samples[index]
