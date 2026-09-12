"""CubiCasa5K ingestion and annotation rasterisation."""

from .cubicasa import CubiCasaDataset, CubiCasaSample, load_official_split
from .masks import AnnotationResult, rasterize_annotation

__all__ = [
    "AnnotationResult",
    "CubiCasaDataset",
    "CubiCasaSample",
    "load_official_split",
    "rasterize_annotation",
]
