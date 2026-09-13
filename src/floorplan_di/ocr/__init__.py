"""Tesseract OCR, preprocessing and benchmark utilities."""

from .entities import classify_entity
from .tesseract_engine import OCRResult, configure_tesseract, recognise_crop, recognise_page

__all__ = [
    "OCRResult",
    "classify_entity",
    "configure_tesseract",
    "recognise_crop",
    "recognise_page",
]
