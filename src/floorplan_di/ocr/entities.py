from __future__ import annotations

import re

ROOM_LABELS = {
    "BATHROOM",
    "BEDROOM",
    "DINING",
    "GARAGE",
    "HALL",
    "KITCHEN",
    "LIVING ROOM",
    "STORE",
    "UTILITY",
    "WC",
}
DIMENSION_PATTERN = re.compile(r"^\d+(?:[.,]\d+)?\s*(?:M|MM)?$", re.IGNORECASE)
DRAWING_CODE_PATTERN = re.compile(r"^[A-Z]{1,3}-\d{2,4}$", re.IGNORECASE)


def normalise_text(value: str) -> str:
    return " ".join(value.upper().strip().split())


def classify_entity(value: str) -> str:
    text = normalise_text(value)
    if text in ROOM_LABELS or re.fullmatch(r"BEDROOM\s+\d+", text):
        return "ROOM_LABEL"
    if DIMENSION_PATTERN.fullmatch(text) or text.startswith("FFL "):
        return "DIMENSION_LIKE"
    if DRAWING_CODE_PATTERN.fullmatch(text):
        return "DRAWING_CODE"
    return "OTHER_TEXT"
