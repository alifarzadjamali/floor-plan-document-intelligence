"""Selection rules for transparent, repeatable qualitative error analysis."""

from __future__ import annotations


def choose_error_examples(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Choose evidence using fixed extrema rather than hand-picked favourable examples."""
    if not records:
        raise ValueError("At least one record is required")
    return {
        "strong_segmentation": max(records, key=lambda row: float(row["foreground_iou"])),
        "weak_segmentation": min(records, key=lambda row: float(row["foreground_iou"])),
        "missed_thin_wall_or_opening": max(records, key=lambda row: int(row["missed_openings"])),
        "false_door_or_window": max(records, key=lambda row: int(row["false_openings"])),
        "successful_text_to_room": max(records, key=lambda row: int(row["text_links"])),
        "ambiguous_text_to_room": max(records, key=lambda row: int(row["unassigned_text"])),
        "successful_doorway_relationship": max(records, key=lambda row: int(row["resolved_doors"])),
        "unresolved_doorway_relationship": max(
            records, key=lambda row: int(row["unresolved_doors"])
        ),
    }
