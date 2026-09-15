from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path


def structured_document(
    input_path: Path,
    image_shape: tuple[int, int],
    geometry: dict[str, list[dict[str, object]]],
    text_entities: list[dict[str, object]],
    text_links: list[dict[str, object]],
    door_links: list[dict[str, object]],
    room_graph: list[dict[str, object]],
    page_number: int = 1,
) -> dict[str, object]:
    """Create portable JSON without implying metric or CAD accuracy."""
    warnings = []
    for category in ("doors", "windows"):
        for item in geometry[category]:
            if item["confidence"] is not None and item["confidence"] < 0.5:
                warnings.append({"entity_id": item["id"], "reason": "low_segmentation_confidence"})
    warnings.extend(
        {"entity_id": link["door_id"], "reason": "unresolved_door_to_room"}
        for link in door_links
        if not link["resolved"]
    )
    linked_text_ids = {link["text_id"] for link in text_links}
    warnings.extend(
        {"entity_id": token["id"], "reason": "unassigned_text"}
        for token in text_entities
        if token["id"] not in linked_text_ids
    )
    return {
        "schema_version": "1.0",
        "created_utc": datetime.now(UTC).isoformat(),
        "coordinate_system": {
            "unit": "pixel",
            "origin": "top_left",
            "image_width": image_shape[1],
            "image_height": image_shape[0],
        },
        "input": {"path": str(input_path), "page_number": page_number},
        **geometry,
        "text_entities": text_entities,
        "text_to_room_links": text_links,
        "door_to_room_links": door_links,
        "room_connectivity": room_graph,
        "review_warnings": warnings,
    }
