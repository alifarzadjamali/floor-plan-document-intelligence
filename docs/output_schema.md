# Structured Output Schema

The inference pipeline produces a JSON document with schema version `1.0`.

Coordinates are expressed in pixels relative to the original input image. The
coordinate origin is the top-left corner. The output does not represent CAD,
BIM, physical dimensions, or real-world scale.

Consumers should treat `schema_version` as the compatibility boundary and keep
unknown fields forward-compatible. Review warnings are part of the normal
output contract, so callers should surface them rather than silently dropping
the associated predictions.

## Top-level fields

- `schema_version`: structured-output format version.
- `created_utc`: UTC timestamp when the document was generated.
- `coordinate_system`: pixel coordinate metadata and source image dimensions.
- `input`: source path and PDF page number when applicable.
- `rooms`: detected room regions.
- `walls`: detected wall regions.
- `doors`: detected door regions.
- `windows`: detected window regions.
- `text_entities`: OCR-derived text entities.
- `text_to_room_links`: conservative associations between OCR text and rooms.
- `door_to_room_links`: candidate door-to-room associations.
- `room_connectivity`: room connections supported by resolved door links.
- `review_warnings`: predictions requiring manual review.

## Geometry entities

Room, wall, door and window entities can include:

- `id`
- `polygon`
- `bbox`
- `centroid`
- `area_pixels`
- `polygon_area_pixels`
- `geometry_valid`
- `confidence`
- `confidence_band`

Wall entities can additionally include an approximate `centerline`.

Door and window entities can additionally include `orientation_degrees`.

## Confidence bands

Predictions are mapped to three review-oriented confidence bands:

- `high`
- `medium`
- `low_review`

These labels support review prioritisation. They should not be interpreted as
guarantees that the extracted geometry is correct.

## Review warnings

The pipeline explicitly records unresolved or uncertain output instead of
silently treating it as reliable. Current warnings include low-confidence
openings, unresolved door-to-room links and OCR text that could not be linked
to a room.

A complete generated example is available under
`results/phase4/test_1191/structured_output.json`.
