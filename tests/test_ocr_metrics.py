from floorplan_di.ocr.entities import classify_entity, normalise_text
from floorplan_di.ocr.metrics import benchmark_metrics, levenshtein_distance


def test_entity_classification() -> None:
    assert classify_entity(" kitchen ") == "ROOM_LABEL"
    assert classify_entity("3.20 m") == "DIMENSION_LIKE"
    assert classify_entity("A-102") == "DRAWING_CODE"
    assert classify_entity("north") == "OTHER_TEXT"


def test_ocr_metrics_are_exact_for_exact_text() -> None:
    metrics = benchmark_metrics([("KITCHEN", " kitchen "), ("3.20 m", "3.20 M")])
    assert metrics["cer"] == 0.0
    assert metrics["wer"] == 0.0
    assert metrics["exact_match_accuracy"] == 1.0
    assert metrics["numeric_token_exact_match_accuracy"] == 1.0


def test_levenshtein_distance() -> None:
    assert levenshtein_distance(list("KITCHEN"), list("KICTHEN")) == 2
    assert normalise_text("  living\n room ") == "LIVING ROOM"
