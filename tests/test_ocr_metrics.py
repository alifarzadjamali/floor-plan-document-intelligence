from itertools import product

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
    assert levenshtein_distance("A", "A" * 10_000) == 9_999
    assert normalise_text("  living\n room ") == "LIVING ROOM"


def test_ocr_metrics_accept_a_single_pass_iterator() -> None:
    pairs = ((reference, prediction) for reference, prediction in [("BEDROOM", "BEDROOM")])

    metrics = benchmark_metrics(pairs)

    assert metrics["samples"] == 1
    assert metrics["room_label_exact_match_accuracy"] == 1.0


def test_edit_distance_trims_long_matching_affixes() -> None:
    prefix, suffix = "A" * 10_000, "Z" * 10_000
    assert levenshtein_distance(prefix + "BC" + suffix, prefix + "D" + suffix) == 2
    assert levenshtein_distance(prefix, prefix) == 0
    assert levenshtein_distance(["ROOM", "ONE"], ["ROOM", "TWO"]) == 1


def test_edit_distance_matches_full_matrix_for_short_strings() -> None:
    strings = ["".join(chars) for length in range(4) for chars in product("AB", repeat=length)]
    for reference, hypothesis in product(strings, repeat=2):
        matrix = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
        for row in range(len(reference) + 1):
            matrix[row][0] = row
        for column in range(len(hypothesis) + 1):
            matrix[0][column] = column
        for row, source in enumerate(reference, 1):
            for column, target in enumerate(hypothesis, 1):
                matrix[row][column] = min(
                    matrix[row - 1][column] + 1,
                    matrix[row][column - 1] + 1,
                    matrix[row - 1][column - 1] + (source != target),
                )
        assert levenshtein_distance(reference, hypothesis) == matrix[-1][-1]
