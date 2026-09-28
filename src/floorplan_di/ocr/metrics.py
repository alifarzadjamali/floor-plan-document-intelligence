from __future__ import annotations

from collections.abc import Iterable, Sequence

from floorplan_di.ocr.entities import classify_entity, normalise_text


def levenshtein_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, source in enumerate(reference, start=1):
        current = [row]
        for column, target in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + int(source != target),
                )
            )
        previous = current
    return previous[-1]


def benchmark_metrics(pairs: Iterable[tuple[str, str]]) -> dict[str, float | int]:
    materialised = [
        (normalise_text(reference), normalise_text(prediction)) for reference, prediction in pairs
    ]
    character_errors = sum(levenshtein_distance(ref, pred) for ref, pred in materialised)
    characters = sum(len(ref) for ref, _ in materialised)
    tokenised = [(ref.split(), pred.split()) for ref, pred in materialised]
    word_errors = sum(levenshtein_distance(ref, pred) for ref, pred in tokenised)
    words = sum(len(ref) for ref, _ in tokenised)
    exact = sum(reference == prediction for reference, prediction in materialised)
    numeric: list[tuple[str, str]] = []
    rooms: list[tuple[str, str]] = []
    for pair in materialised:
        category = classify_entity(pair[0])
        if category == "DIMENSION_LIKE":
            numeric.append(pair)
        elif category == "ROOM_LABEL":
            rooms.append(pair)
    return {
        "samples": len(materialised),
        "cer": character_errors / max(characters, 1),
        "wer": word_errors / max(words, 1),
        "exact_match_accuracy": exact / max(len(materialised), 1),
        "numeric_token_exact_match_accuracy": sum(ref == pred for ref, pred in numeric)
        / max(len(numeric), 1),
        "room_label_exact_match_accuracy": sum(ref == pred for ref, pred in rooms)
        / max(len(rooms), 1),
        "numeric_samples": len(numeric),
        "room_label_samples": len(rooms),
    }
