from __future__ import annotations

from collections.abc import Iterable

from floorplan_di.ocr.entities import classify_entity, normalise_text


def levenshtein_distance(reference: list[str], hypothesis: list[str]) -> int:
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
    character_errors = sum(
        levenshtein_distance(list(ref), list(pred)) for ref, pred in materialised
    )
    characters = sum(len(ref) for ref, _ in materialised)
    word_errors = sum(levenshtein_distance(ref.split(), pred.split()) for ref, pred in materialised)
    words = sum(len(ref.split()) for ref, _ in materialised)
    exact = sum(reference == prediction for reference, prediction in materialised)
    numeric = [pair for pair in materialised if classify_entity(pair[0]) == "DIMENSION_LIKE"]
    rooms = [pair for pair in materialised if classify_entity(pair[0]) == "ROOM_LABEL"]
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
