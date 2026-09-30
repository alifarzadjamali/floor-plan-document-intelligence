from __future__ import annotations

from collections.abc import Iterable, Sequence

from floorplan_di.ocr.entities import classify_entity, normalise_text


def levenshtein_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Return edit distance using memory proportional to the shorter input."""
    if len(hypothesis) > len(reference):
        reference, hypothesis = hypothesis, reference
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
    samples = 0
    character_errors = 0
    characters = 0
    word_errors = 0
    words = 0
    exact = 0
    numeric_samples = 0
    numeric_exact = 0
    room_samples = 0
    room_exact = 0
    for raw_reference, raw_prediction in pairs:
        reference = normalise_text(raw_reference)
        prediction = normalise_text(raw_prediction)
        samples += 1
        character_errors += levenshtein_distance(reference, prediction)
        characters += len(reference)
        reference_words = reference.split()
        word_errors += levenshtein_distance(reference_words, prediction.split())
        words += len(reference_words)
        is_exact = reference == prediction
        exact += is_exact
        category = classify_entity(reference)
        if category == "DIMENSION_LIKE":
            numeric_samples += 1
            numeric_exact += is_exact
        elif category == "ROOM_LABEL":
            room_samples += 1
            room_exact += is_exact
    return {
        "samples": samples,
        "cer": character_errors / max(characters, 1),
        "wer": word_errors / max(words, 1),
        "exact_match_accuracy": exact / max(samples, 1),
        "numeric_token_exact_match_accuracy": numeric_exact / max(numeric_samples, 1),
        "room_label_exact_match_accuracy": room_exact / max(room_samples, 1),
        "numeric_samples": numeric_samples,
        "room_label_samples": room_samples,
    }
