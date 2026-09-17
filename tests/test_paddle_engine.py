from floorplan_di.ocr.paddle_engine import _normalise_payload, paddle_available


def test_paddle_result_normalisation_is_backend_independent() -> None:
    rows = _normalise_payload(
        {"res": {"rec_texts": ["KITCHEN"], "rec_scores": [0.8], "rec_boxes": [[1, 2, 3, 4]]}}
    )
    assert rows[0]["text"] == "KITCHEN"
    assert rows[0]["confidence"] == 0.8


def test_paddle_backend_is_optional() -> None:
    assert isinstance(paddle_available(), bool)
