from pathlib import Path

import pytest

from floorplan_di.data.cubicasa import load_official_split


def test_official_split_resolves_samples_inside_dataset_root(tmp_path: Path) -> None:
    (tmp_path / "train.txt").write_text("high_quality/123\n", encoding="utf-8")

    samples = load_official_split(tmp_path, "train")

    assert samples[0].sample_id == "high_quality/123"
    assert samples[0].directory == (tmp_path / "high_quality" / "123").resolve()


@pytest.mark.parametrize("entry", ["../outside", "/absolute/path"])
def test_official_split_rejects_paths_outside_dataset_root(tmp_path: Path, entry: str) -> None:
    (tmp_path / "train.txt").write_text(entry, encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe path on line 1"):
        load_official_split(tmp_path, "train")
