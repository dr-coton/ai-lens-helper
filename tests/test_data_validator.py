"""Tests for dataset validation helpers."""

from __future__ import annotations

from pathlib import Path

from ai_lens_helper.train.datamodule import DataValidator


def _touch_images(folder: Path, count: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (folder / f"img_{index:03d}.jpg").write_bytes(b"")


def test_validation_reports_issues_for_small_items(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _touch_images(data_root / "A" / "exhibit_red", 5)
    _touch_images(data_root / "A" / "exhibit_green", 12)
    (data_root / "B").mkdir(parents=True, exist_ok=True)

    validator = DataValidator(data_root=data_root, min_images=10)
    report = validator.validate()

    assert len(report.place_summaries) == 2
    place_a = next(summary for summary in report.place_summaries if summary.place == "A")
    assert place_a.item_count == 2
    assert place_a.items_meeting_requirement == 1

    # Issues should mention the under-populated item and the empty place.
    assert any("exhibit_red" in issue for issue in report.issues)
    assert any("contains no exhibit folders" in issue for issue in report.issues)


def test_validation_handles_missing_root(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"
    validator = DataValidator(data_root=missing_root, min_images=10)
    report = validator.validate()

    assert report.place_summaries == []
    assert report.issues == [f"Data root '{missing_root}' does not exist."]
