"""Integration style tests for the colour-based inference workflow."""

from __future__ import annotations

from pathlib import Path

from ai_lens_helper import Lens
from ai_lens_helper.infer.index import IndexBuilder, IndexOptions
from ai_lens_helper.utils.image import write_solid_ppm


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    write_solid_ppm(path, color)


def test_build_index_and_run_inference(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    for idx in range(3):
        _make_image(data_root / "HallA" / "red_exhibit" / f"r_{idx}.ppm", (255, 0, 0))
        _make_image(data_root / "HallA" / "green_exhibit" / f"g_{idx}.ppm", (0, 255, 0))

    options = IndexOptions(
        model_path=tmp_path / "dummy.ckpt",
        data_root=data_root,
        place="HallA",
        save_path=tmp_path / "hall.index.json",
    )
    summary = IndexBuilder(options=options).build()

    assert summary.item_count == 2
    assert summary.image_count == 6
    assert summary.save_path.exists()

    lens = Lens(model_path=summary.save_path)

    query_accept = tmp_path / "query_accept.ppm"
    _make_image(query_accept, (250, 10, 10))
    result_accept = lens.infer(place="HallA", image_path=query_accept, topk=2)

    assert result_accept.decision == "accept"
    assert result_accept.selected_item == "red_exhibit"
    assert result_accept.confidence is not None and result_accept.confidence > 0.7
    assert len(result_accept.predictions) == 2

    query_reject = tmp_path / "query_reject.ppm"
    _make_image(query_reject, (0, 0, 255))
    result_reject = lens.infer(place="HallA", image_path=query_reject, topk=2)

    assert result_reject.decision == "recollect"
    assert result_reject.hints and len(result_reject.hints) <= 2

    forced_accept = lens.infer(
        place="HallA",
        image_path=query_reject,
        topk=1,
        reject_threshold=0.0,
    )
    assert forced_accept.decision == "accept"
