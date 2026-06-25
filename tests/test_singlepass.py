"""Tests for the single-pass video pipeline (artvid/pipeline/singlepass.py).

The long-term frame-selection logic and the filename helpers are pure
(torch-free) ports of ``artistic_video.lua:159-187`` and
``artistic_video_core.lua:558-578``; they are tested directly without torch.
Tensor-level behaviour (warp/temporal wiring) is gated on the conftest ``torch``
fixture so the suite still runs where torch is unavailable.
"""

from __future__ import annotations

import pytest

from artvid.config import Config
from artvid.pipeline.singlepass import (
    build_out_filename,
    format_flow_filename,
    select_previous_indices,
)


# ---------------------------------------------------------------------------
# select_previous_indices — long-term J selection (artistic_video.lua:159-187)
# ---------------------------------------------------------------------------

def test_first_frame_has_no_previous():
    # frame_idx == start_number -> empty (legacy guard at :159).
    assert select_previous_indices(1, 1, (1,)) == []


def test_temporal_weight_zero_disables():
    # temporal_weight == 0 -> no temporal targets (legacy guard at :159).
    assert select_previous_indices(5, 1, (1, 2, 4), temporal_weight=0) == []


def test_single_relative_index_immediate_previous():
    # Default flow_relative_indices=(1,): the immediately previous frame.
    assert select_previous_indices(5, 1, (1,)) == [4]


def test_longterm_indices_sorted_descending():
    # (1, 2, 4) from frame 10 -> {9, 8, 6}, sorted descending (closest first).
    assert select_previous_indices(10, 1, (1, 2, 4)) == [9, 8, 6]


def test_indices_clamped_to_start_number():
    # frame 3, start 1, steps (1, 2, 4): 3-4 = -1 < start -> dropped.
    assert select_previous_indices(3, 1, (1, 2, 4)) == [2, 1]


def test_indices_respect_nondefault_start_number():
    # start_number=10: 12-1=11 (ok), 12-4=8 (< start, dropped).
    assert select_previous_indices(12, 10, (1, 4)) == [11]


def test_use_flow_every_adds_strided_previous_frames():
    # frame 10, start 1, steps (1,), use_flow_every=3:
    #   from relative: 9
    #   strided down by 3 from 10-3=7: 7, 4, 1
    # combined + sorted desc.
    got = select_previous_indices(10, 1, (1,), use_flow_every=3)
    assert got == [9, 7, 4, 1]


def test_use_flow_every_dedups_against_relative():
    # frame 7, start 1, steps (3,) -> relative gives 4.
    # use_flow_every=3: strided from 7-3=4 (already present, skipped), then 1.
    got = select_previous_indices(7, 1, (3,), use_flow_every=3)
    assert got == [4, 1]


def test_relative_index_duplicates_are_kept():
    # The legacy code does NOT de-dup duplicate relative indices against each
    # other (only use_flow_every additions are de-duped). (1, 1) keeps both.
    got = select_previous_indices(5, 1, (1, 1))
    assert got == [4, 4]


def test_use_flow_every_disabled_by_default():
    # use_flow_every=-1 (default) adds nothing beyond the relative indices.
    assert select_previous_indices(20, 1, (1, 2)) == [19, 18]


# ---------------------------------------------------------------------------
# format_flow_filename — getFormatedFlowFileName (artistic_video_core.lua:571)
# ---------------------------------------------------------------------------

def test_format_flow_filename_default_backward_pattern():
    # Default flow_pattern 'backward_[%d]_{%d}.flo':
    #   [...] = to = frame_idx, {...} = from = prev_index.
    # Legacy call: getFormatedFlowFileName(pattern, prev_index, frame_idx).
    got = format_flow_filename("backward_[%d]_{%d}.flo", 6, 7)
    assert got == "backward_7_6.flo"


def test_format_flow_filename_zero_padded():
    got = format_flow_filename("backward_[%04d]_{%04d}.flo", 6, 7)
    assert got == "backward_0007_0006.flo"


def test_format_flow_filename_reliable_pattern():
    got = format_flow_filename("reliable_[%d]_{%d}.pgm", 3, 5)
    assert got == "reliable_5_3.pgm"


def test_format_flow_filename_with_directory():
    got = format_flow_filename("flow/backward_[%d]_{%d}.flo", 1, 2)
    assert got == "flow/backward_2_1.flo"


# ---------------------------------------------------------------------------
# build_out_filename — build_OutFilename (artistic_video_core.lua:558-569)
# ---------------------------------------------------------------------------

def test_build_out_filename_default():
    cfg = Config(output_image="out.png", number_format="%d", output_folder="")
    assert build_out_filename(cfg, 1) == "out-1.png"


def test_build_out_filename_zero_padded_number_format():
    cfg = Config(output_image="out.png", number_format="%04d", output_folder="")
    assert build_out_filename(cfg, 7) == "out-0007.png"


def test_build_out_filename_with_folder():
    cfg = Config(output_image="result.jpg", number_format="%d", output_folder="frames/")
    assert build_out_filename(cfg, 12) == "frames/result-12.jpg"


def test_build_out_filename_strips_directory_from_output_image():
    # Only the basename + ext of output_image is used; the directory comes from
    # output_folder (legacy uses paths.basename / paths.extname).
    cfg = Config(output_image="some/dir/out.png", number_format="%d", output_folder="o/")
    assert build_out_filename(cfg, 3) == "o/out-3.png"


# ---------------------------------------------------------------------------
# Torch-gated: temporal-loss wiring and warp integration
# ---------------------------------------------------------------------------

def test_warp_previous_output_shape_and_validity(torch):
    from artvid.pipeline.singlepass import _warp_previous_output

    prev = torch.rand(3, 8, 10)
    flow = torch.zeros(2, 8, 10)  # identity flow -> warped == prev, all valid.
    res = _warp_previous_output(prev, flow)
    assert res.image.shape == (3, 8, 10)
    assert res.valid.shape == (1, 8, 10)
    assert torch.allclose(res.image, prev, atol=1e-5)
    assert bool(res.valid.all())


def test_temporal_loss_zero_when_matching_warped_target(torch):
    # A WeightedContentLoss against a perfectly-matching target should be ~0,
    # confirming the pixel-space temporal term is wired against the image var.
    from artvid.losses.temporal import WeightedContentLoss

    target = torch.rand(3, 8, 10)
    weights = torch.ones(3, 8, 10)
    loss_mod = WeightedContentLoss(target, weights=weights, strength=1.0)
    out = loss_mod(target.clone())
    assert float(out) == pytest.approx(0.0, abs=1e-6)


def test_combine_longterm_weights_closest_first_ordering(torch):
    # The pipeline passes reliability masks closest-previous-frame first; verify
    # combine_longterm_weights keeps the closest frame's weight untouched.
    from artvid.flow.consistency import combine_longterm_weights

    closest = torch.ones(2, 2)
    farther = torch.ones(2, 2)
    combined = combine_longterm_weights([closest, farther], method="closestFirst")
    assert torch.allclose(combined[0], closest)  # closest untouched
    assert torch.all(combined[1] <= farther)     # farther reduced where overlap
