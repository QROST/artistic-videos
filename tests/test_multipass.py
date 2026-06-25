"""Tests for the multi-pass video pipeline (artvid/pipeline/multipass.py).

The pass-direction sequencing and the blend-weight selection are pure
(torch-free) ports of ``artistic_video_multiPass.lua:147-150`` (``flag``-based
direction) and ``:234`` / ``:243`` (``blendWeight`` vs ``blendWeight_lastPass``);
they are tested directly without torch. Tensor-level behaviour (neighbour warp,
blended init) is gated on the conftest ``torch`` fixture so the suite still runs
where torch is unavailable.
"""

from __future__ import annotations

import pytest

from artvid.config import Config
from artvid.pipeline.multipass import (
    blend_weight_for_neighbour,
    build_pass_out_filename,
    is_forward_pass,
    pass_direction,
    pass_frame_order,
    temporal_loss_enabled,
)


# ---------------------------------------------------------------------------
# Pass direction / sequencing (artistic_video_multiPass.lua:147-150)
# ---------------------------------------------------------------------------

def test_odd_pass_is_forward():
    # run % 2 == 1 -> forward (flag == 1).
    assert is_forward_pass(1) is True
    assert is_forward_pass(3) is True
    assert is_forward_pass(15) is True


def test_even_pass_is_backward():
    # run % 2 == 0 -> backward (flag == 0).
    assert is_forward_pass(2) is False
    assert is_forward_pass(4) is False


def test_pass_direction_forward_on_odd():
    # Odd pass sweeps start..end step +1 (legacy :148-150 flag==1 branch).
    assert pass_direction(1, start_number=1, end_image_idx=5) == (1, 5, 1)


def test_pass_direction_backward_on_even():
    # Even pass sweeps end..start step -1 (legacy :148-150 flag==0 branch).
    assert pass_direction(2, start_number=1, end_image_idx=5) == (5, 1, -1)


def test_pass_frame_order_forward():
    assert pass_frame_order(1, 1, 5) == [1, 2, 3, 4, 5]


def test_pass_frame_order_backward():
    assert pass_frame_order(2, 1, 5) == [5, 4, 3, 2, 1]


def test_pass_frame_order_respects_start_number():
    # start_number=10, 3 frames -> end_image_idx 12.
    assert pass_frame_order(1, 10, 12) == [10, 11, 12]
    assert pass_frame_order(2, 10, 12) == [12, 11, 10]


def test_alternating_directions_over_passes():
    # Passes alternate forward/backward strictly by parity.
    dirs = [is_forward_pass(run) for run in range(1, 7)]
    assert dirs == [True, False, True, False, True, False]


def test_pass_frame_order_single_frame():
    # A degenerate one-frame sequence: order is just [start] either direction.
    assert pass_frame_order(1, 7, 7) == [7]
    assert pass_frame_order(2, 7, 7) == [7]


# ---------------------------------------------------------------------------
# Blend-weight selection (artistic_video_multiPass.lua:234, :243)
# ---------------------------------------------------------------------------

def _blend_cfg(blend=1.0, last=0.0):
    return Config(blend_weight=blend, blend_weight_last_pass=last)


def test_prev_neighbour_gets_blend_weight_on_forward_pass():
    # Legacy :234: flag == 1 (forward) -> prev weight scaled by blendWeight.
    cfg = _blend_cfg(blend=1.0, last=0.0)
    assert blend_weight_for_neighbour(cfg, run=1, neighbour="prev") == 1.0


def test_prev_neighbour_gets_last_pass_weight_on_backward_pass():
    # Legacy :234: flag == 0 (backward) -> prev weight scaled by blendWeight_lastPass.
    cfg = _blend_cfg(blend=1.0, last=0.0)
    assert blend_weight_for_neighbour(cfg, run=2, neighbour="prev") == 0.0


def test_next_neighbour_gets_blend_weight_on_backward_pass():
    # Legacy :243: flag == 0 (backward) -> next weight scaled by blendWeight.
    cfg = _blend_cfg(blend=1.0, last=0.0)
    assert blend_weight_for_neighbour(cfg, run=2, neighbour="next") == 1.0


def test_next_neighbour_gets_last_pass_weight_on_forward_pass():
    # Legacy :243: flag == 1 (forward) -> next weight scaled by blendWeight_lastPass.
    cfg = _blend_cfg(blend=1.0, last=0.0)
    assert blend_weight_for_neighbour(cfg, run=1, neighbour="next") == 0.0


def test_blend_weights_use_configured_values():
    # Non-default blend/last weights are passed through (not hard-coded).
    cfg = _blend_cfg(blend=0.7, last=0.3)
    assert blend_weight_for_neighbour(cfg, run=1, neighbour="prev") == 0.7
    assert blend_weight_for_neighbour(cfg, run=1, neighbour="next") == 0.3
    assert blend_weight_for_neighbour(cfg, run=2, neighbour="prev") == 0.3
    assert blend_weight_for_neighbour(cfg, run=2, neighbour="next") == 0.7


def test_in_direction_neighbour_always_gets_full_blend_weight():
    # The neighbour behind the sweep direction (prev on fwd, next on bwd) always
    # gets blend_weight; the one ahead always gets blend_weight_last_pass.
    cfg = _blend_cfg(blend=0.9, last=0.1)
    for run in range(1, 7):
        if is_forward_pass(run):
            assert blend_weight_for_neighbour(cfg, run, neighbour="prev") == 0.9
            assert blend_weight_for_neighbour(cfg, run, neighbour="next") == 0.1
        else:
            assert blend_weight_for_neighbour(cfg, run, neighbour="next") == 0.9
            assert blend_weight_for_neighbour(cfg, run, neighbour="prev") == 0.1


def test_blend_weight_invalid_neighbour_raises():
    cfg = _blend_cfg()
    with pytest.raises(ValueError):
        blend_weight_for_neighbour(cfg, run=1, neighbour="sideways")


# ---------------------------------------------------------------------------
# Temporal-loss gate (artistic_video_multiPass.lua:174)
# ---------------------------------------------------------------------------

def test_temporal_disabled_before_threshold_pass():
    # run < use_temporalLoss_after -> off even with a warped neighbour.
    assert temporal_loss_enabled(3, use_temporal_loss_after=8, has_warped_neighbour=True) is False


def test_temporal_enabled_at_and_after_threshold():
    # run >= threshold and a neighbour exists -> on.
    assert temporal_loss_enabled(8, 8, True) is True
    assert temporal_loss_enabled(12, 8, True) is True


def test_temporal_disabled_without_warped_neighbour():
    # No warped neighbour (e.g. first frame in the sweep direction) -> off.
    assert temporal_loss_enabled(12, 8, False) is False


# ---------------------------------------------------------------------------
# Per-(frame, pass) output filename (build_OutFilename multi-pass form)
# ---------------------------------------------------------------------------

def test_pass_out_filename_appends_run():
    cfg = Config(output_image="out.png", number_format="%d", output_folder="")
    assert build_pass_out_filename(cfg, frame_idx=3, run=2) == "out-3_2.png"


def test_pass_out_filename_zero_padded():
    cfg = Config(output_image="out.png", number_format="%04d", output_folder="")
    assert build_pass_out_filename(cfg, frame_idx=7, run=5) == "out-0007_5.png"


def test_pass_out_filename_with_folder():
    cfg = Config(output_image="result.jpg", number_format="%d", output_folder="frames/")
    assert build_pass_out_filename(cfg, frame_idx=12, run=1) == "frames/result-12_1.jpg"


def test_pass_out_filename_uses_absolute_frame_index():
    # Multi-pass names by the ABSOLUTE frame index (not relative), so a non-1
    # start_number still produces that absolute number in the filename.
    cfg = Config(
        output_image="out.png",
        number_format="%d",
        output_folder="",
        start_number=10,
    )
    assert build_pass_out_filename(cfg, frame_idx=10, run=1) == "out-10_1.png"


# ---------------------------------------------------------------------------
# Torch-gated: neighbour warp + blended init wiring
# ---------------------------------------------------------------------------

def test_warp_neighbour_identity_flow_round_trips(torch, tmp_path, monkeypatch):
    # With identity flow and an all-reliable mask, warping a neighbour output
    # returns it unchanged and a (H, W) reliability of ~1 everywhere.
    import artvid.pipeline.multipass as mp

    cfg = Config()
    device = torch.device("cpu")
    content_rgb = torch.rand(3, 6, 8)
    neighbour_out = torch.rand(3, 6, 8)
    outputs_rgb = {(1, 1): neighbour_out}

    # Force identity flow + all-ones reliability without touching disk.
    monkeypatch.setattr(
        mp, "_neighbour_flow", lambda *a, **k: torch.zeros(2, 6, 8)
    )
    monkeypatch.setattr(
        mp, "_neighbour_reliability", lambda *a, **k: torch.ones(6, 8)
    )

    warped_rgb, reliab = mp._warp_neighbour(
        cfg,
        frame_idx=2,
        neighbour="prev",
        source_run=1,
        content_rgb=content_rgb,
        outputs_rgb=outputs_rgb,
        device=device,
        flow_source="raft",
    )
    assert warped_rgb.shape == (3, 6, 8)
    assert reliab.shape == (6, 8)
    assert torch.allclose(warped_rgb, neighbour_out, atol=1e-5)
    assert bool((reliab > 0.99).all())


def test_init_blended_with_zero_weight_returns_previous_pass(torch, monkeypatch):
    # With blend_weight=0 and last=0 the neighbours contribute nothing, so the
    # blended init reduces to this frame's previous-pass result (divisor == 1).
    import artvid.pipeline.multipass as mp
    from artvid.io.image import deprocess, preprocess

    cfg = Config(blend_weight=0.0, blend_weight_last_pass=0.0)
    device = torch.device("cpu")
    prev_pass_rgb = torch.rand(3, 5, 7)
    outputs_rgb = {(2, 1): prev_pass_rgb}

    prev_warp = (torch.rand(3, 5, 7), torch.ones(5, 7))
    next_warp = (torch.rand(3, 5, 7), torch.ones(5, 7))

    image_var = mp._init_blended(
        cfg,
        run=2,
        frame_idx=2,
        end_image_idx=3,
        content_rgb=torch.rand(3, 5, 7),
        prev_warp=prev_warp,
        next_warp=next_warp,
        outputs_rgb=outputs_rgb,
        mode="torchvision",
        device=device,
    )
    # image_var is in preprocessed space; deprocess back and compare to the
    # previous-pass RGB result.
    recovered = deprocess(image_var.detach(), mode="torchvision")
    assert torch.allclose(recovered, prev_pass_rgb, atol=1e-4)
    assert image_var.requires_grad


def test_init_blended_neighbour_pulls_toward_warped(torch, monkeypatch):
    # With a full blend weight on the in-direction (prev) neighbour, the blended
    # init moves toward the warped neighbour relative to the previous-pass result.
    import artvid.pipeline.multipass as mp
    from artvid.io.image import deprocess

    cfg = Config(blend_weight=1.0, blend_weight_last_pass=0.0)
    device = torch.device("cpu")
    prev_pass_rgb = torch.zeros(3, 4, 4)  # previous-pass result is black
    outputs_rgb = {(2, 1): prev_pass_rgb}

    warped = torch.ones(3, 4, 4)  # neighbour warps to white
    prev_warp = (warped, torch.ones(4, 4))  # fully reliable

    # forward pass (run=3) -> prev neighbour scaled by blend_weight=1.0.
    image_var = mp._init_blended(
        cfg,
        run=3,
        frame_idx=2,
        end_image_idx=2,  # no next neighbour
        content_rgb=torch.rand(3, 4, 4),
        prev_warp=prev_warp,
        next_warp=None,
        outputs_rgb=outputs_rgb,
        mode="torchvision",
        device=device,
    )
    recovered = deprocess(image_var.detach(), mode="torchvision")
    # (0 + 1*1) / (1 + 1) = 0.5 everywhere.
    assert torch.allclose(recovered, torch.full((3, 4, 4), 0.5), atol=1e-4)
