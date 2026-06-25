"""CPU end-to-end tests that EXERCISE the torch code paths without downloads.

Phase-1/2 modules are written with lazy torch imports so the rest of the suite
collects without torch. These tests, by contrast, *run* the torch code on CPU so
the upcoming CI (which installs CPU torch) actually covers the real code paths:

* VGG feature extraction (``weights=None`` — untrained, NO weight download).
* A few L-BFGS iterations of single-image stylize on an 8x8 synthetic image.
* A 2-frame single-pass video run with a tiny **precomputed** synthetic flow
  (so RAFT — and its weight download — is never touched).
* loss-module gradient flow (content / style / TV / temporal).
* latent-warp scale-factor + reliability gradient sanity on synthetic latents.

Marker policy (registered in ``conftest.pytest_configure``)
-----------------------------------------------------------
* Anything that would download weights is marked ``@pytest.mark.network``:
  - VGG with ``weights="torchvision"`` (downloads ImageNet VGG-19),
  - RAFT (``compute_flow`` downloads Raft_Large), and
  - the diffusion engine (downloads SDXL / ControlNet / IP-Adapter).
  CI runs the offline subset with ``-m 'not network'``.
* Anything needing the Apple-Silicon Metal backend is marked ``@pytest.mark.mps``
  and additionally uses the ``mps`` fixture so it self-skips on Linux.

Every torch-touching test depends on the ``torch`` fixture from ``conftest.py``
so the file stays importable / collectable when torch is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Torch-free import: Config does not import torch.
#
# NOTE: ``artvid.models.vgg`` imports torch at *module* top level (it defines an
# ``nn.Module``), so its symbols are imported lazily *inside* the torch-gated
# test bodies — importing them here would break collection without torch.
from artvid.config import Config

# ---------------------------------------------------------------------------
# Small synthetic-image helpers (CPU, no I/O, no downloads)
# ---------------------------------------------------------------------------

def _write_synthetic_image(torch, path: Path, h: int, w: int, seed: int) -> None:
    """Write a tiny deterministic RGB PNG so the pipeline can ``load_image`` it."""
    from artvid.io.image import save_image

    g = torch.Generator().manual_seed(seed)
    # save_image deprocesses then clamps; feed it a *preprocessed* tensor so the
    # round-trip lands in a sane [0,1] range. torchvision mode keeps it simple.
    rgb = torch.rand(3, h, w, generator=g)
    from artvid.io.image import MODE_TORCHVISION, preprocess

    save_image(preprocess(rgb, mode=MODE_TORCHVISION), path, mode=MODE_TORCHVISION)


# ===========================================================================
# VGG feature extractor — untrained (weights=None), no download.
# ===========================================================================

def test_vgg_untrained_feature_shapes(torch):
    """VGGFeatures(weights=None) runs on CPU and returns correctly-shaped acts.

    Untrained VGG-19 (random init) avoids any weight download. We verify the
    forward pass produces one activation per requested layer at the expected
    spatial downsample (relu1_1 full-res, relu2_1 /2, relu3_1 /4, ...).
    """
    from artvid.models.vgg import VGGFeatures

    net = VGGFeatures(
        layers=("relu1_1", "relu2_1", "relu3_1", "relu4_2"),
        weights=None,
    ).eval()

    x = torch.randn(1, 3, 32, 32)
    acts = net(x)

    assert set(acts) == {"relu1_1", "relu2_1", "relu3_1", "relu4_2"}
    # relu1_1: no pooling yet -> full res. Each pool halves H/W.
    assert acts["relu1_1"].shape == (1, 64, 32, 32)
    assert acts["relu2_1"].shape == (1, 128, 16, 16)
    assert acts["relu3_1"].shape == (1, 256, 8, 8)
    # relu4_2 is after pool3 -> /8.
    assert acts["relu4_2"].shape == (1, 512, 4, 4)


def test_vgg_untrained_truncates_to_deepest_layer(torch):
    """The features Sequential is truncated to the deepest requested relu index."""
    from artvid.models.vgg import RELU_NAME_TO_INDEX, VGGFeatures

    net = VGGFeatures(layers=("relu2_1",), weights=None)
    # relu2_1 is index 6 -> modules [0..6], i.e. 7 children.
    assert len(net.features) == RELU_NAME_TO_INDEX["relu2_1"] + 1


def test_vgg_untrained_avg_pooling_swaps_maxpool(torch):
    """pooling='avg' replaces MaxPool2d with AvgPool2d (legacy -pooling avg)."""
    from torch import nn

    from artvid.models.vgg import VGGFeatures

    net = VGGFeatures(layers=("relu3_1",), weights=None, pooling="avg")
    has_maxpool = any(isinstance(m, nn.MaxPool2d) for m in net.features)
    has_avgpool = any(isinstance(m, nn.AvgPool2d) for m in net.features)
    assert not has_maxpool
    assert has_avgpool


def test_vgg_untrained_activations_are_differentiable(torch):
    """Activations keep the input's autograd graph (loss can backprop to image)."""
    from artvid.models.vgg import VGGFeatures

    net = VGGFeatures(layers=("relu1_1",), weights=None).eval()
    x = torch.randn(1, 3, 16, 16, requires_grad=True)
    acts = net(x)
    acts["relu1_1"].pow(2).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_build_feature_net_untrained_union_and_split(torch):
    """build_feature_net(weights=None) exposes content+style layers; split routes them."""
    from artvid.models.vgg import (
        DEFAULT_CONTENT_LAYERS,
        DEFAULT_STYLE_LAYERS,
        build_feature_net,
        split_activations,
    )

    net = build_feature_net(weights=None).eval()
    x = torch.randn(1, 3, 16, 16)
    acts = net(x)
    content, style = split_activations(
        acts, DEFAULT_CONTENT_LAYERS, DEFAULT_STYLE_LAYERS
    )
    assert set(content) == set(DEFAULT_CONTENT_LAYERS)
    assert set(style) == set(DEFAULT_STYLE_LAYERS)


# ===========================================================================
# Loss-module gradients (content / style / TV / temporal) — pure CPU.
# ===========================================================================

def test_content_loss_gradient_flows_and_zero_at_target(torch):
    from artvid.losses.content import ContentLoss

    target = torch.randn(1, 4, 5, 6)
    loss_mod = ContentLoss(target, strength=2.0)

    # Zero at the target.
    assert float(loss_mod(target.clone())) == pytest.approx(0.0, abs=1e-7)

    # Non-zero + finite gradient away from the target.
    x = torch.randn(1, 4, 5, 6, requires_grad=True)
    out = loss_mod(x)
    assert float(out.detach()) > 0.0
    out.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_style_loss_gram_gradient_flows(torch):
    from artvid.losses.style import StyleLoss, gram_matrix

    feat_target = torch.randn(1, 8, 4, 4)
    loss_mod = StyleLoss(feat_target, strength=3.0)
    # gram(target) matches the cached buffer -> zero loss.
    g = gram_matrix(feat_target)
    torch.testing.assert_close(loss_mod.target, g)

    x = torch.randn(1, 8, 4, 4, requires_grad=True)
    out = loss_mod(x)
    out.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_tv_loss_l2_gradient_matches_legacy_stencil(torch):
    """The L2 TV energy's autograd gradient equals the legacy finite-diff stencil.

    Legacy ``updateGradInput`` (artistic_video_core.lua:415-429) accumulates, over
    the top-left (H-1)x(W-1) block: +x_diff+y_diff at the anchor, -x_diff at the
    right neighbour, -y_diff at the bottom neighbour. We rebuild that by hand and
    assert autograd reproduces it bit-for-bit.
    """
    from artvid.losses.tv import TVLoss

    x = torch.randn(1, 3, 5, 7, requires_grad=True)
    strength = 0.5
    TVLoss(strength=strength, form="l2")(x).backward()

    xd = x.detach()
    x_diff = xd[..., :-1, :-1] - xd[..., :-1, 1:]
    y_diff = xd[..., :-1, :-1] - xd[..., 1:, :-1]
    expected = torch.zeros_like(xd)
    expected[..., :-1, :-1] += x_diff + y_diff
    expected[..., :-1, 1:] -= x_diff
    expected[..., 1:, :-1] -= y_diff
    expected *= strength

    torch.testing.assert_close(x.grad, expected, rtol=1e-5, atol=1e-6)


def test_temporal_loss_weighted_square_semantics(torch):
    """WeightedContentLoss realizes w*err^2 via the sqrt(w) pre-multiply trick.

    With constant weight ``w`` and MSE criterion, the loss must equal
    ``strength * mean(w * (x - target)^2)`` — confirming the legacy
    ``(sqrt(w)*err)^2 = w*err^2`` parity (artistic_video_core.lua:296-322).
    """
    from artvid.losses.temporal import WeightedContentLoss

    target = torch.randn(3, 4, 5)
    w_val = 0.25
    weights = torch.full((3, 4, 5), w_val)
    loss_mod = WeightedContentLoss(target, weights=weights, strength=2.0)

    x = torch.randn(3, 4, 5, requires_grad=True)
    out = loss_mod(x)

    expected = 2.0 * (w_val * (x.detach() - target).pow(2)).mean()
    assert float(out) == pytest.approx(float(expected), rel=1e-5)

    out.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ===========================================================================
# Single-image stylize: a few L-BFGS iterations on an 8x8 synthetic image.
# Uses weights=None VGG via Config.vgg_weights=None -> NO download.
# ===========================================================================

def _tiny_stylize_config(tmp_path: Path) -> Config:
    """A Config that runs single-image stylize tiny + offline (untrained VGG)."""
    return Config(
        vgg_weights=None,          # untrained VGG-19 -> no weight download
        content_layers=("relu2_1",),  # shallow -> cheap on an 8x8 image
        style_layers=("relu1_1", "relu2_1"),
        num_iterations=(3, 3),     # a *few* L-BFGS iters
        optimizer="lbfgs",
        tol_loss_relative=0.0,     # do not early-stop; run the requested iters
        print_iter=0,
        save_iter=0,
        tv_weight=1e-3,
        init=("image", "image"),   # start from content (deterministic, fast)
    )


def test_stylize_image_lbfgs_runs_on_8x8(torch, tmp_path):
    """A few L-BFGS steps of single-image stylize on an 8x8 image, offline.

    Exercises the full single-image path: load -> preprocess -> untrained VGG
    feature net -> content/style/TV losses -> run_optimization (L-BFGS). We only
    assert it completes, produces a finite preprocessed image of the right shape,
    and actually decreased the total loss over the few iterations.
    """
    content = tmp_path / "content.png"
    style = tmp_path / "style.png"
    _write_synthetic_image(torch, content, 8, 8, seed=1)
    _write_synthetic_image(torch, style, 8, 8, seed=2)

    from artvid.pipeline.stylize_image import stylize_image

    cfg = _tiny_stylize_config(tmp_path)
    image_pre, result = stylize_image(
        content, style, cfg, device="cpu",
    )

    # stylize_image returns the optimized image variable in CHW (the Phase 1
    # convention; VGG adds the batch dim internally), not NCHW.
    assert image_pre.shape == (3, 8, 8)
    assert torch.isfinite(image_pre).all()
    assert result.num_iterations >= 1
    # L-BFGS should have made progress: final total <= initial total.
    assert len(result.loss_history) >= 1
    assert result.loss_history[-1] <= result.loss_history[0] + 1e-6


def test_stylize_image_adam_runs_on_8x8(torch, tmp_path):
    """The Adam optimizer branch also runs end-to-end offline on 8x8."""
    content = tmp_path / "c.png"
    style = tmp_path / "s.png"
    _write_synthetic_image(torch, content, 8, 8, seed=3)
    _write_synthetic_image(torch, style, 8, 8, seed=4)

    from artvid.pipeline.stylize_image import stylize_image

    cfg = _tiny_stylize_config(tmp_path)
    cfg.optimizer = "adam"
    cfg.learning_rate = 1.0
    cfg.num_iterations = (3, 3)

    out = tmp_path / "out.png"
    image_pre, result = stylize_image(content, style, cfg, device="cpu", output_path=out)

    assert image_pre.shape == (3, 8, 8)  # CHW (see lbfgs test)
    assert torch.isfinite(image_pre).all()
    assert out.is_file()  # save_fn callback fired


# ===========================================================================
# 2-frame single-pass video with a tiny PRECOMPUTED synthetic flow.
# flow_source="precomputed" => RAFT (and its download) is never touched.
# ===========================================================================

def _setup_singlepass_sequence(torch, tmp_path: Path, h: int, w: int):
    """Write 2 content frames + a synthetic backward .flo + reliability .pgm.

    Returns a Config whose patterns point at the written files, configured to use
    the precomputed flow path (no RAFT). The flow is a small constant shift; the
    reliability mask is all-ones (fully reliable).
    """
    import numpy as np

    from artvid.io.flow_io import write_flo
    from artvid.io.image import MODE_TORCHVISION, preprocess, save_image

    # Content frames frame1, frame2 (1-indexed, %d).
    g = torch.Generator().manual_seed(7)
    for idx in (1, 2):
        rgb = torch.rand(3, h, w, generator=g)
        save_image(
            preprocess(rgb, mode=MODE_TORCHVISION),
            tmp_path / f"frame{idx}.png",
            mode=MODE_TORCHVISION,
        )

    # Backward flow for (prev=1 -> current=2). Default pattern is
    # backward_[%d]_{%d}.flo with [to]=frame_idx=2, {from}=prev=1 -> backward_2_1.flo.
    flow = np.zeros((2, h, w), dtype=np.float32)
    flow[0] = 1.0  # +1 px u shift -> exercises the warp, stays mostly in-border
    write_flo(tmp_path / "backward_2_1.flo", flow)

    # Reliability mask reliable_[%d]_{%d}.pgm -> reliable_2_1.pgm. All-ones.
    rel = torch.ones(3, h, w)
    save_image(
        # save_image deprocesses; feed preprocessed white so it round-trips to 1.0
        preprocess(rel, mode=MODE_TORCHVISION),
        tmp_path / "reliable_2_1.pgm",
        mode=MODE_TORCHVISION,
    )

    style = tmp_path / "style.png"
    _write_synthetic_image(torch, style, h, w, seed=11)

    return Config(
        vgg_weights=None,
        content_pattern=str(tmp_path / "frame%d.png"),
        flow_pattern=str(tmp_path / "backward_[%d]_{%d}.flo"),
        flow_weight_pattern=str(tmp_path / "reliable_[%d]_{%d}.pgm"),
        style_image=str(style),
        output_folder=str(tmp_path) + "/",
        output_image="out.png",
        number_format="%d",
        start_number=1,
        num_images=2,
        content_layers=("relu2_1",),
        style_layers=("relu1_1", "relu2_1"),
        num_iterations=(2, 2),
        tol_loss_relative=0.0,
        print_iter=0,
        tv_weight=1e-3,
        temporal_weight=1e3,
        flow_relative_indices=(1,),
        init=("image", "prevWarped"),
    )


def test_singlepass_two_frames_precomputed_flow(torch, tmp_path):
    """A 2-frame single-pass run using precomputed synthetic flow (no RAFT).

    Exercises: frame loop, per-frame content target, precomputed flow loading,
    warp_image of the previous output, reliability mask loading, the temporal
    WeightedContentLoss wiring, prevWarped init, and per-frame save. Asserts both
    frames are produced and the 2nd frame actually used the 1st as a temporal
    target.
    """
    cfg = _setup_singlepass_sequence(torch, tmp_path, h=8, w=8)

    from artvid.pipeline.singlepass import stylize_video

    results = stylize_video(cfg, device="cpu", flow_source="precomputed")

    assert len(results) == 2
    # First frame: no temporal target.
    assert results[0].frame_idx == 1
    assert results[0].previous_indices == []
    # Second frame: temporal target on the immediately previous frame (idx 1).
    assert results[1].frame_idx == 2
    assert results[1].previous_indices == [1]
    # Both output files were written.
    assert Path(results[0].output_path).is_file()
    assert Path(results[1].output_path).is_file()


# ===========================================================================
# latent_warp on synthetic latents — complements test_latent_warp.py with the
# scale-factor + reliability-gradient angle for the diffusion path.
# ===========================================================================

def test_latent_warp_scale_factor_non_divisible(torch):
    """Per-axis magnitude rescale stays exact when H/W aren't multiples of f.

    flow_to_latent uses the exact per-axis ratios (h/H, w/W) rather than a single
    1/f, so a non-divisible image size still maps a 1-cell motion correctly. Here
    H=20, W=28 with target latent (5, 7): ratios are 5/20=0.25 and 7/28=0.25, so a
    4-px u-flow becomes exactly 1.0 latent cell.
    """
    from artvid.diffusion.latent_warp import flow_to_latent

    H, W = 20, 28
    h, w = 5, 7
    flow_px = torch.zeros((2, H, W), dtype=torch.float32)
    flow_px[0] = 4.0  # 4 px * (7/28) = 1.0 latent cell
    flow_px[1] = 8.0  # 8 px * (5/20) = 2.0 latent cells

    flow_lat = flow_to_latent(flow_px, (h, w))
    assert flow_lat.shape == (2, h, w)
    torch.testing.assert_close(flow_lat[0], torch.ones(h, w), rtol=0, atol=1e-5)
    torch.testing.assert_close(flow_lat[1], torch.full((h, w), 2.0), rtol=0, atol=1e-5)


def test_warp_latent_does_not_mutate_input_flow(torch):
    """warp_latent must not mutate the caller's flow tensor (out-of-place rescale)."""
    from artvid.diffusion.latent_warp import warp_latent

    z = torch.randn(1, 4, 5, 7)
    flow_px = torch.zeros((2, 40, 56), dtype=torch.float32)
    flow_px[0] = 8.0
    flow_ref = flow_px.clone()

    warp_latent(z, flow_px, vae_factor=8)
    torch.testing.assert_close(flow_px, flow_ref)


# ===========================================================================
# NETWORK-marked tests: these DOWNLOAD weights and are deselected on offline CI.
# ===========================================================================

@pytest.mark.network
def test_vgg_torchvision_weights_download(torch):
    """(network) Loading torchvision VGG-19 weights downloads the checkpoint."""
    from artvid.models.vgg import (
        DEFAULT_CONTENT_LAYERS,
        DEFAULT_STYLE_LAYERS,
        build_feature_net,
    )

    net = build_feature_net(weights="torchvision").eval()
    x = torch.randn(1, 3, 16, 16)
    acts = net(x)
    assert set(acts) == set(DEFAULT_STYLE_LAYERS) | set(DEFAULT_CONTENT_LAYERS)


@pytest.mark.network
def test_raft_compute_flow_downloads(torch):
    """(network) RAFT-large flow on CPU downloads the Raft_Large checkpoint."""
    from artvid.flow.raft import compute_flow

    img1 = torch.rand(3, 64, 64)
    img2 = torch.rand(3, 64, 64)
    flow = compute_flow(img1, img2, device="cpu")
    assert flow.shape == (2, 64, 64)
    assert torch.isfinite(flow).all()


@pytest.mark.network
def test_singlepass_two_frames_raft_flow(torch, tmp_path):
    """(network) Same 2-frame single-pass loop but with on-the-fly RAFT flow."""
    cfg = _setup_singlepass_sequence(torch, tmp_path, h=64, w=64)

    from artvid.pipeline.singlepass import stylize_video

    results = stylize_video(cfg, device="cpu", flow_source="raft")
    assert len(results) == 2
    assert results[1].previous_indices == [1]


@pytest.mark.network
def test_diffusion_engine_constructs(torch):
    """(network) Constructing the diffusion engine downloads SDXL/ControlNet/IP.

    Kept minimal and import-guarded: skips cleanly if diffusers is not installed
    even on a network-enabled runner.
    """
    pytest.importorskip("diffusers")
    from artvid.diffusion.engine import DiffusionEngine

    engine = DiffusionEngine.from_config(Config())
    # Trigger the lazy pipeline build (this is the part that downloads weights).
    engine.load()


# ===========================================================================
# MPS-marked test: requires the Apple-Silicon Metal backend (skipped on Linux).
# ===========================================================================

@pytest.mark.mps
def test_warp_image_runs_on_mps(torch, mps):
    """(mps) The flow warp runs unchanged on the Metal backend.

    The ``mps`` fixture self-skips when MPS is unavailable (e.g. Linux CI), so
    this only executes on Apple Silicon. It pins the device-agnostic claim in the
    warp module docstring: identity flow -> warped == input, fully valid.
    """
    from artvid.flow.warp import warp_image

    img = torch.rand(3, 8, 8, device=mps)
    flow = torch.zeros(2, 8, 8, device=mps)
    res = warp_image(img, flow)
    assert res.image.device.type == "mps"
    torch.testing.assert_close(res.image, img, rtol=0, atol=1e-5)
    assert bool(res.valid.all())
