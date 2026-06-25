"""Deepening tests for the Phase 2 diffusion temporal-consistency mechanisms.

What this file covers (``docs/07-phase2-design.md`` §2.5 / §2.6)
---------------------------------------------------------------
CPU-runnable, **synthetic-tensor** tests for the four pieces that were deepened
in Phase 2 and that the CPU-torch CI can actually exercise (no model weights, no
network):

* **Mechanism-1 masked init MATH** (§2.5, resolves docs/06 #15). The
  reliability-masked warped-latent init blends the renoised warped latent against
  a fresh-noise init: ``start = rel * renoised + (1 - rel) * fresh``. This blend
  is *inlined* inside :meth:`DiffusionEngine.denoise_frame` (it needs a loaded
  SDXL pipeline + scheduler to reach), so — per the assignment's "replicate the
  documented formula" path — we pin the pure tensor formula directly on synthetic
  latents: ``rel == 1`` -> exactly the renoised warp; ``rel == 0`` ->
  ``noise * sigma_init``; intermediate -> a linear blend. A drift in the engine's
  inlined seam (e.g. dropping the masking, or using ``+`` instead of a convex
  blend) would diverge from this reference.
* **combine_anchor_reliability** (§2.6) — shapes preserved, and that with a
  zero anchor reliability it reduces to ``prev_rel`` (the closest-first long-term
  weight scheme of :func:`artvid.flow.consistency.combine_longterm_weights`).
* **warp_previous_pixel** (``warp_space='pixel'``) — zero flow is identity in
  pixel space (VAE-encode-agnostic: we assert on the warped RGB the engine would
  re-encode), and the shape is preserved.
* **cross_frame_attention processor** — the KV-concat in ``"inject"`` mode
  produces the expected ``current ++ reference`` key/value sequence length on
  synthetic ``(B, seq, dim)`` tensors, and the ``"off"`` mode is a no-op
  passthrough (output equals stock self-attention).

Which modules this builds on
----------------------------
:mod:`artvid.diffusion.engine` (mechanism-1 seam), :mod:`artvid.diffusion.latent_warp`
(``combine_anchor_reliability`` / ``warp_previous_pixel``),
:mod:`artvid.diffusion.cross_frame_attention` (the processor), and the Phase 1
:mod:`artvid.flow.consistency` long-term-weight semantics they reuse.

torch gating
------------
torch is not installable in the authoring/CI-scaffold env, so every test body
lazy-imports torch via the ``torch`` fixture from ``conftest.py`` and skips when
torch is absent. The module itself imports no torch/diffusers at collection time,
so it COLLECTS cleanly without torch. None of these tests touch the network (no
model downloads); they run on the CPU-torch CI.
"""

from __future__ import annotations

import pytest

from artvid.diffusion.cross_frame_attention import CrossFrameAttnProcessor
from artvid.diffusion.latent_warp import (
    combine_anchor_reliability,
    warp_previous_pixel,
)

# ---------------------------------------------------------------------------
# Mechanism-1: reliability-masked warped-latent init MATH (engine.py §2.5).
#
# The blend is inlined in DiffusionEngine.denoise_frame as
#     fresh_init = noise * scheduler.init_noise_sigma
#     renoised   = scheduler.add_noise(init, noise, t0)
#     latents    = rel * renoised + (1 - rel) * fresh_init      # rel in [0,1]
# Reaching that seam needs a loaded SDXL pipeline (network); we instead pin the
# documented pure-tensor formula on synthetic latents so a regression in the
# inlined math (dropped masking / non-convex blend) is caught by CPU CI.
# ---------------------------------------------------------------------------


def _masked_init(rel, renoised, fresh):
    """The documented mechanism-1 masked-init blend (mirrors engine.py)."""
    return rel * renoised + (1.0 - rel) * fresh


def test_masked_init_reliability_one_is_renoised_warp(torch):
    """rel == 1 everywhere -> start latent is exactly the renoised warped latent."""
    h, w = 4, 6
    renoised = torch.randn(1, 4, h, w)
    fresh = torch.randn(1, 4, h, w)
    rel = torch.ones(1, 1, h, w)  # broadcasts over the latent channel axis

    start = _masked_init(rel, renoised, fresh)

    assert start.shape == renoised.shape
    torch.testing.assert_close(start, renoised, rtol=0, atol=0)


def test_masked_init_reliability_zero_is_fresh_noise_times_sigma(torch):
    """rel == 0 everywhere -> start latent is the fresh-noise init (noise*sigma).

    ``fresh_init`` in the engine is ``noise * scheduler.init_noise_sigma``; we
    construct that product synthetically and assert the masked init returns it
    untouched when nothing is reliable (disocclusion fallback).
    """
    h, w = 4, 6
    sigma_init = 14.6  # representative Euler max-sigma; DDIM would be 1.0
    noise = torch.randn(1, 4, h, w)
    fresh = noise * sigma_init
    renoised = torch.randn(1, 4, h, w)
    rel = torch.zeros(1, 1, h, w)

    start = _masked_init(rel, renoised, fresh)

    torch.testing.assert_close(start, fresh, rtol=0, atol=0)
    # And it is genuinely the noise scaled by sigma, not the warp.
    torch.testing.assert_close(start, noise * sigma_init, rtol=0, atol=1e-6)


def test_masked_init_intermediate_blends_linearly(torch):
    """A constant intermediate reliability is a linear (convex) blend of the two.

    Pinning convexity (coefficients ``rel`` and ``1-rel`` summing to 1) guards
    against the seam regressing to an additive / unnormalised mix.
    """
    h, w = 3, 5
    renoised = torch.randn(1, 4, h, w)
    fresh = torch.randn(1, 4, h, w)
    alpha = 0.3
    rel = torch.full((1, 1, h, w), alpha)

    start = _masked_init(rel, renoised, fresh)

    expected = alpha * renoised + (1.0 - alpha) * fresh
    torch.testing.assert_close(start, expected, rtol=0, atol=1e-6)
    # Convexity sanity: at alpha, start lies on the segment fresh->renoised.
    torch.testing.assert_close(
        start, fresh + alpha * (renoised - fresh), rtol=0, atol=1e-6
    )


def test_masked_init_per_cell_reliability_selects_per_cell(torch):
    """A spatially-varying mask blends each latent cell with its own weight."""
    h, w = 2, 2
    renoised = torch.ones(1, 4, h, w) * 5.0
    fresh = torch.ones(1, 4, h, w) * -1.0
    rel = torch.tensor([[[[1.0, 0.0], [0.5, 0.25]]]])  # (1,1,2,2)

    start = _masked_init(rel, renoised, fresh)

    # Each cell == rel*5 + (1-rel)*(-1); broadcast across all 4 channels.
    expected_cell = rel * 5.0 + (1.0 - rel) * (-1.0)
    expected = expected_cell.expand(1, 4, h, w)
    torch.testing.assert_close(start, expected, rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# combine_anchor_reliability (latent_warp.py §2.6).
# ---------------------------------------------------------------------------


def test_combine_anchor_reliability_shapes_preserved(torch):
    """Both returned weights keep the (N, 1, h, w) input shape."""
    n, h, w = 1, 4, 5
    prev_rel = torch.rand(n, 1, h, w)
    anchor_rel = torch.rand(n, 1, h, w)

    prev_w, anchor_w = combine_anchor_reliability(prev_rel, anchor_rel)

    assert prev_w.shape == (n, 1, h, w)
    assert anchor_w.shape == (n, 1, h, w)


def test_combine_anchor_reliability_zero_anchor_reduces_to_prev(torch):
    """anchor_rel == 0 -> prev weight is prev_rel and anchor weight is 0.

    Under the default ``closestFirst`` scheme prev (closest) is untouched and the
    anchor only claims cells prev does not see: ``anchor_w = clamp(anchor-prev,0)``.
    With a zero anchor that is identically 0, so the combination collapses to the
    previous-frame reliability alone.
    """
    n, h, w = 1, 3, 3
    prev_rel = torch.rand(n, 1, h, w)
    anchor_rel = torch.zeros(n, 1, h, w)

    prev_w, anchor_w = combine_anchor_reliability(prev_rel, anchor_rel)

    torch.testing.assert_close(prev_w, prev_rel, rtol=0, atol=0)
    torch.testing.assert_close(anchor_w, torch.zeros_like(anchor_rel), rtol=0, atol=0)


def test_combine_anchor_reliability_matches_closestfirst_semantics(torch):
    """anchor weight == clamp(anchor_rel - prev_rel, 0) (closestFirst math)."""
    n, h, w = 1, 2, 4
    prev_rel = torch.rand(n, 1, h, w)
    anchor_rel = torch.rand(n, 1, h, w)

    prev_w, anchor_w = combine_anchor_reliability(prev_rel, anchor_rel)

    torch.testing.assert_close(prev_w, prev_rel, rtol=0, atol=0)
    torch.testing.assert_close(
        anchor_w, torch.clamp(anchor_rel - prev_rel, min=0.0), rtol=0, atol=1e-6
    )


def test_combine_anchor_reliability_does_not_mutate_inputs(torch):
    """The combination returns new tensors; the caller's masks are untouched."""
    n, h, w = 1, 2, 2
    prev_rel = torch.rand(n, 1, h, w)
    anchor_rel = torch.rand(n, 1, h, w)
    prev_before = prev_rel.clone()
    anchor_before = anchor_rel.clone()

    combine_anchor_reliability(prev_rel, anchor_rel)

    torch.testing.assert_close(prev_rel, prev_before, rtol=0, atol=0)
    torch.testing.assert_close(anchor_rel, anchor_before, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# warp_previous_pixel (latent_warp.py, warp_space='pixel').
# ---------------------------------------------------------------------------


def test_warp_previous_pixel_zero_flow_is_identity(torch):
    """A zero pixel flow returns the previous RGB frame unchanged, all-valid.

    This is the pixel-space (pre-VAE-encode) input the engine would re-encode;
    asserting on it is VAE-encode-agnostic per the assignment.
    """
    C, H, W = 3, 8, 10
    prev_rgb = torch.rand(C, H, W)
    flow_px = torch.zeros(2, H, W)

    result = warp_previous_pixel(prev_rgb, flow_px)

    # WarpResult.image is the warped RGB; zero flow => identity.
    torch.testing.assert_close(result.image.squeeze(0), prev_rgb, rtol=0, atol=1e-5)
    # Everything sampled from inside the source => fully valid.
    assert bool(result.valid.all())


def test_warp_previous_pixel_shape_preserved(torch):
    """The warped RGB keeps the input spatial size and 3 channels."""
    C, H, W = 3, 6, 9
    prev_rgb = torch.rand(C, H, W)
    flow_px = torch.zeros(2, H, W)
    flow_px[0] = 1.5  # a non-trivial constant shift (stays in-shape)

    result = warp_previous_pixel(prev_rgb, flow_px)

    assert result.image.shape[-3:] == (C, H, W)
    assert result.valid.shape[-2:] == (H, W)


# ---------------------------------------------------------------------------
# CrossFrameAttnProcessor: KV-concat shapes + disabled passthrough.
#
# We synthesize a minimal stand-in for a diffusers ``Attention`` module so the
# processor's KV-concat / off-mode paths run on CPU without diffusers. The
# stand-in exposes exactly the attributes the processor touches: ``heads``,
# ``to_q``/``to_k``/``to_v`` (identity linears so we can reason about shapes and
# values), ``to_out`` (linear + dropout), and the optional norm/flags read via
# ``getattr`` (left unset -> their stock-off defaults).
# ---------------------------------------------------------------------------


def _make_attn(torch, dim: int, heads: int):
    """A tiny diffusers-Attention-like module exercising the processor paths."""
    nn = torch.nn

    class _Identity(nn.Module):
        def forward(self, x):
            return x

    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = heads
            # Identity projections keep K==hidden so the concat is checkable.
            self.to_q = _Identity()
            self.to_k = _Identity()
            self.to_v = _Identity()
            # to_out is [linear, dropout]; identity linear + no-op dropout.
            self.to_out = nn.ModuleList([_Identity(), nn.Dropout(0.0)])
            # Flags the processor reads via getattr; explicit for clarity.
            self.spatial_norm = None
            self.group_norm = None
            self.residual_connection = False
            self.rescale_output_factor = 1.0

    attn = _Attn()
    assert dim % heads == 0, "dim must be divisible by heads"
    return attn


def _capture_seq_len(torch, processor, attn):
    """Run the processor once, capturing the key sequence length seen by SDPA.

    We monkeypatch ``processor._sdpa`` to record ``key``'s sequence length
    (dim 1 of the ``(B, S_kv, inner_dim)`` key) before delegating to the real
    SDPA, so we can assert ``current ++ reference`` concatenation without
    needing attention weights out of SDPA.
    """
    seen = {}
    real_sdpa = processor._sdpa

    def _spy(attn_, F, q, key, value, attention_mask, batch_size, head_dim):
        seen["kv_seq"] = key.shape[1]
        return real_sdpa(attn_, F, q, key, value, attention_mask, batch_size, head_dim)

    processor._sdpa = _spy
    return seen, real_sdpa


def test_cross_frame_inject_concats_reference_kv_seq(torch):
    """inject mode with a reference -> SDPA sees S_cur + S_ref keys."""
    B, S, dim, heads = 2, 7, 8, 2
    attn = _make_attn(torch, dim, heads)
    proc = CrossFrameAttnProcessor(mix=1.0)

    cur = torch.randn(B, S, dim)
    ref = torch.randn(B, S, dim)  # same seq length here; concat -> 2S

    # Record the reference frame's hidden states for this layer.
    proc.set_mode("record")
    proc(attn, ref)  # self-attention (encoder_hidden_states is None)
    bank = proc.take_recorded()
    assert id(attn) in bank

    # Inject: current frame attends over current ++ reference keys.
    proc.set_reference(bank)
    proc.set_mode("inject")
    seen, _ = _capture_seq_len(torch, proc, attn)
    out = proc(attn, cur)

    assert seen["kv_seq"] == S + S  # current S keys + reference S keys
    assert out.shape == (B, S, dim)


def test_cross_frame_inject_different_reference_length(torch):
    """The reference may have a different seq length; KV concat is S_cur+S_ref."""
    B, dim, heads = 1, 8, 2
    S_cur, S_ref = 6, 9
    attn = _make_attn(torch, dim, heads)
    proc = CrossFrameAttnProcessor(mix=1.0)

    ref = torch.randn(B, S_ref, dim)
    proc.set_mode("record")
    proc(attn, ref)
    bank = proc.take_recorded()

    proc.set_reference(bank)
    proc.set_mode("inject")
    seen, _ = _capture_seq_len(torch, proc, attn)
    proc(attn, torch.randn(B, S_cur, dim))

    assert seen["kv_seq"] == S_cur + S_ref


def test_cross_frame_inject_batch_mismatch_skips_reference(torch):
    """A reference whose batch differs from the current frame is skipped (no concat).

    Guards the CFG batch-alignment safeguard: rather than mis-align uncond/cond,
    the processor falls back to plain self-attention (S_kv == S_cur).
    """
    dim, heads, S = 8, 2, 5
    attn = _make_attn(torch, dim, heads)
    proc = CrossFrameAttnProcessor(mix=1.0)

    # Record with batch 2, inject with batch 1 -> mismatch.
    proc.set_mode("record")
    proc(attn, torch.randn(2, S, dim))
    proc.set_reference(proc.take_recorded())
    proc.set_mode("inject")

    seen, _ = _capture_seq_len(torch, proc, attn)
    proc(attn, torch.randn(1, S, dim))

    assert seen["kv_seq"] == S  # reference skipped; current keys only


def test_cross_frame_off_mode_is_self_attention_passthrough(torch):
    """off mode == stock self-attention: SDPA sees only the current keys.

    With identity projections and ``mix=1`` the off-mode output also equals a
    direct single-head-agnostic SDPA over the current frame's own q/k/v, i.e. it
    does not diverge from plain self-attention.
    """
    B, S, dim, heads = 1, 5, 8, 2
    attn = _make_attn(torch, dim, heads)
    proc = CrossFrameAttnProcessor(mix=1.0)
    proc.set_mode("off")

    x = torch.randn(B, S, dim)
    seen, _ = _capture_seq_len(torch, proc, attn)
    out = proc(attn, x)

    # No reference concat in off mode -> only the current S keys.
    assert seen["kv_seq"] == S
    assert out.shape == (B, S, dim)

    # Reference SDPA over the current frame's own (identity) q/k/v.
    head_dim = dim // heads
    q = x.view(B, S, heads, head_dim).transpose(1, 2)
    k = q
    v = q
    F = torch.nn.functional
    ref_out = (
        F.scaled_dot_product_attention(q, k, v)
        .transpose(1, 2)
        .reshape(B, S, dim)
    )
    torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-5)


def test_cross_frame_off_mode_records_nothing(torch):
    """off mode leaves the record sink empty (no accidental capture)."""
    dim, heads, S = 8, 2, 4
    attn = _make_attn(torch, dim, heads)
    proc = CrossFrameAttnProcessor()
    proc.set_mode("off")

    proc(attn, torch.randn(1, S, dim))

    assert proc.take_recorded() == {}


def test_cross_frame_inject_without_reference_is_plain_self_attention(torch):
    """inject mode with an empty bank falls back to self-only keys (no crash)."""
    dim, heads, S = 8, 2, 6
    attn = _make_attn(torch, dim, heads)
    proc = CrossFrameAttnProcessor(mix=1.0)
    proc.set_mode("inject")
    proc.set_reference(None)  # no reference recorded

    seen, _ = _capture_seq_len(torch, proc, attn)
    out = proc(attn, torch.randn(1, S, dim))

    assert seen["kv_seq"] == S
    assert out.shape == (1, S, dim)


def test_set_mode_rejects_unknown_mode(torch):
    """The processor guards its mode control surface."""
    proc = CrossFrameAttnProcessor()
    with pytest.raises(ValueError):
        proc.set_mode("bogus")
