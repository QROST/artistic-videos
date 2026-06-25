"""Cross-frame (extended / reference) self-attention for the SDXL UNet (P2-M2 alt).

What this module does
---------------------
Provides an **optional, complementary** temporal-consistency mechanism to the
latent optical-flow warp + per-step fusion of ``docs/07-phase2-design.md`` §2.5
(implemented in :mod:`artvid.diffusion.latent_warp` / :mod:`artvid.diffusion.video`):
**cross-frame self-attention**, a.k.a. *extended* / *reference* / *sparse-causal*
attention. It is the implementation hook for the P2-M4 "cross-frame-attention
backbone" comparison (docs §6, milestone P2-M4) — here realised as a drop-in
diffusers attention processor on the *existing* SDXL UNet rather than a separate
video backbone, so it composes with the same ControlNet + IP-Adapter + flow stack.

The idea (no optical flow involved)
-----------------------------------
In a vanilla self-attention layer the current frame's latent activations attend
only to **themselves**: ``Q, K, V`` are all projected from the current frame's
``hidden_states``. Independent per-frame denoising of those activations is the
root cause of flicker — there is no information path coupling frame *t* to frame
*t-1* (or to an anchor keyframe).

Cross-frame attention couples them by **concatenating the KEYS and VALUES of a
reference frame** (the previous or anchor frame's activations *at the same UNet
layer and timestep*) onto the current frame's keys/values before the attention
softmax. The query stays the current frame's. Concretely, per attention head:

    Q_cur  = x_cur  @ W_q                      # (B, S,        d)
    K_cur  = x_cur  @ W_k ;  V_cur  = x_cur @ W_v
    K_ref  = x_ref  @ W_k ;  V_ref  = x_ref @ W_v     # reference frame's K/V
    K_cat  = concat([K_cur, K_ref], dim=seq)   # (B, S + S_ref, d)
    V_cat  = concat([V_cur, V_ref], dim=seq)   # (B, S + S_ref, d)
    out    = softmax(Q_cur @ K_cat^T / sqrt(d)) @ V_cat

Because the softmax now ranges over ``S + S_ref`` keys, each current-frame token
can *copy* appearance from whichever reference token it most resembles. Tokens
that have a strong match in the reference (static background, slowly-moving
regions) get pulled toward the reference's appearance → less flicker; genuinely
new content (disocclusions) has no good reference match and is synthesized
freshly. This is the diffusion analogue of "trust the warped previous frame where
it is reliable", achieved through learned feature similarity instead of an
explicit flow warp — which is exactly why it is **complementary**: it needs no
flow and degrades gracefully where the flow warp is unreliable, but it is weaker
at enforcing pixel-exact correspondence than the §2.5 warp. The two can be
stacked (flow warp for geometric lock + cross-frame attention for appearance
coherence).

The reference key/value bank
----------------------------
``W_k`` / ``W_v`` are the layer's *own* (frozen) projection weights, so the
reference K/V are produced by re-projecting cached reference *hidden_states*. We
cache the reference frame's per-layer ``hidden_states`` (the input to each self-
attention layer) rather than its K/V tensors, because (a) it is projection-weight
agnostic, and (b) under classifier-free guidance the batch carries an (uncond,
cond) pair whose K/V we must match positionally — caching hidden_states lets us
re-project with the right weights on each call. See :class:`CrossFrameAttnProcessor`.

diffusers integration
---------------------
A diffusers UNet exposes its attention processors as a ``dict`` keyed by dotted
module path via ``unet.attn_processors`` and lets you swap them with
``unet.set_attn_processor(dict)``. A processor is a callable with the
``AttnProcessor2_0`` signature

    __call__(self, attn, hidden_states, encoder_hidden_states=None,
             attention_mask=None, temb=None, **kwargs) -> Tensor

where ``attn`` is the owning ``diffusers.models.attention.Attention`` module
(carrying ``to_q`` / ``to_k`` / ``to_v`` / ``to_out``, ``heads``, group-norm,
etc.). We subclass / wrap the stock SDPA processor and only diverge in the
**self-attention** case (``encoder_hidden_states is None``): cross-attention
layers (where ``encoder_hidden_states`` is the text / IP-Adapter embeds) are left
**untouched** — coupling frames through the *text* cross-attention would be wrong.

Install/remove helpers (:func:`install_cross_frame_attention` /
:func:`remove_cross_frame_attention`) walk ``unet.attn_processors``, replace only
the self-attention entries (keys ending in ``attn1.processor``) with a shared
:class:`CrossFrameAttnProcessor`, and restore the originals respectively. They
return the live processor(s) so the caller can drive the per-frame protocol:

    proc = install_cross_frame_attention(unet)
    # frame 0 (anchor): record the reference bank, no injection
    proc.set_mode("record"); denoise(frame0); proc.clear()
    # frame t>0: replay the recorded bank as the reference K/V
    proc.set_reference(bank); proc.set_mode("inject"); denoise(frame_t)

Which modules this builds on
----------------------------
* :mod:`diffusers` — ``Attention`` module API + ``unet.attn_processors`` /
  ``set_attn_processor`` (lazy-imported; the env has no diffusers/torch).
* Composes with :mod:`artvid.diffusion.engine` (the UNet lives at
  ``DiffusionEngine.unet``) and is an alternative/addition to the flow path in
  :mod:`artvid.diffusion.latent_warp` / :mod:`artvid.diffusion.video`.
* It does **not** import torch/diffusers at module load: every framework symbol is
  imported inside a method/function, mirroring the rest of Phase 2, so this file
  is ``py_compile``-able and importable without the frameworks.

Hard-constraint / status note
-----------------------------
This is FOUNDATION/scaffolding written against the documented diffusers attention-
processor API. It is **not validated on hardware** (no GPU/torch here). Every
quality-sensitive choice — *which* layers to apply it to, *prev vs anchor* as the
reference, the bank-collection protocol, and interaction with the SDPA fast path —
is marked ``TODO(tuning)`` with what to verify on the M5 Max.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only; torch/diffusers optional at import
    import torch


# ---------------------------------------------------------------------------
# Tuning constants (starting points; verify on hardware).
# ---------------------------------------------------------------------------

#: Suffix that marks a *self*-attention processor key in ``unet.attn_processors``.
#: In diffusers' UNet2DConditionModel, transformer blocks expose ``attn1`` (self-
#: attention; ``encoder_hidden_states is None``) and ``attn2`` (cross-attention to
#: text / IP-Adapter). Only ``attn1`` should become cross-frame. The processor
#: dict keys are the dotted module paths with a trailing ``.processor`` (e.g.
#: ``down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor``).
SELF_ATTENTION_KEY_SUFFIX = "attn1.processor"

#: Default blend between the cross-frame-attended output and the original (self-
#: only) output: ``out = (1 - mix) * self_only + mix * cross_frame``. 1.0 = full
#: cross-frame attention. Exposing a mix lets the caller dial strength without
#: re-installing. TODO(tuning): the right strength is layer- and content-
#: dependent; 0.6-1.0 is a sane sweep. Too high over-couples (ghosting of the
#: reference into moving regions); too low does nothing.
DEFAULT_CROSS_FRAME_MIX = 1.0


# ---------------------------------------------------------------------------
# The processor.
# ---------------------------------------------------------------------------


class CrossFrameAttnProcessor:
    """Self-attention processor that concatenates a reference frame's K/V.

    Implements the diffusers ``AttnProcessor2_0`` call signature and uses
    ``torch.nn.functional.scaled_dot_product_attention`` (the SDPA fast path) for
    the attention itself. In **self-attention** layers and when enabled with a
    populated reference bank, it appends the reference frame's keys/values to the
    current frame's before the attention so each current-frame query attends over
    ``current ++ reference`` (the KV-concat math in the module docstring).

    Operating modes (set via :meth:`set_mode`):

    * ``"off"``    — behaves exactly like the stock SDPA self-attention processor
      (no recording, no injection). Safe default so installing the processor is a
      no-op until explicitly driven. This is what the anchor frame uses when the
      caller does not even want to record.
    * ``"record"`` — runs normal self-attention **and** caches this layer's input
      ``hidden_states`` into the reference bank, keyed by the layer id. Used while
      denoising the reference frame (prev or anchor) so its activations can be
      replayed for the next frame. No injection.
    * ``"inject"`` — runs cross-frame attention: re-projects the cached reference
      ``hidden_states`` for *this* layer through the layer's own ``to_k`` / ``to_v``
      and concatenates onto the current K/V. Falls back to plain self-attention
      for any layer that has no recorded reference (e.g. first frame, or a layer
      that was not recorded). Used while denoising frames ``t > 0``.

    Reference-bank keying
    ---------------------
    diffusers shares one processor *instance* across all the self-attention layers
    we install onto (so a single ``set_reference`` / ``set_mode`` call drives the
    whole UNet). To keep per-layer references apart we key the bank by ``id(attn)``
    (the owning :class:`~diffusers.models.attention.Attention` module's identity),
    which is stable across the record→inject calls for a fixed UNet. The cached
    value is the raw input ``hidden_states`` tensor for that layer (detached);
    re-projection happens at inject time with that layer's frozen weights so K/V
    are always consistent with the current layer.

    CFG / batch-alignment note
    --------------------------
    Under classifier-free guidance the UNet is called on a stacked (uncond, cond)
    batch, so ``hidden_states`` has batch ``2B`` and the recorded reference has the
    same batch layout. Concatenating along the *sequence* dimension therefore lines
    up uncond-with-uncond and cond-with-cond automatically **iff** the reference
    was recorded with the same CFG batch layout (it is, since both frames run the
    same loop). If the reference batch size does not match the current batch, we
    skip injection for that layer rather than mis-align (guarded in :meth:`__call__`).

    TODO(tuning) — to validate on the M5 Max:
      * **Which layers.** Applying cross-frame attention to *every* self-attention
        layer is the strongest but can over-smooth / ghost. Common practice limits
        it to the mid + up blocks (coarser, more semantic) and/or to a subset of
        timesteps. :func:`install_cross_frame_attention` accepts a ``layer_filter``
        for exactly this; sweep all-layers vs mid/up-only.
      * **Reference choice: prev vs anchor.** ``record`` the *previous* stylized
        frame for local coherence, or the *anchor* (frame 0) to fight long-range
        drift, or **both** (concatenate two references — supported: call
        :meth:`set_reference` with a merged bank that stacks two refs along seq).
      * **Mix / strength** (:data:`DEFAULT_CROSS_FRAME_MIX`) and whether to ramp it
        down over timesteps (more coupling early, freer synthesis late — mirrors the
        §2.5 fuse window).
      * **SDPA vs manual softmax.** SDPA cannot return attention weights; if a
        future variant needs to *reweight* reference keys (e.g. by flow reliability)
        switch to an explicit ``softmax`` path. Kept on SDPA here for speed.
      * Interaction with **IP-Adapter**: IP-Adapter installs its *own* processors on
        the cross-attention (``attn2``) layers. Because we only touch ``attn1`` the
        two are orthogonal, but confirm IP-Adapter did not also wrap ``attn1`` in the
        pinned diffusers version (it should not).
    """

    def __init__(self, *, mix: float = DEFAULT_CROSS_FRAME_MIX) -> None:
        #: One of "off" | "record" | "inject".
        self.mode: str = "off"
        #: Blend factor for the cross-frame vs self-only output (see module const).
        self.mix: float = float(mix)
        #: id(attn) -> reference hidden_states tensor (detached). Populated in
        #: "record" mode (live capture) or via :meth:`set_reference` (replay).
        self._reference_bank: dict[int, "torch.Tensor"] = {}
        #: Recording sink used in "record" mode; separate from the replay bank so a
        #: caller can record frame t while still holding frame t-1's replay bank.
        self._record_sink: dict[int, "torch.Tensor"] = {}

    # -- control surface ----------------------------------------------------

    def set_mode(self, mode: str) -> "CrossFrameAttnProcessor":
        """Set the operating mode: ``"off"`` | ``"record"`` | ``"inject"``."""
        if mode not in ("off", "record", "inject"):
            raise ValueError(
                f"mode must be 'off', 'record' or 'inject'; got {mode!r}."
            )
        if mode == "record":
            self._record_sink = {}
        self.mode = mode
        return self

    @property
    def enabled(self) -> bool:
        """True when the processor diverges from stock self-attention."""
        return self.mode in ("record", "inject")

    def set_reference(self, bank: "dict[int, torch.Tensor] | None") -> None:
        """Install the reference K/V-source bank used in ``"inject"`` mode.

        Args:
            bank: A mapping ``id(attn) -> hidden_states`` as produced by
                :meth:`take_recorded` after a ``"record"`` pass over the reference
                frame, or ``None`` to clear it (injection then no-ops to plain
                self-attention for every layer).
        """
        self._reference_bank = dict(bank) if bank else {}

    def take_recorded(self) -> "dict[int, torch.Tensor]":
        """Return the bank captured during the last ``"record"`` pass and reset it.

        The returned dict is what you hand back via :meth:`set_reference` before the
        next frame's ``"inject"`` pass. Returned by value (a shallow copy) so the
        caller owns it across the mode switch.
        """
        recorded = self._record_sink
        self._record_sink = {}
        return recorded

    def clear(self) -> None:
        """Drop both the replay bank and any in-progress recording."""
        self._reference_bank = {}
        self._record_sink = {}

    # -- the AttnProcessor2_0 call ------------------------------------------

    def __call__(
        self,
        attn: Any,
        hidden_states: "torch.Tensor",
        encoder_hidden_states: "Optional[torch.Tensor]" = None,
        attention_mask: "Optional[torch.Tensor]" = None,
        temb: "Optional[torch.Tensor]" = None,
        **kwargs: Any,
    ) -> "torch.Tensor":
        """diffusers ``AttnProcessor2_0``-compatible forward with optional KV-concat.

        Mirrors diffusers' stock ``AttnProcessor2_0`` (group-norm / spatial-norm
        pre-processing, q/k/v projection, head reshape, SDPA, ``to_out``) and only
        diverges to (a) cache ``hidden_states`` in ``"record"`` mode and (b)
        concatenate the reference K/V in ``"inject"`` mode. **Cross-attention layers
        are passed straight through to self-only behaviour** (we never couple frames
        via text/IP-Adapter attention).

        The body re-implements the stock processor rather than calling ``super()``
        because diffusers' ``AttnProcessor2_0`` is a plain object we wrap, not a base
        class. Keep this in sync with the pinned diffusers ``AttnProcessor2_0`` —
        TODO(tuning): diff it against the installed version on first hardware run;
        the residual-connection / rescale / ``group_norm`` ordering occasionally
        shifts across diffusers releases.
        """
        import torch
        import torch.nn.functional as F

        is_self_attention = encoder_hidden_states is None

        # ---- stock pre-processing (spatial norm, optional 4-D flatten) -------
        residual = hidden_states
        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, hh, ww = hidden_states.shape
            hidden_states = hidden_states.view(b, c, hh * ww).transpose(1, 2)

        batch_size = hidden_states.shape[0]
        # ``prepare_attention_mask`` keys off the *key* sequence length, which is the
        # current hidden_states for self-attention and the encoder_hidden_states for
        # cross-attention (matches the stock AttnProcessor2_0). SDXL is unmasked so
        # this only matters if a masked layer ever appears.
        kv_seq_len = (
            hidden_states.shape[1]
            if encoder_hidden_states is None
            else encoder_hidden_states.shape[1]
        )

        attention_mask = self._prepare_mask(
            attn, attention_mask, kv_seq_len, batch_size
        )

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        # --- record: cache this layer's (self-attention) input activations ---
        # We capture *before* projection so the replayed reference is projection-
        # weight agnostic (re-projected at inject time with the layer's own to_k/
        # to_v). Only meaningful for self-attention (attn1).
        if is_self_attention and self.mode == "record":
            self._record_sink[id(attn)] = hidden_states.detach()

        # ---- q/k/v projection ------------------------------------------------
        kv_input = (
            hidden_states if encoder_hidden_states is None else encoder_hidden_states
        )

        query = attn.to_q(hidden_states)
        key = attn.to_k(kv_input)
        value = attn.to_v(kv_input)

        # ---- cross-frame KV concat (self-attention + inject + have ref) ------
        # Build the (possibly concatenated) key/value. We keep the pre-concat
        # ``key``/``value`` for the optional ``mix < 1`` self-only blend.
        ref_used = False
        key_cat, value_cat = key, value
        if is_self_attention and self.mode == "inject":
            ref_hidden = self._reference_bank.get(id(attn))
            if ref_hidden is not None and ref_hidden.shape[0] == batch_size:
                # Re-project the cached reference activations through THIS layer's
                # frozen key/value weights, then concat along the sequence axis.
                ref_hidden = ref_hidden.to(device=query.device, dtype=query.dtype)
                ref_key = attn.to_k(ref_hidden)
                ref_value = attn.to_v(ref_hidden)
                key_cat = torch.cat([key, ref_key], dim=1)
                value_cat = torch.cat([value, ref_value], dim=1)
                ref_used = True
                # The appended reference keys are all valid; a non-None mask would
                # need extending by S_ref True columns. SDXL self-attention is
                # unmasked, so we flag rather than silently mis-mask.
                if attention_mask is not None:
                    raise NotImplementedError(
                        "cross-frame KV-concat with a non-None attention_mask is "
                        "not supported (SDXL self-attention is unmasked). "
                        "TODO(tuning): extend the mask by S_ref True columns if a "
                        "masked self-attention layer ever appears."
                    )

        head_dim = query.shape[-1] // attn.heads
        q = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = self._sdpa(
            attn, F, q, key_cat, value_cat, attention_mask, batch_size, head_dim
        )

        # ---- optional mix with the self-only output --------------------------
        # When ref_used and mix < 1, blend toward the plain (pre-concat) self-
        # attention result so strength is dial-able without re-installing. The
        # default mix == 1 skips this second SDPA for speed.
        if ref_used and self.mix < 1.0:
            self_only = self._sdpa(
                attn, F, q, key, value, attention_mask, batch_size, head_dim
            )
            hidden_states = (1.0 - self.mix) * self_only + self.mix * hidden_states

        # ---- output projection + stock residual ------------------------------
        hidden_states = attn.to_out[0](hidden_states)  # linear
        hidden_states = attn.to_out[1](hidden_states)  # dropout

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, c, hh, ww
            )

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual

        rescale = getattr(attn, "rescale_output_factor", 1.0)
        if rescale != 1.0:
            hidden_states = hidden_states / rescale

        return hidden_states

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _prepare_mask(
        attn: Any,
        attention_mask: "Optional[torch.Tensor]",
        kv_seq_len: int,
        batch_size: int,
    ) -> "Optional[torch.Tensor]":
        """Stock ``prepare_attention_mask`` + SDPA head-broadcast reshape.

        Returns ``None`` for the (overwhelmingly common in SDXL) unmasked case so
        the KV-concat path stays simple. TODO(tuning): confirm no SDXL self-
        attention layer passes a mask in the pinned diffusers (none should).
        """
        if attention_mask is None:
            return None
        mask = attn.prepare_attention_mask(attention_mask, kv_seq_len, batch_size)
        # SDPA expects (batch, heads, query, key); broadcast the head dim.
        return mask.view(batch_size, attn.heads, -1, mask.shape[-1])

    @staticmethod
    def _sdpa(
        attn: Any,
        F: Any,
        q: "torch.Tensor",
        key: "torch.Tensor",
        value: "torch.Tensor",
        attention_mask: "Optional[torch.Tensor]",
        batch_size: int,
        head_dim: int,
    ) -> "torch.Tensor":
        """Head-reshape ``key``/``value``, run SDPA against pre-reshaped ``q``, flatten.

        ``q`` is already ``(B, heads, S, head_dim)``; ``key``/``value`` arrive as
        ``(B, S_kv, inner_dim)`` (S_kv = S for self-only, S + S_ref for the cross-
        frame concat) and are reshaped here. Returns ``(B, S, heads*head_dim)`` in
        ``q``'s dtype. Shared by the cross-frame and the ``mix < 1`` self-only paths
        so the attention math lives in one place.
        """
        k = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        v = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        return (
            out.transpose(1, 2)
            .reshape(batch_size, -1, attn.heads * head_dim)
            .to(q.dtype)
        )


# ---------------------------------------------------------------------------
# Install / remove on a UNet (self-attention layers only).
# ---------------------------------------------------------------------------


def install_cross_frame_attention(
    unet: Any,
    *,
    mix: float = DEFAULT_CROSS_FRAME_MIX,
    layer_filter: Optional[Callable[[str], bool]] = None,
    processor: Optional[CrossFrameAttnProcessor] = None,
) -> CrossFrameAttnProcessor:
    """Install a shared :class:`CrossFrameAttnProcessor` on the UNet's self-attention.

    Walks ``unet.attn_processors`` and replaces only the **self-attention** entries
    (keys ending in :data:`SELF_ATTENTION_KEY_SUFFIX`, i.e. ``attn1.processor``)
    with one shared processor, leaving cross-attention (``attn2`` — text /
    IP-Adapter) processors untouched. A single shared instance means one
    :meth:`~CrossFrameAttnProcessor.set_mode` / ``set_reference`` call drives the
    whole UNet.

    The original processor dict is stashed on the UNet as
    ``unet._artvid_cfa_saved_processors`` so :func:`remove_cross_frame_attention`
    can restore it exactly (idempotent: a second install reuses the existing save).

    Args:
        unet: A diffusers ``UNet2DConditionModel`` (e.g.
            ``DiffusionEngine.unet``) exposing ``attn_processors`` /
            ``set_attn_processor``.
        mix: Forwarded to the processor (see :data:`DEFAULT_CROSS_FRAME_MIX`).
        layer_filter: Optional predicate ``key -> bool`` to restrict which self-
            attention layers get the cross-frame processor (the others keep their
            original). E.g. ``lambda k: k.startswith(("mid_block", "up_blocks"))``
            to apply only to mid/up blocks. TODO(tuning): which subset is best —
            docs §6 / this module's class docstring.
        processor: Optionally reuse an existing processor instance (e.g. to keep a
            recorded bank across an install/remove cycle). A new one is created when
            ``None``.

    Returns:
        The shared :class:`CrossFrameAttnProcessor` (drive it per-frame via
        ``set_mode`` / ``set_reference`` / ``take_recorded``).
    """
    proc = processor if processor is not None else CrossFrameAttnProcessor(mix=mix)

    current = dict(unet.attn_processors)

    # Stash originals once so remove() restores the pre-install state exactly.
    if not hasattr(unet, "_artvid_cfa_saved_processors"):
        unet._artvid_cfa_saved_processors = dict(current)

    new_processors: dict[str, Any] = {}
    for key, original in current.items():
        is_self_attn = key.endswith(SELF_ATTENTION_KEY_SUFFIX)
        if is_self_attn and (layer_filter is None or layer_filter(key)):
            new_processors[key] = proc
        else:
            new_processors[key] = original

    unet.set_attn_processor(new_processors)
    unet._artvid_cfa_processor = proc
    return proc


def remove_cross_frame_attention(unet: Any) -> None:
    """Restore the UNet's original attention processors saved at install time.

    Reverses :func:`install_cross_frame_attention` by re-applying the dict stashed
    on ``unet._artvid_cfa_saved_processors``. No-op (with a cleared handle) if no
    install is recorded, so it is safe to call unconditionally in a ``finally``.
    """
    saved = getattr(unet, "_artvid_cfa_saved_processors", None)
    if saved is not None:
        unet.set_attn_processor(dict(saved))
        del unet._artvid_cfa_saved_processors
    if hasattr(unet, "_artvid_cfa_processor"):
        del unet._artvid_cfa_processor


def self_attention_keys(unet: Any) -> list:
    """Return the ``attn_processors`` keys that are self-attention (``attn1``).

    Convenience for callers / tests that want to know which layers
    :func:`install_cross_frame_attention` would touch (and to build a
    ``layer_filter``). Pure dict-key inspection; no torch math.
    """
    return [
        key
        for key in unet.attn_processors
        if key.endswith(SELF_ATTENTION_KEY_SUFFIX)
    ]
