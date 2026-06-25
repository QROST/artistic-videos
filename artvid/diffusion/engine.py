"""Single-frame diffusion stylization engine (Phase 2 foundation).

What this module does
---------------------
Implements :class:`DiffusionStylizer`, the per-frame diffusion stylizer that
Phase 2 builds its temporal video loop on top of (see ``docs/07-phase2-design.md``
§3.1). It assembles the pinned model stack:

* **Base T2I** — Stable Diffusion XL (``stabilityai/stable-diffusion-xl-base-1.0``).
* **Structure ControlNet** — SDXL depth/canny/lineart ControlNet, locking the
  *geometry* of the content frame while leaving texture free to be restyled
  (the diffusion analogue of Phase 1's content loss).
* **IP-Adapter** — encodes an arbitrary style reference image once and injects it
  into the UNet cross-attention (the diffusion analogue of Phase 1's arbitrary
  Gram-matrix style, with zero training).

The engine produces a stylized frame from a content frame + style reference, and
deliberately exposes the lower-level handles the *video* module needs to graft the
Phase 1 optical-flow temporal-consistency idea into latent space:

* ``encode`` / ``decode`` — VAE round-trip (pixels <-> latents), so the video
  module can warp the *previous latent* (``docs/07`` §2.3) and decode the result.
* ``scheduler`` / ``add_noise`` / ``timesteps`` — so the video module can re-noise
  a flow-warped latent to the correct noise level and fuse it between denoise
  steps (``docs/07`` §2.5 mechanisms 1 & 2).
* ``denoise_frame`` — the K-step inner loop for one frame, accepting an optional
  warped-latent init + per-step reliability-masked fusion.

Which Phase 1 modules this builds on
------------------------------------
Per the design doc, ``engine.py`` is intentionally **thin on Phase 1 imports**:
only :mod:`artvid.device` (device/dtype/MPS-fallback policy) and
:mod:`artvid.io.image` (style-reference & frame I/O in RGB ``[0,1]`` CHW). The
flow stack (:mod:`artvid.flow.raft` / ``warp`` / ``consistency``) is consumed by
``artvid/diffusion/latent_warp.py`` and ``video.py``, **not** here — this keeps
the engine a clean diffusers wrapper and the flow reuse localised.

Which diffusers/transformers components this builds on
------------------------------------------------------
* ``diffusers.StableDiffusionXLControlNetPipeline`` (+ ``ControlNetModel``).
* ``pipe.load_ip_adapter`` / ``pipe.set_ip_adapter_scale`` /
  ``pipe.prepare_ip_adapter_image_embeds`` (IP-Adapter wiring).
* The pipeline's ``vae`` / ``unet`` / ``controlnet`` / dual text encoders /
  scheduler, accessed directly for the custom denoise loop.

Hard constraints honoured here
------------------------------
* ``torch`` / ``diffusers`` are **not** installed in the authoring environment and
  are **lazy-imported inside methods**, mirroring the Phase 1 pattern
  (``artvid/flow/raft.py``, ``artvid/cli.py``). This file is therefore
  ``py_compile``-able and ``--help``-safe without torch.
* This is FOUNDATION/scaffolding written against the documented diffusers API; it
  is meant to be **run and tuned on the user's M5 Max**. Every numerically- or
  quality-sensitive choice is marked ``TODO(tuning)`` with what to verify.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only; torch/diffusers optional at import
    import torch

from artvid import device as _device

# ---------------------------------------------------------------------------
# Defaults. These mirror the documented ``Config`` "diffusion" field group in
# ``docs/07-phase2-design.md`` §4.1. ``Config`` is owned by the config agent;
# we read every value via ``getattr(cfg, name, DEFAULT)`` so this engine works
# whether or not the config fields have landed yet, and so the contract is
# self-documenting here. If/when ``Config`` gains these fields, they win.
# ---------------------------------------------------------------------------

#: HF id of the base SDXL text-to-image model. TODO(tuning): confirm id/revision
#: is the mid-2026 SOTA + available; pin ``revision=`` once chosen (docs §1.1).
DEFAULT_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
#: HF id of the structure ControlNet. Depth is the default (best
#: structure-preserving-but-restylable signal). TODO(tuning): benchmark
#: depth vs canny/lineart vs a maintained ControlNet-Union checkpoint (docs §1.1).
DEFAULT_CONTROLNET_MODEL = "diffusers/controlnet-depth-sdxl-1.0"
DEFAULT_CONTROLNET_KIND = "depth"  # depth|canny|lineart|hed|tile (preprocessor side)
#: ControlNet conditioning scale: content/restyle tradeoff. TODO(tuning) docs §5.6.
DEFAULT_CONTROLNET_SCALE = 0.7

DEFAULT_IP_ADAPTER_REPO = "h94/IP-Adapter"
DEFAULT_IP_ADAPTER_SUBFOLDER = "sdxl_models"
#: ``ip-adapter_sdxl.bin`` (OpenCLIP-ViT-bigG) is the default; the lighter
#: ``ip-adapter_sdxl_vit-h.bin`` (CLIP-ViT-H) uses ~40% less encoder memory.
#: TODO(tuning): bigG-vs-ViT-H memory/fidelity tradeoff (docs §5.7).
DEFAULT_IP_ADAPTER_WEIGHT = "ip-adapter_sdxl.bin"
#: IP-Adapter style strength. TODO(tuning) docs §5.7.
DEFAULT_IP_ADAPTER_SCALE = 0.7

DEFAULT_PROMPT = ""
DEFAULT_NEGATIVE_PROMPT = ""
#: K denoising steps. TODO(tuning): fewest steps that keep quality (docs §5.8).
DEFAULT_STEPS = 30
#: Classifier-free-guidance scale. TODO(tuning) docs §5 / §4.1.
DEFAULT_GUIDANCE_SCALE = 6.0
#: euler|ddim|dpm. TODO(tuning): fastest scheduler that keeps quality (docs §5.8).
DEFAULT_SCHEDULER = "euler"

#: VAE spatial downsample factor (SDXL & SD1.5 both use 8). Don't change unless the
#: backbone differs. Exposed so ``latent_warp`` can rescale flow to the latent grid.
DEFAULT_VAE_FACTOR = 8


def _get(cfg: Any, name: str, default: Any) -> Any:
    """Read ``cfg.<name>`` falling back to ``default``.

    ``Config`` (owned by another agent) may or may not yet carry the diffusion
    field group from ``docs/07`` §4.1; reading defensively keeps this engine
    decoupled from that landing order.
    """
    if cfg is None:
        return default
    return getattr(cfg, name, default)


# ---------------------------------------------------------------------------
# Style reference handle. Kept as a small struct so the video module can encode
# the style once (it is frame-invariant) and reuse the embeds for every frame.
# ---------------------------------------------------------------------------


@dataclass
class StyleReference:
    """A once-encoded IP-Adapter style reference.

    Attributes:
        embeds: The IP-Adapter image embeds as produced by
            ``pipe.prepare_ip_adapter_image_embeds`` — typically a list of
            tensors (cond/uncond stacked), passed straight back to the UNet via
            ``added_cond_kwargs["image_embeds"]`` / the pipeline ``ip_adapter_image_embeds=``
            argument. Opaque to us; we just carry it.
        source: The path/identifier the embeds were computed from (for logging).
    """

    embeds: Any
    source: str


class DiffusionEngine:
    """SDXL + ControlNet + IP-Adapter single-frame stylizer (lazy-loaded).

    Construct with :meth:`from_config` (or the ``__init__`` kwargs directly),
    then call :meth:`load` (lazy; also auto-invoked by the first op) before
    :meth:`stylize` / :meth:`denoise_frame` / :meth:`encode` / :meth:`decode`.

    The heavy attributes (``pipe``, ``vae``, ``unet``, ``controlnet``,
    ``scheduler``) are ``None`` until :meth:`load` runs, so importing /
    constructing the stylizer never touches torch or downloads weights.
    """

    def __init__(
        self,
        *,
        base_model: str = DEFAULT_BASE_MODEL,
        controlnet_model: str = DEFAULT_CONTROLNET_MODEL,
        controlnet_scale: float = DEFAULT_CONTROLNET_SCALE,
        ip_adapter_repo: str = DEFAULT_IP_ADAPTER_REPO,
        ip_adapter_subfolder: str = DEFAULT_IP_ADAPTER_SUBFOLDER,
        ip_adapter_weight: str = DEFAULT_IP_ADAPTER_WEIGHT,
        ip_adapter_scale: float = DEFAULT_IP_ADAPTER_SCALE,
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        scheduler: str = DEFAULT_SCHEDULER,
        vae_factor: int = DEFAULT_VAE_FACTOR,
        device: str | None = None,
    ) -> None:
        self.base_model = base_model
        self.controlnet_model = controlnet_model
        self.controlnet_scale = float(controlnet_scale)
        self.ip_adapter_repo = ip_adapter_repo
        self.ip_adapter_subfolder = ip_adapter_subfolder
        self.ip_adapter_weight = ip_adapter_weight
        self.ip_adapter_scale = float(ip_adapter_scale)
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.steps = int(steps)
        self.guidance_scale = float(guidance_scale)
        self.scheduler_name = scheduler
        self.vae_factor = int(vae_factor)
        self._prefer_device = device

        # Resolved on load().
        self.device: "torch.device | None" = None
        self.dtype: "torch.dtype | None" = None
        self.pipe: Any = None
        self.loaded: bool = False

    # -- construction -------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Any) -> "DiffusionEngine":
        """Build a stylizer from an :class:`artvid.config.Config`.

        Reads the ``docs/07`` §4.1 diffusion field group defensively (see
        :func:`_get`) so it works regardless of whether those fields have landed
        in ``Config`` yet.
        """
        return cls(
            base_model=_get(cfg, "diff_base_model", DEFAULT_BASE_MODEL),
            controlnet_model=_get(cfg, "controlnet_model", DEFAULT_CONTROLNET_MODEL),
            controlnet_scale=_get(cfg, "controlnet_scale", DEFAULT_CONTROLNET_SCALE),
            ip_adapter_repo=_get(cfg, "ip_adapter_repo", DEFAULT_IP_ADAPTER_REPO),
            ip_adapter_subfolder=_get(
                cfg, "ip_adapter_subfolder", DEFAULT_IP_ADAPTER_SUBFOLDER
            ),
            ip_adapter_weight=_get(cfg, "ip_adapter_weight", DEFAULT_IP_ADAPTER_WEIGHT),
            ip_adapter_scale=_get(cfg, "ip_adapter_scale", DEFAULT_IP_ADAPTER_SCALE),
            prompt=_get(cfg, "diff_prompt", DEFAULT_PROMPT),
            negative_prompt=_get(cfg, "diff_negative_prompt", DEFAULT_NEGATIVE_PROMPT),
            steps=_get(cfg, "diff_steps", DEFAULT_STEPS),
            guidance_scale=_get(cfg, "guidance_scale", DEFAULT_GUIDANCE_SCALE),
            scheduler=_get(cfg, "diff_scheduler", DEFAULT_SCHEDULER),
            vae_factor=_get(cfg, "vae_factor", DEFAULT_VAE_FACTOR),
            device=_get(cfg, "device", None),
        )

    # -- dtype/device policy ------------------------------------------------

    def _resolve_dtype(self):
        """Pick the compute dtype for the diffusion weights.

        Policy (separate from Phase 1's float32-optimized-image policy, which does
        not apply to inference): run the UNet/ControlNet/VAE in low precision for
        speed and memory. On MPS, fp16 is the broadly-supported fast path; bf16
        support on MPS has historically been spotty. On CUDA, bf16 is generally
        preferable. On CPU, stay float32 (fp16 CPU kernels are slow/unsupported).

        TODO(tuning): On the M5 Max, verify fp16 is numerically stable through
        the full SDXL + ControlNet + IP-Adapter stack (some VAEs need fp32 or the
        ``madebyollin/sdxl-vae-fp16-fix`` to avoid black/NaN decodes). If the VAE
        decode produces NaNs/artifacts in fp16, either load the fp16-fix VAE or
        run *just* the VAE in fp32 (``pipe.vae.to(torch.float32)`` + cast latents)
        and keep the UNet in fp16. Also try bf16 on MPS and compare speed/quality.
        """
        import torch

        dev_type = self.device.type if self.device is not None else "cpu"
        if dev_type == "cuda":
            return torch.bfloat16
        if dev_type == "mps":
            # TODO(tuning): bf16-vs-fp16 on MPS; default fp16 for op coverage.
            return torch.float16
        return torch.float32  # cpu

    # -- lazy load ----------------------------------------------------------

    def load(self) -> "DiffusionEngine":
        """Build the diffusers pipeline and move it to the device. Idempotent.

        Lazy-imports ``torch`` / ``diffusers`` so import-time stays torch-free.
        Downloads weights from the HF hub on first run (like RAFT in Phase 1);
        document the first-run network requirement + licenses per docs §7.
        """
        if self.loaded:
            return self

        # Make MPS fall back to CPU for any unimplemented op before the first
        # dispatch (matches Phase 1; SDXL+ControlNet hit a few uncovered ops).
        _device.enable_mps_fallback()

        import torch  # noqa: F401  (imported for side-effect parity / dtype)
        from diffusers import (
            ControlNetModel,
            StableDiffusionXLControlNetPipeline,
        )

        self.device = _device.get_device(self._prefer_device)
        self.dtype = self._resolve_dtype()

        # --- ControlNet (structure) ---
        # TODO(tuning): pin ``revision=`` once the SOTA checkpoint is confirmed
        # (docs §1.1 / §5.9). ``variant="fp16"`` pulls the fp16-sharded weights
        # when published, halving download/peak; fall back if the variant is
        # missing for a given checkpoint.
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_model,
            torch_dtype=self.dtype,
        )

        # --- Base SDXL + ControlNet pipeline ---
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.base_model,
            controlnet=controlnet,
            torch_dtype=self.dtype,
            # ``use_safetensors=True`` is the safe default; ``variant="fp16"``
            # likewise when available. TODO(tuning): set variant once confirmed.
            use_safetensors=True,
        )

        # --- Scheduler selection ---
        pipe.scheduler = self._build_scheduler(pipe.scheduler.config)

        pipe = pipe.to(self.device)

        # --- IP-Adapter (style-from-reference) ---
        # ``load_ip_adapter`` also loads the CLIP/OpenCLIP image encoder named in
        # the IP-Adapter repo; ``image_encoder_folder`` defaults correctly for the
        # h94 layout. TODO(tuning): if the encoder is huge (bigG), consider the
        # ``ip-adapter_sdxl_vit-h.bin`` weight + matching encoder (docs §5.7).
        pipe.load_ip_adapter(
            self.ip_adapter_repo,
            subfolder=self.ip_adapter_subfolder,
            weight_name=self.ip_adapter_weight,
        )
        pipe.set_ip_adapter_scale(self.ip_adapter_scale)

        # --- Memory knobs ---
        # On 128 GB unified memory the full stack fits resident; CPU offload is
        # slow on MPS so we do NOT enable it. TODO(tuning): at 1024²+ batched
        # frames, ``enable_attention_slicing()`` / ``enable_vae_tiling()`` trade a
        # little speed for lower peak — measure peak with MPS counters and decide
        # (docs §1.3). Left OFF by default.

        self.pipe = pipe
        self.loaded = True
        return self

    def _build_scheduler(self, base_config: Any):
        """Instantiate the requested scheduler from the pipeline's config.

        Maps ``cfg.diff_scheduler`` (euler|ddim|dpm) to a diffusers scheduler,
        preserving the model's training config (betas, prediction type, etc.) via
        ``from_config``. TODO(tuning): the fastest scheduler that keeps quality
        for stylization is hardware-dependent (docs §5.8). Euler is a robust
        default; DPM++ (``DPMSolverMultistepScheduler``) often needs fewer steps.
        """
        from diffusers import (
            DDIMScheduler,
            DPMSolverMultistepScheduler,
            EulerDiscreteScheduler,
        )

        name = (self.scheduler_name or DEFAULT_SCHEDULER).lower()
        if name == "ddim":
            return DDIMScheduler.from_config(base_config)
        if name in ("dpm", "dpmpp", "dpm++", "dpmsolver"):
            return DPMSolverMultistepScheduler.from_config(base_config)
        if name == "euler":
            return EulerDiscreteScheduler.from_config(base_config)
        raise ValueError(
            f"Unknown scheduler {self.scheduler_name!r}; expected euler|ddim|dpm."
        )

    def _ensure_loaded(self) -> None:
        if not self.loaded:
            self.load()

    # -- handles the video module needs ------------------------------------

    @property
    def scheduler(self) -> Any:
        """The active diffusers scheduler (for ``set_timesteps`` / ``step`` /
        ``add_noise`` / ``scale_model_input``). ``video.py`` drives the latent
        temporal fusion through this object (docs §2.5)."""
        self._ensure_loaded()
        return self.pipe.scheduler

    @property
    def vae(self) -> Any:
        self._ensure_loaded()
        return self.pipe.vae

    @property
    def unet(self) -> Any:
        self._ensure_loaded()
        return self.pipe.unet

    @property
    def controlnet(self) -> Any:
        self._ensure_loaded()
        return self.pipe.controlnet

    # -- VAE round-trip (so video.py can warp/decode latents) ---------------

    def encode(self, image_rgb: "torch.Tensor") -> "torch.Tensor":
        """Encode an RGB ``[0,1]`` CHW image into a scaled VAE latent.

        Used by the ``warp_space="pixel"`` variant (docs §2.5: warp the previous
        *decoded* frame, then re-encode) and to seed/init latents.

        Args:
            image_rgb: ``(3, H, W)`` or ``(N, 3, H, W)`` float in ``[0, 1]`` RGB
                (as produced by :func:`artvid.io.image.load_image` / :meth:`decode`).

        Returns:
            ``(N, C, H/f, W/f)`` latent already multiplied by
            ``vae.config.scaling_factor`` (the convention diffusers' schedulers
            and UNet expect). ``N`` is 1 for unbatched input.
        """
        self._ensure_loaded()
        import torch

        x = image_rgb if image_rgb.dim() == 4 else image_rgb.unsqueeze(0)
        x = x.to(device=self.device, dtype=self.vae.dtype)
        # VAE expects inputs in [-1, 1].
        x = x * 2.0 - 1.0
        with torch.no_grad():
            posterior = self.vae.encode(x).latent_dist
            # ``.mode()`` (deterministic) is preferable to ``.sample()`` for a
            # reproducible warp/re-encode round-trip. TODO(tuning): if you want
            # the encode to inject a little noise diversity, switch to ``.sample()``.
            latent = posterior.mode()
        latent = latent * self.vae.config.scaling_factor
        return latent

    def decode(self, latent: "torch.Tensor") -> "torch.Tensor":
        """Decode a scaled VAE latent into an RGB ``[0,1]`` CHW image.

        Inverse of :meth:`encode`; the per-frame output step of docs §2.5.

        Args:
            latent: ``(N, C, h, w)`` or ``(C, h, w)`` latent **already scaled by**
                ``vae.config.scaling_factor`` (as returned by :meth:`encode` and
                the denoise loop).

        Returns:
            ``(3, H, W)`` (unbatched in) or ``(N, 3, H, W)`` float32 RGB in
            ``[0, 1]`` on CPU-friendly dtype, ready for
            :func:`artvid.io.image.save_image` (which writes raw RGB; pass
            ``mode``-less / RGB save path — see ``video.py``).
        """
        self._ensure_loaded()
        import torch

        was_3d = latent.dim() == 3
        z = latent.unsqueeze(0) if was_3d else latent
        z = z.to(device=self.device, dtype=self.vae.dtype)
        z = z / self.vae.config.scaling_factor
        with torch.no_grad():
            img = self.vae.decode(z).sample  # in [-1, 1]
        img = (img / 2.0 + 0.5).clamp(0.0, 1.0).to(torch.float32)
        return img[0] if was_3d else img

    def add_noise(
        self,
        latent: "torch.Tensor",
        noise: "torch.Tensor",
        timestep: "torch.Tensor",
    ) -> "torch.Tensor":
        """Re-noise ``latent`` to the noise level of ``timestep`` via the scheduler.

        Thin pass-through to ``scheduler.add_noise`` so ``video.py`` can put a
        flow-warped latent onto the same manifold as the running latent before
        fusing (docs §2.5 mechanisms 1 & 2) without reaching into ``self.pipe``.
        """
        self._ensure_loaded()
        return self.scheduler.add_noise(latent, noise, timestep)

    # -- text / style conditioning -----------------------------------------

    def encode_style(self, style_image: "str | Path | torch.Tensor") -> StyleReference:
        """Encode the IP-Adapter style reference **once** (frame-invariant).

        Args:
            style_image: A path to a style image, or a ``(3, H, W)`` RGB ``[0,1]``
                tensor / PIL-convertible. The video module calls this a single
                time and reuses the result for every frame (docs §2.5).

        Returns:
            A :class:`StyleReference` carrying the IP-Adapter image embeds.

        Notes:
            We call ``pipe.prepare_ip_adapter_image_embeds`` to precompute the
            embeds rather than re-running the image encoder per frame. The exact
            signature varies across diffusers versions; this passes the documented
            arguments. TODO(tuning): confirm the arg names/return shape against the
            pinned diffusers version — some versions want ``ip_adapter_image=`` as a
            list (one entry per adapter) and a ``num_images_per_prompt`` /
            ``do_classifier_free_guidance`` pair, returning a list of cond/uncond
            stacked tensors.
        """
        self._ensure_loaded()

        pil = self._to_pil(style_image)

        # ``prepare_ip_adapter_image_embeds`` exists on the IP-Adapter mixin and
        # returns embeds ready to hand back to the call/UNet. TODO(tuning): verify
        # signature on pinned diffusers; below is the documented mid-2026 form.
        embeds = self.pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=[pil],
            ip_adapter_image_embeds=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=self.guidance_scale > 1.0,
        )
        src = str(style_image) if isinstance(style_image, (str, Path)) else "<tensor>"
        return StyleReference(embeds=embeds, source=src)

    def _to_pil(self, image: "str | Path | torch.Tensor"):
        """Coerce a path / RGB[0,1] CHW tensor into a PIL.Image (encoder input)."""
        from PIL import Image

        if isinstance(image, (str, Path)):
            from artvid.io.image import load_image

            image = load_image(image)  # (3,H,W) float32 [0,1]

        if hasattr(image, "dim"):  # torch tensor
            import numpy as np

            t = image
            if t.dim() == 4:
                t = t[0]
            arr = (
                t.detach().to("cpu").clamp(0.0, 1.0).mul(255.0).round()
                .permute(1, 2, 0).to("cpu")
            )
            return Image.fromarray(arr.numpy().astype("uint8"), mode="RGB")
        # Already PIL / array-like.
        return image

    # -- the single-frame core ---------------------------------------------

    def stylize(
        self,
        content_image: "str | Path | torch.Tensor",
        *,
        style: "StyleReference | str | Path | torch.Tensor | None" = None,
        control_image: "torch.Tensor | None" = None,
        init_latents: "torch.Tensor | None" = None,
        strength: float = 1.0,
        seed: int | None = None,
        return_latent: bool = False,
    ) -> "torch.Tensor":
        """Stylize one content frame. The P2-M0 single-image entry point.

        This is the convenience wrapper the CLI single-image path and quick tests
        use. The video module instead drives :meth:`denoise_frame` directly so it
        can inject warped latents between steps.

        Args:
            content_image: Content frame as a path or ``(3, H, W)`` RGB ``[0,1]``
                tensor. Its structure (geometry) is what gets preserved.
            style: The style reference — either a pre-encoded :class:`StyleReference`
                (preferred; encode once) or a raw path/tensor (encoded here).
            control_image: Optional precomputed ControlNet conditioning image
                ``(3, H, W)`` RGB ``[0,1]`` (e.g. a depth/lineart map from
                ``artvid/diffusion/preprocess.py``). If ``None``, the structure
                map is built from ``content_image`` by :meth:`_build_control`.
            init_latents: Optional latent to start denoising from (the
                flow-warped-previous-latent init of docs §2.5 mechanism 1). When
                given with ``strength < 1`` it acts like img2img: only the last
                ``strength``-fraction of the schedule is run, re-noising
                ``init_latents`` to that start timestep.
            strength: img2img-style denoising strength in ``[0, 1]``. ``1.0`` =
                full generation from noise (anchor/first frame); ``<1.0`` keeps
                more of ``init_latents`` (used for temporal continuity).
                TODO(tuning): the strength sweet spot is the main fidelity/coherence
                knob for the warped-init mechanism (docs §5.3).
            seed: Per-frame RNG seed for the base noise. TODO(tuning): fixed seed
                across frames vs warped-noise vs per-frame random — docs §5.5.
            return_latent: If True, return the final **latent** (so the video loop
                can carry it forward) instead of the decoded image.

        Returns:
            ``(3, H, W)`` RGB ``[0,1]`` image (default) or the final
            ``(1, C, h, w)`` latent if ``return_latent`` is True.
        """
        self._ensure_loaded()

        content = self._load_rgb(content_image)
        style_ref = self._resolve_style(style)
        control = control_image if control_image is not None else self._build_control(content)

        z = self.denoise_frame(
            content,
            control_image=control,
            style=style_ref,
            init_latents=init_latents,
            strength=strength,
            seed=seed,
        )
        if return_latent:
            return z
        return self.decode(z)

    def denoise_frame(
        self,
        content_rgb: "torch.Tensor",
        *,
        control_image: "torch.Tensor",
        style: "StyleReference | None",
        init_latents: "torch.Tensor | None" = None,
        reliability: "torch.Tensor | None" = None,
        warped_latent: "torch.Tensor | None" = None,
        strength: float = 1.0,
        steps: int | None = None,
        seed: int | None = None,
        fuse_steps: "set[int] | None" = None,
        temporal_strength: float = 0.0,
    ) -> "torch.Tensor":
        """Run the K-step ControlNet+IP-Adapter denoise loop for one frame.

        This is the §2.5 inner loop. It is written as a **hand-rolled loop** over
        ``scheduler.timesteps`` (rather than a single ``pipe(...)`` call) precisely
        so the video module can:

          * start from a flow-**warped previous latent** (``init_latents`` +
            ``strength < 1``) — mechanism 1; and
          * **fuse** the warped latent back in at selected steps in reliable
            regions (``warped_latent`` + ``reliability`` + ``fuse_steps`` +
            ``temporal_strength``) — mechanism 2.

        When ``warped_latent``/``reliability``/``fuse_steps`` are all unset and
        ``strength == 1`` and ``init_latents is None``, this degenerates to a
        plain per-frame ControlNet+IP-Adapter generation (the P2-M0 / anchor path).

        Args:
            content_rgb: ``(3, H, W)`` RGB ``[0,1]`` content frame (drives the
                SDXL resolution; latent is ``(1, 4, H/f, W/f)``).
            control_image: ``(3, H, W)`` RGB ``[0,1]`` ControlNet conditioning map.
            style: Pre-encoded :class:`StyleReference` (or ``None`` to skip
                IP-Adapter — discouraged; the style comes from it).
            init_latents: Optional latent to renoise+denoise (mechanism 1).
            reliability: ``(1, 1, h, w)`` float ``[0,1]`` latent-grid reliability
                mask (from ``latent_warp.latent_reliability``), used by the fusion.
            warped_latent: ``(1, C, h, w)`` flow-warped previous latent, the fusion
                target (mechanism 2).
            strength: img2img strength in ``[0,1]`` (see :meth:`stylize`).
            steps: Override K; defaults to ``self.steps``.
            seed: Base-noise seed.
            fuse_steps: Iteration indices ``i`` at which to fuse the warped latent.
            temporal_strength: Cap on the fusion blend (multiplied by
                ``reliability``). TODO(tuning): the §5.3 knob.

        Returns:
            The final ``(1, C, h, w)`` denoised latent (scaled). Decode with
            :meth:`decode`.
        """
        self._ensure_loaded()
        import torch

        K = int(steps) if steps is not None else self.steps
        device = self.device
        dtype = self.dtype

        content = self._load_rgb(content_rgb)
        _, H, W = content.shape
        h, w = H // self.vae_factor, W // self.vae_factor

        # --- conditioning embeds ---
        text = self._encode_prompts()
        style_embeds = style.embeds if style is not None else None
        control = self._prep_control(control_image, H, W)

        # --- scheduler timesteps (with optional img2img truncation) ---
        self.scheduler.set_timesteps(K, device=device)
        timesteps = self.scheduler.timesteps
        # img2img start offset: with strength<1 we skip the first
        # (1-strength) fraction of steps and renoise init_latents to that point.
        t_start_idx = 0
        if init_latents is not None and strength < 1.0:
            t_start_idx = max(0, int(round(K * (1.0 - strength))))
        timesteps = timesteps[t_start_idx:]

        # --- base noise ---
        generator = None
        if seed is not None:
            # MPS generators must live on CPU in some torch builds; using a CPU
            # generator for randn then moving is the portable path.
            # TODO(tuning): confirm MPS RNG path on the M5 Max torch build.
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(
            (1, self._latent_channels(), h, w),
            generator=generator,
            dtype=dtype,
            device="cpu" if generator is not None else device,
        ).to(device)

        # --- initial latent ---
        if init_latents is not None:
            init = init_latents.to(device=device, dtype=dtype)
            t0 = timesteps[0]
            latents = self.scheduler.add_noise(init, noise, t0)
        else:
            # Pure-noise start scaled to the scheduler's initial sigma (Euler) /
            # convention. ``init_noise_sigma`` is 1.0 for DDIM and the max sigma
            # for Euler; multiplying covers both.
            latents = noise * self.scheduler.init_noise_sigma

        do_cfg = self.guidance_scale > 1.0

        for i, t in enumerate(timesteps):
            model_in = self.scheduler.scale_model_input(latents, t)

            noise_pred = self._unet_step(
                model_in,
                t,
                text=text,
                style_embeds=style_embeds,
                control=control,
                do_cfg=do_cfg,
            )

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # --- mechanism 2: per-step warped-latent fusion ---
            if (
                warped_latent is not None
                and reliability is not None
                and temporal_strength > 0.0
                and fuse_steps is not None
                and i in fuse_steps
            ):
                # Re-noise the warped target to the CURRENT level so it lives on
                # the same manifold as ``latents`` at step i+1's input timestep.
                # Using the same per-frame ``noise`` keeps the fusion coherent.
                # NOTE: after ``step`` we are conceptually at the *next* timestep;
                # using ``t`` here is a close approximation. TODO(tuning): whether
                # to renoise to ``t`` or to ``timesteps[i+1]`` materially affects
                # the blend — verify on hardware (docs §2.5 / §5.3).
                warped_t = self.scheduler.add_noise(
                    warped_latent.to(device=device, dtype=dtype), noise, t
                )
                blend = (temporal_strength * reliability).to(device=device, dtype=dtype)
                latents = (1.0 - blend) * latents + blend * warped_t

        return latents

    # -- UNet / ControlNet plumbing ----------------------------------------

    def _unet_step(
        self,
        model_in: "torch.Tensor",
        t: "torch.Tensor",
        *,
        text: dict,
        style_embeds: Any,
        control: "torch.Tensor",
        do_cfg: bool,
    ) -> "torch.Tensor":
        """One ControlNet→UNet forward with classifier-free guidance.

        Builds the duplicated (uncond, cond) batch for CFG, runs the ControlNet to
        get the down/mid residuals from the structure map, then the UNet with the
        SDXL ``added_cond_kwargs`` (time ids + text-pooled + IP-Adapter
        ``image_embeds``), and combines with the guidance scale.

        TODO(tuning): this is the most version-sensitive block. Confirm against the
        pinned diffusers:
          * SDXL ``added_cond_kwargs`` keys are ``text_embeds`` (pooled) and
            ``time_ids``; IP-Adapter adds ``image_embeds``.
          * ControlNet returns ``down_block_res_samples`` (a tuple) and
            ``mid_block_res_sample``; the UNet consumes them as
            ``down_block_additional_residuals`` / ``mid_block_additional_residual``.
          * Whether the high-level ``pipe(..., callback_on_step_end=...)`` (which
            can mutate ``latents`` between steps) is cleaner than this hand-rolled
            path — docs §3.1 prefers the callback if it cleanly exposes mid-loop
            latents. This method is the explicit fallback.
        """
        import torch

        if do_cfg:
            latent_in = torch.cat([model_in, model_in], dim=0)
            control_in = torch.cat([control, control], dim=0)
            prompt_embeds = text["prompt_embeds"]  # already (2, ...) cond/uncond stacked
            added = dict(text["added_cond_kwargs"])
        else:
            latent_in = model_in
            control_in = control
            prompt_embeds = text["prompt_embeds_cond"]
            added = dict(text["added_cond_kwargs_cond"])

        # Inject IP-Adapter style embeds (the encoder ran once in encode_style).
        if style_embeds is not None:
            added["image_embeds"] = style_embeds

        down_res, mid_res = self.controlnet(
            latent_in,
            t,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=control_in,
            conditioning_scale=self.controlnet_scale,
            added_cond_kwargs=added,
            return_dict=False,
        )

        noise_pred = self.unet(
            latent_in,
            t,
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
            added_cond_kwargs=added,
            return_dict=False,
        )[0]

        if do_cfg:
            uncond, cond = noise_pred.chunk(2, dim=0)
            noise_pred = uncond + self.guidance_scale * (cond - uncond)
        return noise_pred

    def _encode_prompts(self) -> dict:
        """Encode the SDXL dual text encoders + build ``added_cond_kwargs``.

        SDXL needs: two text encoders → concatenated ``prompt_embeds`` plus a
        ``pooled_prompt_embeds`` (from encoder-2), and ``time_ids`` (original size
        / crop / target size). ``pipe.encode_prompt`` returns the four embed
        tensors; we assemble both the CFG-stacked and cond-only variants so
        :meth:`_unet_step` can pick per ``do_cfg``.

        TODO(tuning): confirm ``encode_prompt`` return order on the pinned
        diffusers (``prompt_embeds, negative_prompt_embeds, pooled,
        negative_pooled``) and the ``time_ids`` construction (uses the frame H/W,
        which we don't have here — built lazily in :meth:`_prep_control`/the loop
        if needed). For the foundation we cache prompt embeds and compute time_ids
        from the control image size at call time; see ``_build_time_ids``.
        """
        self._ensure_loaded()
        import torch

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled,
            negative_pooled,
        ) = self.pipe.encode_prompt(
            prompt=self.prompt,
            prompt_2=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=self.guidance_scale > 1.0,
            negative_prompt=self.negative_prompt,
        )

        # time_ids depend on resolution; filled per-frame in _prep_control via a
        # stash. We expose a builder and let the loop set the right size. For the
        # scaffold we default to a square; _prep_control overwrites self._time_ids.
        time_ids = getattr(self, "_time_ids", None)
        if time_ids is None:
            time_ids = self._build_time_ids(1024, 1024)

        cond_added = {"text_embeds": pooled, "time_ids": time_ids}
        # CFG-stacked variants (uncond first, matching the latent cat order).
        stacked_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        stacked_pooled = torch.cat([negative_pooled, pooled], dim=0)
        stacked_time_ids = torch.cat([time_ids, time_ids], dim=0)
        cfg_added = {"text_embeds": stacked_pooled, "time_ids": stacked_time_ids}

        return {
            "prompt_embeds": stacked_embeds,
            "added_cond_kwargs": cfg_added,
            "prompt_embeds_cond": prompt_embeds,
            "added_cond_kwargs_cond": cond_added,
        }

    def _build_time_ids(self, height: int, width: int) -> "torch.Tensor":
        """SDXL ``add_time_ids`` = (orig_h, orig_w, crop_top, crop_left, tgt_h, tgt_w).

        TODO(tuning): match the pipeline's own ``_get_add_time_ids`` (it also
        accounts for the text-encoder projection dim); for the foundation we use
        the no-crop, original==target convention which is the standard full-frame
        stylization case.
        """
        import torch

        ids = [height, width, 0, 0, height, width]
        return torch.tensor([ids], dtype=self.dtype, device=self.device)

    def _prep_control(self, control_image: "torch.Tensor", H: int, W: int) -> "torch.Tensor":
        """Move/scale the ControlNet conditioning image to ``(1,3,H,W)`` on device.

        Also stashes resolution-correct SDXL ``time_ids`` for this frame so
        :meth:`_encode_prompts` builds them at the right size.
        """
        import torch
        import torch.nn.functional as F

        c = control_image
        if c.dim() == 3:
            c = c.unsqueeze(0)
        c = c.to(device=self.device, dtype=self.dtype)
        if c.shape[-2:] != (H, W):
            c = F.interpolate(c, size=(H, W), mode="bilinear", align_corners=False)
        self._time_ids = self._build_time_ids(H, W)
        return c

    # -- helpers ------------------------------------------------------------

    def _latent_channels(self) -> int:
        """VAE latent channel count (4 for SDXL/SD1.5). Read from config."""
        self._ensure_loaded()
        return int(getattr(self.unet.config, "in_channels", 4))

    def _load_rgb(self, image: "str | Path | torch.Tensor") -> "torch.Tensor":
        """Coerce a path / tensor into a ``(3, H, W)`` RGB ``[0,1]`` float tensor."""
        if isinstance(image, (str, Path)):
            from artvid.io.image import load_image

            return load_image(image)
        if image.dim() == 4:
            return image[0]
        return image

    def _resolve_style(
        self, style: "StyleReference | str | Path | torch.Tensor | None"
    ) -> "StyleReference | None":
        """Return a :class:`StyleReference`, encoding a raw input if needed."""
        if style is None:
            return None
        if isinstance(style, StyleReference):
            return style
        return self.encode_style(style)

    def _build_control(self, content_rgb: "torch.Tensor") -> "torch.Tensor":
        """Build the ControlNet structure map from a content frame.

        Delegates to ``artvid/diffusion/preprocess.py`` (owned by the preprocess
        agent) when present; that module wraps ``controlnet_aux`` depth/lineart/
        canny extractors. We import it lazily so this engine does not hard-depend
        on it during the foundation increment.

        TODO(tuning): which structure signal best preserves content while allowing
        restyling — start with ``depth`` (docs §5.6). If ``preprocess`` is not yet
        available, raise a clear error pointing the caller to pass ``control_image=``.
        """
        try:
            from artvid.diffusion import preprocess as _pp
        except Exception as exc:  # pragma: no cover - foundation: module may not exist yet
            raise NotImplementedError(
                "Structure preprocessing lives in artvid.diffusion.preprocess "
                "(controlnet_aux depth/lineart/canny). It is not importable; pass "
                "an explicit control_image= to stylize()/denoise_frame()."
            ) from exc
        return _pp.structure_map(content_rgb, DEFAULT_CONTROLNET_KIND)


# Alias: ``docs/07`` §3.1 names the class ``DiffusionEngine`` (used by the
# package ``__init__`` / ``video`` / ``cli`` agents); the prompt suggested
# ``DiffusionStylizer``. Expose both so either name works.
DiffusionStylizer = DiffusionEngine

