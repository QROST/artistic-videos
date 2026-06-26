# 07 · Phase 2 Design: Diffusion Video Stylization with Latent Optical-Flow Consistency

> **Status:** concrete, implementable spec. Turns the directional `docs/04-phase2-plan.md`
> into the design the implementation agents follow.
> **Scope of this increment:** FOUNDATION / scaffolding. The code is written against the
> documented `diffusers` / `torch` API and is meant to be **run and tuned on your Apple
> Silicon (M-series) Mac with the Metal/MPS backend**. Numerically/qualitatively sensitive
> choices are marked `TODO(tuning)` with exactly what to verify on hardware.
> **Hard dependency note:** `torch` / `diffusers` are *not* installed in the authoring
> environment. Nothing here may be executed end-to-end here; only `py_compile` and
> torch-free unit tests run. Treat every numeric default below as a starting point.

This phase grafts the 2016 optical-flow temporal-consistency idea (Ruder et al., which
Phase 1 ports) onto a modern diffusion stylizer, **reusing the Phase 1 flow stack**
(`artvid.flow.raft`, `artvid.flow.warp`, `artvid.flow.consistency`) but moving the warp +
reliability masking from pixel space into **VAE latent space**. The result is the project's
differentiator: structure-locked, reference-styled, temporally-coherent diffusion video on
Apple Silicon, with the same RAFT/consistency machinery that drives the optimization engine.

---

## 0. Relationship to the rest of the codebase

| Existing module | Phase 2 use | Concrete call sites in this design |
|---|---|---|
| `artvid.flow.raft.compute_flow_pair(img1, img2, *, device) -> FlowPair{forward,backward}` | flow source for latent warp & masking | §2 per-frame loop |
| `artvid.flow.warp.warp_image(img, flow, *, fill, align_corners) -> WarpResult{image,valid}` and `warp.flow_to_grid(flow, *, align_corners)` | latent warp (we call `flow_to_grid` directly at latent resolution; `warp_image` for pixel-space previews) | §2.3 `latent_warp.warp_latent` |
| `artvid.flow.consistency.consistency_mask(fwd, bwd, *, smooth_sigma, check_motion_boundaries) -> (H,W) [0,1]` | per-pixel reliability → downsampled to latent grid | §2.4 |
| `artvid.flow.consistency.combine_longterm_weights(weights, method, *, invert)` | optional multi-reference (anchor + prev) weighting | §2.6 |
| `artvid.device.{enable_mps_fallback, get_device}` | device/dtype policy, MPS op fallback | engine init |
| `artvid.io.image.{load_image, save_image}` | style-reference & frame I/O (RGB [0,1] CHW) | engine I/O |
| `artvid.io.video.{extract_frames, encode_video}` | video ⇄ frames (fully reused) | §3 `video.py` |
| `artvid.config.Config` | extended with a diffusion field group | §4 (owned by config agent) |
| `artvid.cli` `--engine diffusion` | dispatch switch (currently `NotImplementedError`) | §4 (owned by cli agent) |

Phase 1 conventions this design **must** honor:
- Flow is `(2, H, W)` float32 in `(u, v) = (dx, dy)` order; `flow[:, y, x]` maps pixel
  `(x, y)` in `img1` to `img2`. `warp_image` is a **backward** warp: given the
  *backward* flow (current → previous) it samples the previous image into the current
  frame's coordinates and returns `valid` (True where the bilinear support stayed inside).
- `consistency_mask(primary, crosscheck, ...)`: to validate the **backward** warp (the
  one we use to pull the previous frame forward), call
  `consistency_mask(backward, forward, ...)` — matching `cli.cmd_flow`'s `rel_back`.
- Images are RGB in `[0,1]`, CHW, **not** caffe-preprocessed (RAFT and the VAE both want
  plain RGB [0,1]; the VGG caffe path is irrelevant to Phase 2).

---

## 1. Model stack

Design target: **image diffusion + ControlNet (structure) + IP-Adapter (style-from-reference)
+ latent optical-flow consistency**, applied per frame. This maximizes reuse of the Phase 1
flow stack, gives strong/interpretable control, and is the cheapest thing to get correct on
MPS. Native video-diffusion backbones (AnimateDiff / SVD / DiT) are explicitly deferred to a
later comparison milestone (§6) because their style-control interfaces differ and they don't
exercise our flow stack.

### 1.1 Pinned default stack (SDXL family)

| Role | Model (HF id) | Why | License |
|---|---|---|---|
| Base T2I | `stabilityai/stable-diffusion-xl-base-1.0` | strong, ubiquitous, best ControlNet/IP-Adapter ecosystem; runs on MPS | **CreativeML Open RAIL++-M** (open weights, use-based restrictions; redistribution allowed with the license + use restrictions attached) |
| Structure ControlNet | `diffusers/controlnet-depth-sdxl-1.0` (default) — alt: `diffusers/controlnet-canny-sdxl-1.0`, lineart/HED via `controlnet_aux` | depth gives the best *structure-preserving but restylable* signal for stylization (locks geometry, frees texture); canny/lineart for line-art targets; tile for high-fidelity refinement | **OpenRAIL** (Stability/diffusers-published SDXL ControlNets) |
| Style-from-reference | `h94/IP-Adapter` → `sdxl_models/ip-adapter_sdxl.bin` (uses image encoder `models/image_encoder`, OpenCLIP-ViT-bigG-14) or the lighter `ip-adapter_sdxl_vit-h.bin` (CLIP-ViT-H, ~40% less encoder memory) | zero-training "arbitrary style image" — the diffusion analogue of Phase 1's arbitrary-style strength | **Apache-2.0** (IP-Adapter weights) |
| Structure preprocessors | `controlnet_aux` (MiDaS/DPT depth, Canny, LineartDetector, HEDdetector) | produce the ControlNet conditioning image per frame | Apache-2.0 / model-specific |

`TODO(tuning)`: confirm the **exact current HF ids/revisions exist and are the SOTA choice
mid-2026** before downloading — model availability and recommended ControlNet checkpoints
move quickly. The SDXL-depth ControlNet vs a unified "ControlNet-Union" SDXL checkpoint
should be benchmarked; if a maintained union model is available it simplifies swapping
structure signals. Version-pin every id with `revision=` once chosen.

### 1.2 SD 1.5 fallback stack (smaller / faster, lower fidelity)

For fast iteration and as a low-memory fallback: `runwayml/stable-diffusion-v1-5`
(or `stable-diffusion-v1-5/stable-diffusion-v1-5`) + `lllyasviel/control_v11f1p_sd15_depth`
(or `_lineart`/`_softedge`) + `h94/IP-Adapter` `models/ip-adapter_sd15.bin`. Same algorithm,
fewer parameters. The engine is written backbone-agnostic (§3) so SD1.5 ↔ SDXL is a config
swap, not a code fork.

### 1.3 Unified-memory footprint on an Apple Silicon Mac

Rough resident footprints (fp16 weights; activations dominate at high resolution):

| Component | fp16 weights (approx) |
|---|---|
| SDXL UNet | ~5.0 GB |
| SDXL VAE | ~0.3 GB |
| 2× text encoders (CLIP-L + OpenCLIP-bigG) | ~1.8 GB |
| SDXL depth ControlNet | ~2.5 GB |
| IP-Adapter image encoder (bigG) + adapter | ~2.5 GB (≈1.3 GB with ViT-H variant) |
| RAFT-large (Phase 1, resident concurrently) | ~0.1 GB |
| **Total resident** | **~12–13 GB weights** |

Plus per-step activations (SDXL at 1024² with a ControlNet roughly 6–10 GB peak transient).
**Conclusion:** Apple Silicon uses unified memory, so the GPU shares your Mac's RAM and
there is no separate VRAM budget — the practical cap is your Mac's RAM minus what the OS and
apps use, not a fixed device limit. On a Mac with enough RAM this holds the full SDXL +
ControlNet + IP-Adapter + RAFT stack resident simultaneously with headroom and no CPU offload
(good, because MPS↔CPU offload is slow); on a lower-RAM Mac, keep fp16, enable attention
slicing and VAE tiling, and lower the resolution. More RAM = higher resolution and more
headroom. `TODO(tuning)`: confirm real peak with MPS memory counters at the target
resolution; decide whether `enable_attention_slicing` / `enable_vae_tiling` is worth enabling
(they trade ~speed for lower peak; helpful on tighter RAM or at 1024²+ batched frames).

---

## 2. The latent temporal-consistency algorithm

This is the heart of Phase 2 and the part that reuses Phase 1. We move the 2016 idea —
"warp the previous stylized frame into the current frame via optical flow, trust it only
where the forward-backward consistency check says it's reliable, and pull the current result
toward it" — from pixel space into the **VAE latent grid**, following the now-established
Rerender-A-Video / LatentWarp / MGLD line of work (see References).

### 2.1 Why latent space (and the key gotchas)

- The diffusion process operates on latents `z` of shape `(1, C, h, w)` where for SDXL
  `C=4` and `h = H/8, w = W/8` (VAE downsample factor `f=8`). We warp **`z`**, not pixels,
  so the warp directly constrains the denoising trajectory.
- **Flow must be rescaled to latent resolution.** RAFT flow is in *pixel* displacement at
  image resolution `(H, W)`. To warp a latent at `(h, w) = (H/f, W/f)` we both (a) bilinearly
  resize the flow field to `(h, w)` **and** (b) divide the displacement magnitudes by `f`
  (a 16 px motion is a 2-latent-cell motion at `f=8`). This mirrors exactly what
  `raft._postprocess_flow` does when it rescales vectors after a resolution change — we apply
  the same `scale = target/source` logic, here `1/f`.
- **`warp.flow_to_grid` is resolution-agnostic** (it normalizes to `[-1,1]` over whatever
  H×W the flow has) and uses the same `align_corners=True` convention as `warp_image`. So we
  build the latent sampling grid by calling `flow_to_grid(flow_latent)` and `grid_sample`
  the latent — reusing Phase 1's grid math instead of re-deriving it.
- **Reliability mask must be downsampled to the latent grid**, conservatively. We compute
  `consistency_mask` at full resolution (most accurate; reuses the validated Phase 1 code)
  then area-downsample to `(h, w)` and **threshold/erode** so that a latent cell counts as
  reliable only if (nearly) all the pixels feeding it are reliable. We also AND-in the
  `valid` mask returned by the latent warp (out-of-border latent cells).

### 2.2 Two complementary mechanisms (both implemented; selectable)

1. **Warped-latent initialization** (`init` mechanism, cheap, always on for `t>0`):
   Initialize the current frame's denoising from the **warped previous latent** plus the
   correctly-scaled noise for the chosen start timestep, instead of from pure noise. This is
   the latent-space generalization of Phase 1's `init = prevWarped`. Strong stabilizer, near
   free.

2. **Per-step warped-latent guidance / fusion** (`fuse` mechanism, the real temporal
   constraint): at selected denoising steps, blend the running latent toward the warped
   previous latent in reliable regions. This is the latent analogue of the 2016 temporal
   *loss*, applied as a hard blend (Rerender-style "pixel-aware"/temporal fusion) rather than
   a gradient term, which is more stable and avoids backprop through the UNet.

Both consume the **same** warped latent and **same** reliability mask; they differ only in
*when/how* they apply it.

### 2.3 `warp_latent` (new, in `artvid/diffusion/latent_warp.py`)

```text
warp_latent(z_prev, backward_flow_px, *, vae_factor=8, align_corners=True) -> WarpResult-like
  # z_prev:           (1, C, h, w)  previous frame's latent (in the CURRENT frame's denoise loop)
  # backward_flow_px: (2, H, W)     RAFT backward flow (current -> previous), pixel units
  # 1. resize flow to latent grid and rescale magnitude:
  #      flow_lat = F.interpolate(flow[None], size=(h, w), mode="bilinear",
  #                               align_corners=False)[0] / vae_factor
  #      (channel 0 /=1, channel 1 /=1 after the /vae_factor; both axes share f)
  #      NOTE: if H/f != h exactly (odd sizes) use the per-axis scale h/H, w/W like
  #            raft._postprocess_flow, not a single 1/f, to stay exact.  TODO(tuning)
  # 2. grid = artvid.flow.warp.flow_to_grid(flow_lat[None], align_corners=align_corners)
  # 3. warped = F.grid_sample(z_prev, grid, mode="bilinear",
  #                           padding_mode="zeros", align_corners=align_corners)
  # 4. valid  = (grid_sample(ones_like, grid) >= 1 - 1e-6)   # out-of-border latent cells
  # return (warped, valid)            # do NOT VGG-mean-fill latents; keep zeros, mask handles it
```

We deliberately **do not** reuse `warp_image` directly for latents because its default
`fill=VGG_MEAN_PIXEL_RGB_01` is a pixel-space RGB constant meaningless in latent space. We
*do* reuse `flow_to_grid` and replicate the ones-mask `valid` logic (identical to
`warp_image` lines that build `mask`/`valid`). `warp_image` is still used unchanged for
optional **pixel-space** debug previews and for the "warp the previous *decoded* frame"
fidelity check.

### 2.4 `latent_reliability` (new, in `latent_warp.py`)

```text
latent_reliability(forward_flow_px, backward_flow_px, valid_latent, *,
                   smooth_sigma=0.8, latent_hw, erode=True) -> (1,1,h,w) float [0,1]
  # 1. rel_px = artvid.flow.consistency.consistency_mask(backward_flow_px, forward_flow_px,
  #                                                      smooth_sigma=smooth_sigma)
  #    (validates the BACKWARD warp — same arg order as cli.cmd_flow rel_back)
  # 2. downsample to latent grid CONSERVATIVELY:
  #      rel_lat = F.avg_pool2d(rel_px[None,None], kernel=f, stride=f)   # mean reliability
  #      or F.adaptive_avg_pool2d(...) to (h,w) for non-divisible sizes
  # 3. if erode: treat a cell reliable only if mean >= thresh (default 0.9) -> soft via
  #      rel_lat = (rel_lat.clamp(0,1)) ; optionally rel_lat = rel_lat ** gamma  (gamma>1 erodes)
  # 4. rel_lat = rel_lat * valid_latent.float()
  # return rel_lat
```

`TODO(tuning)`: the **downsample-then-threshold** is the single most important knob.
Too lenient → flicker bleeds through occlusion edges; too strict → temporal constraint
vanishes and you get the per-frame-independent flicker we're trying to kill. Verify on a
clip with fast motion / disocclusion. Candidates: mean+`gamma` erosion (above), min-pool
(hardest), or a learned threshold. Default `gamma=2.0`, `thresh≈0.9`.

### 2.5 Per-frame denoising loop (pseudocode — the spec)

For an image-diffusion backbone with a scheduler exposing `timesteps`, `add_noise`,
`scale_model_input`, and `step`. `K` = denoising steps (e.g. 30 DDIM/Euler). We implement
this as a **custom callback-style loop** around the ControlNet+IP-Adapter UNet rather than
calling the high-level `pipe(...)` once, because we must inject the warped latent *between*
steps. (diffusers `callback_on_step_end` can mutate `latents` between steps and is the
lighter-weight alternative — see §3 note.)

```text
# --- one-time setup ---
pipe = build_pipeline(cfg)                 # SDXL + depth ControlNet + IP-Adapter, fp16, on device
style_embeds = pipe.prepare ip_adapter(style_image)   # IP-Adapter style ref, encoded ONCE
prev_latent = None                          # z of previous stylized frame (denoised, pre-decode)
prev_frame_rgb = None                       # decoded previous stylized frame (for flow if needed)

for t_idx, frame in enumerate(frames):      # frames: content RGB [0,1] CHW
    control = preprocess_structure(frame)   # depth/lineart map via controlnet_aux  -> ControlNet cond
    z0_noise = randn_latent(seed_for(t_idx))                 # base noise (per-frame seed; see §5)
    scheduler.set_timesteps(K)

    if prev_latent is None:
        # ---- anchor / first frame: plain ControlNet+IP-Adapter generation ----
        latents = z0_noise * sigma_init
        warped = rel = None
    else:
        # ---- optical flow between CONTENT frames (NOT stylized) ----
        fp = raft.compute_flow_pair(frame_rgb, prev_frame_content_rgb, device=dev)
        #     forward  = cur_content -> prev_content
        #     backward = prev_content -> cur_content
        # We need the BACKWARD warp (pull prev INTO cur): that is the flow
        # cur->prev, i.e. fp.forward here (current->previous). Name carefully:
        bwd = fp.forward     # current -> previous  == "backward warp" flow for warp_image
        fwd = fp.backward    # previous -> current  (cross-check)
        warped = warp_latent(prev_latent, bwd, vae_factor=f).image          # (1,C,h,w)
        valid  = warp_latent(...).valid
        rel    = latent_reliability(fwd, bwd, valid, latent_hw=(h,w))       # (1,1,h,w)

        # ---- MECHANISM 1: warped-latent init ----
        # Start denoising from the warped previous latent renoised to the start sigma,
        # blended with fresh noise init in UNRELIABLE regions.
        t_start = scheduler.timesteps[0]
        renoised = scheduler.add_noise(warped, z0_noise, t_start)
        latents  = rel * renoised + (1 - rel) * (z0_noise * sigma_init)

    for i, t in enumerate(scheduler.timesteps):
        model_in = scheduler.scale_model_input(latents, t)
        # ControlNet residuals from the structure map:
        down, mid = controlnet(model_in, t, encoder_hidden_states=text_embeds,
                               controlnet_cond=control, conditioning_scale=cfg.controlnet_scale)
        # UNet with IP-Adapter style embeds injected (added_cond_kwargs / ip_adapter scale):
        noise_pred = unet(model_in, t, encoder_hidden_states=text_embeds,
                          down_block_additional_residuals=down,
                          mid_block_additional_residual=mid,
                          added_cond_kwargs={**sdxl_time_ids, "image_embeds": style_embeds})
        noise_pred = cfg_guidance(noise_pred, cfg.guidance_scale)        # classifier-free guidance
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # ---- MECHANISM 2: per-step warped-latent fusion (temporal constraint) ----
        if warped is not None and (i in fuse_steps):     # fuse_steps e.g. early+mid steps only
            # Re-noise the warped target to the CURRENT noise level so it lives on the
            # same manifold as `latents` at timestep t, then blend in reliable regions.
            warped_t = scheduler.add_noise(warped, z0_noise, t)
            blend = cfg.temporal_strength * rel          # (1,1,h,w) in [0,1]
            latents = (1 - blend) * latents + blend * warped_t

    prev_latent = latents                       # carry the (denoised) latent forward
    out_rgb = vae_decode(latents)               # (3,H,W) RGB [0,1]
    save_image(out_rgb, output_path(t_idx))
    prev_frame_content_rgb = frame_rgb          # content frame drives next flow
```

Key design decisions baked into the loop:
- **Flow is computed on the *content* frames**, not the stylized outputs. The content video
  carries the true motion; stylization changes appearance, not geometry. (Rerender uses the
  original video's flow for the same reason.) This is also why we keep `prev_frame_content_rgb`.
- **Warp target is the previous *latent*** `prev_latent` (post-denoise, pre-VAE-decode). An
  alternative — warp the previous *decoded* frame, re-encode with the VAE — is provided as a
  `warp_space="pixel"` option for the fidelity-oriented variant (§2.6) but defaults to
  `warp_space="latent"` (cheaper, no extra VAE round-trip).
- **`fuse_steps` restricts fusion to early/mid steps** (e.g. first 60–70% of steps). Fusing
  at the very last steps over-smooths and can reintroduce the warped frame's VAE artifacts;
  fusing only early lets the UNet still synthesize fresh high-frequency detail in the current
  frame. `TODO(tuning)`: sweep the `fuse_steps` window and `temporal_strength`.

### 2.6 Long-term / anchor consistency (optional, reuses `combine_longterm_weights`)

To fight long-range drift (not just frame-to-frame), keep an **anchor latent** (frame 0 or
keyframe) and warp *both* `prev_latent` and the anchor into the current frame, then combine
their reliability masks with `consistency.combine_longterm_weights([rel_prev, rel_anchor],
method="closestFirst")` — exactly the Phase 1 long-term-weight scheme, lifted to latent
space. Default off (`anchor=False`); enable for long clips. `TODO(tuning)`: whether anchor
warp over many frames is reliable enough or whether keyframe re-anchoring (Rerender-style
keyframe + interpolation) is needed.

---

## 3. Pipeline architecture & module layout (`artvid/diffusion/`)

New package, **framework-touching but self-contained**; nothing here is imported by the
Phase 1 optim path. Lazy-imports `torch`/`diffusers` inside functions so `--help` and the
config/cli stay torch-free (matching the Phase 1 pattern in `cli.py`).

```
artvid/diffusion/
  __init__.py          # re-exports DiffusionEngine, stylize_video_diffusion; lazy
  engine.py            # DiffusionEngine: owns pipeline construction + single-frame stylize
  latent_warp.py       # warp_latent(), latent_reliability() — the §2.3/§2.4 reuse layer
  video.py             # stylize_video_diffusion(): the §2.5 per-frame loop over a sequence
  preprocess.py        # structure-signal extractors (depth/lineart/canny via controlnet_aux)
```

### 3.1 `engine.py` — `DiffusionEngine`

- `DiffusionEngine.from_config(cfg) -> DiffusionEngine`
  - `device.enable_mps_fallback(); dev = device.get_device(cfg.device)`
  - builds `StableDiffusionXLControlNetPipeline.from_pretrained(cfg.diff_base_model,
    controlnet=ControlNetModel.from_pretrained(cfg.controlnet_model), torch_dtype=fp16)`
    `.to(dev)`; selects scheduler (`cfg.diff_scheduler`, default Euler/DDIM);
    `pipe.load_ip_adapter(cfg.ip_adapter_repo, subfolder=..., weight_name=...)` and
    `pipe.set_ip_adapter_scale(cfg.ip_adapter_scale)`.
  - holds the scheduler, unet, controlnet, vae, text encoders, image encoder.
- `encode_style(style_image_path) -> style_embeds` (IP-Adapter image embeds, computed once).
- `prepare_text_embeds(prompt, negative_prompt)` — SDXL dual-encoder + pooled + time ids.
- `denoise_frame(content_rgb, *, warped_latent=None, reliability=None, seed=None) -> z`
  — the inner per-frame loop body of §2.5 (one frame's K-step denoise with optional
  init+fuse). Returns the final latent.
- `decode(z) -> (3,H,W) RGB[0,1]` (VAE decode + `[-1,1]→[0,1]`); `encode(rgb) -> z` for the
  pixel-warp variant.
- Pure-torch; **no Phase 1 imports except `artvid.device` and `artvid.io.image`**.

`TODO(tuning)`: exact diffusers class for the SDXL ControlNet+IP-Adapter combo and whether
the high-level `pipe(..., callback_on_step_end=...)` (mutating `latents`) is sufficient vs a
hand-rolled loop. Prefer the callback path if it cleanly exposes mid-loop `latents`; fall
back to the manual loop in §2.5 otherwise. Pin the diffusers version in `pyproject`.

### 3.2 `latent_warp.py`

Implements §2.3 `warp_latent` and §2.4 `latent_reliability`. **This is the file that imports
the Phase 1 flow stack** (`artvid.flow.warp.flow_to_grid`, `artvid.flow.consistency.
consistency_mask`, optionally `combine_longterm_weights`). Framework-agnostic torch ops only
(no diffusers) → unit-testable with synthetic latents/flows on CPU once torch is present, and
`py_compile`-able now.

### 3.3 `video.py`

`stylize_video_diffusion(cfg, *, flow_source="raft") -> list[FrameResult]` — the §2.5 loop:
iterate the content frames (`cli`/`Config` patterns, reusing `discover_frame_count` /
`format_frame_path` from `cli`), call `raft.compute_flow_pair` per adjacent pair (or load
precomputed `.flo` via `io.flow_io` when `flow_source="precomputed"`, reusing the same
`Config.flow_pattern` plumbing the optim path uses), drive `DiffusionEngine.denoise_frame`,
decode, and `io.image.save_image`. Output naming mirrors the single-pass engine so
`cli.cmd_run`'s `encode_pattern` re-encode step works **unchanged**.

### 3.4 `preprocess.py`

`structure_map(rgb, kind) -> control_rgb` for `kind in {"depth","canny","lineart","hed","tile"}`
via `controlnet_aux`. Cached per frame. `TODO(tuning)`: which signal best preserves content
while allowing restyling — start with `depth`.

---

## 4. CLI / Config integration (owned by other agents — spec only)

This doc does **not** edit `config.py` or `cli.py`. It specifies the contract their owning
agents implement.

### 4.1 `Config` additions (new "diffusion" field group)

Add a clearly-grouped block (all with sane defaults; torch-free):

```python
# --- Diffusion engine (Phase 2) ---
diff_base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet_model: str = "diffusers/controlnet-depth-sdxl-1.0"
controlnet_kind: str = "depth"          # depth|canny|lineart|hed|tile
controlnet_scale: float = 0.7           # TODO(tuning)
ip_adapter_repo: str = "h94/IP-Adapter"
ip_adapter_subfolder: str = "sdxl_models"
ip_adapter_weight: str = "ip-adapter_sdxl.bin"
ip_adapter_scale: float = 0.7           # style strength; TODO(tuning)
diff_prompt: str = ""                   # optional text prompt (style/content hints)
diff_negative_prompt: str = ""
diff_steps: int = 30                    # K
guidance_scale: float = 6.0             # TODO(tuning)
diff_scheduler: str = "euler"           # euler|ddim|dpm
# temporal (latent flow consistency)
temporal_strength: float = 0.6          # per-step fusion blend cap; TODO(tuning)
temporal_fuse_start: float = 0.0        # fraction of steps where fusion begins
temporal_fuse_end: float = 0.7          # ...and ends (fuse early/mid only); TODO(tuning)
latent_reliability_gamma: float = 2.0   # erosion of downsampled reliability; TODO(tuning)
warp_space: str = "latent"              # latent|pixel
use_anchor: bool = False                # long-term anchor warp (combine_longterm_weights)
vae_factor: int = 8                     # SDXL/SD1.5 VAE downsample (don't change unless model differs)
```

These reuse the existing `device`, `start_number`, `num_images`, `content_pattern`,
`style_image`, `output_image`, `output_folder`, `number_format`, and the `flow_*` patterns
unchanged. The flow-relative-indices / reliability plumbing is shared with the optim engine.

### 4.2 `cli` dispatch

`--engine diffusion` currently routes through `_engine_guard` → `NotImplementedError`.
The cli agent replaces that guard for `stylize`/`run` so that when `engine == "diffusion"`:
- `cmd_stylize` dispatches to `artvid.diffusion.video.stylize_video_diffusion(config,
  flow_source=...)` instead of the optim single/multi-pass pipelines;
- `cmd_run` keeps steps 1 (extract), 2 (flow precompute — **fully reused**, the diffusion
  engine consumes the same `.flo`/reliability files), 4 (re-encode) and only swaps the
  step-3 stylize call to the diffusion path.

No new subcommand. The diffusion-specific flags above are added to `_add_stylize_flags`
(all `default=None`, applied only when set — same pattern as every existing flag) and mapped
in `_STYLIZE_CONFIG_MAP`. `--style` (positional) is reused as the **IP-Adapter style
reference**; `--diff-prompt` supplies optional text.

Because flow precompute, frame discovery, output naming, and re-encode are all shared, the
diffusion engine slots into the *existing* `run` orchestration with a single dispatch branch.

---

## 5. Open questions / what MUST be tuned on hardware

Everything here needs your Apple Silicon Mac (any M-series with MPS); defaults above are starting points.

1. **`warp_latent` exactness at the latent grid.** Verify the flow rescale (`/vae_factor` vs
   per-axis `h/H, w/W`) and `align_corners` give a *pixel-accurate* latent warp on a synthetic
   translation (warp a known latent by a known integer-pixel shift, compare to a manual roll).
   Reuse the Phase 1 warp unit-test methodology. **Highest-risk correctness item.**
2. **Reliability downsampling / erosion** (§2.4 `TODO`): mean+gamma vs min-pool vs threshold.
   Objectively score with the Phase 1 stability metric (warp-residual after stylization) on a
   fixed clip — reuse Phase 1's evaluation harness so optim vs diffusion are comparable.
3. **`temporal_strength` × `fuse_steps` window.** Too much/too late → over-smoothed, washed
   detail; too little/too early → flicker. Sweep on a fast-motion clip and a slow-pan clip.
4. **Latent vs pixel warp** (`warp_space`): does warping the latent directly beat the
   VAE-decode→warp→re-encode round-trip? Latent is cheaper; confirm quality is comparable.
5. **Seed strategy per frame.** Fixed seed across frames (more coherent base noise, less
   flicker but risk of "frozen texture") vs flow-warped noise (integral noise warping) vs
   per-frame random (relies entirely on fusion). Default: fixed seed + warped-init; test
   noise warping if flicker persists. `TODO(tuning)`.
6. **ControlNet structure signal** (depth vs lineart vs canny vs tile) and `controlnet_scale`
   — the content/restyle tradeoff. Start `depth @ 0.7`.
7. **IP-Adapter scale** (style strength) and ViT-bigG vs ViT-H encoder (memory vs fidelity).
8. **MPS op coverage & speed.** Confirm SDXL + ControlNet + IP-Adapter run on MPS without
   falling to CPU on hot ops (we set `PYTORCH_ENABLE_MPS_FALLBACK`); measure sec/frame and
   peak memory at 768²/1024². Decide attention-slicing/VAE-tiling. Choose the fastest
   scheduler that keeps quality (Euler/DPM).
9. **Model id/revision confirmation** (§1.1 `TODO`): confirm the pinned HF ids are the
   mid-2026 SOTA + available + license-clean before any download; pin `revision=`.
10. **Anchor / long-term** (§2.6): is single-prev-frame warp enough, or is keyframe
    re-anchoring needed for >a few-hundred-frame clips?

---

## 6. Milestones (concrete, supersedes the §5 draft in `04-phase2-plan.md`)

- **P2-M0** — `DiffusionEngine` single-image stylize (SDXL + depth ControlNet + IP-Adapter)
  runs on MPS; `decode`/`encode` round-trip verified. No temporal code.
- **P2-M1** — `latent_warp.warp_latent` + `latent_reliability` implemented and unit-tested
  (synthetic latent + known flow); pixel-accurate warp confirmed.
- **P2-M2** — `stylize_video_diffusion` per-frame loop (init + fuse) produces a temporally
  stable short clip; tune items 1–5 above.
- **P2-M3** — wire `--engine diffusion` into `cli` (stylize + run); shared flow precompute &
  re-encode; benchmark sec/frame & stability vs the optim engine; document.
- **P2-M4 (deferred)** — comparison against a native video-diffusion / cross-frame-attention
  backbone (AnimateDiff / SVD); pick the most stable.

---

## 7. Licensing & redistribution notes

- **SDXL base / SDXL ControlNets:** CreativeML Open RAIL++-M — open weights, **use-based
  restrictions** (no listed harmful uses), redistribution allowed if the license + use
  restrictions travel with the weights. We **do not** vendor weights; they download from HF
  at first run (like RAFT in Phase 1). Document the license and the first-run network
  requirement in `docs/usage.md`.
- **IP-Adapter weights / `controlnet_aux`:** Apache-2.0 — permissive.
- SD1.5 fallback: CreativeML Open RAIL-M (same family of restrictions).
- Action item for impl: surface a one-line license/acknowledgement banner on first
  diffusion run and list model ids + licenses in the docs. `TODO`: re-verify each model's
  current license at implementation time (licenses occasionally change across revisions).

---

## References (consulted mid-2026; confirm versions at implementation)

- Ruder, Dosovitskiy, Brox, *Artistic Style Transfer for Videos* (2016) — the optical-flow
  temporal-consistency idea Phase 1 ports and Phase 2 lifts to latent space.
- Yang et al., *Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation* (2023) —
  optical-flow latent warp of the previous keyframe, occlusion-masked fusion, ControlNet
  structure guidance. Primary template for §2. https://arxiv.org/abs/2306.07954
- *LatentWarp: Consistent Diffusion Latents for Zero-Shot Video-to-Video Translation* (2023)
  — latent-space warping constraint during denoising. https://arxiv.org/abs/2311.00353
- *Motion-Guided Latent Diffusion (MGLD)* (ECCV 2024) — flow-warped latent alignment with a
  warp-residual loss.
- Xie et al., *Synchronized Multi-Frame Diffusion for Temporally Consistent Video
  Stylization* (CGF 2025) — flow-as-correspondence latent sharing across frames; an
  alternative to per-frame warp considered for P2-M4. https://doi.org/10.1111/cgf.70095
- diffusers IP-Adapter + ControlNet usage (`load_ip_adapter`, `set_ip_adapter_scale`,
  `StableDiffusionXLControlNetPipeline`); IP-Adapter weights `h94/IP-Adapter`.
- SDXL on Apple Silicon / MPS memory & `PYTORCH_ENABLE_MPS_FALLBACK` guidance.
