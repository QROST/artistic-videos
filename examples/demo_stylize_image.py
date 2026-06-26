#!/usr/bin/env python3
"""Demo: single-image neural style transfer with artvid.

Runs :func:`artvid.pipeline.stylize_image.stylize_image` on an example content
frame and a style image, then writes the stylized result to disk. This is the
M0 acceptance demo from ``docs/03-phase1-plan.md`` ("a reasonable stylized still
from an example/ frame + a style image").

It is *not* executed in the build/scaffold environment (torch is unavailable
there); run it on your Apple Silicon Mac (any M-series with MPS), e.g.::

    python examples/demo_stylize_image.py \
        --content example/marple8_01.ppm \
        --style example/seated-nude.jpg \
        --output out/marple8_stylized.png \
        --num-iterations 500 \
        --device mps

By default it uses torchvision VGG-19 weights (RGB/ImageNet preprocessing); pass
``--vgg-weights /path/to/caffe_vgg19.pth`` for the (currently stubbed) caffe
parity path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from artvid.config import Config
from artvid.pipeline.stylize_image import stylize_image


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent.parent
    default_content = repo_root / "example" / "marple8_01.ppm"
    default_style = repo_root / "example" / "seated-nude.jpg"

    p = argparse.ArgumentParser(
        description="Single-image neural style transfer (artvid M0 demo).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--content",
        type=str,
        default=str(default_content),
        help="Path to the content image (an example/ frame).",
    )
    p.add_argument(
        "--style",
        type=str,
        nargs="+",
        default=[str(default_style)],
        help="One or more style image paths (multiple are blended).",
    )
    p.add_argument(
        "--style-blend-weights",
        type=str,
        default=None,
        help="Comma-separated blend weights for multiple style images "
        "(default: equal weighting).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="out_stylized.png",
        help="Where to write the stylized image.",
    )
    p.add_argument(
        "--num-iterations",
        type=int,
        default=1000,
        help="Number of optimizer iterations for this (first) frame.",
    )
    p.add_argument(
        "--init",
        type=str,
        choices=("random", "image"),
        default="random",
        help="Initialization for the optimized image.",
    )
    p.add_argument(
        "--optimizer",
        type=str,
        choices=("lbfgs", "adam"),
        default="lbfgs",
        help="Optimizer to use.",
    )
    p.add_argument("--content-weight", type=float, default=5e0)
    p.add_argument("--style-weight", type=float, default=1e2)
    p.add_argument("--tv-weight", type=float, default=1e-3)
    p.add_argument("--style-scale", type=float, default=1.0)
    p.add_argument(
        "--pooling", type=str, choices=("max", "avg"), default="max"
    )
    p.add_argument(
        "--vgg-weights",
        type=str,
        default="torchvision",
        help="'torchvision' (option A) or a path to caffe VGG-19 weights "
        "(option B).",
    )
    p.add_argument(
        "--device",
        type=str,
        choices=("mps", "cuda", "cpu"),
        default=None,
        help="Device to run on (default: autodetect mps > cuda > cpu).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed for the 'random' init (-1 = unseeded).",
    )
    p.add_argument(
        "--print-iter",
        type=int,
        default=100,
        help="Print losses every N iterations (0 disables).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    config = Config(
        content_pattern=args.content,
        style_image=",".join(args.style),
        style_blend_weights=args.style_blend_weights,
        # Single image: only the first-frame entry of each pair is used.
        num_iterations=(args.num_iterations, args.num_iterations),
        init=(args.init, args.init),
        optimizer=args.optimizer,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        style_scale=args.style_scale,
        pooling=args.pooling,
        vgg_weights=args.vgg_weights,
        device=args.device,
        seed=args.seed,
        print_iter=args.print_iter,
        output_image=args.output,
    )

    print(f"Content: {args.content}")
    print(f"Style:   {args.style}")
    print(f"Output:  {args.output}")

    _image_pre, result = stylize_image(
        content=args.content,
        style=args.style,
        config=config,
        output_path=args.output,
    )

    print(
        f"Done in {result.elapsed_seconds:.0f}s over {result.num_iterations} "
        f"iterations; final total loss = {result.last_losses.get('total', float('nan')):.4f}"
    )
    print(f"Wrote stylized image to {args.output}")


if __name__ == "__main__":
    main()
