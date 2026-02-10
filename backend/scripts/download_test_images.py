#!/usr/bin/env python3
"""Download ADE20K validation images in bulk for local test gallery."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk download ADE20K validation images")
    parser.add_argument("--count", type=int, default=100, help="Number of images to download")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("app/static/test_images"),
        help="Directory to store downloaded images",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep existing image files instead of cleaning output directory",
    )
    return parser.parse_args()


def clean_output_dir(output_dir: Path) -> int:
    removed = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            path.unlink()
            removed += 1
    return removed


def main() -> int:
    args = parse_args()

    if args.count <= 0:
        raise SystemExit("--count must be > 0")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.keep_existing:
        removed = clean_output_dir(output_dir)
        print(f"cleaned existing images: {removed}")

    print("loading dataset: scene_parse_150 (validation split, streaming)")
    dataset = load_dataset(
        "scene_parse_150",
        split="validation",
        trust_remote_code=True,
        streaming=True,
    )

    downloaded = 0
    for idx, record in enumerate(dataset):
        if idx >= args.count:
            break
        image = record["image"]
        out_name = f"ade20k_val_{idx + 1:04d}.jpg"
        image.save(output_dir / out_name, format="JPEG", quality=95)
        downloaded += 1

    print(f"downloaded images: {downloaded}")
    print(f"output dir: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
