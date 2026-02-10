from pathlib import Path

import numpy as np
from PIL import Image

from app.services.visualization_service import VisualizationService


def test_to_overlay_keeps_size(tmp_path: Path):
    service = VisualizationService(tmp_path)
    image = Image.new("RGB", (4, 4), color=(128, 128, 128))
    seg = np.array(
        [
            [0, 0, 1, 1],
            [0, 2, 2, 1],
            [0, 2, 1, 1],
            [2, 2, 1, 0],
        ],
        dtype=np.int32,
    )

    overlay = service.to_overlay(image, seg)
    assert overlay.size == image.size


def test_save_image_and_class_masks(tmp_path: Path):
    service = VisualizationService(tmp_path)
    image = Image.new("RGB", (3, 3), color=(10, 20, 30))

    url = service.save_image(image, "orig")
    assert url.startswith("/static/results/orig_")
    assert len(list(tmp_path.glob("orig_*.png"))) == 1

    seg = np.array([[0, 1, 1], [2, 2, 1], [0, 2, 0]], dtype=np.int32)
    masks = service.class_mask_urls(seg, {0: "wall", 1: "floor", 2: "door"})

    assert len(masks) == 3
    assert {m["label"] for m in masks} == {"wall", "floor", "door"}
