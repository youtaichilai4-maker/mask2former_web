import uuid
from pathlib import Path

import numpy as np
from PIL import Image


class VisualizationService:
    def __init__(self, result_dir: Path) -> None:
        self.result_dir = result_dir

    @staticmethod
    def color_for_class(class_id: int) -> tuple[int, int, int]:
        rng = np.random.default_rng(class_id + 17)
        color = rng.integers(40, 230, size=3, dtype=np.uint8)
        return int(color[0]), int(color[1]), int(color[2])

    def to_overlay(self, image: Image.Image, seg: np.ndarray) -> Image.Image:
        arr = np.array(image.convert("RGB"))
        color_mask = np.zeros_like(arr)

        for class_id in np.unique(seg):
            color_mask[seg == class_id] = np.array(self.color_for_class(int(class_id)), dtype=np.uint8)

        alpha = 0.45
        overlay = (arr * (1 - alpha) + color_mask * alpha).astype(np.uint8)
        return Image.fromarray(overlay)

    def save_image(self, img: Image.Image, prefix: str) -> str:
        name = f"{prefix}_{uuid.uuid4().hex[:10]}.png"
        path = self.result_dir / name
        img.save(path, format="PNG")
        return f"/static/results/{name}"

    def class_mask_urls(self, seg: np.ndarray, id2label: dict[int, str]) -> list[dict[str, str | int]]:
        urls: list[dict[str, str | int]] = []
        for class_id in sorted(np.unique(seg).tolist()):
            binary = (seg == class_id).astype(np.uint8) * 255
            img = Image.fromarray(binary, mode="L")
            url = self.save_image(img, f"mask_{class_id}")
            urls.append(
                {
                    "class_id": int(class_id),
                    "label": id2label.get(int(class_id), str(class_id)),
                    "mask_url": url,
                }
            )

        return urls
