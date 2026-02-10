from pathlib import Path


class ImageCatalogService:
    def __init__(self, test_image_dir: Path) -> None:
        self.test_image_dir = test_image_dir

    def list_images(self) -> list[dict[str, str | list[str]]]:
        images: list[dict[str, str | list[str]]] = []
        for path in sorted(self.test_image_dir.glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            images.append(
                {
                    "id": path.name,
                    "name": path.stem,
                    "thumbnail_url": f"/static/test_images/{path.name}",
                    "image_url": f"/static/test_images/{path.name}",
                    "tags": ["ade20k-real", "local"],
                }
            )
        return images

    def resolve(self, image_id: str) -> Path:
        return self.test_image_dir / image_id
