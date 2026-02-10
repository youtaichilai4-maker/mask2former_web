from pathlib import Path

from app.services.image_catalog_service import ImageCatalogService


def test_list_images_filters_and_sorts(tmp_path: Path):
    (tmp_path / "z_last.webp").write_bytes(b"x")
    (tmp_path / "a_first.jpg").write_bytes(b"x")
    (tmp_path / "b_mid.png").write_bytes(b"x")
    (tmp_path / "ignore.txt").write_text("nope")

    service = ImageCatalogService(tmp_path)
    images = service.list_images()

    assert [img["id"] for img in images] == ["a_first.jpg", "b_mid.png", "z_last.webp"]
    assert all(img["tags"] == ["ade20k-real", "local"] for img in images)


def test_resolve_returns_target_path(tmp_path: Path):
    service = ImageCatalogService(tmp_path)
    resolved = service.resolve("abc.jpg")
    assert resolved == tmp_path / "abc.jpg"
