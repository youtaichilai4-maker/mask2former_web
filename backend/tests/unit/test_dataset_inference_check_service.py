from pathlib import Path

from PIL import Image

from app.services.dataset_inference_check_service import DatasetInferenceCheckService, format_cli_report


class FakeImageCatalogService:
    def __init__(self, root: Path):
        self.root = root

    def list_images(self):
        return [
            {"id": "a.jpg"},
            {"id": "b.jpg"},
            {"id": "c.jpg"},
        ]

    def resolve(self, image_id: str) -> Path:
        return self.root / image_id


class FakeInferenceService:
    def __init__(self):
        self.calls = 0

    def run_prediction(self, image, model_key: str):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("boom")
        return {
            "inference_ms": 10.5 * self.calls,
            "top_classes": [{"label": "wall"}],
            "labels": [{"class_id": 0, "label": "wall"}],
        }


def test_run_collects_success_and_failures(tmp_path: Path):
    for name in ["a.jpg", "b.jpg", "c.jpg"]:
        Image.new("RGB", (2, 2), color=(128, 128, 128)).save(tmp_path / name, format="JPEG")

    runner = DatasetInferenceCheckService(
        image_catalog_service=FakeImageCatalogService(tmp_path),
        inference_service=FakeInferenceService(),
        model_key="ade20k_official",
    )

    summary = runner.run()
    assert summary["total"] == 3
    assert summary["success"] == 2
    assert summary["failed"] == 1
    assert summary["results"][0]["top_label"] == "wall"
    assert "boom" in summary["failures"][0]["error"]


def test_format_cli_report_contains_key_fields():
    report = format_cli_report(
        {
            "total": 2,
            "success": 2,
            "failed": 0,
            "avg_inference_ms": 12.3,
            "results": [
                {"image_id": "x.jpg", "inference_ms": 11.1, "top_label": "wall", "num_labels": 4},
                {"image_id": "y.jpg", "inference_ms": 13.5, "top_label": "floor", "num_labels": 5},
            ],
            "failures": [],
        }
    )
    assert "Dataset Inference Check" in report
    assert "x.jpg" in report
    assert "top_label=wall" in report
