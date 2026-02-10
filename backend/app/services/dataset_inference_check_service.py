from pathlib import Path
from typing import Any

from PIL import Image

from app.services.image_catalog_service import ImageCatalogService
from app.services.inference_service import InferenceService


class DatasetInferenceCheckService:
    def __init__(
        self,
        image_catalog_service: ImageCatalogService,
        inference_service: InferenceService,
        model_key: str,
    ) -> None:
        self.image_catalog_service = image_catalog_service
        self.inference_service = inference_service
        self.model_key = model_key

    def run(self, limit: int | None = None) -> dict[str, Any]:
        entries = self.image_catalog_service.list_images()
        if limit is not None:
            entries = entries[:limit]

        results: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []

        for entry in entries:
            image_id = str(entry["id"])
            image_path = self.image_catalog_service.resolve(image_id)
            try:
                with Image.open(image_path) as image:
                    prediction = self.inference_service.run_prediction(image.convert("RGB"), model_key=self.model_key)

                top_label = "-"
                if prediction.get("top_classes"):
                    top_label = str(prediction["top_classes"][0].get("label", "-"))

                results.append(
                    {
                        "image_id": image_id,
                        "inference_ms": float(prediction.get("inference_ms", 0.0)),
                        "top_label": top_label,
                        "num_labels": len(prediction.get("labels", [])),
                    }
                )
            except Exception as exc:
                failures.append({"image_id": image_id, "error": str(exc)})

        avg_ms = 0.0
        if results:
            avg_ms = sum(item["inference_ms"] for item in results) / len(results)

        return {
            "total": len(entries),
            "success": len(results),
            "failed": len(failures),
            "avg_inference_ms": round(avg_ms, 2),
            "results": results,
            "failures": failures,
        }


def format_cli_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=== Dataset Inference Check ===")
    lines.append(f"total={summary['total']} success={summary['success']} failed={summary['failed']}")
    lines.append(f"avg_inference_ms={summary['avg_inference_ms']}")
    lines.append("--- per image ---")

    for item in summary["results"]:
        lines.append(
            f"{item['image_id']} | ms={item['inference_ms']} | top_label={item['top_label']} | labels={item['num_labels']}"
        )

    if summary["failures"]:
        lines.append("--- failures ---")
        for item in summary["failures"]:
            lines.append(f"{item['image_id']} | error={item['error']}")

    return "\n".join(lines)
