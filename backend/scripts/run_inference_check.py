#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import ADE20K_MODEL_KEY, RESULT_DIR, TEST_IMAGE_DIR
from app.models.registry import ModelRegistry
from app.services.dataset_inference_check_service import DatasetInferenceCheckService, format_cli_report
from app.services.image_catalog_service import ImageCatalogService
from app.services.inference_service import InferenceService
from app.services.metrics_service import MetricsService
from app.services.visualization_service import VisualizationService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference check over local test images and print CLI report")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of images to run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_registry = ModelRegistry()
    visualization_service = VisualizationService(RESULT_DIR)
    metrics_service = MetricsService()
    inference_service = InferenceService(model_registry, visualization_service, metrics_service)
    image_catalog_service = ImageCatalogService(TEST_IMAGE_DIR)

    runner = DatasetInferenceCheckService(
        image_catalog_service=image_catalog_service,
        inference_service=inference_service,
        model_key=ADE20K_MODEL_KEY,
    )
    summary = runner.run(limit=args.limit)
    print(format_cli_report(summary))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
