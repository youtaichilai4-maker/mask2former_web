import time
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from app.models.registry import ModelRegistry
from app.services.metrics_service import MetricsService
from app.services.visualization_service import VisualizationService


class InferenceService:
    def __init__(
        self,
        model_registry: ModelRegistry,
        visualization_service: VisualizationService,
        metrics_service: MetricsService,
    ) -> None:
        self.model_registry = model_registry
        self.visualization_service = visualization_service
        self.metrics_service = metrics_service
        self._model_cache: dict[str, Any] = {}

    @staticmethod
    def _device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def load_model(
        self, model_key: str
    ) -> tuple[AutoImageProcessor, Mask2FormerForUniversalSegmentation, torch.device]:
        if model_key in self._model_cache:
            return self._model_cache[model_key]

        hf_id = self.model_registry.hf_id(model_key)
        processor = AutoImageProcessor.from_pretrained(hf_id)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(hf_id)
        device = self._device()
        model.to(device)
        model.eval()

        self._model_cache[model_key] = (processor, model, device)
        return processor, model, device

    def run_prediction(self, image: Image.Image, model_key: str) -> dict[str, Any]:
        processor, model, device = self.load_model(model_key)

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs)

        post = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])
        seg = post[0].detach().cpu().numpy().astype(np.int32)
        semantic_probs = self.metrics_service.semantic_probabilities(outputs, image.size[::-1]).detach().cpu()

        elapsed_ms = (time.perf_counter() - start) * 1000

        overlay = self.visualization_service.to_overlay(image, seg)
        original_url = self.visualization_service.save_image(image, "orig")
        overlay_url = self.visualization_service.save_image(overlay, "overlay")

        id2label = model.config.id2label or {}
        labels = [
            {"class_id": int(class_id), "label": id2label.get(int(class_id), str(class_id))}
            for class_id in sorted(np.unique(seg).tolist())
        ]
        top_classes, area_stats = self.metrics_service.class_stats(seg, semantic_probs, id2label)
        class_masks = self.visualization_service.class_mask_urls(seg, id2label)

        return {
            "model_key": model_key,
            "model_hf_id": self.model_registry.hf_id(model_key),
            "inference_ms": round(elapsed_ms, 2),
            "original_url": original_url,
            "overlay_url": overlay_url,
            "labels": labels,
            "top_classes": top_classes,
            "area_stats": area_stats,
            "class_masks": class_masks,
            "width": image.width,
            "height": image.height,
        }
