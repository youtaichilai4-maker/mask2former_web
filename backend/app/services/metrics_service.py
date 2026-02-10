from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class MetricsService:
    @staticmethod
    def semantic_probabilities(outputs: Any, target_size: tuple[int, int]) -> torch.Tensor:
        class_logits = outputs.class_queries_logits[0]
        mask_logits = outputs.masks_queries_logits[0]

        class_probs = torch.softmax(class_logits, dim=-1)[:, :-1]
        mask_probs = torch.sigmoid(mask_logits)

        semantic_logits = torch.einsum("qc,qhw->chw", class_probs, mask_probs)
        semantic_logits = F.interpolate(
            semantic_logits.unsqueeze(0),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return torch.softmax(semantic_logits, dim=0)

    @staticmethod
    def class_stats(
        seg: np.ndarray,
        semantic_probs: torch.Tensor,
        id2label: dict[int, str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        total_pixels = seg.size
        top_classes: list[dict[str, Any]] = []
        area_stats: list[dict[str, Any]] = []

        for class_id in sorted(np.unique(seg).tolist()):
            mask = seg == class_id
            if not mask.any():
                continue

            area_ratio = (mask.sum() / total_pixels) * 100.0
            confidence = float(semantic_probs[class_id][torch.from_numpy(mask)].mean().item())
            label = id2label.get(int(class_id), str(class_id))

            top_classes.append(
                {
                    "class_id": int(class_id),
                    "label": label,
                    "confidence": round(confidence, 4),
                }
            )
            area_stats.append(
                {
                    "class_id": int(class_id),
                    "label": label,
                    "area_ratio": round(float(area_ratio), 2),
                }
            )

        top_classes.sort(key=lambda x: x["confidence"], reverse=True)
        area_stats.sort(key=lambda x: x["area_ratio"], reverse=True)
        return top_classes[:8], area_stats[:12]
