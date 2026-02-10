import numpy as np
import torch

from app.services.metrics_service import MetricsService


class FakeOutputs:
    def __init__(self):
        # Q=2, classes=3(+no-object)
        self.class_queries_logits = torch.tensor([[[2.0, 1.0, 0.5, -1.0], [1.2, 2.1, 0.7, -0.5]]])
        self.masks_queries_logits = torch.tensor(
            [[[[2.0, 1.0], [0.1, -0.2]], [[1.5, -0.1], [0.2, 0.3]]]]
        )


def test_semantic_probabilities_shape_and_normalization():
    service = MetricsService()
    probs = service.semantic_probabilities(FakeOutputs(), (2, 2))

    assert probs.shape == (3, 2, 2)
    per_pixel_sum = probs.sum(dim=0)
    assert torch.allclose(per_pixel_sum, torch.ones_like(per_pixel_sum), atol=1e-5)


def test_class_stats_returns_sorted_lists():
    service = MetricsService()
    seg = np.array([[0, 1], [1, 2]], dtype=np.int32)
    semantic_probs = torch.tensor(
        [
            [[0.9, 0.1], [0.2, 0.1]],
            [[0.05, 0.8], [0.7, 0.1]],
            [[0.05, 0.1], [0.1, 0.8]],
        ]
    )
    id2label = {0: "wall", 1: "floor", 2: "door"}

    top_classes, area_stats = service.class_stats(seg, semantic_probs, id2label)

    assert top_classes[0]["label"] in {"wall", "floor", "door"}
    assert area_stats[0]["label"] == "floor"
    assert area_stats[0]["area_ratio"] == 50.0
