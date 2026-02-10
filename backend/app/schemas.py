from typing import Any

from pydantic import BaseModel


class PredictByIdRequest(BaseModel):
    image_id: str


class DescribeRequest(BaseModel):
    top_classes: list[dict[str, Any]]
    area_stats: list[dict[str, Any]]
    inference_ms: float | None = None
