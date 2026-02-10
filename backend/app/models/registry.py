from fastapi import HTTPException

from app.core.config import MODELS


class ModelRegistry:
    def list_models(self) -> list[dict[str, str]]:
        return [
            {
                "model_key": model_key,
                "hf_id": meta["hf_id"],
                "label_space": meta["label_space"],
                "note": meta["note"],
            }
            for model_key, meta in MODELS.items()
        ]

    def hf_id(self, model_key: str) -> str:
        if model_key not in MODELS:
            raise HTTPException(status_code=400, detail=f"Unknown model_key: {model_key}")
        return MODELS[model_key]["hf_id"]
