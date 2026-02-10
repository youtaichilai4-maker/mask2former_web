import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.core.config import ADE20K_MODEL_KEY, RESULT_DIR, STATIC_DIR, TEST_IMAGE_DIR
from app.models.registry import ModelRegistry
from app.schemas import DescribeRequest, PredictByIdRequest
from app.services.description_service import DescriptionService
from app.services.inference_service import InferenceService
from app.services.metrics_service import MetricsService
from app.services.image_catalog_service import ImageCatalogService
from app.services.visualization_service import VisualizationService

model_registry = ModelRegistry()
visualization_service = VisualizationService(RESULT_DIR)
metrics_service = MetricsService()
inference_service = InferenceService(model_registry, visualization_service, metrics_service)
description_service = DescriptionService()
test_image_service = ImageCatalogService(TEST_IMAGE_DIR)


def create_app() -> FastAPI:
    app = FastAPI(title="Mask2Former ADE20K Demo", version="0.3.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/models")
    def models() -> dict[str, list[dict[str, str]]]:
        return {"models": model_registry.list_models()}

    @app.get("/test-images")
    def test_images() -> dict[str, list[dict[str, str | list[str]]]]:
        return {"images": test_image_service.list_images()}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), model_key: str = Form(ADE20K_MODEL_KEY)) -> dict:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Upload an image file")

        raw = await file.read()
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid image") from exc

        return inference_service.run_prediction(image, model_key=model_key)

    @app.post("/predict-by-id")
    def predict_by_id(req: PredictByIdRequest) -> dict:
        image_path = test_image_service.resolve(req.image_id)
        if not image_path.exists() or not image_path.is_file():
            raise HTTPException(status_code=404, detail=f"Unknown image_id: {req.image_id}")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid stored image") from exc

        return inference_service.run_prediction(image, model_key=ADE20K_MODEL_KEY)

    @app.post("/describe")
    def describe(req: DescribeRequest) -> dict:
        return description_service.describe(req)

    @app.get("/")
    def root() -> dict[str, str]:
        return {"message": "Mask2Former ADE20K API", "docs": "/docs"}

    return app


app = create_app()
