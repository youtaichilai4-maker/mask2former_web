import io

from fastapi.testclient import TestClient
from PIL import Image

import app.main as main

client = TestClient(main.app)


def _png_bytes(width=2, height=2):
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fake_predict_payload():
    return {
        "model_key": "ade20k_official",
        "model_hf_id": "facebook/mask2former-swin-large-ade-semantic",
        "inference_ms": 12.3,
        "original_url": "/static/results/orig.png",
        "overlay_url": "/static/results/overlay.png",
        "labels": [
            {"class_id": 0, "label": "wall"},
            {"class_id": 1, "label": "floor"},
        ],
        "top_classes": [
            {"class_id": 0, "label": "wall", "confidence": 0.91},
            {"class_id": 1, "label": "floor", "confidence": 0.81},
        ],
        "area_stats": [
            {"class_id": 0, "label": "wall", "area_ratio": 35.0},
            {"class_id": 1, "label": "floor", "area_ratio": 27.0},
        ],
        "class_masks": [
            {"class_id": 0, "label": "wall", "mask_url": "/static/results/mask_0.png"},
        ],
        "width": 2,
        "height": 2,
    }


def test_health_ok():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_models_has_ade20k_official():
    res = client.get("/models")
    assert res.status_code == 200
    assert res.json()["models"][0]["model_key"] == "ade20k_official"


def test_predict_success_with_mocked_inference(monkeypatch):
    monkeypatch.setattr(main.inference_service, "run_prediction", lambda image, model_key: _fake_predict_payload())

    files = {"file": ("sample.png", _png_bytes(), "image/png")}
    data = {"model_key": "ade20k_official"}

    res = client.post("/predict", files=files, data=data)
    assert res.status_code == 200
    body = res.json()
    assert body["model_key"] == "ade20k_official"
    assert body["top_classes"][0]["label"] == "wall"


def test_predict_rejects_non_image_upload():
    files = {"file": ("sample.txt", b"not image", "text/plain")}
    res = client.post("/predict", files=files, data={"model_key": "ade20k_official"})
    assert res.status_code == 400


def test_test_images_endpoint(monkeypatch):
    monkeypatch.setattr(
        main.test_image_service,
        "list_images",
        lambda: [
            {
                "id": "demo.png",
                "name": "demo",
                "thumbnail_url": "/x",
                "image_url": "/x",
                "tags": ["ade20k-real", "local"],
            }
        ],
    )
    res = client.get("/test-images")
    assert res.status_code == 200
    assert res.json()["images"][0]["id"] == "demo.png"
    assert res.json()["images"][0]["tags"] == ["ade20k-real", "local"]


def test_predict_by_id_not_found(monkeypatch):
    class FakePath:
        def exists(self):
            return False

        def is_file(self):
            return False

    monkeypatch.setattr(main.test_image_service, "resolve", lambda image_id: FakePath())
    res = client.post("/predict-by-id", json={"image_id": "missing.png"})
    assert res.status_code == 404


def test_describe_success(monkeypatch):
    monkeypatch.setattr(
        main.description_service,
        "describe",
        lambda payload: {"summary_ja": "ok", "highlights": ["h1"], "cautions": ["c1"]},
    )
    req = {
        "top_classes": [{"class_id": 0, "label": "wall", "confidence": 0.9}],
        "area_stats": [{"class_id": 0, "label": "wall", "area_ratio": 34.0}],
        "inference_ms": 10.0,
    }
    res = client.post("/describe", json=req)
    assert res.status_code == 200
    assert res.json()["summary_ja"] == "ok"
