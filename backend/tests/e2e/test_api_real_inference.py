import os
from pathlib import Path

from fastapi.testclient import TestClient

import app.main as main

# Force offline usage of already cached model artifacts during this real inference test.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

client = TestClient(main.app)
RESULT_DIR = Path("app/static/results")


def _result_files() -> set[str]:
    return {p.name for p in RESULT_DIR.glob("*.png")}


def test_predict_by_id_real_inference_and_artifact_save():
    images_res = client.get("/test-images")
    assert images_res.status_code == 200
    images = images_res.json().get("images", [])
    assert len(images) >= 1

    image_id = images[0]["id"]
    before_files = _result_files()

    pred_res = client.post("/predict-by-id", json={"image_id": image_id})
    assert pred_res.status_code == 200
    body = pred_res.json()

    assert body["inference_ms"] > 0
    assert len(body["labels"]) >= 1
    assert len(body["top_classes"]) >= 1
    assert body["original_url"].startswith("/static/results/")
    assert body["overlay_url"].startswith("/static/results/")
    assert len(body["class_masks"]) >= 1

    created_names: list[str] = []
    created_names.append(body["original_url"].split("/")[-1])
    created_names.append(body["overlay_url"].split("/")[-1])
    created_names.extend(mask["mask_url"].split("/")[-1] for mask in body["class_masks"])

    for name in created_names:
        assert (RESULT_DIR / name).exists(), f"Expected saved artifact missing: {name}"

    after_files = _result_files()
    assert len(after_files - before_files) >= 2
