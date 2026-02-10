from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = BASE_DIR.parent
load_dotenv(BACKEND_DIR / ".env")

STATIC_DIR = BASE_DIR / "static"
RESULT_DIR = STATIC_DIR / "results"
TEST_IMAGE_DIR = STATIC_DIR / "test_images"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
TEST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

ADE20K_MODEL_KEY = "ade20k_official"
MODELS: dict[str, dict[str, str]] = {
    ADE20K_MODEL_KEY: {
        "hf_id": "facebook/mask2former-swin-large-ade-semantic",
        "note": "Official Mask2Former checkpoint on ADE20K semantic segmentation.",
        "label_space": "ADE20K-150",
    }
}
