# Mask2Former Floorplan Web Demo

建設図面向けに、推論前後を比較可視化する `Next.js + FastAPI` デモです。

## What this proves

- Web実装力: 画像アップロード、推論API連携、可視化UI
- MLキャッチアップ力: Mask2Formerの学習済み重みを切り替えて挙動を比較
- 実務感: モデルの信頼度を「公式/コミュニティ」で明示

## Model status (important)

- `ade20k_official`
  - Hugging Face: `facebook/mask2former-swin-large-ade-semantic`
  - 公式のMask2Former公開重み（ADE20K）
- `floorplan_community`
  - Hugging Face: `Hyunwoo1605/mask2former-floorplan-instance-segmentation`
  - 図面向けのコミュニティ重み（本番利用前に検証必須）

`CubiCasa5K` / `Structured3D` そのものの「公式Mask2Former重み」は見つけにくいため、
本デモは公式重み+コミュニティ重みの比較構成にしています。

## Run backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp env.template .env
# Edit .env and set GEMINI_API_KEY manually
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Download ADE20K test images (bulk)

```bash
cd backend
source .venv/bin/activate
python scripts/download_test_images.py --count 100
```

## Run frontend

```bash
cd frontend
npm install
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

Open: `http://localhost:3000`

## Backend env for Gemini

- `GEMINI_API_KEY`: Gemini API key (required)
- `GEMINI_ENABLED`: `true` or `false` (default: `true`)
- `GEMINI_MODEL`: Gemini model name (default: `gemini-2.0-flash`)
- `GEMINI_API_BASE`: API base URL (default: `https://generativelanguage.googleapis.com/v1beta/models`)
- `GEMINI_TIMEOUT_SEC`: request timeout in seconds (default: `20`)

## API

- `GET /models`: 利用可能モデル一覧
- `POST /predict`:
  - form-data: `file`, `model_key`
  - returns: 推論時間、推論前画像URL、オーバーレイ画像URL、検出クラス

## CLI inference check

```bash
cd backend
source .venv/bin/activate
python scripts/run_inference_check.py --limit 3
```

This runs inference over local `app/static/test_images` and prints a per-image summary to CLI.

## Notes for MacBook Air

- 推論は `mps` -> `cuda` -> `cpu` の順で自動選択
- フル学習は重いので、このデモは推論中心
- まず顧客図面で失敗パターンを集め、入社後にチューニング計画へつなぐ想定
