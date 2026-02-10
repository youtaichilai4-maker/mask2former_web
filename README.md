# Mask2Former ADE20K Web Demo

ADE20K公式重みの `Mask2Former` を使って、推論前後の可視化・画像選択推論・指標表示を行う `Next.js + FastAPI` デモです。

## What this proves

- Web実装力: 画像アップロード、推論API連携、可視化UI
- MLキャッチアップ力: Mask2Formerの公式学習済み重みで推論・評価
- 実務感: テスト画像選択→HTTP推論→可視化保存までの一連経路を検証

## Model status (important)

- `ade20k_official`
  - Hugging Face: `facebook/mask2former-swin-large-ade-semantic`
  - 公式のMask2Former公開重み（ADE20K）

現在の実装は `ade20k_official` のみを使用します。

## Project path

```bash
/Users/yutaakase/Documents/GitHub/mask2former_web
```

## Run backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp env.template .env
# Edit .env and set GEMINI_API_KEY manually
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uvicorn app.main:app --host 127.0.0.1 --port 18000
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
NEXT_PUBLIC_API_BASE=http://127.0.0.1:18000 npm run dev -- --hostname 127.0.0.1 --port 13000
```

Open: `http://127.0.0.1:13000`

## Backend env for Gemini

- `GEMINI_API_KEY`: Gemini API key (required)
- `GEMINI_ENABLED`: `true` or `false` (default: `true`)
- `GEMINI_MODEL`: Gemini model name (default: `gemini-2.0-flash`)
- `GEMINI_API_BASE`: API base URL (default: `https://generativelanguage.googleapis.com/v1beta/models`)
- `GEMINI_TIMEOUT_SEC`: request timeout in seconds (default: `20`)

## API

- `GET /models`: 利用可能モデル一覧
- `GET /test-images`: ギャラリー表示用のテスト画像一覧
- `POST /predict-by-id`: ギャラリーで選択した画像IDで推論
- `POST /predict`:
  - form-data: `file`, `model_key`
  - returns: 推論時間、推論前画像URL、オーバーレイ画像URL、検出クラス

## CLI inference check

```bash
cd backend
source .venv/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_inference_check.py --limit 3
```

This runs inference over local `app/static/test_images` and prints a per-image summary to CLI.

## Notes for MacBook Air

- 推論は `mps` -> `cuda` -> `cpu` の順で自動選択
- フル学習は重いので、このデモは推論中心
- まず顧客図面で失敗パターンを集め、入社後にチューニング計画へつなぐ想定
