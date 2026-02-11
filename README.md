# mask2former_web メモ

このディレクトリは **Mask2Former モデルを使った推論フローのキャッチアップ** が目的。
やりたいことは、

- フロントで画像を選ぶ/アップロードする
- FastAPI で推論を呼ぶ
- 推論結果（オーバーレイ、クラス面積など）を確認する

までを一気通貫で追うこと。

## プロジェクト場所

```bash
/Users/yutaakase/Documents/GitHub/mask2former_web
```

## 使用モデル

- `ade20k_official`
  - Hugging Face: `facebook/mask2former-swin-large-ade-semantic`
  - ADE20K 用の公式重み

今の実装はこのモデルキー固定で使う。

## 起動手順（最短）

### 1) Backend 起動

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp env.template .env

# 必要なら .env に GEMINI_API_KEY を入れる
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uvicorn app.main:app --host 127.0.0.1 --port 18000
```

API docs: `http://127.0.0.1:18000/docs`

### 2) Frontend 起動

```bash
cd frontend
npm install
NEXT_PUBLIC_API_BASE=http://127.0.0.1:18000 npm run dev -- --hostname 127.0.0.1 --port 13000
```

画面: `http://127.0.0.1:13000`

## テスト画像の準備

ADE20K の validation 画像をローカルに落とすと、ギャラリー推論ですぐ試せる。

```bash
cd backend
source .venv/bin/activate
python scripts/download_test_images.py --count 100
```

出力先: `backend/app/static/test_images`

## API メモ

ベース URL: `http://127.0.0.1:18000`

- `GET /health`
  - ヘルスチェック
- `GET /models`
  - 利用可能モデル一覧
- `GET /test-images`
  - ギャラリー用の画像一覧
- `POST /predict`
  - アップロード画像で推論
  - `multipart/form-data`: `file`, `model_key`（省略時 `ade20k_official`）
- `POST /predict-by-id`
  - テスト画像IDで推論
  - JSON body: `{ "image_id": "..." }`
- `POST /describe`
  - 推論結果の要約文を生成
  - JSON body: `{ "top_classes": [...], "area_stats": [...], "inference_ms": number|null }`

## ざっくりディレクトリ構造

```text
mask2former_web/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI エントリ
│   │   ├── core/config.py           # 定数・パス・モデルキー定義
│   │   ├── models/registry.py       # モデル管理
│   │   ├── services/                # 推論・可視化・説明文生成のロジック
│   │   └── static/
│   │       ├── test_images/         # テスト入力画像
│   │       └── results/             # 推論結果画像（orig/overlay/mask）
│   ├── scripts/
│   │   ├── download_test_images.py  # ADE20K画像の一括DL
│   │   └── run_inference_check.py   # CLIでの推論疎通確認
│   ├── tests/
│   ├── requirements.txt
│   └── env.template
└── frontend/
    ├── src/
    │   ├── app/                     # Next.js App Router
    │   ├── components/              # UI部品
    │   ├── hooks/                   # 画面ロジック
    │   ├── lib/api.ts               # Backend呼び出し
    │   └── types/                   # 型定義
    ├── tests/
    └── package.json
```

## CLI で推論だけ確認したい時

```bash
cd backend
source .venv/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/run_inference_check.py --limit 3
```

`app/static/test_images` を順番に推論して、結果サマリを CLI で確認できる。

## 補足メモ

- 推論デバイスは基本 `mps -> cuda -> cpu` の順で使えるものを選ぶ設計
- このリポジトリは学習よりも **推論フロー理解** が主目的
- まずは API 入出力と可視化の流れを把握するのを優先
