# 実装TODOリスト

## 0. セットアップ

- [ ] `backend` の依存をインストール
- [ ] `frontend` の依存をインストール
- [ ] `.env` を作成（必要なら `GEMINI_API_KEY` を設定）
- [ ] `backend/app/static/test_images` にテスト画像を3枚以上配置

## 1. Backend API

- [ ] `GET /test-images` を実装
- [ ] `POST /predict-by-id` を実装
- [ ] `POST /predict` のレスポンスを拡張
  - [ ] `class_masks[]`
  - [ ] `top_classes[]`
  - [ ] `area_stats[]`
- [ ] `POST /describe` を実装（Gemini連携）
- [ ] `GEMINI_API_KEY` 未設定時のフォールバックを実装

## 2. 推論・可視化ロジック

- [ ] ADE20K公式重みを単一デフォルトに固定
- [ ] クラス別2値マスク生成を実装
- [ ] オーバーレイ生成を改善（凡例と色の一貫性）
- [ ] Top-K信頼度算出を実装
- [ ] クラス面積比（%）算出を実装
- [ ] 推論時間計測を統一

## 3. Frontend UI（Next.js + React）

- [ ] `TestImageCarousel` を実装（スクロール選択）
- [ ] `UploadForm` とギャラリー選択の切替導線を実装
- [ ] 3ペインを実装
  - [ ] 入力画像
  - [ ] 予測オーバーレイ
  - [ ] クラス別マスク一覧
- [ ] 指標テーブルを実装
  - [ ] Top-K信頼度
  - [ ] 面積比（%）
  - [ ] 推論時間
- [ ] `DescriptionPanel` を実装（Gemini結果表示）
- [ ] エラーハンドリング（画像不正、推論失敗、Gemini失敗）

## 4. テスト

- [x] `GET /health` のAPIテスト
- [x] `GET /models` のAPIテスト
- [x] `POST /predict` の正常系テスト（モデルモック）
- [x] `POST /predict` の異常系テスト（非画像）
- [ ] `GET /test-images` テスト
- [ ] `POST /predict-by-id` テスト
- [ ] `POST /describe` テスト（キーなし/ありモック）

## 5. 仕上げ

- [ ] READMEを最終仕様に更新
- [ ] 面接デモ用スクリーンショットを保存
- [ ] 3枚の固定テスト画像で結果比較を記録
- [ ] 想定質問への回答メモを作成（信頼性、限界、次の一手）
