import json
import os
from typing import Any

import requests
from fastapi import HTTPException

from app.schemas import DescribeRequest


class DescriptionService:
    def __init__(self, endpoint: str | None = None) -> None:
        api_base = os.getenv(
            "GEMINI_API_BASE",
            "https://generativelanguage.googleapis.com/v1beta/models",
        ).rstrip("/")
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
        self.endpoint = endpoint or f"{api_base}/{model}:generateContent"
        timeout_raw = os.getenv("GEMINI_TIMEOUT_SEC", "20").strip()
        try:
            self.timeout_sec = float(timeout_raw)
        except ValueError:
            self.timeout_sec = 20.0

    def describe(self, payload: DescribeRequest) -> dict[str, Any]:
        enabled_raw = os.getenv("GEMINI_ENABLED", "true").strip().lower()
        if enabled_raw in {"0", "false", "no", "off"}:
            raise HTTPException(status_code=503, detail="Gemini description is disabled (GEMINI_ENABLED=false)")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not set")

        lines = [
            "あなたは『空間を読むナレーター』です。",
            "Mask2Formerのセグメンテーション統計から、読んで楽しい短い紹介文を日本語で作成してください。",
            "出力はJSONのみ（前置き/後置き/Markdown禁止）。",
            "キーは必ず summary_ja, highlights, cautions。",
            "summary_ja は 80〜140文字、1段落。",
            "highlights は 2〜4件、各30文字以内。",
            "cautions は 1〜3件、各40文字以内。",
            "数値・事実は入力にあるものだけを使い、創作しない。",
            "断定しすぎず『〜と見られる』『〜の可能性』を適宜使う。",
            "語り口は明るく知的、展示会のガイド説明のように。",
            "summary_ja は『この空間は』で始める。",
            "比喩は1つまでに抑える。",
            "出力スキーマ:",
            '{"summary_ja":"string","highlights":["string"],"cautions":["string"]}',
            "top_classes:",
        ]
        lines.extend([f"- {row}" for row in payload.top_classes])
        lines.append("area_stats:")
        lines.extend([f"- {row}" for row in payload.area_stats])
        if payload.inference_ms is not None:
            lines.append(f"inference_ms: {payload.inference_ms}")

        req_body = {
            "contents": [{"parts": [{"text": "\n".join(lines)}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }

        res = requests.post(
            self.endpoint,
            params={"key": api_key},
            json=req_body,
            timeout=self.timeout_sec,
        )
        if res.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Gemini API error: {res.text[:200]}")

        text = (
            res.json()
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )
        if not text:
            raise HTTPException(status_code=502, detail="Empty response from Gemini")

        try:
            parsed = json.loads(text)
            return {
                "summary_ja": parsed.get("summary_ja", ""),
                "highlights": parsed.get("highlights", []),
                "cautions": parsed.get("cautions", []),
            }
        except Exception:
            return {"summary_ja": text, "highlights": [], "cautions": []}
