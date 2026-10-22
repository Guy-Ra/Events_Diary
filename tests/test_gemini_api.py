"""Focused adapter regressions for Gemini's production video request boundary."""

import base64
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from events_diary import gemini_api
from events_diary.config import VLMAPISettings
from events_diary.gemini_api import (
    GEMINI_RESPONSE_SCHEMA,
    MAX_INLINE_VIDEO_BYTES,
    GeminiAPIRuntime,
)
from events_diary.vlm import VLMRequest
from events_diary.vlm_prompts import (
    VLM_BOOLEAN_FIELDS,
    VLM_TEXT_FIELDS,
    build_vlm_prompt,
)

SECRET = "fake-gemini-secret-not-real"


class FakeResponse:
    """Provide a JSON HTTP response for deterministic adapter tests."""

    status = 200

    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.payload


def make_settings(**overrides) -> VLMAPISettings:
    """Build central Gemini settings with a non-secret test credential."""
    values = {
        "provider": "gemini", "api_key": SECRET,
        "base_url": "https://generativelanguage.googleapis.com", "api_version": "v1beta",
        "model_name": "gemini-flash-latest", "request_timeout_s": 23.0,
    }
    values.update(overrides)
    return VLMAPISettings(**values)


def make_request(path: Path) -> VLMRequest:
    """Create one continuous-video request with the unchanged production prompt."""
    return VLMRequest(
        investigation_index=0,
        source_video=Path("data/videos/ticket_scroll_test.mp4"),
        video_clip_path=path,
        source_start_s=0.0,
        source_end_s=8.4,
        prompt=build_vlm_prompt("fr24_vlm_v1"),
        prompt_version="fr24_vlm_v1",
    )


def response_payload(text: str) -> dict:
    """Build a compact successful GenerateContent response envelope."""
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
        "modelVersion": "gemini-flash-test-version",
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 8, "totalTokenCount": 20},
    }


def test_request_sends_one_video_and_prompt_and_preserves_utf8_response(tmp_path, monkeypatch):
    """Protect direct-video payload shape, secret header, schema, and source text."""
    clip = tmp_path / "investigation.mp4"
    video_bytes = b"complete continuous source video"
    clip.write_bytes(video_bytes)
    generated_text = '{"track":"281°"}'
    captured = []

    def fake_urlopen(request, *, timeout):
        captured.append((request, timeout))
        return FakeResponse(json.dumps(response_payload(generated_text), ensure_ascii=False).encode("utf-8"))

    monkeypatch.setattr(gemini_api, "urlopen", fake_urlopen)
    result = GeminiAPIRuntime(make_settings()).run_with_diagnostics(make_request(clip))

    http_request, timeout = captured[0]
    body = json.loads(http_request.data)
    parts = body["contents"][0]["parts"]
    inline_video = [part["inlineData"] for part in parts if "inlineData" in part]
    prompt_parts = [part["text"] for part in parts if "text" in part]
    config = body["generationConfig"]
    assert http_request.full_url.endswith("/v1beta/models/gemini-flash-latest:generateContent")
    assert http_request.get_header("X-goog-api-key") == SECRET
    assert timeout == 23.0
    assert len(inline_video) == 1
    assert inline_video[0]["mimeType"] == "video/mp4"
    assert base64.b64decode(inline_video[0]["data"]) == video_bytes
    assert prompt_parts == [build_vlm_prompt("fr24_vlm_v1")]
    assert set(body) == {"contents", "generationConfig"}
    assert config == {"responseMimeType": "application/json", "responseSchema": GEMINI_RESPONSE_SCHEMA}
    assert set(GEMINI_RESPONSE_SCHEMA["required"]) == set(VLM_TEXT_FIELDS + VLM_BOOLEAN_FIELDS)
    assert all(prop["nullable"] for prop in GEMINI_RESPONSE_SCHEMA["properties"].values())
    assert result.raw_result.raw_text == generated_text
    assert result.raw_result.model_revision == "gemini-flash-test-version"
    assert result.diagnostics.total_token_count == 20
    assert SECRET not in repr(result)


def test_schema_rejection_retries_once_without_schema_but_keeps_json_mode(tmp_path, monkeypatch):
    """Keep the documented single retry for Gemini schema incompatibility."""
    clip = tmp_path / "investigation.mp4"
    clip.write_bytes(b"video")
    replies = [
        (400, b'{"error":{"message":"responseSchema unsupported"}}'),
        (200, json.dumps(response_payload("{}")).encode("utf-8")),
    ]
    sent_bodies = []

    def fake_urlopen(request, *, timeout):
        sent_bodies.append(json.loads(request.data))
        status, payload = replies.pop(0)
        if status != 200:
            raise HTTPError("https://invalid", status, "bad request", {}, io.BytesIO(payload))
        return FakeResponse(payload)

    monkeypatch.setattr(gemini_api, "urlopen", fake_urlopen)
    result = GeminiAPIRuntime(make_settings()).run_with_diagnostics(make_request(clip))

    assert result.raw_result.execution_status == "success"
    assert len(sent_bodies) == 2
    assert "responseSchema" in sent_bodies[0]["generationConfig"]
    assert "responseSchema" not in sent_bodies[1]["generationConfig"]
    assert sent_bodies[1]["generationConfig"] == {"responseMimeType": "application/json"}
    assert result.diagnostics.response_schema_rejected is True


def test_provider_error_does_not_leak_credential(tmp_path, monkeypatch):
    """Ensure provider error bodies and credentials never enter VLM results."""
    clip = tmp_path / "investigation.mp4"
    clip.write_bytes(b"video")

    def fail_http(*_, **__):
        raise HTTPError("https://invalid", 403, "failed", {}, io.BytesIO(SECRET.encode()))

    monkeypatch.setattr(gemini_api, "urlopen", fail_http)
    result = GeminiAPIRuntime(make_settings()).run(make_request(clip))
    assert result.execution_status == "runtime_error"
    assert "HTTP 403" in result.error
    assert SECRET not in repr(result)


def test_oversized_video_fails_before_http_request(tmp_path, monkeypatch):
    """Reject inline payloads beyond Gemini's supported size before transport."""
    clip = tmp_path / "oversized.mp4"
    with clip.open("wb") as stream:
        stream.seek(MAX_INLINE_VIDEO_BYTES)
        stream.write(b"x")
    monkeypatch.setattr(gemini_api, "urlopen", lambda *_a, **_k: pytest.fail("unexpected HTTP request"))
    result = GeminiAPIRuntime(make_settings()).run(make_request(clip))
    assert result.execution_status == "input_error"
