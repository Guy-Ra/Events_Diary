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
from events_diary.vlm import VLMImageInput, VLMRequest
from events_diary.vlm_prompts import (
    VLM_BOOLEAN_FIELDS,
    VLM_ROUTE_FIELDS,
    VLM_TEXT_FIELDS,
    build_vlm_prompt,
)

SECRET = "fake-gemini-secret-not-real"


class FakeResponse:
    """Provide a JSON HTTP response for deterministic adapter tests."""

    status = 200
    def __init__(self, payload: bytes):
        self.payload = payload
        self.headers = {}

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


def make_request(path: Path, *, ticket_images=()) -> VLMRequest:
    """Create one continuous-video request with the unchanged production prompt."""
    return VLMRequest(
        investigation_index=0,
        source_video=Path("data/videos/ticket_scroll_test.mp4"),
        video_clip_path=path,
        source_start_s=0.0,
        source_end_s=8.4,
        prompt=build_vlm_prompt("fr24_vlm_v1"),
        prompt_version="fr24_vlm_v1",
        ticket_images=tuple(ticket_images),
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
    assert set(GEMINI_RESPONSE_SCHEMA["required"]) == set(VLM_TEXT_FIELDS + VLM_BOOLEAN_FIELDS) | {"visual_route"}
    assert all(GEMINI_RESPONSE_SCHEMA["properties"][name]["nullable"]
               for name in VLM_TEXT_FIELDS + VLM_BOOLEAN_FIELDS)
    assert set(GEMINI_RESPONSE_SCHEMA["properties"]["visual_route"]["required"]) == set(VLM_ROUTE_FIELDS)
    assert result.raw_result.raw_text == generated_text
    assert result.raw_result.model_revision == "gemini-flash-test-version"
    assert result.diagnostics.total_token_count == 20
    assert SECRET not in repr(result)


def test_vlm_only_request_sends_video_and_selected_ticket_images_without_ocr_or_fusion(
    tmp_path, monkeypatch,
):
    """Attach separate selected ticket crops while preserving the Pass 1 schema."""
    clip = tmp_path / "investigation.mp4"
    image = tmp_path / "ticket_upper.png"
    clip.write_bytes(b"complete continuous source video")
    image.write_bytes(b"selected ticket image bytes")
    ticket_input = VLMImageInput(
        image, 3.2, "upper_ticket_header_and_route_times", "flight identity, route, and times",
    )
    captured = []

    def fake_urlopen(request, *, timeout):
        captured.append(json.loads(request.data))
        return FakeResponse(json.dumps(response_payload("{}")).encode())

    monkeypatch.setattr(gemini_api, "urlopen", fake_urlopen)
    result = GeminiAPIRuntime(make_settings()).run_with_diagnostics(
        make_request(clip, ticket_images=(ticket_input,)),
    )

    parts = captured[0]["contents"][0]["parts"]
    images = [part["inlineData"] for part in parts if "inlineData" in part]
    text_parts = [part["text"] for part in parts if "text" in part]
    assert [part["mimeType"] for part in images] == ["video/mp4", "image/png"]
    assert base64.b64decode(images[0]["data"]) == b"complete continuous source video"
    assert base64.b64decode(images[1]["data"]) == b"selected ticket image bytes"
    assert text_parts == [
        (
            "Ticket image 1; source video time 3.200s; "
            "role=upper_ticket_header_and_route_times; section=flight identity, route, and times."
        ),
        build_vlm_prompt("fr24_vlm_v1"),
    ]
    assert "ocr" not in " ".join(text_parts).casefold()
    assert "fusion" not in " ".join(text_parts).casefold()
    assert captured[0]["generationConfig"]["responseSchema"] == GEMINI_RESPONSE_SCHEMA
    assert result.raw_result.execution_status == "success"


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


def _queued_http_responses(monkeypatch, responses):
    """Install deterministic HTTP outcomes and return the number of transport calls."""
    remaining = list(responses)
    calls = []
    monkeypatch.setattr(gemini_api.random, "uniform", lambda *_: 0.0)

    def fake_urlopen(*_, **__):
        calls.append(None)
        response = remaining.pop(0)
        status, payload = response[:2]
        headers = response[2] if len(response) > 2 else {}
        if isinstance(status, Exception):
            raise status
        if status != 200:
            raise HTTPError("https://invalid", status, "provider error", headers, io.BytesIO(payload))
        return FakeResponse(payload)

    monkeypatch.setattr(gemini_api, "urlopen", fake_urlopen)
    return calls


def test_503_then_success_retries_once_after_30_seconds(tmp_path, monkeypatch):
    """Retry one HTTP 503 and use the first approved backoff interval."""
    clip = tmp_path / "investigation.mp4"
    clip.write_bytes(b"video")
    calls = _queued_http_responses(monkeypatch, [
        (503, b'{"error":{"status":"UNAVAILABLE"}}'),
        (200, json.dumps(response_payload("{}")).encode()),
    ])
    delays = []
    monkeypatch.setattr(gemini_api.time, "sleep", delays.append)

    result = GeminiAPIRuntime(make_settings()).run(make_request(clip))

    assert result.execution_status == "success"
    assert len(calls) == 2
    assert delays == [5.0]


def test_two_503_failures_then_success_use_30_and_90_second_backoffs(tmp_path, monkeypatch):
    """Apply the bounded backoff schedule across consecutive transient failures."""
    clip = tmp_path / "investigation.mp4"
    clip.write_bytes(b"video")
    calls = _queued_http_responses(monkeypatch, [
        (503, b"{}"), (503, b"{}"),
        (200, json.dumps(response_payload("{}" )).encode()),
    ])
    delays = []
    monkeypatch.setattr(gemini_api.time, "sleep", delays.append)

    result = GeminiAPIRuntime(make_settings()).run(make_request(clip))

    assert result.execution_status == "success"
    assert len(calls) == 3
    assert delays == [5.0, 10.0]


def test_three_503_failures_return_final_provider_error(tmp_path, monkeypatch):
    """Stop after three attempts and preserve the final HTTP failure."""
    clip = tmp_path / "investigation.mp4"
    clip.write_bytes(b"video")
    calls = _queued_http_responses(monkeypatch, [(503, b"{}") for _ in range(3)])
    monkeypatch.setattr(gemini_api.time, "sleep", lambda _: None)

    result = GeminiAPIRuntime(make_settings()).run(make_request(clip))

    assert result.execution_status == "runtime_error"
    assert result.error.startswith("Gemini API returned HTTP 503")
    assert len(calls) == 3


def test_request_timeout_then_success_retries(tmp_path, monkeypatch):
    """Retry a request timeout and succeed on the next transport attempt."""
    clip = tmp_path / "investigation.mp4"
    clip.write_bytes(b"video")
    calls = _queued_http_responses(monkeypatch, [
        (TimeoutError("socket timed out"), b""),
        (200, json.dumps(response_payload("{}" )).encode()),
    ])
    delays = []
    monkeypatch.setattr(gemini_api.time, "sleep", delays.append)

    result = GeminiAPIRuntime(make_settings()).run(make_request(clip))

    assert result.execution_status == "success"
    assert len(calls) == 2
    assert delays == [5.0]


def test_nonretryable_http_4xx_is_not_retried(tmp_path, monkeypatch):
    """Do not repeat permanent request or authentication failures."""
    clip = tmp_path / "investigation.mp4"
    clip.write_bytes(b"video")
    calls = _queued_http_responses(monkeypatch, [(400, b'{"error":{"status":"INVALID_ARGUMENT"}}')])
    monkeypatch.setattr(gemini_api.time, "sleep", lambda _: pytest.fail("unexpected retry"))

    result = GeminiAPIRuntime(make_settings()).run(make_request(clip))

    assert result.execution_status == "runtime_error"
    assert "HTTP 400" in result.error
    assert len(calls) == 1


def test_retry_logs_and_errors_do_not_expose_secret(tmp_path, monkeypatch, caplog):
    """Keep credentials out of retry diagnostics and final provider errors."""
    clip = tmp_path / "investigation.mp4"
    clip.write_bytes(b"video")
    calls = _queued_http_responses(monkeypatch, [(503, SECRET.encode()) for _ in range(3)])
    delays = []
    monkeypatch.setattr(gemini_api.time, "sleep", delays.append)

    result = GeminiAPIRuntime(make_settings()).run(make_request(clip))

    assert len(calls) == 3
    assert delays == [5.0, 10.0]
    assert "retrying attempt 2/3 after 5.00 s" in caplog.text
    assert "retrying attempt 3/3 after 10.00 s" in caplog.text
    assert SECRET not in caplog.text
    assert SECRET not in repr(result)


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


def test_http_error_preserves_sanitized_provider_details(tmp_path, monkeypatch):
    """Retain useful HTTP error fields while redacting keys and payload-like data."""
    clip = tmp_path / "investigation.mp4"
    clip.write_bytes(b"video")
    provider_body = {
        "error": {
            "code": 400,
            "status": "INVALID_ARGUMENT",
            "message": f"Invalid request for {SECRET}",
            "details": [{"reason": "BAD_SCHEMA", "data": "A" * 600}],
        }
    }

    def fail_http(*_, **__):
        raise HTTPError(
            "https://invalid", 400, "bad request", {},
            io.BytesIO(json.dumps(provider_body).encode()),
        )

    monkeypatch.setattr(gemini_api, "urlopen", fail_http)
    result = GeminiAPIRuntime(make_settings()).run_with_diagnostics(make_request(clip))

    sanitized = json.loads(result.diagnostics.provider_response_json)
    assert result.raw_result.execution_status == "runtime_error"
    assert sanitized["error"]["code"] == 400
    assert sanitized["error"]["status"] == "INVALID_ARGUMENT"
    assert SECRET not in result.diagnostics.provider_response_json
    assert "[REDACTED]" in result.diagnostics.provider_response_json


def test_429_retry_honors_retry_after_and_keeps_exact_attempt_diagnostics(tmp_path, monkeypatch):
    """Retry a transient 429 once and preserve quota, timing, and request context."""
    clip = tmp_path / "investigation.mp4"
    clip.write_bytes(b"video")
    body = {
        "error": {
            "code": 429, "status": "RESOURCE_EXHAUSTED", "message": f"Rate exceeded {SECRET}",
            "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                         "violations": [{"quotaMetric": "generate_requests_per_minute",
                                         "quotaId": "RequestsPerMinutePerProject",
                                         "quotaValue": "10", "quotaDimensions": {"model": "test"}}]},
                        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"}],
        },
        "usageMetadata": {"promptTokenCount": 22, "totalTokenCount": 22},
    }
    calls = _queued_http_responses(monkeypatch, [
        (429, json.dumps(body).encode(), {"Retry-After": "12"}),
        (200, json.dumps(response_payload("{}")).encode()),
    ])
    delays = []
    monkeypatch.setattr(gemini_api.time, "sleep", delays.append)
    result = GeminiAPIRuntime(make_settings()).run(make_request(clip))
    diagnostic = json.loads(result.provider_diagnostics_json)

    assert result.execution_status == "success"
    assert len(calls) == 2
    assert delays == [17.0]
    assert diagnostic["attempt_count"] == 2
    assert diagnostic["attempts"][0]["provider_error"]["quota_metric"] == "generate_requests_per_minute"
    assert diagnostic["attempts"][0]["provider_error"]["quota_id"] == "RequestsPerMinutePerProject"
    assert diagnostic["attempts"][0]["provider_error"]["quota_limit"] == "10"
    assert diagnostic["attempts"][0]["provider_error"]["classification"] == "RPM"
    assert diagnostic["attempts"][0]["provider_error"]["provider_code"] == 429
    assert diagnostic["attempts"][0]["provider_error"]["provider_status"] == "RESOURCE_EXHAUSTED"
    assert diagnostic["attempts"][0]["provider_error"]["usage_metadata"]["promptTokenCount"] == 22
    assert SECRET not in result.provider_diagnostics_json
    assert "[REDACTED]" in result.provider_diagnostics_json
    assert diagnostic["attempts"][0]["provider_error"]["retry_delay"] == "17s"
    assert diagnostic["video_clip_duration_s"] == pytest.approx(8.4)
    assert diagnostic["image_crop_count"] == 0
    assert diagnostic["model"] == "gemini-flash-latest"
    assert diagnostic["api_method"] == "models.generateContent"
    assert diagnostic["investigation_id"] == 0
    assert diagnostic["request_timestamp_utc"]
    assert diagnostic["total_api_request_s"] >= 0
    assert diagnostic["total_elapsed_s"] >= diagnostic["total_api_request_s"]
    assert diagnostic["sdk_retries_enabled"] is False


def test_daily_quota_exhaustion_is_not_retried(tmp_path, monkeypatch):
    """Recognize explicit per-day quota exhaustion and preserve it as diagnostic data."""
    clip = tmp_path / "investigation.mp4"
    clip.write_bytes(b"video")
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                      "message": "Quota exceeded for requests per day",
                      "details": [{"quotaMetric": "generate_requests_per_day"}]}}
    calls = _queued_http_responses(monkeypatch, [(429, json.dumps(body).encode())])
    monkeypatch.setattr(gemini_api.time, "sleep", lambda _: pytest.fail("daily quota must not retry"))

    result = GeminiAPIRuntime(make_settings()).run(make_request(clip))
    diagnostic = json.loads(result.provider_diagnostics_json)

    assert len(calls) == 1
    assert result.execution_status == "runtime_error"
    assert diagnostic["attempt_count"] == 1
    assert diagnostic["attempts"][0]["provider_error"]["classification"] == "RPD/daily_quota"


def test_three_429_failures_never_make_a_fourth_request(tmp_path, monkeypatch):
    """Apply one shared three-request ceiling to all transient HTTP 429 outcomes."""
    clip = tmp_path / "investigation.mp4"
    clip.write_bytes(b"video")
    calls = _queued_http_responses(monkeypatch, [(429, b'{"error":{"status":"RESOURCE_EXHAUSTED"}}') for _ in range(3)])
    monkeypatch.setattr(gemini_api.time, "sleep", lambda _: None)

    result = GeminiAPIRuntime(make_settings()).run(make_request(clip))

    assert result.execution_status == "runtime_error"
    assert len(calls) == 3
    assert json.loads(result.provider_diagnostics_json)["attempt_count"] == 3
    assert SECRET not in result.provider_diagnostics_json


def test_schema_compatibility_attempts_share_the_three_call_ceiling(tmp_path, monkeypatch):
    """Do not let schema fallback plus transient retries exceed three HTTP calls."""
    clip = tmp_path / "investigation.mp4"
    clip.write_bytes(b"video")
    calls = _queued_http_responses(monkeypatch, [
        (400, b'{"error":{"message":"responseSchema unsupported"}}'),
        (503, b"{}"), (503, b"{}"),
    ])
    monkeypatch.setattr(gemini_api.time, "sleep", lambda _: None)

    result = GeminiAPIRuntime(make_settings()).run(make_request(clip))

    assert result.execution_status == "runtime_error"
    assert len(calls) == 3
    diagnostic = json.loads(result.provider_diagnostics_json)
    assert diagnostic["attempt_count"] == diagnostic["max_attempts"] == 3


def test_oversized_video_fails_before_http_request(tmp_path, monkeypatch):
    """Reject inline payloads beyond Gemini's supported size before transport."""
    clip = tmp_path / "oversized.mp4"
    with clip.open("wb") as stream:
        stream.seek(MAX_INLINE_VIDEO_BYTES)
        stream.write(b"x")
    monkeypatch.setattr(gemini_api, "urlopen", lambda *_a, **_k: pytest.fail("unexpected HTTP request"))
    result = GeminiAPIRuntime(make_settings()).run(make_request(clip))
    assert result.execution_status == "input_error"
