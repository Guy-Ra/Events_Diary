"""Google Gemini GenerateContent adapter for continuous FR24 Investigation video.

This provider boundary sends one MP4 as Gemini inlineData plus the unchanged
production prompt. Credentials are read only from the configured environment
variable, never added to results or logs. The adapter returns raw generated text
and provider diagnostics; semantic interpretation remains in ``vlm_parsing``.
"""

import base64
import json
import re
import time
from dataclasses import dataclass
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from events_diary.config import VLMAPISettings
from events_diary.vlm import ExecutionStatus, RawVLMResult, VLMRequest, VLMRuntime
from events_diary.vlm_prompts import VLM_BOOLEAN_FIELDS, VLM_TEXT_FIELDS
from events_diary.vlm_review import VLMReviewRequest

GEMINI_API_KEY_ENV = "EVENTS_DIARY_VLM_API_KEY"
MAX_INLINE_VIDEO_BYTES = 20 * 1024 * 1024


def _response_schema() -> dict[str, Any]:
    """Build the nullable response schema from the production prompt field list."""
    properties: dict[str, dict[str, Any]] = {}
    for name in VLM_TEXT_FIELDS:
        properties[name] = {"type": "STRING", "nullable": True}
    for name in VLM_BOOLEAN_FIELDS:
        properties[name] = {"type": "BOOLEAN", "nullable": True}
    return {
        "type": "OBJECT",
        "properties": properties,
        "required": list(VLM_TEXT_FIELDS + VLM_BOOLEAN_FIELDS),
    }


GEMINI_RESPONSE_SCHEMA = _response_schema()


@dataclass(frozen=True)
class GeminiRunDiagnostics:
    """Sanitized provider response and timing metadata for validation artifacts."""

    provider_response_json: str | None
    model_version: str | None
    prompt_token_count: int | None
    candidate_token_count: int | None
    total_token_count: int | None
    finish_reason: str | None
    generation_config: dict[str, Any]
    response_schema_used: bool
    response_schema_rejected: bool
    elapsed_s: float


@dataclass(frozen=True)
class GeminiRunResult:
    """Raw model execution plus provider details not held by the VLM contract."""

    raw_result: RawVLMResult
    diagnostics: GeminiRunDiagnostics


class GeminiAPIRuntime(VLMRuntime):
    """Call Gemini GenerateContent using one continuous MP4 inlineData part."""

    def __init__(self, settings: VLMAPISettings) -> None:
        """Retain centrally loaded configuration without reading environment values."""
        if not isinstance(settings, VLMAPISettings):
            raise TypeError("settings must be VLMAPISettings.")
        self.settings = settings
        self.request_timeout_s = settings.request_timeout_s
        self.model_name = settings.model_name
        self.endpoint_url = settings.endpoint_url

    def _generation_config(self) -> dict[str, Any]:
        """Request JSON output without imposing sampling or token-budget options."""
        return {"responseMimeType": "application/json"}

    def run(self, request: VLMRequest | VLMReviewRequest) -> RawVLMResult:
        """Execute a continuous-video request through Gemini and return raw text."""
        return self.run_with_diagnostics(request).raw_result

    def run_with_diagnostics(
        self, request: VLMRequest | VLMReviewRequest
    ) -> GeminiRunResult:
        """Run a video request and retain sanitized response metadata for validation."""
        if not isinstance(request, (VLMRequest, VLMReviewRequest)):
            raise TypeError("request must be VLMRequest or VLMReviewRequest.")
        started = time.perf_counter()
        path = request.video_clip_path
        try:
            size = path.stat().st_size
        except OSError:
            return self._failed(request, "input_error", started, "Continuous Investigation video clip could not be read.")
        if size > MAX_INLINE_VIDEO_BYTES:
            return self._failed(request, "input_error", started,
                "Continuous Investigation video exceeds the 20 MiB inline limit; Gemini Files API is required.")
        try:
            video_bytes = path.read_bytes()
        except OSError:
            return self._failed(request, "input_error", started, "Continuous Investigation video clip could not be read.")
        if not video_bytes:
            return self._failed(request, "input_error", started, "Continuous Investigation video clip is empty.")
        key = self.settings.api_key
        if not key.strip():
            return self._failed(request, "runtime_error", started,
                f"{GEMINI_API_KEY_ENV} is unset.")

        generation_config = self._generation_config()
        schema_used = not isinstance(request, VLMReviewRequest)
        payload = self._video_payload(
            request, video_bytes, generation_config, include_schema=schema_used,
        )
        status_code, response_bytes, transport_error = self._post(payload, key)
        schema_rejected = False
        if schema_used and status_code == 400 and self._is_schema_rejection(response_bytes):
            schema_rejected = True
            schema_used = False
            payload = self._video_payload(request, video_bytes, generation_config, include_schema=False)
            status_code, response_bytes, transport_error = self._post(payload, key)
        elapsed = time.perf_counter() - started
        if transport_error is not None:
            status, error = transport_error
            return self._failed(request, status, started, error, elapsed=elapsed,
                response_schema_used=schema_used, response_schema_rejected=schema_rejected,
                generation_config=generation_config)
        if status_code != 200:
            safe_response = self._sanitize_provider_error(response_bytes, key)
            return self._failed(request, "runtime_error", started, f"Gemini API returned HTTP {status_code}.",
                elapsed=elapsed, response_schema_used=schema_used,
                response_schema_rejected=schema_rejected, generation_config=generation_config,
                provider_response_json=safe_response)
        return self._result_from_response(
            request, response_bytes, elapsed, generation_config,
            schema_used=schema_used, schema_rejected=schema_rejected,
        )

    def _video_payload(
        self, request: VLMRequest | VLMReviewRequest, video_bytes: bytes,
        generation_config: dict[str, Any], *, include_schema: bool,
    ) -> dict[str, Any]:
        """Build Gemini JSON with the entire video bytes and current production prompt."""
        config = dict(generation_config)
        if include_schema:
            config["responseSchema"] = GEMINI_RESPONSE_SCHEMA
        return {
            "contents": [{"parts": [
                {"inlineData": {"mimeType": "video/mp4", "data": base64.b64encode(video_bytes).decode("ascii")}},
                {"text": request.prompt},
            ]}],
            "generationConfig": config,
        }

    def _post(self, payload: dict[str, Any], key: str) -> tuple[int | None, bytes, tuple[str, str] | None]:
        """POST JSON to Gemini without logging or returning the secret header."""
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = Request(self.endpoint_url, data=body, headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-goog-api-key": key,
        }, method="POST")
        try:
            with urlopen(request, timeout=self.request_timeout_s) as response:
                return response.status, response.read(), None
        except HTTPError as error:
            return error.code, error.read(), None
        except (TimeoutError, URLError) as error:
            if isinstance(error, TimeoutError) or (
                isinstance(error, URLError) and isinstance(error.reason, TimeoutError)
            ):
                return None, b"", ("timeout", "Gemini API request timed out.")
            return None, b"", ("runtime_error", "Gemini API request failed due to a network error.")
        except (OSError, HTTPException):
            return None, b"", ("runtime_error", "Gemini API request failed due to an I/O or HTTP error.")

    def _result_from_response(
        self, request: VLMRequest | VLMReviewRequest, response_bytes: bytes,
        elapsed: float, generation_config: dict[str, Any],
        *, schema_used: bool, schema_rejected: bool,
    ) -> GeminiRunResult:
        """Extract candidate text and provider telemetry without semantic parsing."""
        response_text = response_bytes.decode("utf-8", errors="replace")
        try:
            body = json.loads(response_text)
            text, model_version, usage, finish_reason = self._extract_response(body)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            return self._failed(request, "invalid_response", time.perf_counter() - elapsed,
                self._safe_response_error(None, error), elapsed=elapsed,
                provider_response_json=response_text, generation_config=generation_config,
                response_schema_used=schema_used, response_schema_rejected=schema_rejected)
        usage = usage or {}
        diagnostics = GeminiRunDiagnostics(
            provider_response_json=response_text,
            model_version=model_version,
            prompt_token_count=usage.get("promptTokenCount"),
            candidate_token_count=usage.get("candidatesTokenCount"),
            total_token_count=usage.get("totalTokenCount"),
            finish_reason=finish_reason,
            generation_config=dict(generation_config),
            response_schema_used=schema_used,
            response_schema_rejected=schema_rejected,
            elapsed_s=elapsed,
        )
        if text is None or not text.strip():
            raw = self._make_raw(request, "invalid_response", elapsed, model_version=model_version,
                error="Gemini API returned no candidate text.")
        else:
            raw = self._make_raw(request, "success", elapsed, raw_text=text,
                generated_tokens=usage.get("candidatesTokenCount"), model_version=model_version)
        return GeminiRunResult(raw, diagnostics)

    @staticmethod
    def _extract_response(body: object) -> tuple[str | None, str | None, dict[str, int] | None, str | None]:
        """Read Gemini candidate text, model version, token usage and finish reason."""
        if not isinstance(body, dict):
            raise TypeError("Gemini response root must be an object.")
        candidates = body.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            feedback = body.get("promptFeedback")
            reason = feedback.get("blockReason") if isinstance(feedback, dict) else None
            safe_reason = reason if isinstance(reason, str) and reason.replace("_", "").isalnum() else None
            raise ValueError(f"Gemini response has no candidates{f' ({safe_reason})' if safe_reason else ''}.")
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise TypeError("Gemini candidate has an invalid shape.")
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            raise TypeError("Gemini candidate has no text parts.")
        text_parts = [part["text"] for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str)]
        text = "".join(text_parts) if text_parts else None
        usage = body.get("usageMetadata")
        usage_counts = None
        if isinstance(usage, dict):
            usage_counts = {
                key: value for key in ("promptTokenCount", "candidatesTokenCount", "totalTokenCount")
                if isinstance((value := usage.get(key)), int) and not isinstance(value, bool) and value >= 0
            }
        model_version = body.get("modelVersion")
        if not isinstance(model_version, str):
            model_version = None
        finish_reason = candidate.get("finishReason")
        if not isinstance(finish_reason, str):
            finish_reason = None
        return text, model_version, usage_counts, finish_reason

    @staticmethod
    def _is_schema_rejection(response_bytes: bytes) -> bool:
        """Recognize a 400 responseSchema rejection without exposing its body."""
        try:
            body = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        error = body.get("error") if isinstance(body, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        return isinstance(message, str) and "schema" in message.lower()

    @staticmethod
    def _sanitize_provider_error(response_bytes: bytes, api_key: str) -> str:
        """Preserve provider diagnostics while removing credentials and media payloads."""
        raw = response_bytes.decode("utf-8", errors="replace")
        try:
            body: object = json.loads(raw)
        except json.JSONDecodeError:
            body = raw

        def sanitize(value: object) -> object:
            if isinstance(value, dict):
                cleaned = {}
                for key, item in value.items():
                    normalized_key = re.sub(r"[^a-z0-9]", "", str(key).lower())
                    if normalized_key in {
                        "data", "apikey", "authorization", "access_token", "refreshtoken",
                    }:
                        cleaned[str(key)] = "[REDACTED]"
                    else:
                        cleaned[str(key)] = sanitize(item)
                return cleaned
            if isinstance(value, list):
                return [sanitize(item) for item in value]
            if isinstance(value, str):
                cleaned = value.replace(api_key, "[REDACTED]") if api_key else value
                return re.sub(r"[A-Za-z0-9+/]{512,}={0,2}", "[REDACTED_BASE64]", cleaned)
            return value

        cleaned_body = sanitize(body)
        if isinstance(cleaned_body, str):
            return cleaned_body
        return json.dumps(cleaned_body, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _safe_response_error(body: object, error: Exception) -> str:
        """Return a sanitized fixed-shape error without echoing model/provider text."""
        if isinstance(error, ValueError) and isinstance(body, dict):
            feedback = body.get("promptFeedback")
            reason = feedback.get("blockReason") if isinstance(feedback, dict) else None
            if isinstance(reason, str) and reason.replace("_", "").isalnum():
                return f"Gemini returned no usable candidate ({reason})."
        return "Gemini API response did not match the expected GenerateContent shape."

    def _make_raw(
        self, request: VLMRequest | VLMReviewRequest, status: ExecutionStatus,
        elapsed: float, *, raw_text: str | None = None,
        generated_tokens: int | None = None, model_version: str | None = None, error: str | None = None,
    ) -> RawVLMResult:
        """Build the existing raw VLM contract without storing provider secrets."""
        return RawVLMResult(
            execution_status=status, investigation_index=request.investigation_index,
            source_video=request.source_video, video_clip_path=request.video_clip_path,
            source_start_s=request.source_start_s, source_end_s=request.source_end_s,
            model_name=self.model_name, model_revision=model_version, prompt_version=request.prompt_version,
            raw_text=raw_text, inference_time_s=elapsed, ttft_s=None,
            generated_tokens=generated_tokens, error=error,
        )

    def _failed(
        self, request: VLMRequest | VLMReviewRequest, status: ExecutionStatus,
        started: float, error: str, *,
        elapsed: float | None = None, provider_response_json: str | None = None,
        generation_config: dict[str, Any] | None = None, response_schema_used: bool = False,
        response_schema_rejected: bool = False,
    ) -> GeminiRunResult:
        """Represent an expected request failure and preserve safe diagnostics."""
        duration = time.perf_counter() - started if elapsed is None else elapsed
        raw = self._make_raw(request, status, duration, error=error)
        diagnostics = GeminiRunDiagnostics(
            provider_response_json=provider_response_json, model_version=None,
            prompt_token_count=None, candidate_token_count=None, total_token_count=None,
            finish_reason=None, generation_config=generation_config or self._generation_config(),
            response_schema_used=response_schema_used, response_schema_rejected=response_schema_rejected,
            elapsed_s=duration,
        )
        return GeminiRunResult(raw, diagnostics)
