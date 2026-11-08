"""Google Gemini GenerateContent adapter for continuous FR24 Investigation review.

This provider boundary sends one complete MP4 and, for Pass 2 v3, selected ticket
detail crops in the same Gemini GenerateContent request. Credentials are read only from the configured environment
variable, never added to results or logs. The adapter returns raw generated text
and provider diagnostics; semantic interpretation remains in ``vlm_parsing``.
"""

import base64
import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from events_diary.config import VLMAPISettings
from events_diary.vlm import ExecutionStatus, RawVLMResult, VLMRequest, VLMRuntime
from events_diary.vlm_prompts import (
    VLM_BOOLEAN_FIELDS,
    VLM_ROUTE_FIELDS,
    VLM_TEXT_FIELDS,
)
from events_diary.vlm_review import VLMReviewRequest

GEMINI_API_KEY_ENV = "EVENTS_DIARY_VLM_API_KEY"
MAX_INLINE_VIDEO_BYTES = 20 * 1024 * 1024
MAX_REQUEST_ATTEMPTS = 3
logger = logging.getLogger(__name__)


def _response_schema() -> dict[str, Any]:
    """Build the nullable response schema from the production prompt field list."""
    properties: dict[str, dict[str, Any]] = {}
    for name in VLM_TEXT_FIELDS:
        properties[name] = {"type": "STRING", "nullable": True}
    for name in VLM_BOOLEAN_FIELDS:
        properties[name] = {"type": "BOOLEAN", "nullable": True}
    properties["visual_route"] = {
        "type": "OBJECT",
        "properties": {
            name: {
                "type": "OBJECT",
                "properties": {
                    "value": {"type": "STRING", "nullable": True},
                    "status": {"type": "STRING", "enum": ["observed", "inferred", "not_visible"]},
                },
                "required": ["value", "status"],
            }
            for name in VLM_ROUTE_FIELDS
        },
        "required": list(VLM_ROUTE_FIELDS),
    }
    return {
        "type": "OBJECT",
        "properties": properties,
        "required": list(VLM_TEXT_FIELDS + VLM_BOOLEAN_FIELDS) + ["visual_route"],
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
        attempts: list[dict[str, Any]] = []

        def post_with_retry(current_payload: dict[str, Any]):
            """Retry classified transient failures without exceeding the request budget."""
            while True:
                attempt_number = len(attempts) + 1
                requested_at = datetime.now(timezone.utc).isoformat()
                request_started = time.perf_counter()
                logger.info(
                    "Gemini Investigation %s Attempt %s/%s.", request.investigation_index,
                    attempt_number, min(self.settings.max_attempts, MAX_REQUEST_ATTEMPTS),
                )
                response = self._post(current_payload, key)
                status, body, transport, headers = response
                elapsed = time.perf_counter() - request_started
                provider = self._provider_error_details(body, key)
                daily_exhausted = self._daily_quota_exhausted(provider)
                retryable = (
                    not daily_exhausted
                    and (status in (429, 503) or (transport is not None and transport[0] == "timeout"))
                )
                retry_after = self._retry_after_seconds(headers, provider)
                delay = None
                if retryable and attempt_number < min(self.settings.max_attempts, MAX_REQUEST_ATTEMPTS):
                    exponential = min(
                        self.settings.retry_base_delay_s * (2 ** (attempt_number - 1)),
                        self.settings.retry_max_delay_s,
                    )
                    delay = max(exponential, retry_after or 0.0) + random.uniform(
                        0.0, self.settings.retry_jitter_s,
                    )
                attempts.append({
                    "attempt": attempt_number,
                    "request_timestamp_utc": requested_at,
                    "duration_s": round(elapsed, 3),
                    "http_status": status,
                    "transport_error": transport[0] if transport else None,
                    "provider_error": provider,
                    "retryable": retryable,
                    "retry_delay_s": round(delay, 3) if delay is not None else None,
                })
                if status != 200 or transport is not None:
                    logger.warning(
                        "Gemini Investigation %s attempt %s/%s failed: HTTP %s / %s; elapsed %.3f s; retryable=%s.",
                        request.investigation_index, attempt_number,
                        min(self.settings.max_attempts, MAX_REQUEST_ATTEMPTS), status,
                        provider.get("classification", transport[0] if transport else "unknown"),
                        elapsed, retryable,
                    )
                if delay is None:
                    return response
                logger.warning(
                    "Gemini request failed for Investigation %s (HTTP %s, %s); retrying attempt %s/%s after %.2f s.",
                    request.investigation_index, status, provider.get("classification", "unknown"),
                    attempt_number + 1, min(self.settings.max_attempts, MAX_REQUEST_ATTEMPTS), delay,
                )
                time.sleep(delay)

        status_code, response_bytes, transport_error, _ = post_with_retry(payload)
        schema_rejected = False
        if (schema_used and status_code == 400 and self._is_schema_rejection(response_bytes)
                and len(attempts) < min(self.settings.max_attempts, MAX_REQUEST_ATTEMPTS)):
            schema_rejected = True
            schema_used = False
            payload = self._video_payload(request, video_bytes, generation_config, include_schema=False)
            status_code, response_bytes, transport_error, _ = post_with_retry(payload)
        elapsed = time.perf_counter() - started
        diagnostics_payload = self._request_diagnostics(
            request, attempts, elapsed, response_bytes, key,
        )
        if transport_error is not None:
            status, error = transport_error
            return self._failed(request, status, started, error, elapsed=elapsed,
                response_schema_used=schema_used, response_schema_rejected=schema_rejected,
                generation_config=generation_config, provider_diagnostics_json=diagnostics_payload)
        if status_code != 200:
            safe_response = self._sanitize_provider_error(response_bytes, key)
            provider = self._provider_error_details(response_bytes, key)
            reason = provider.get("message") or provider.get("status") or "provider error"
            safe_reason = " ".join(str(reason).split())[:240]
            error = f"Gemini API returned HTTP {status_code} — {safe_reason}."
            return self._failed(request, "runtime_error", started, error,
                elapsed=elapsed, response_schema_used=schema_used,
                response_schema_rejected=schema_rejected, generation_config=generation_config,
                provider_response_json=safe_response, provider_diagnostics_json=diagnostics_payload)
        logger.info(
            "Gemini request succeeded for Investigation %s on attempt %s/%s.",
            request.investigation_index, len(attempts),
            min(self.settings.max_attempts, MAX_REQUEST_ATTEMPTS),
        )
        result = self._result_from_response(
            request, response_bytes, elapsed, generation_config,
            schema_used=schema_used, schema_rejected=schema_rejected,
        )
        from dataclasses import replace
        return GeminiRunResult(
            replace(result.raw_result, provider_diagnostics_json=diagnostics_payload),
            result.diagnostics,
        )

    def _video_payload(
        self, request: VLMRequest | VLMReviewRequest, video_bytes: bytes,
        generation_config: dict[str, Any], *, include_schema: bool,
    ) -> dict[str, Any]:
        """Build one Gemini request with the complete video and optional selected ticket crops."""
        config = dict(generation_config)
        if include_schema:
            config["responseSchema"] = GEMINI_RESPONSE_SCHEMA
        parts: list[dict[str, Any]] = [
            {"inlineData": {"mimeType": "video/mp4", "data": base64.b64encode(video_bytes).decode("ascii")}},
        ]
        for index, image in enumerate(request.ticket_images, start=1):
            try:
                image_bytes = image.image_path.read_bytes()
            except OSError as error:
                raise ValueError(f"Selected ticket image {index} could not be read.") from error
            if not image_bytes:
                raise ValueError(f"Selected ticket image {index} is empty.")
            image_label = "Ticket detail image" if isinstance(request, VLMReviewRequest) else "Ticket image"
            parts.append({"text": (
                f"{image_label} {index}; source video time "
                f"{image.source_timestamp_s:.3f}s; role={image.role}; section={image.section}."
            )})
            parts.append({"inlineData": {"mimeType": "image/png", "data": base64.b64encode(image_bytes).decode("ascii")}})
        parts.append({"text": request.prompt})
        return {
            "contents": [{"parts": parts}],
            "generationConfig": config,
        }

    def _post(self, payload: dict[str, Any], key: str):
        """POST JSON to Gemini without logging or returning the secret header."""
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = Request(self.endpoint_url, data=body, headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-goog-api-key": key,
        }, method="POST")
        try:
            with urlopen(request, timeout=self.request_timeout_s) as response:
                headers = getattr(response, "headers", {}) or {}
                return response.status, response.read(), None, dict(headers.items())
        except HTTPError as error:
            return error.code, error.read(), None, dict(error.headers.items()) if error.headers else {}
        except (TimeoutError, URLError) as error:
            if isinstance(error, TimeoutError) or (
                isinstance(error, URLError) and isinstance(error.reason, TimeoutError)
            ):
                return None, b"", ("timeout", "Gemini API request timed out."), {}
            return None, b"", ("runtime_error", "Gemini API request failed due to a network error."), {}
        except (OSError, HTTPException):
            return None, b"", ("runtime_error", "Gemini API request failed due to an I/O or HTTP error."), {}

    def _provider_error_details(self, response_bytes: bytes, api_key: str) -> dict[str, Any]:
        """Extract sanitized Gemini error, quota, delay, and usage metadata."""
        try:
            body = json.loads(self._sanitize_provider_error(response_bytes, api_key))
        except (json.JSONDecodeError, TypeError):
            body = {}
        error = body.get("error", {}) if isinstance(body, dict) else {}
        error = error if isinstance(error, dict) else {}
        details = error.get("details", [])
        details = details if isinstance(details, list) else []
        all_dicts = []
        def visit(value):
            if isinstance(value, dict):
                all_dicts.append(value)
                for nested in value.values():
                    visit(nested)
            elif isinstance(value, list):
                for nested in value:
                    visit(nested)
        visit(error)
        def find(names):
            return next((item[name] for item in all_dicts for name in names if name in item), None)
        quota_metric = find(("quotaMetric", "quota_metric"))
        quota_id = find(("quotaId", "quota_id"))
        quota_limit = find(("quotaValue", "quotaLimit", "quota_limit"))
        dimensions = find(("quotaDimensions", "quota_dimensions"))
        retry_delay = find(("retryDelay", "retry_delay"))
        if retry_delay is None:
            for item in details:
                if isinstance(item, dict) and item.get("@type", "").endswith("RetryInfo"):
                    retry_delay = item.get("retryDelay")
                    break
        message = error.get("message")
        provider_status = error.get("status")
        classification = self._quota_classification(quota_metric, quota_id, dimensions, message)
        result = {
            "provider_code": error.get("code"),
            "provider_status": provider_status,
            "message": message if isinstance(message, str) else None,
            "quota_metric": quota_metric,
            "quota_id": quota_id,
            "quota_limit": quota_limit,
            "quota_dimensions": dimensions,
            "retry_delay": retry_delay,
            "classification": classification,
            "usage_metadata": body.get("usageMetadata") if isinstance(body, dict) else None,
        }
        if error:
            result["sanitized_error"] = self._sanitize_provider_error(response_bytes, api_key)
        return result

    @staticmethod
    def _quota_classification(metric, quota_id, dimensions, message) -> str:
        """Classify quota failures only when provider metadata supports it."""
        text = " ".join(str(value) for value in (metric, quota_id, dimensions, message) if value).casefold()
        if any(term in text for term in ("tokenperday", "tokens_per_day", "tpd")):
            return "TPD/daily_token_limit"
        if any(term in text for term in ("perday", "per_day", "daily", "per day", "rpd")):
            return "RPD/daily_quota"
        if any(term in text for term in ("tokenperminute", "tokens_per_minute", "tpm")):
            return "TPM"
        if any(term in text for term in ("requestperminute", "requests_per_minute", "rpm")):
            return "RPM"
        if any(term in text for term in ("capacity", "overloaded", "unavailable")):
            return "provider_capacity"
        return "unknown"

    @staticmethod
    def _daily_quota_exhausted(provider: dict[str, Any]) -> bool:
        """Stop retrying when explicit quota metadata identifies a daily limit."""
        if provider.get("classification") in ("RPD/daily_quota", "TPD/daily_token_limit"):
            return True
        status = str(provider.get("provider_status", "")).casefold()
        message = str(provider.get("message", "")).casefold()
        return status == "resource_exhausted" and any(
            term in message for term in ("per day", "daily quota", "requests per day", "daily limit")
        )

    @staticmethod
    def _retry_after_seconds(headers: dict[str, str], provider: dict[str, Any]) -> float | None:
        """Read Retry-After or Gemini RetryInfo metadata when supplied."""
        value = next((v for k, v in headers.items() if k.casefold() == "retry-after"), None)
        header_delay = None
        if value is not None:
            try:
                parsed = float(value)
                if math.isfinite(parsed):
                    header_delay = max(0.0, parsed)
            except ValueError:
                try:
                    moment = parsedate_to_datetime(value)
                    if moment.tzinfo is None:
                        moment = moment.replace(tzinfo=timezone.utc)
                    header_delay = max(0.0, (moment - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    pass
        value = provider.get("retry_delay")
        provider_delay = None
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value)):
            provider_delay = max(0.0, float(value))
        elif isinstance(value, str):
            match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)s\s*", value)
            if match:
                provider_delay = float(match.group(1))
        delays = [delay for delay in (header_delay, provider_delay) if delay is not None]
        return max(delays) if delays else None

    def _request_diagnostics(self, request, attempts, elapsed, final_body, api_key) -> str:
        """Serialize non-secret attempt, request-shape, and final provider diagnostics."""
        final_provider = self._provider_error_details(final_body, api_key) if final_body else None
        if final_provider and not any(final_provider.get(key) for key in (
            "provider_code", "provider_status", "message", "sanitized_error",
        )):
            final_provider = None
        payload = {
            "provider": "gemini",
            "model": self.model_name,
            "api_method": "models.generateContent",
            "investigation_id": request.investigation_index,
            "request_timestamp_utc": attempts[0]["request_timestamp_utc"] if attempts else None,
            "video_clip_duration_s": round(request.source_end_s - request.source_start_s, 3),
            "image_crop_count": len(request.ticket_images),
            "max_attempts": min(self.settings.max_attempts, MAX_REQUEST_ATTEMPTS),
            "attempt_count": len(attempts),
            "total_api_request_s": round(sum(item["duration_s"] for item in attempts), 3),
            "total_backoff_s": round(sum(item["retry_delay_s"] or 0 for item in attempts), 3),
            "total_elapsed_s": round(elapsed, 3),
            "attempts": attempts,
            "final_provider_error": final_provider,
            "sdk_retries_enabled": False,
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

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
        provider_diagnostics_json: str | None = None,
    ) -> RawVLMResult:
        """Build the existing raw VLM contract without storing provider secrets."""
        return RawVLMResult(
            execution_status=status, investigation_index=request.investigation_index,
            source_video=request.source_video, video_clip_path=request.video_clip_path,
            source_start_s=request.source_start_s, source_end_s=request.source_end_s,
            model_name=self.model_name, model_revision=model_version, prompt_version=request.prompt_version,
            raw_text=raw_text, inference_time_s=elapsed, ttft_s=None,
            generated_tokens=generated_tokens, error=error,
            provider_diagnostics_json=provider_diagnostics_json,
        )

    def _failed(
        self, request: VLMRequest | VLMReviewRequest, status: ExecutionStatus,
        started: float, error: str, *,
        elapsed: float | None = None, provider_response_json: str | None = None,
        generation_config: dict[str, Any] | None = None, response_schema_used: bool = False,
        response_schema_rejected: bool = False,
        provider_diagnostics_json: str | None = None,
    ) -> GeminiRunResult:
        """Represent an expected request failure and preserve safe diagnostics."""
        duration = time.perf_counter() - started if elapsed is None else elapsed
        raw = self._make_raw(
            request, status, duration, error=error,
            provider_diagnostics_json=provider_diagnostics_json,
        )
        diagnostics = GeminiRunDiagnostics(
            provider_response_json=provider_response_json, model_version=None,
            prompt_token_count=None, candidate_token_count=None, total_token_count=None,
            finish_reason=None, generation_config=generation_config or self._generation_config(),
            response_schema_used=response_schema_used, response_schema_rejected=response_schema_rejected,
            elapsed_s=duration,
        )
        return GeminiRunResult(raw, diagnostics)
