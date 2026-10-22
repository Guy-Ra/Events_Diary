"""Run one inspectable Gemini request for the complete Investigation video.

This validation/demo tool uses the production Gemini adapter and parser. Its
middle-frame preview is saved only for human inspection and is never sent to
Gemini. The only retry is up to two additional attempts after HTTP 503.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2

from events_diary.config import load_vlm_api_settings
from events_diary.gemini_api import MAX_INLINE_VIDEO_BYTES, GeminiAPIRuntime
from events_diary.vlm import VLMInvestigationResult, VLMRequest
from events_diary.vlm_parsing import parse_vlm_result
from events_diary.vlm_prompts import build_vlm_prompt

ROOT = Path(__file__).resolve().parents[2]
SOURCE_VIDEO = ROOT / "data/videos/ticket_scroll_test.mp4"
OUTPUT_DIR = ROOT / "data/debug/vlm/gemini_api_validation"
RUN_DIR = OUTPUT_DIR / "latest_run"
SOURCE_START_S = 0.0
SOURCE_END_S = 8.4


def _write_json(path: Path, payload: object) -> None:
    """Write indented UTF-8 JSON to a validation artifact path."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _video_metadata(path: Path) -> dict[str, object]:
    """Read video duration metadata and file size without sampling frames."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError("Unable to open the Investigation source video.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    return {
        "path": str(path), "duration_seconds": frames / fps if fps else None,
        "fps": fps, "frame_count": frames, "width": width, "height": height,
        "file_size_bytes": path.stat().st_size,
    }


def _create_preview(source: Path, output: Path, timestamp_s: float) -> None:
    """Save one middle frame for human inspection only, never for model input."""
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise RuntimeError("Unable to open the Investigation video for its preview.")
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000)
        success, frame = capture.read()
        if not success or not cv2.imwrite(str(output), frame):
            raise RuntimeError("Unable to create the human-inspection preview image.")
    finally:
        capture.release()


def _parsed_payload(
    result: VLMInvestigationResult,
    *,
    model_version: str | None,
    finish_reason: str | None,
    usage: dict[str, object],
) -> dict[str, object]:
    """Serialize parser output and provenance without adding interpretation."""
    raw = result.raw_result
    diagnostics = result.parse_diagnostics
    return {
        "raw_result": {
            "execution_status": raw.execution_status,
            "investigation_index": raw.investigation_index,
            "source_video": str(raw.source_video),
            "video_clip_path": str(raw.video_clip_path),
            "source_start_s": raw.source_start_s,
            "source_end_s": raw.source_end_s,
            "model_name": raw.model_name,
            "model_revision": raw.model_revision,
            "prompt_version": raw.prompt_version,
            "raw_text": raw.raw_text,
            "inference_time_s": raw.inference_time_s,
            "ttft_s": raw.ttft_s,
            "generated_tokens": raw.generated_tokens,
            "error": raw.error,
        },
        "observations": [
            {
                "field_name": item.field_name,
                "raw_value": item.raw_value,
                "status": item.status,
                "investigation_index": item.investigation_index,
                "source_video": str(item.source_video),
                "source_start_s": item.source_start_s,
                "source_end_s": item.source_end_s,
                "model_name": item.model_name,
                "model_revision": item.model_revision,
                "prompt_version": item.prompt_version,
                "evidence_time_s": item.evidence_time_s,
            }
            for item in result.observations
        ],
        "parse_diagnostics": {
            "status": diagnostics.status if diagnostics else None,
            "json_format": diagnostics.json_format if diagnostics else None,
            "missing_keys": diagnostics.missing_keys if diagnostics else (),
            "unknown_keys": diagnostics.unknown_keys if diagnostics else (),
            "error": diagnostics.error if diagnostics else None,
        },
        "provider_diagnostics": {
            "model_version": model_version,
            "finish_reason": finish_reason,
            "usage": usage,
        },
    }


def _provider_usage_metadata(response_json: str | None) -> dict[str, object]:
    """Return provider-reported usage details without interpreting the values."""
    if response_json is None:
        return {}
    try:
        body = json.loads(response_json)
    except json.JSONDecodeError:
        return {}
    usage = body.get("usageMetadata") if isinstance(body, dict) else None
    return usage if isinstance(usage, dict) else {}


def main(argv: list[str] | None = None) -> int:
    """Save inspectable inputs, call Gemini with 503-only retries, and report results."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-only", action="store_true",
        help="load and display safe project configuration without making a network request",
    )
    args = parser.parse_args(argv)
    settings = load_vlm_api_settings()
    if args.config_only:
        print(f"Provider: {settings.provider}")
        print(f"Model: {settings.model_name}")
        print(f"Base URL: {settings.base_url}")
        print(f"API version: {settings.api_version}")
        print(f"Timeout: {settings.request_timeout_s:g} seconds")
        print(f"API key configured: {'yes' if settings.api_key_configured else 'no'}")
        return 0
    if not settings.api_key_configured:
        print("Set EVENTS_DIARY_VLM_API_KEY in the local .env file before running Gemini validation.")
        return 2
    if not SOURCE_VIDEO.is_file():
        print(f"Required continuous Investigation video is missing: {SOURCE_VIDEO}")
        return 1

    metadata = _video_metadata(SOURCE_VIDEO)
    if metadata["file_size_bytes"] > MAX_INLINE_VIDEO_BYTES:
        print("Video exceeds the 20 MiB inline limit; a Gemini Files API path is required. Stopping.")
        return 1
    if metadata["duration_seconds"] != SOURCE_END_S:
        print(f"Source video duration is not the expected complete [0.0, 8.4) clip: {metadata['duration_seconds']}")
        return 1

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    payload_video = RUN_DIR / "investigation_input.mp4"
    shutil.copyfile(SOURCE_VIDEO, payload_video)
    preview_image = RUN_DIR / "investigation_preview.png"
    _create_preview(SOURCE_VIDEO, preview_image, (SOURCE_START_S + SOURCE_END_S) / 2)
    prompt_text = build_vlm_prompt("fr24_vlm_v1")
    prompt_artifact = RUN_DIR / "prompt.txt"
    prompt_artifact.write_text(prompt_text, encoding="utf-8")

    request = VLMRequest(
        investigation_index=0,
        source_video=SOURCE_VIDEO,
        video_clip_path=payload_video,
        source_start_s=SOURCE_START_S,
        source_end_s=SOURCE_END_S,
        prompt=prompt_text,
        prompt_version="fr24_vlm_v1",
    )
    duration = SOURCE_END_S - SOURCE_START_S
    print("=== VLM INPUT ===")
    print(f"Source video: {SOURCE_VIDEO}")
    print("Investigation: 0 (ticket_scroll / Investigation 0)")
    print(f"Start: {SOURCE_START_S:.3f} s")
    print(f"End: {SOURCE_END_S:.3f} s")
    print(f"Duration: {duration:.3f} s")
    print(f"Payload video: {payload_video}")
    print(f"Preview image: {preview_image}")
    print(f"Prompt artifact: {prompt_artifact}")
    print(f"File size: {metadata['file_size_bytes']} bytes ({metadata['file_size_bytes'] / 1_000_000:.3f} MB)")
    print("Provider: Gemini")
    print(f"Requested model: {settings.model_name}")
    print(f"Prompt version: {request.prompt_version}")
    sys.stdout.flush()

    runtime = GeminiAPIRuntime(settings)
    retry_waits = (5, 10)
    total_started = time.perf_counter()
    total_api_runtime_s = 0.0
    execution = None
    successful_attempt = None
    for attempt in range(1, 4):
        print(f"Attempt {attempt}/3", flush=True)
        execution = runtime.run_with_diagnostics(request)
        total_api_runtime_s += execution.diagnostics.elapsed_s
        raw_attempt = execution.raw_result
        if raw_attempt.execution_status == "success":
            successful_attempt = attempt
            break
        is_http_503 = (
            raw_attempt.execution_status == "runtime_error"
            and raw_attempt.error == "Gemini API returned HTTP 503."
        )
        if not is_http_503 or attempt == 3:
            break
        time.sleep(retry_waits[attempt - 1])
    total_elapsed_s = time.perf_counter() - total_started
    if execution is None:
        raise RuntimeError("Gemini adapter did not produce an execution result.")

    raw_result = execution.raw_result
    parsed = parse_vlm_result(request, raw_result)
    raw_response_path = RUN_DIR / "raw_response.txt"
    raw_response_path.write_text(raw_result.raw_text or "", encoding="utf-8")
    diagnostics = execution.diagnostics
    usage = {
        "prompt_tokens": diagnostics.prompt_token_count,
        "candidate_tokens": diagnostics.candidate_token_count,
        "total_tokens": diagnostics.total_token_count,
        "provider_usage_metadata": _provider_usage_metadata(diagnostics.provider_response_json),
    }
    parsed_result_path = RUN_DIR / "parsed_result.json"
    _write_json(parsed_result_path, _parsed_payload(
        parsed,
        model_version=diagnostics.model_version,
        finish_reason=diagnostics.finish_reason,
        usage=usage,
    ))

    print("=== GEMINI EXECUTION ===")
    print(f"Runtime status: {raw_result.execution_status}")
    print(f"Successful attempt: {successful_attempt if successful_attempt is not None else 'none'}")
    print(f"Requested model: {settings.model_name}")
    print(f"Returned model version: {diagnostics.model_version}")
    print(f"Elapsed time: {diagnostics.elapsed_s:.3f} seconds")
    print(f"Finish reason: {diagnostics.finish_reason}")
    print(f"Usage metadata: {json.dumps(usage, ensure_ascii=False, sort_keys=True)}")

    print("\n=== RAW GEMINI RESPONSE ===\n")
    if raw_result.raw_text is not None:
        sys.stdout.write(raw_result.raw_text)
        if not raw_result.raw_text.endswith("\n"):
            sys.stdout.write("\n")

    observed = sum(item.status == "observed" for item in parsed.observations)
    unavailable = sum(item.status == "unavailable" for item in parsed.observations)
    ambiguous = sum(item.status == "ambiguous" for item in parsed.observations)
    parse_diagnostics = parsed.parse_diagnostics
    parser_valid = bool(
        parse_diagnostics
        and parse_diagnostics.status in ("valid_json", "recovered_markdown_fence")
        and not parse_diagnostics.missing_keys
    )
    print("\n=== PARSED VLM RESULT ===")
    print(f"{'FIELD':<40} {'STATUS':<14} VALUE")
    print("-" * 88)
    for item in parsed.observations:
        value = "null" if item.raw_value is None else str(item.raw_value)
        print(f"{item.field_name:<40} {item.status:<14} {value}")
    print(f"Observed: {observed}")
    print(f"Unavailable: {unavailable}")
    print(f"Ambiguous: {ambiguous}")

    json_format = parse_diagnostics.json_format if parse_diagnostics else None
    print("\n=== PARSER DIAGNOSTICS ===")
    print(f"Strict JSON valid: {'yes' if json_format == 'strict_json' and parser_valid else 'no'}")
    print(f"Fence recovery used: {'yes' if json_format == 'markdown_fence' else 'no'}")
    print(f"Missing expected keys: {parse_diagnostics.missing_keys if parse_diagnostics else ()}")
    print(f"Unknown keys: {parse_diagnostics.unknown_keys if parse_diagnostics else ()}")
    print(f"Other diagnostics: {parse_diagnostics}")
    print(f"Provider/runtime error: {raw_result.error}")
    print(f"Schema used: {diagnostics.response_schema_used}")
    print(f"Schema rejected: {diagnostics.response_schema_rejected}")

    print("\n=== RUN SUMMARY ===")
    print(f"Analyzed source: {SOURCE_VIDEO}")
    print("Investigation: 0")
    print(f"Analyzed interval: {SOURCE_START_S:.3f} - {SOURCE_END_S:.3f} s")
    print(f"Duration sent: {duration:.3f} s")
    print(f"Exact payload copy: {payload_video}")
    print(f"Preview image: {preview_image}")
    print(f"Prompt: {prompt_artifact}")
    print(f"Requested model: {settings.model_name}")
    print(f"Returned model: {diagnostics.model_version}")
    print(f"Successful attempt: {successful_attempt if successful_attempt is not None else 'none'}")
    print(f"Observed fields: {observed} / {len(parsed.observations)}")
    print(f"Unavailable: {unavailable}")
    print(f"Ambiguous: {ambiguous}")
    print(f"Total API runtime: {total_api_runtime_s:.3f} seconds")
    print(f"Total elapsed including backoff: {total_elapsed_s:.3f} seconds")
    print("OCR used: no")
    print("Fusion used: no")
    print(f"Raw response artifact: {raw_response_path}")
    print(f"Parsed result artifact: {parsed_result_path}")
    return 0 if raw_result.execution_status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
