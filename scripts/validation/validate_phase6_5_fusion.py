"""Run one full real-video Phase 6.5 fusion validation and save inspectable evidence.

This manual diagnostic composes existing production Investigation selection,
RapidOCR extraction, independent Gemini Pass 1, deterministic field comparison,
temporal awareness, Gemini Pass 2 review, and Phase 6.5f resolution. It never
injects expected values or edits source observations. The current pairwise fusion
API accepts one OCR observation per field; when production selects multiple
physical representatives, all occurrences are preserved and displayed, while
the latest usable occurrence is projected into that pairwise API. This adapter
choice is reported and is not a production multi-view merge policy.

Gemini calls use the configured model and the existing adapter, with at most
three total attempts for HTTP 503 and request timeouts. A failed or incomplete
VLM pass writes the evidence gathered so far and stops before dependent stages
are fabricated. Artifacts contain no API credentials.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from events_diary.config import UIConfig, load_vlm_api_settings
from events_diary.evidence_candidate_timeline import build_evidence_candidate_timeline
from events_diary.frame_materialization import (
    VideoFrameResolver,
    materialize_evidence_candidates,
)
from events_diary.fusion import (
    FusedFieldObservation,
    FusedInvestigationResult,
    FusionSummary,
    apply_temporal_awareness,
    compare_field_observations,
    resolve_investigation,
)
from events_diary.gemini_api import MAX_INLINE_VIDEO_BYTES, GeminiAPIRuntime
from events_diary.ocr_fields import CANONICAL_LABELS, extract_ocr_fields_batch
from events_diary.representative_selection import select_investigation_representatives
from events_diary.ticket_ocr import run_ocr_for_representatives
from events_diary.vlm import VLMRequest
from events_diary.vlm_parsing import parse_vlm_result
from events_diary.vlm_prompts import build_vlm_prompt
from events_diary.vlm_review import (
    FR24_VLM_REVIEW_V1,
    REVIEW_FIELD_NAMES,
    REVIEW_ROUTE_FIELDS,
    build_vlm_review_request,
    parse_vlm_review_result,
)
from scripts.evaluation.evaluate_ticket_stable_views import _build_contexts

VIDEO = ROOT / "data/videos/ticket_scroll_test.mp4"
INVESTIGATION = 0
START_S = 0.0
END_S = 8.4
RUN_DIR = ROOT / "data/debug/fusion/phase_6_5/latest_run"
RETRY_WAITS = (5, 10)


def _json_write(path: Path, payload: object) -> None:
    """Write indented UTF-8 JSON with a trailing newline."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _jsonable(value: object) -> object:
    """Convert paths, frozen records, and mappings into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _field_value(value: object) -> str:
    """Format a field value for readable report tables without losing false/null."""
    if value is None:
        return "null"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    """Render a compact Markdown-style pipe table for console and text report."""
    widths = [len(header) for header in headers]
    rendered = [[_field_value(value) for value in row] for row in rows]
    for row in rendered:
        widths = [max(width, len(value)) for width, value in zip(widths, row, strict=True)]
    line = lambda row: "| " + " | ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)) + " |"
    return "\n".join((line(headers), line(tuple("-" * width for width in widths)), *(line(row) for row in rendered)))


def _write_report_section(lines: list[str], title: str, content: str) -> None:
    """Append one named section and its complete table to the report."""
    lines.extend((f"\n=== {title} ===", content))


def _ocr_record(item) -> dict[str, object]:
    """Serialize one raw OCR field occurrence and its physical provenance."""
    return {
        "field_name": item.canonical_label,
        "status": item.status,
        "raw_label": item.raw_label_text,
        "raw_value": item.raw_value_text,
        "candidate_index": item.provenance.candidate_index,
        "frame_index": item.provenance.frame_index,
        "source_time_s": item.provenance.actual_timestamp_seconds,
        "engine": item.provenance.engine_name,
        "preprocessing": item.provenance.preprocessing_name,
        "label_detection_index": item.label_detection_index,
        "value_detection_index": item.value_detection_index,
        "reason": item.reason,
    }


def _vlm_record(item) -> dict[str, object]:
    """Serialize a source-specific parsed Pass 1 field observation."""
    return {"field_name": item.field_name, "status": item.status, "raw_value": item.raw_value,
            "evidence_time_s": item.evidence_time_s}


def _review_record(item) -> dict[str, object]:
    """Serialize a source-specific Pass 2 review without normalizing its value."""
    return {
        "field_name": item.field_name,
        "reviewed_value": item.reviewed_value,
        "review_status": item.review_status,
        "evidence_quality": item.evidence_quality,
        "evidence_time_s": item.evidence_time_s,
        "notes": item.notes,
    }


def _raw_payload(parsed, diagnostics, successful_attempt, attempt_records, total_api_s, total_elapsed_s) -> dict[str, object]:
    """Serialize VLM output, parser/provider measurements, and safe runtime details."""
    raw = parsed.raw_result
    return {
        "raw_result": {
            "execution_status": raw.execution_status,
            "investigation_index": raw.investigation_index,
            "source_video": str(raw.source_video),
            "video_clip_path": str(raw.video_clip_path),
            "source_interval_s": [raw.source_start_s, raw.source_end_s],
            "model_name": raw.model_name,
            "model_revision": raw.model_revision,
            "prompt_version": raw.prompt_version,
            "raw_text": raw.raw_text,
            "inference_time_s": raw.inference_time_s,
            "ttft_s": raw.ttft_s,
            "generated_tokens": raw.generated_tokens,
            "error": raw.error,
        },
        "observations": [_vlm_record(item) for item in parsed.observations],
        "visual_route": _jsonable(parsed.visual_route),
        "parse_diagnostics": _jsonable(parsed.parse_diagnostics),
        "provider_diagnostics": {
            "model_version": diagnostics.model_version,
            "finish_reason": diagnostics.finish_reason,
            "prompt_token_count": diagnostics.prompt_token_count,
            "candidate_token_count": diagnostics.candidate_token_count,
            "total_token_count": diagnostics.total_token_count,
            "usage_metadata": {
                "prompt_tokens": diagnostics.prompt_token_count,
                "candidate_tokens": diagnostics.candidate_token_count,
                "total_tokens": diagnostics.total_token_count,
            },
            "attempt_count": len(attempt_records),
            "successful_attempt": successful_attempt,
            "attempts": attempt_records,
            "attempt_runtime_s": [item["elapsed_api_s"] for item in attempt_records],
            "total_api_request_time_s": total_api_s,
            "total_elapsed_including_backoff_s": total_elapsed_s,
            "final_attempt_elapsed_s": diagnostics.elapsed_s,
        },
    }


def _run_gemini(runtime, request, parser):
    """Retry only HTTP 503 and request timeouts, at most three total attempts."""
    started = time.perf_counter()
    attempt_records = []
    execution = None
    successful_attempt = None
    for attempt in range(1, 4):
        print(f"Attempt {attempt}/3", flush=True)
        execution = runtime.run_with_diagnostics(request)
        raw = execution.raw_result
        elapsed = execution.diagnostics.elapsed_s
        returned_model = execution.diagnostics.model_version
        if raw.execution_status == "success":
            attempt_records.append({
                "attempt": attempt, "status": "success", "failure_type": None,
                "elapsed_api_s": elapsed, "backoff_s": 0,
                "returned_model_version": returned_model,
            })
            print(
                f"Attempt status: success\nFailure type: none\n"
                f"Elapsed request time: {elapsed:.3f}s\n"
                f"Returned model version: {returned_model}", flush=True,
            )
            successful_attempt = attempt
            break
        failure_type = (
            "503" if raw.error == "Gemini API returned HTTP 503."
            else "timeout" if raw.execution_status == "timeout" or raw.error == "Gemini API request timed out."
            else "other"
        )
        retryable = failure_type in ("503", "timeout")
        backoff_s = RETRY_WAITS[attempt - 1] if retryable and attempt < 3 else 0
        attempt_records.append({
            "attempt": attempt,
            "status": raw.execution_status,
            "failure_type": failure_type,
            "elapsed_api_s": elapsed,
            "backoff_s": backoff_s,
            "error": raw.error,
            "returned_model_version": returned_model,
        })
        print(f"Failure type: {failure_type}\nElapsed request time: {elapsed:.3f}s", flush=True)
        if not retryable or attempt == 3:
            break
        print(f"Backoff: {backoff_s} seconds", flush=True)
        time.sleep(backoff_s)
    if execution is None:
        raise RuntimeError("Gemini runtime returned no attempt result.")
    total_elapsed_s = time.perf_counter() - started
    total_api_s = sum(item["elapsed_api_s"] for item in attempt_records)
    print("=== REQUEST ATTEMPT SUMMARY ===")
    print(f"Number of attempts: {len(attempt_records)}")
    for record in attempt_records:
        print(
            f"Attempt {record['attempt']}: status={record['status']}; "
            f"failure_type={record['failure_type'] or 'none'}; "
            f"elapsed_api_time={record['elapsed_api_s']:.3f}s; "
            f"returned_model={record['returned_model_version']}; backoff={record['backoff_s']}s"
        )
    print(f"Total API request time: {total_api_s:.3f}s")
    print(f"Total elapsed including backoff: {total_elapsed_s:.3f}s")
    return (
        parser(request, execution.raw_result), execution.diagnostics,
        successful_attempt, attempt_records, total_api_s, total_elapsed_s,
    )


def _show_configuration(settings) -> str:
    """Print and return the required redacted active VLM configuration block."""
    block = "\n".join((
        "=== ACTIVE VLM CONFIGURATION ===",
        f"Provider: {settings.provider}",
        f"Configured model: {settings.model_name}",
        f"API version: {settings.api_version}",
        f"Timeout: {settings.request_timeout_s:g}",
        f"API key configured: {'yes' if settings.api_key_configured else 'no'}",
    ))
    print(block, flush=True)
    return block


def _inspect_pass2_request(runtime, request) -> dict[str, object]:
    """Persist and print safe structural metadata for the exact Pass 2 payload."""
    video_bytes = request.video_clip_path.read_bytes()
    payload = runtime._video_payload(
        request, video_bytes, runtime._generation_config(), include_schema=False,
    )
    serialized = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    parts = payload["contents"][0]["parts"]
    video_part = next(part["inlineData"] for part in parts if "inlineData" in part)
    pass1_request = VLMRequest(
        investigation_index=request.investigation_index,
        source_video=request.source_video,
        video_clip_path=request.video_clip_path,
        source_start_s=request.source_start_s,
        source_end_s=request.source_end_s,
        prompt=build_vlm_prompt("fr24_vlm_v1"),
        prompt_version="fr24_vlm_v1",
    )
    pass1_payload = runtime._video_payload(
        pass1_request, video_bytes, runtime._generation_config(), include_schema=True,
    )
    pass1_schema = pass1_payload["generationConfig"]["responseSchema"]
    pass1_schema_size = len(json.dumps(
        pass1_schema, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8"))
    summary: dict[str, object] = {
        "endpoint_model": runtime.model_name,
        "endpoint_path": runtime.endpoint_url.split("?", 1)[0],
        "prompt_version": request.prompt_version,
        "video_mime_type": video_part["mimeType"],
        "video_byte_size": len(video_bytes),
        "prompt_character_count": len(request.prompt),
        "prompt_utf8_byte_count": len(request.prompt.encode("utf-8")),
        "review_field_count": len(request.context.fields),
        "json_response_mode": payload["generationConfig"].get("responseMimeType"),
        "json_response_mode_location": "generationConfig.responseMimeType",
        "response_schema_present": False,
        "response_schema_serialized_byte_size": 0,
        "generation_config_keys": sorted(payload["generationConfig"]),
        "top_level_request_keys": sorted(payload),
        "request_json_serializable": json.loads(serialized) == payload,
        "request_json_byte_size_without_headers": len(serialized),
        "pass1_comparison": {
            "prompt_version": pass1_request.prompt_version,
            "prompt_character_count": len(pass1_request.prompt),
            "prompt_utf8_byte_count": len(pass1_request.prompt.encode("utf-8")),
            "response_schema_present": True,
            "response_schema_serialized_byte_size": pass1_schema_size,
            "schema_field_count": len(pass1_schema["properties"]),
            "generation_config_keys": sorted(pass1_payload["generationConfig"]),
            "video_mime_type": pass1_payload["contents"][0]["parts"][0]["inlineData"]["mimeType"],
            "top_level_request_keys": sorted(pass1_payload),
        },
        "pass1_pass2_structural_differences": [
            "prompt version and prompt/context content",
            "Pass 1 includes its responseSchema; Pass 2 uses JSON mode without responseSchema",
        ],
    }
    _json_write(RUN_DIR / "pass2_request_summary.json", summary)
    print("=== PASS 2 REQUEST STRUCTURE ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def _ocr_and_fusion_inputs():
    """Run existing Investigation selection, physical OCR, and canonical parsing."""
    metadata, anomalies, contexts = _build_contexts("ticket_scroll")
    if anomalies:
        raise RuntimeError(f"Unexpected lifecycle anomalies: {anomalies}")
    if metadata.path.resolve() != VIDEO.resolve() or metadata.duration_seconds != END_S:
        raise RuntimeError("Video metadata does not match the requested complete Investigation.")
    window, plan, views, _ = next(item for item in contexts if item[0].investigation_index == INVESTIGATION)
    candidates = build_evidence_candidate_timeline(window=window, sampling_plan=plan, stable_views=views)
    config = UIConfig()
    with VideoFrameResolver(VIDEO, metadata=metadata) as resolver:
        materialized = materialize_evidence_candidates(
            video_path=VIDEO, candidates=candidates, ui_config=config, resolver=resolver,
        )
        selection = select_investigation_representatives(
            window=window, sampling_plan=plan, candidates=candidates,
            materialized_candidates=materialized, stable_views=views,
        )
        raw_results = run_ocr_for_representatives(
            video_path=VIDEO, investigation_index=INVESTIGATION,
            selections=selection.selections, candidates=candidates,
            materialized_candidates=materialized, ui_config=config,
            resolver=resolver,
        )
    observations = extract_ocr_fields_batch(raw_results)
    flattened = tuple(item for group in observations for item in group)
    if any(item.provenance.investigation_index != INVESTIGATION for item in flattened):
        raise RuntimeError("OCR returned evidence for the wrong Investigation.")
    by_name: dict[str, list] = defaultdict(list)
    for item in flattened:
        by_name[item.canonical_label].append(item)
    selected_by_name = {}
    for name, occurrences in by_name.items():
        usable = [item for item in occurrences if item.status == "extracted" and item.raw_value_text is not None]
        pool = usable or occurrences
        selected_by_name[name] = max(pool, key=lambda item: (item.provenance.actual_timestamp_seconds, item.provenance.candidate_index))
    return metadata, raw_results, flattened, by_name, selected_by_name


def _ocr_table(by_name):
    """Build the complete OCR canonical table, including every selected occurrence."""
    rows = []
    usable = unavailable = 0
    for name in CANONICAL_LABELS:
        items = by_name.get(name, [])
        if not items:
            rows.append((name, "unavailable", "null", "no selected OCR occurrence"))
            unavailable += 1
            continue
        for item in items:
            status = item.status
            value = item.raw_value_text if item.raw_value_text is not None else "null"
            provenance = (
                f"candidate={item.provenance.candidate_index}; frame={item.provenance.frame_index}; "
                f"t={item.provenance.actual_timestamp_seconds:.3f}s; label={item.raw_label_text!r}; reason={item.reason}"
            )
            rows.append((name, status, value, provenance))
            if status == "extracted" and item.raw_value_text is not None:
                usable += 1
            else:
                unavailable += 1
    return rows, usable, unavailable


def _pass1_table(observations):
    """Build complete Pass 1 output rows in canonical key order."""
    by_name = {item.field_name: item for item in observations}
    return [(name, by_name[name].status, _field_value(by_name[name].raw_value)) for name in REVIEW_FIELD_NAMES]


def _crosscheck_rows(baseline_fields, fused_fields):
    """Render all fields with raw and normalized values and both statuses."""
    rows = []
    for name, item in fused_fields.items():
        ocr_raw = item.ocr_observation.raw_value_text if item.ocr_observation else None
        vlm_raw = item.vlm_observation.raw_value if item.vlm_observation else None
        rows.append((name, ocr_raw, vlm_raw, item.ocr_normalized_value,
                     item.vlm_normalized_value, baseline_fields[name].status, item.status,
                     item.diagnostics[0] if item.diagnostics else ""))
    return rows


def _review_table(review_result):
    """Build every Pass 2 field row without dropping null/abstention results."""
    fields = review_result.reviewed_fields
    return [
        (name, fields[name].reviewed_value if name in fields else None,
         fields[name].review_status if name in fields else "missing",
         fields[name].evidence_quality if name in fields else "missing",
         fields[name].evidence_time_s if name in fields else None,
         fields[name].notes if name in fields else "missing observation")
        for name in REVIEW_FIELD_NAMES
    ]


def _final_table(result):
    """Build complete final-resolution rows including source review evidence."""
    rows = []
    for name, item in result.fields.items():
        review = item.vlm_review_observation
        rows.append((
            name,
            item.fused_observation.ocr_observation.raw_value_text if item.fused_observation.ocr_observation else None,
            item.fused_observation.vlm_observation.raw_value if item.fused_observation.vlm_observation else None,
            item.fused_observation.ocr_normalized_value,
            item.fused_observation.vlm_normalized_value,
            item.fused_observation.status,
            review.reviewed_value if review else None,
            review.review_status if review else None,
            review.evidence_quality if review else None,
            review.evidence_time_s if review else None,
            item.temporal_status,
            item.final_status,
            item.final_value,
            item.resolution_reason,
        ))
    return rows


def _effect(item):
    """Describe the deterministic Pass 2 effect category for one final field."""
    status = item.fused_observation.status
    review = item.vlm_review_observation
    if status == "agreement":
        return "unchanged agreement" if item.final_status == "resolved" else "agreement challenged"
    if status == "conflict":
        return "conflict resolved" if item.final_status == "resolved" else "conflict unresolved"
    if status == "temporal_mismatch":
        return "temporal mismatch resolved" if item.final_status == "resolved" else "temporal mismatch unresolved"
    if status == "ocr_only":
        return "OCR-only confirmed" if item.final_status == "resolved_single_source" else "OCR-only unresolved"
    if status == "vlm_only":
        return "VLM-only confirmed" if item.final_status == "resolved_single_source" else "VLM-only unresolved"
    if status == "unavailable" and item.final_status == "resolved_single_source":
        return "Pass 2 found a new value"
    if status == "unavailable":
        return "still unavailable"
    if review and review.reviewed_value is not None:
        return "Pass 2 found a new value"
    return "unresolved after review"


def _interesting_cases(result):
    """Select only fields matching the nine requested evidence-pattern categories."""
    categories = (
        ("agreement challenged", lambda f: f.fused_observation.status == "agreement" and f.final_status == "unresolved"),
        ("conflict resolved", lambda f: f.fused_observation.status == "conflict" and f.final_status == "resolved"),
        ("conflict unresolved", lambda f: f.fused_observation.status == "conflict" and f.final_status == "unresolved"),
        ("temporal mismatch resolved", lambda f: f.fused_observation.status == "temporal_mismatch" and f.final_status == "resolved"),
        ("temporal mismatch unresolved", lambda f: f.fused_observation.status == "temporal_mismatch" and f.final_status == "unresolved"),
        ("OCR-only confirmed", lambda f: f.fused_observation.status == "ocr_only" and f.final_status == "resolved_single_source"),
        ("VLM-only confirmed", lambda f: f.fused_observation.status == "vlm_only" and f.final_status == "resolved_single_source"),
        ("both initial sources unavailable; Pass 2 found value", lambda f: f.fused_observation.status == "unavailable" and f.final_status == "resolved_single_source"),
        ("Pass 2 third value", lambda f: f.vlm_review_observation is not None
            and f.vlm_review_observation.reviewed_value is not None
            and f.vlm_review_observation.evidence_quality == "clear"
            and f.vlm_review_observation.review_status in ("confirmed", "corrected")
            and f.vlm_review_observation.reviewed_value not in (
                f.fused_observation.ocr_normalized_value, f.fused_observation.vlm_normalized_value)
            and f.fused_observation.status in ("conflict", "agreement")),
    )
    return [(label, [field for field in result.fields.values() if predicate(field)]) for label, predicate in categories]


def _table_text(headers, rows):
    """Render a generic table using the common readable pipe layout."""
    return _table(tuple(headers), list(rows))


def _provider_summary(parsed, diagnostics, attempt, runtimes):
    """Return printable provider runtime details without raw credential-bearing state."""
    raw = parsed.raw_result
    return {
        "requested_model": raw.model_name,
        "returned_model": diagnostics.model_version,
        "successful_attempt": attempt,
        "attempt_runtime_s": runtimes,
        "runtime_s": diagnostics.elapsed_s,
        "finish_reason": diagnostics.finish_reason,
        "usage": {
            "prompt_tokens": diagnostics.prompt_token_count,
            "candidate_tokens": diagnostics.candidate_token_count,
            "total_tokens": diagnostics.total_token_count,
        },
        "execution_status": raw.execution_status,
        "parse_status": parsed.parse_diagnostics.status if parsed.parse_diagnostics else None,
    }


def main() -> int:
    """Run the bounded real Investigation validation, stopping on either VLM failure."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    settings = load_vlm_api_settings()
    config_block = _show_configuration(settings)
    if not settings.api_key_configured:
        raise RuntimeError("API key is not configured; no Gemini request was made.")
    if not VIDEO.is_file():
        raise FileNotFoundError(f"Validation video is missing: {VIDEO}")
    video_size = VIDEO.stat().st_size
    if video_size > MAX_INLINE_VIDEO_BYTES:
        raise RuntimeError("Complete Investigation video exceeds Gemini's inline-data size limit.")

    start = time.perf_counter()
    metadata, ocr_results, ocr_fields, ocr_by_name, chosen_ocr = _ocr_and_fusion_inputs()
    ocr_elapsed = time.perf_counter() - start
    ocr_rows, ocr_usable, ocr_unavailable = _ocr_table(ocr_by_name)
    print("\n=== OCR RESULT ===")
    print(_table(("FIELD", "STATUS", "RAW VALUE", "SOURCE TIME / PROVENANCE"), ocr_rows))
    print(f"usable OCR observations: {ocr_usable}")
    print(f"unavailable/non-usable OCR observations: {ocr_unavailable}")
    print(f"OCR production representatives: {len(ocr_results)}; runtime: {ocr_elapsed:.3f}s")
    _json_write(RUN_DIR / "ocr_result.json", {
        "video": str(VIDEO), "investigation_id": INVESTIGATION,
        "interval_s": [START_S, END_S], "ocr_runtime_s": ocr_elapsed,
        "selected_representative_count": len(ocr_results),
        "representatives": [_jsonable(result) for result in ocr_results],
        "field_occurrences": [_ocr_record(item) for item in ocr_fields],
        "fusion_projection": {
            "policy": "latest usable physical OCR occurrence per matching canonical field; if none usable, latest occurrence",
            "fields": {name: _ocr_record(item) for name, item in chosen_ocr.items()},
        },
        "usable_occurrence_count": ocr_usable,
        "unavailable_nonusable_occurrence_count": ocr_unavailable,
    })

    request = VLMRequest(
        investigation_index=INVESTIGATION,
        source_video=VIDEO,
        video_clip_path=VIDEO,
        source_start_s=START_S,
        source_end_s=END_S,
        prompt=build_vlm_prompt("fr24_vlm_v1"),
        prompt_version="fr24_vlm_v1",
    )
    runtime = GeminiAPIRuntime(settings)
    print("\n=== VLM PASS 1 ===")
    print(f"Requested model: {settings.model_name}")
    print(f"Video input: {VIDEO} [{START_S:.3f}, {END_S:.3f})")
    pass1, p1diag, p1attempt, p1records, p1api_s, pass1_elapsed = _run_gemini(runtime, request, parse_vlm_result)
    pass1_raw = pass1.raw_result
    (RUN_DIR / "vlm_pass1_raw.txt").write_text(pass1_raw.raw_text or "", encoding="utf-8")
    pass1_json = _raw_payload(pass1, p1diag, p1attempt, p1records, p1api_s, pass1_elapsed)
    _json_write(RUN_DIR / "vlm_pass1_result.json", pass1_json)
    p1_table = _pass1_table(pass1.observations) if len(pass1.observations) == len(REVIEW_FIELD_NAMES) else []
    print(f"Returned model: {p1diag.model_version}")
    print(f"Successful attempt: {p1attempt}")
    print(f"Runtime: {p1diag.elapsed_s:.3f}s (total including retries/backoff {pass1_elapsed:.3f}s)")
    print(f"Finish reason: {p1diag.finish_reason}")
    print(f"Usage metadata: {json.dumps(pass1_json['provider_diagnostics']['usage_metadata'], ensure_ascii=False)}")
    print("FIELD | STATUS | RAW VALUE\n" + "\n".join(" | ".join(map(_field_value, row)) for row in p1_table))
    p1_counts = Counter(item.status for item in pass1.observations)
    print(f"Observed: {p1_counts['observed']}; unavailable: {p1_counts['unavailable']}; ambiguous: {p1_counts['ambiguous']}")
    p1_valid = (
        pass1_raw.execution_status == "success"
        and pass1.parse_diagnostics is not None
        and pass1.parse_diagnostics.status in ("valid_json", "recovered_markdown_fence")
        and len(pass1.observations) == len(REVIEW_FIELD_NAMES)
        and pass1.visual_route is not None
    )
    if not p1_valid:
        attempt_rows = [
            (item["attempt"], item["status"], item["failure_type"],
             f"{item['elapsed_api_s']:.3f}s", item["returned_model_version"], f"{item['backoff_s']}s")
            for item in p1records
        ]
        incomplete_report = [
            "Phase 6.5g validation — INCOMPLETE",
            f"Video: {VIDEO}", f"Investigation: {INVESTIGATION}",
            f"Interval: [{START_S:.3f}, {END_S:.3f}) s",
            f"Configured model: {settings.model_name}",
            f"API version: {settings.api_version}; timeout: {settings.request_timeout_s:g}s; API key configured: yes",
            f"OCR runtime: {ocr_elapsed:.3f}s; usable occurrences={ocr_usable}; non-usable occurrences={ocr_unavailable}",
            "",
            "Complete OCR occurrences:",
            _table(("FIELD", "STATUS", "RAW VALUE", "SOURCE TIME / PROVENANCE"), ocr_rows),
            "",
            "VLM Pass 1 attempts:",
            _table(("ATTEMPT", "STATUS", "FAILURE TYPE", "API TIME", "RETURNED MODEL", "BACKOFF"), attempt_rows),
            f"Total API request time: {p1api_s:.3f}s",
            f"Total elapsed including backoff: {pass1_elapsed:.3f}s",
            f"Failure: {pass1_raw.error}",
            f"Provider finish reason: {p1diag.finish_reason}; returned model: {p1diag.model_version}",
            f"Usage: prompt={p1diag.prompt_token_count}; candidate={p1diag.candidate_token_count}; total={p1diag.total_token_count}",
            "No cross-check, Pass 2, or final resolution was generated.",
            "Phase 6.5 remains OPEN.",
        ]
        (RUN_DIR / "incomplete_validation_report.txt").write_text(
            "\n".join(incomplete_report) + "\n", encoding="utf-8",
        )
        failure = {
            "phase_status": "INCOMPLETE",
            "reason": "VLM Pass 1 execution or complete canonical parsing failed; dependent results were not fabricated.",
            "execution_status": pass1_raw.execution_status,
            "error": pass1_raw.error,
            "parse_diagnostics": _jsonable(pass1.parse_diagnostics),
            "provider_diagnostics": _jsonable(p1diag),
            "attempts": p1records,
            "total_api_request_time_s": p1api_s,
            "total_elapsed_including_backoff_s": pass1_elapsed,
            "incomplete_report": "incomplete_validation_report.txt",
        }
        _json_write(RUN_DIR / "phase_status.json", failure)
        print("Phase 6.5g INCOMPLETE: Pass 1 did not produce a complete valid result.")
        return 1

    vlm_by_name = {item.field_name: item for item in pass1.observations}
    baseline_fused: dict[str, FusedFieldObservation] = {}
    fused: dict[str, FusedFieldObservation] = {}
    for name in REVIEW_FIELD_NAMES:
        ocr = chosen_ocr.get(name)
        vlm = vlm_by_name[name]
        baseline = compare_field_observations(name, ocr, vlm)
        baseline_fused[name] = baseline
        fused[name] = apply_temporal_awareness(baseline)
    fusion_counts = Counter(item.status for item in fused.values())
    fusion_result = FusedInvestigationResult(
        INVESTIGATION, VIDEO, START_S, END_S, fused,
        FusionSummary(
            total_fields=len(fused), agreement_count=fusion_counts["agreement"],
            conflict_count=fusion_counts["conflict"], ocr_only_count=fusion_counts["ocr_only"],
            vlm_only_count=fusion_counts["vlm_only"], unavailable_count=fusion_counts["unavailable"],
            temporal_mismatch_count=fusion_counts["temporal_mismatch"],
        ),
    )
    cross_rows = _crosscheck_rows(baseline_fused, fused)
    print("\n=== CROSS-CHECK ===")
    print(_table(("FIELD", "OCR RAW", "VLM1 RAW", "OCR NORM", "VLM1 NORM", "BASE", "TEMPORAL-AWARE", "DIAGNOSTIC"), cross_rows))
    for status in ("agreement", "conflict", "ocr_only", "vlm_only", "unavailable", "temporal_mismatch"):
        print(f"{status} = {fusion_counts[status]}")
    print(f"total = {len(fused)}")
    _json_write(RUN_DIR / "fusion_result.json", {
        "summary": _jsonable(fusion_result.summary),
        "fields": {
            name: {
                "status": item.status,
                "baseline_status": baseline_fused[name].status,
                "ocr_raw_value": item.ocr_observation.raw_value_text if item.ocr_observation else None,
                "vlm1_raw_value": item.vlm_observation.raw_value if item.vlm_observation else None,
                "ocr_normalized_value": item.ocr_normalized_value,
                "vlm_normalized_value": item.vlm_normalized_value,
                "diagnostics": item.diagnostics,
                "ocr_source_time_s": item.ocr_observation.provenance.actual_timestamp_seconds if item.ocr_observation else None,
                "vlm_source_interval_s": [item.vlm_observation.source_start_s, item.vlm_observation.source_end_s] if item.vlm_observation else None,
                "vlm_evidence_time_s": item.vlm_observation.evidence_time_s if item.vlm_observation else None,
            } for name, item in fused.items()
        },
    })

    review_request = build_vlm_review_request(request, fusion_result, pass1_result=pass1)
    _json_write(RUN_DIR / "vlm_review_context.json", {
        "investigation_id": review_request.context.investigation_id,
        "source_video": str(review_request.context.source_video),
        "source_interval_s": [review_request.context.source_start_s, review_request.context.source_end_s],
        "prompt_version": review_request.prompt_version,
        "review_context": {
            name: _jsonable(item) | {"priority": item.priority}
            for name, item in review_request.context.fields.items()
        },
        "visual_route": _jsonable(review_request.context.visual_route),
        "track_direction_cross_check": _jsonable(review_request.context.direction_consistency),
    })

    print("\n=== VLM PASS 2 REVIEW ===")
    print(f"Requested model: {settings.model_name}; prompt: {FR24_VLM_REVIEW_V1}")
    _inspect_pass2_request(runtime, review_request)
    pass2, p2diag, p2attempt, p2records, p2api_s, pass2_elapsed = _run_gemini(runtime, review_request, parse_vlm_review_result)
    safe_provider_response = (
        GeminiAPIRuntime._sanitize_provider_error(
            p2diag.provider_response_json.encode("utf-8"), settings.api_key,
        )
        if p2diag.provider_response_json else None
    )
    pass2_json = {
        "raw_result": {
            "execution_status": pass2.raw_result.execution_status,
            "source_video": str(pass2.raw_result.source_video),
            "source_interval_s": [pass2.raw_result.source_start_s, pass2.raw_result.source_end_s],
            "model_name": pass2.raw_result.model_name,
            "model_revision": pass2.raw_result.model_revision,
            "prompt_version": pass2.raw_result.prompt_version,
            "inference_time_s": pass2.raw_result.inference_time_s,
            "error": pass2.raw_result.error,
        },
        "reviewed_fields": {name: _review_record(item) for name, item in pass2.reviewed_fields.items()},
        "route_reviews": _jsonable(pass2.route_reviews),
        "direction_consistency": _jsonable(pass2.direction_consistency),
        "diagnostics": pass2.diagnostics,
        "provider_diagnostics": {
            "model_version": p2diag.model_version,
            "finish_reason": p2diag.finish_reason,
            "prompt_token_count": p2diag.prompt_token_count,
            "candidate_token_count": p2diag.candidate_token_count,
            "total_token_count": p2diag.total_token_count,
            "attempt_count": len(p2records),
            "successful_attempt": p2attempt,
            "attempts": p2records,
            "attempt_runtime_s": [item["elapsed_api_s"] for item in p2records],
            "total_api_request_time_s": p2api_s,
            "total_elapsed_including_backoff_s": pass2_elapsed,
            "final_attempt_elapsed_s": p2diag.elapsed_s,
            "total_elapsed_s": pass2_elapsed,
            "response_schema_used": p2diag.response_schema_used,
            "response_schema_rejected": p2diag.response_schema_rejected,
            "provider_response_json": safe_provider_response,
        },
    }
    if safe_provider_response:
        print("Sanitized provider response:")
        print(safe_provider_response)
        if pass2.raw_result.error and "HTTP " in pass2.raw_result.error:
            status_code = int(pass2.raw_result.error.split("HTTP ", 1)[1].split(".", 1)[0])
            try:
                provider_body = json.loads(safe_provider_response)
            except json.JSONDecodeError:
                provider_body = {"raw_body": safe_provider_response}
            _json_write(RUN_DIR / "pass2_provider_error.json", {
                "http_status": status_code,
                "provider_error": provider_body.get("error", provider_body)
                if isinstance(provider_body, dict) else provider_body,
                "sanitized_response_body": provider_body,
            })
    (RUN_DIR / "vlm_pass2_raw.txt").write_text(pass2.raw_result.raw_text or "", encoding="utf-8")
    _json_write(RUN_DIR / "vlm_pass2_result.json", pass2_json)
    p2rows = _review_table(pass2)
    print(f"Returned model: {p2diag.model_version}")
    print(f"Successful attempt: {p2attempt}")
    print(f"Runtime: {p2diag.elapsed_s:.3f}s (total including retries/backoff {pass2_elapsed:.3f}s)")
    print(f"Finish reason: {p2diag.finish_reason}")
    print(f"Usage metadata: {json.dumps({k: pass2_json['provider_diagnostics'][k] for k in ('prompt_token_count','candidate_token_count','total_token_count')}, ensure_ascii=False)}")
    print(_table(("FIELD", "REVIEWED VALUE", "REVIEW STATUS", "QUALITY", "EVIDENCE TIME", "NOTES"), p2rows))
    p2_counts = Counter(item.review_status for item in pass2.reviewed_fields.values())
    for status in ("confirmed", "corrected", "unresolved", "not_visible"):
        print(f"{status} = {p2_counts[status]}")
    p2_valid = (
        pass2.raw_result.execution_status == "success"
        and len(pass2.reviewed_fields) == len(REVIEW_FIELD_NAMES)
        and all(name in pass2.reviewed_fields for name in REVIEW_FIELD_NAMES)
        and set(pass2.route_reviews) == set(REVIEW_ROUTE_FIELDS)
        and not pass2.diagnostics
    )
    if not p2_valid:
        p2_attempt_rows = [
            (item["attempt"], item["status"], item["failure_type"],
             f"{item['elapsed_api_s']:.3f}s", item["returned_model_version"], f"{item['backoff_s']}s")
            for item in p2records
        ]
        incomplete_summary = [
            "=== PHASE 6.5 VALIDATION SUMMARY ===",
            f"Video: {VIDEO}", f"Investigation: {INVESTIGATION}",
            f"Interval: [{START_S:.3f}, {END_S:.3f}) s; duration={END_S - START_S:.3f}s",
            f"OCR: usable={ocr_usable}; unavailable/non-usable={ocr_unavailable}",
            f"VLM Pass 1: observed={p1_counts['observed']}; unavailable={p1_counts['unavailable']}; ambiguous={p1_counts['ambiguous']}",
            "Cross-check: " + "; ".join(
                f"{status}={fusion_counts[status]}"
                for status in ("agreement", "conflict", "ocr_only", "vlm_only", "unavailable", "temporal_mismatch")
            ),
            "VLM Pass 2: INCOMPLETE; no complete review result",
            "Final resolution: NOT RUN; no final values were fabricated",
            "Phase 6.5 remains OPEN / INCOMPLETE.",
        ]
        incomplete_report = [
            "Phase 6.5g real-video end-to-end validation — INCOMPLETE",
            f"Exact video path: {VIDEO}", f"Investigation identifier: {INVESTIGATION}",
            f"Exact source interval: [{START_S:.3f}, {END_S:.3f}) s; duration {END_S - START_S:.3f} s",
            f"Video bytes: {video_size}; metadata duration={metadata.duration_seconds}; fps={metadata.fps}; frames={metadata.frame_count}; size={metadata.width}x{metadata.height}",
            config_block,
            f"OCR runtime: {ocr_elapsed:.3f}s; selected representatives={len(ocr_results)}",
            f"VLM Pass 1 requested={settings.model_name}; returned={p1diag.model_version}; successful_attempt={p1attempt}; inference={p1diag.elapsed_s:.3f}s; total={pass1_elapsed:.3f}s; finish={p1diag.finish_reason}; usage={pass1_json['provider_diagnostics']['usage_metadata']}",
            "VLM Pass 2 did not complete; Pass 2 observations and final resolution were not generated.",
        ]
        _write_report_section(
            incomplete_report, "OCR RESULT",
            _table(("FIELD", "STATUS", "RAW VALUE", "SOURCE TIME / PROVENANCE"), ocr_rows),
        )
        _write_report_section(
            incomplete_report, "VLM PASS 1",
            _table(("FIELD", "STATUS", "RAW VALUE"), p1_table),
        )
        _write_report_section(
            incomplete_report, "CROSS-CHECK",
            _table(("FIELD", "OCR RAW", "VLM1 RAW", "OCR NORM", "VLM1 NORM", "BASE STATUS", "TEMPORAL-AWARE STATUS", "DIAGNOSTIC"), cross_rows),
        )
        _write_report_section(
            incomplete_report, "VLM PASS 2 ATTEMPTS",
            _table(("ATTEMPT", "STATUS", "FAILURE TYPE", "API TIME", "RETURNED MODEL", "BACKOFF"), p2_attempt_rows),
        )
        incomplete_report.extend((
            f"Pass 2 total API request time: {p2api_s:.3f}s",
            f"Pass 2 total elapsed including backoff: {pass2_elapsed:.3f}s",
            f"Pass 2 failure: {pass2.raw_result.error}",
        ))
        _write_report_section(
            incomplete_report, "VLM PASS 2 FIELD RESULTS",
            "Unavailable: Pass 2 execution failed before any canonical reviews were produced.",
        )
        _write_report_section(
            incomplete_report, "FINAL RESOLUTION",
            "Not run: Pass 2 did not complete, so no final result or resolution values were generated.",
        )
        _write_report_section(
            incomplete_report, "BEFORE / AFTER PASS 2",
            "Not generated: Pass 2 did not complete.",
        )
        _write_report_section(
            incomplete_report, "INTERESTING CASES",
            "Not generated: final resolution did not run.",
        )
        incomplete_report.extend((
            "Unresolved fields: not determined; final resolution did not run.",
            "Not-available fields: not determined; final resolution did not run.",
        ))
        _write_report_section(incomplete_report, "SUMMARY COUNTS", "\n".join(incomplete_summary))
        report_text = "\n".join(incomplete_report) + "\n"
        (RUN_DIR / "incomplete_validation_report.txt").write_text(report_text, encoding="utf-8")
        (RUN_DIR / "full_validation_report.txt").write_text(report_text, encoding="utf-8")
        (RUN_DIR / "summary.txt").write_text("\n".join(incomplete_summary) + "\n", encoding="utf-8")
        _json_write(RUN_DIR / "phase_status.json", {
            "phase_status": "INCOMPLETE",
            "reason": "VLM Pass 2 execution or complete canonical review parsing failed; final resolution was not fabricated.",
            "execution_status": pass2.raw_result.execution_status,
            "error": pass2.raw_result.error,
            "parse_diagnostics": pass2.diagnostics,
            "provider_diagnostics": pass2_json["provider_diagnostics"],
            "artifacts": sorted(path.name for path in RUN_DIR.iterdir() if path.is_file()),
        })
        print("\n".join(incomplete_summary))
        print(f"\nIncomplete full report: {RUN_DIR / 'full_validation_report.txt'}")
        print("Phase 6.5g INCOMPLETE: Pass 2 did not produce all valid canonical reviews.")
        return 1

    final = resolve_investigation(fusion_result, pass2)
    final_rows = _final_table(final)
    _json_write(RUN_DIR / "final_result.json", {
        "investigation_id": final.investigation_id,
        "source_video": str(final.source_video),
        "source_interval_s": [final.start_time_s, final.end_time_s],
        "summary": _jsonable(final.summary),
        "fields": {
            name: {
                "final_status": item.final_status,
                "final_value": item.final_value,
                "resolution_reason": item.resolution_reason,
                "temporal_status": item.temporal_status,
                "fusion_status": item.fused_observation.status,
                "ocr_raw_value": item.fused_observation.ocr_observation.raw_value_text if item.fused_observation.ocr_observation else None,
                "vlm1_raw_value": item.fused_observation.vlm_observation.raw_value if item.fused_observation.vlm_observation else None,
                "review": _review_record(item.vlm_review_observation) if item.vlm_review_observation else None,
            } for name, item in final.fields.items()
        },
    })

    before_after = []
    for name, item in final.fields.items():
        before = item.fused_observation.resolved_value if item.fused_observation.status == "agreement" else None
        after = item.final_value if item.final_status in ("resolved", "resolved_single_source") else None
        before_after.append((name, before, after, _effect(item)))
    interesting = _interesting_cases(final)
    summary_lines = [
        "=== PHASE 6.5 VALIDATION SUMMARY ===",
        f"Video: {VIDEO}", f"Investigation: {INVESTIGATION}",
        f"Interval: {START_S:.3f}-{END_S:.3f} s", f"Duration: {END_S - START_S:.3f} s",
        f"OCR: usable={ocr_usable}; unavailable/non-usable={ocr_unavailable}",
        f"VLM Pass 1: observed={p1_counts['observed']}; unavailable={p1_counts['unavailable']}; ambiguous={p1_counts['ambiguous']}",
        "Cross-check: " + "; ".join(f"{s}={fusion_counts[s]}" for s in ("agreement", "conflict", "ocr_only", "vlm_only", "unavailable", "temporal_mismatch")),
        "VLM Pass 2: " + "; ".join(f"{s}={p2_counts[s]}" for s in ("confirmed", "corrected", "unresolved", "not_visible")),
        "Final resolution: " + "; ".join(f"{s}={getattr(final.summary, key)}" for s, key in (("resolved", "resolved_count"), ("resolved_single_source", "resolved_single_source_count"), ("unresolved", "unresolved_count"), ("not_available", "not_available_count"))) + f"; total={final.summary.total_fields}",
        "UNRESOLVED FIELDS: " + (", ".join(name for name, item in final.fields.items() if item.final_status == "unresolved") or "none"),
        "NOT AVAILABLE FIELDS: " + (", ".join(name for name, item in final.fields.items() if item.final_status == "not_available") or "none"),
    ]
    report = [
        "Phase 6.5g real-video end-to-end validation",
        f"Exact video path: {VIDEO}", f"Investigation identifier: {INVESTIGATION}",
        f"Exact source interval: [{START_S:.3f}, {END_S:.3f}) s; duration {END_S - START_S:.3f} s",
        f"Video bytes: {video_size}; metadata duration={metadata.duration_seconds}; fps={metadata.fps}; frames={metadata.frame_count}; size={metadata.width}x{metadata.height}",
        config_block,
        "OCR uses the existing stable-view selector and production OCR path. All selected occurrences are preserved. For the single-observation comparison API, the latest usable selected OCR occurrence is projected per matching canonical field; if no usable occurrence exists, the latest non-usable occurrence is retained. This is an evaluation adapter, not a production multi-view policy.",
        f"OCR runtime (upstream analysis + selection + OCR + field extraction): {ocr_elapsed:.3f}s",
        f"VLM Pass 1 requested={settings.model_name}; returned={p1diag.model_version}; successful_attempt={p1attempt}; inference={p1diag.elapsed_s:.3f}s; total={pass1_elapsed:.3f}s; finish={p1diag.finish_reason}; usage={pass1_json['provider_diagnostics']['usage_metadata']}",
        f"VLM Pass 2 requested={settings.model_name}; returned={p2diag.model_version}; successful_attempt={p2attempt}; inference={p2diag.elapsed_s:.3f}s; total={pass2_elapsed:.3f}s; finish={p2diag.finish_reason}; usage={pass2_json['provider_diagnostics']}",
    ]
    _write_report_section(report, "OCR RESULT", _table(("FIELD", "STATUS", "RAW VALUE", "SOURCE TIME / PROVENANCE"), ocr_rows))
    _write_report_section(report, "VLM PASS 1", _table(("FIELD", "STATUS", "RAW VALUE"), p1_table))
    _write_report_section(report, "CROSS-CHECK", _table(("FIELD", "OCR RAW", "VLM1 RAW", "OCR NORM", "VLM1 NORM", "BASE STATUS", "TEMPORAL-AWARE STATUS", "DIAGNOSTIC"), cross_rows))
    _write_report_section(report, "VLM PASS 2 REVIEW", _table(("FIELD", "REVIEWED VALUE", "REVIEW STATUS", "QUALITY", "EVIDENCE TIME", "NOTES"), p2rows))
    _write_report_section(report, "FINAL RESOLUTION", _table(("FIELD", "OCR VALUE", "VLM1 VALUE", "OCR NORMALIZED", "VLM1 NORMALIZED", "FUSION STATUS", "VLM2 REVIEWED VALUE", "VLM2 REVIEW STATUS", "VLM2 EVIDENCE QUALITY", "VLM2 EVIDENCE TIME", "TEMPORAL STATUS", "FINAL STATUS", "FINAL VALUE", "RESOLUTION REASON"), final_rows))
    _write_report_section(report, "BEFORE / AFTER PASS 2", _table(("FIELD", "BEFORE PASS 2", "AFTER PASS 2", "EFFECT"), before_after))
    case_lines = []
    for label, items in interesting:
        case_lines.append(f"\n{label}: {len(items)}")
        if not items:
            case_lines.append("none")
        for item in items:
            review = item.vlm_review_observation
            case_lines.append(_table(("FIELD", "OCR", "VLM1", "FUSION", "VLM2", "REVIEW", "QUALITY", "FINAL", "VALUE", "REASON"), [(
                item.field_name,
                item.fused_observation.ocr_observation.raw_value_text if item.fused_observation.ocr_observation else None,
                item.fused_observation.vlm_observation.raw_value if item.fused_observation.vlm_observation else None,
                item.fused_observation.status,
                review.reviewed_value if review else None,
                review.review_status if review else None,
                review.evidence_quality if review else None,
                item.final_status, item.final_value, item.resolution_reason,
            )]))
    _write_report_section(report, "INTERESTING CASES", "\n".join(case_lines))
    _write_report_section(report, "SUMMARY COUNTS", "\n".join(summary_lines))
    report_text = "\n".join(report) + "\n"
    (RUN_DIR / "full_validation_report.txt").write_text(report_text, encoding="utf-8")
    (RUN_DIR / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    _json_write(RUN_DIR / "phase_status.json", {
        "phase_status": "COMPLETE",
        "video": str(VIDEO),
        "investigation_id": INVESTIGATION,
        "source_interval_s": [START_S, END_S],
        "policy_or_production_code_changed": False,
        "artifacts": sorted(path.name for path in RUN_DIR.iterdir() if path.is_file()),
    })

    print("\n=== FINAL RESOLUTION ===")
    print(_table(("FIELD", "OCR VALUE", "VLM1 VALUE", "OCR NORMALIZED", "VLM1 NORMALIZED", "FUSION STATUS", "VLM2 REVIEWED VALUE", "VLM2 REVIEW STATUS", "VLM2 EVIDENCE QUALITY", "VLM2 EVIDENCE TIME", "TEMPORAL STATUS", "FINAL STATUS", "FINAL VALUE", "RESOLUTION REASON"), final_rows))
    print("\n=== BEFORE / AFTER PASS 2 ===")
    print(_table(("FIELD", "BEFORE PASS 2", "AFTER PASS 2", "EFFECT"), before_after))
    print("\n=== INTERESTING CASES ===")
    print("\n".join(case_lines))
    print("\n" + "\n".join(summary_lines))
    print(f"\nFull report: {RUN_DIR / 'full_validation_report.txt'}")
    print("Phase 6.5 validation run complete; policy and production code unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
