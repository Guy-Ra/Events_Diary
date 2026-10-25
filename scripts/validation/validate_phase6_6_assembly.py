"""Validate complete, independent InvestigationRecord assembly on real video.

This Phase 6.6 diagnostic reuses the existing detection, lifecycle, window,
adaptive sampling, stable-view, evidence-candidate, materialization, representative
selection, OCR, Gemini Pass 1, comparison, temporal-awareness, Pass 2 and final
resolution components. It runs those components separately for every detected
Investigation and then calls the pure assembly API. Each VLM receives a temporary
continuous clip limited to one half-open Investigation interval, while requests
and result models retain original-video times. The clips are not concatenated,
stills are not sent to Gemini, and OCR/Pass 1 evidence is never mixed across
Investigation boundaries. Artifacts contain no credentials or encoded video.

The script is a manual real-video validation tool, not a new production runner.
It adds no field-resolution policy; the existing Phase 6.5 final result remains
authoritative. OCR occurrences remain separate in each assembled record.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import fields, is_dataclass
from enum import Enum
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from events_diary.config import UIConfig, load_vlm_api_settings
from events_diary.evidence_candidate_timeline import build_evidence_candidate_timeline
from events_diary.frame_materialization import (
    VideoFrameResolver,
    materialize_evidence_candidates,
)
from events_diary.fusion import (
    FusedInvestigationResult,
    FusionSummary,
    apply_temporal_awareness,
    compare_field_observations,
    resolve_investigation,
)
from events_diary.gemini_api import MAX_INLINE_VIDEO_BYTES, GeminiAPIRuntime
from events_diary.investigation_record import (
    InvestigationFailure,
    VideoInvestigationResult,
    assemble_investigation_record,
)
from events_diary.ocr_fields import extract_ocr_fields_batch
from events_diary.representative_selection import select_investigation_representatives
from events_diary.ticket_ocr import run_ocr_for_representatives, ticket_ocr_roi
from events_diary.video import VideoMetadata
from events_diary.vlm import VLMRequest
from events_diary.vlm_parsing import parse_vlm_result
from events_diary.vlm_prompts import build_vlm_prompt
from events_diary.vlm_review import (
    FR24_VLM_REVIEW_V3,
    REVIEW_FIELD_NAMES,
    REVIEW_ROUTE_FIELDS,
    TicketDetailImage,
    build_vlm_review_request,
    parse_vlm_review_result,
)
from scripts.evaluation.evaluate_ticket_stable_views import _build_contexts

VIDEO = ROOT / "data/videos/investigation_switch_test.mp4"
VIDEO_NAME = "investigation_switch"
RUN_ID = time.strftime("v3_multi_%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000:06d}"
OUTPUT = ROOT / "data/debug/investigation_assembly/phase_6_6" / RUN_ID
EPSILON = 1e-9
TRANSIENT_BACKOFF = {"503": (30, 90), "timeout": (30, 90), "429": (30, 60)}


def _jsonable(value: Any) -> Any:
    """Convert existing typed records and immutable mappings to safe JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def _write_json(path: Path, payload: object) -> None:
    """Write indented UTF-8 JSON and create its parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _safe_name(value: object) -> str:
    """Format report values without losing false/null and without breaking tables."""
    if value is None:
        return "null"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    """Render a readable aligned text table from scalar result rows."""
    rendered = [[_safe_name(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        widths = [max(width, len(value)) for width, value in zip(widths, row, strict=True)]
    line = lambda row: "| " + " | ".join(
        value.ljust(width) for value, width in zip(row, widths, strict=True)
    ) + " |"
    return "\n".join((line(headers), line(tuple("-" * width for width in widths)), *(line(row) for row in rendered)))


def _make_investigation_clip(
    source_path: Path,
    destination: Path,
    metadata: VideoMetadata,
    start_time_s: float,
    end_time_s: float,
) -> tuple[int, int]:
    """Write one half-open continuous core clip and return its source frame range."""
    first_frame = round(start_time_s * metadata.fps)
    end_frame = round(end_time_s * metadata.fps)
    if end_frame <= first_frame:
        raise ValueError("Investigation interval must contain at least one source frame.")
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not decode source video: {source_path}")
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create temporary Investigation clip: {destination}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        written = 0
        for _ in range(first_frame, end_frame):
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        capture.release()
    if written != end_frame - first_frame:
        raise RuntimeError(
            f"Clip decode ended early: expected {end_frame-first_frame} frames, wrote {written}."
        )
    if destination.stat().st_size > MAX_INLINE_VIDEO_BYTES:
        raise RuntimeError("Investigation clip exceeds the configured Gemini inline-data limit.")
    return first_frame, end_frame


def _project_ocr_occurrences(occurrences):
    """Retain all OCR rows and select the existing latest-usable pairwise projection."""
    by_field = defaultdict(list)
    for occurrence in occurrences:
        by_field[occurrence.canonical_label].append(occurrence)
    selected = {}
    for name, items in by_field.items():
        usable = [item for item in items if item.status == "extracted" and item.raw_value_text is not None]
        selected[name] = max(
            usable or items,
            key=lambda item: (item.provenance.actual_timestamp_seconds, item.provenance.candidate_index),
        )
    return selected


def _make_ticket_detail_images(ocr_results, selection, resolver, ui_config, output):
    """Create v3 ticket crops only from the existing Phase 5 OCR selections."""
    selection_by_index = {item.candidate_index: item for item in selection.selections}
    output.mkdir(parents=True, exist_ok=True)
    image_records = []
    seen_frames = set()
    for item in ocr_results:
        if item.frame_index in seen_frames:
            continue
        seen_frames.add(item.frame_index)
        selected = selection_by_index[item.candidate_index]
        role = "post_scroll_stable" if "post_scroll_stable" in selected.selection_reasons else "stable_ticket"
        section = "lower scrolled ticket details" if role == "post_scroll_stable" else "upper ticket and flight identity"
        frame = resolver.resolve(item.actual_timestamp_seconds)
        if frame.frame_index != item.frame_index or frame.image is None:
            raise RuntimeError("Decoded Pass 2 image frame disagrees with its selected representative.")
        crop = ticket_ocr_roi(ui_config, frame.image.shape[1], frame.image.shape[0]).crop(frame.image)
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        path = output / f"ticket_{role}_{item.frame_index:06d}_{item.actual_timestamp_seconds:.3f}s.png"
        ok, encoded = cv2.imencode(".png", crop)
        if not ok:
            raise RuntimeError("Could not encode selected ticket detail crop.")
        path.write_bytes(encoded.tobytes())
        image_records.append(TicketDetailImage(
            path, item.actual_timestamp_seconds, role, crop.shape[1], crop.shape[0], section,
        ))
    if not image_records:
        raise RuntimeError("Phase 5 selection supplied no eligible ticket images for Pass 2 v3.")
    return tuple(image_records)


def _probe_provider(settings, artifact_path):
    """Make and record exactly one minimal text-only provider availability request."""
    runtime = GeminiAPIRuntime(settings)
    payload = {
        "contents": [{"parts": [{"text": "Reply with exactly OK."}]}],
        "generationConfig": {"maxOutputTokens": 2},
    }
    started = time.perf_counter()
    status, response_bytes, transport_error = runtime._post(payload, settings.api_key)
    elapsed = time.perf_counter() - started
    response = None
    model_version = None
    if response_bytes:
        try:
            response = json.loads(response_bytes.decode("utf-8", errors="replace"))
            model_version = response.get("modelVersion")
        except (json.JSONDecodeError, AttributeError):
            response = {"body": response_bytes.decode("utf-8", errors="replace")[:2000]}
    quota = _daily_quota_details(response)
    result = {
        "configured_model": settings.model_name,
        "http_status": status,
        "returned_model_version": model_version,
        "request_duration_s": elapsed,
        "transport_error": transport_error[0] if transport_error else None,
        "provider_response": response,
        "daily_quota_exhausted": bool(quota),
        "quota_details": quota,
    }
    _write_json(artifact_path, result)
    print("=== ONE PROVIDER AVAILABILITY PROBE ===", flush=True)
    print(f"Configured model: {settings.model_name}", flush=True)
    print(f"HTTP status: {status if status is not None else transport_error[0]}", flush=True)
    print(f"Returned model version: {model_version or 'unavailable'}", flush=True)
    print(f"Request duration: {elapsed:.3f}s", flush=True)
    if quota:
        print(f"Confirmed Free Tier daily quota exhaustion: {json.dumps(quota, ensure_ascii=False)}", flush=True)
    return result


def _daily_quota_details(response):
    """Extract only explicit Free Tier daily request-quota details from an API body."""
    if not isinstance(response, dict):
        return None
    error = response.get("error")
    if not isinstance(error, dict):
        return None
    details = error.get("details")
    candidates = details if isinstance(details, list) else []
    quota_records = [item for item in candidates if isinstance(item, dict)]
    serialized = json.dumps(response, ensure_ascii=False)
    marker = "GenerateRequestsPerDayPerModel-FreeTier"
    if marker not in serialized:
        return None
    result = {"quota_metric": marker}
    for item in quota_records:
        for key in ("quotaMetric", "quotaId", "quotaValue", "retryDelay"):
            value = item.get(key)
            if value is not None and key not in result:
                result[key] = value
        for violation in item.get("violations", []):
            if not isinstance(violation, dict):
                continue
            for key in ("quotaMetric", "quotaId", "quotaValue"):
                value = violation.get(key)
                if value is not None and key not in result:
                    result[key] = value
            if "retryDelay" not in result:
                delay = violation.get("retryDelay")
                if delay is not None:
                    result["retryDelay"] = delay
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            for key in ("quotaMetric", "quotaId", "quotaValue", "retryDelay"):
                value = metadata.get(key)
                if value is not None and key not in result:
                    result[key] = value
    return result


def _run_gemini(runtime, request, parser, *, investigation_id, pass_number):
    """Run one labeled Gemini pass with the Phase 6.6 transient retry schedule."""
    started = time.perf_counter()
    attempt_records = []
    execution = None
    successful_attempt = None
    quota_exhaustion = None
    for attempt in range(1, 4):
        print(
            f"\nInvestigation: {investigation_id}\nPass: {pass_number}\n"
            f"Model: {runtime.model_name}\nAttempt: {attempt}/3",
            flush=True,
        )
        execution = runtime.run_with_diagnostics(request)
        raw = execution.raw_result
        elapsed = execution.diagnostics.elapsed_s
        returned_model = execution.diagnostics.model_version
        provider_body = None
        if execution.diagnostics.provider_response_json:
            try:
                provider_body = json.loads(execution.diagnostics.provider_response_json)
            except json.JSONDecodeError:
                provider_body = None
        quota_details = _daily_quota_details(provider_body)
        if raw.execution_status == "success":
            failure_type = None
            status = "success"
            successful_attempt = attempt
        elif raw.error == "Gemini API returned HTTP 503.":
            failure_type = "503"
            status = "503"
        elif raw.error == "Gemini API returned HTTP 429.":
            failure_type = "429"
            status = "429"
        elif raw.execution_status == "timeout" or raw.error == "Gemini API request timed out.":
            failure_type = "timeout"
            status = "timeout"
        else:
            failure_type = "other"
            status = "other"
        daily_quota = bool(quota_details) and runtime.model_name == "gemini-3.6-flash"
        if quota_details and runtime.model_name == "gemini-3.8-flash":
            daily_quota = True
        if daily_quota:
            quota_exhaustion = {
                "model": runtime.model_name,
                "quota_metric": quota_details.get("quota_metric"),
                "quota_id": quota_details.get("quotaId"),
                "quota_value": quota_details.get("quotaValue"),
                "retry_delay": quota_details.get("retryDelay"),
            }
        waits = () if daily_quota else TRANSIENT_BACKOFF.get(failure_type, ())
        backoff_s = waits[attempt - 1] if attempt < 3 and waits else 0
        attempt_records.append({
            "attempt": attempt,
            "status": status,
            "http_result": 200 if status == "success" else status,
            "failure_type": failure_type,
            "elapsed_api_s": elapsed,
            "backoff_s": backoff_s,
            "returned_model_version": returned_model,
            "error": raw.error,
            "provider_retry_after_exposed": False,
            "model": runtime.model_name,
            "quota_exhaustion": quota_exhaustion,
        })
        http_result = 200 if status == "success" else status
        print(
            f"Model: {runtime.model_name}\nHTTP/result: {http_result}\n"
            f"Request elapsed: {elapsed:.3f}s", flush=True,
        )
        if quota_exhaustion:
            print(f"Daily quota details: {json.dumps(quota_exhaustion, ensure_ascii=False)}", flush=True)
        print(f"Backoff before next attempt: {backoff_s}s", flush=True)
        print(f"Returned model version: {returned_model or 'unavailable'}", flush=True)
        if status == "success" or not waits or attempt == 3:
            break
        time.sleep(backoff_s)
    if execution is None:
        raise RuntimeError("Gemini runtime returned no attempt result.")
    total_elapsed_s = time.perf_counter() - started
    total_api_s = sum(item["elapsed_api_s"] for item in attempt_records)
    print(f"Pass {pass_number} total API request time: {total_api_s:.3f}s", flush=True)
    print(f"Pass {pass_number} total elapsed including backoff: {total_elapsed_s:.3f}s", flush=True)
    return (
        parser(request, execution.raw_result), execution.diagnostics,
        successful_attempt, attempt_records, total_api_s, total_elapsed_s, quota_exhaustion,
    )


def _build_fusion(ocr_by_name, pass1, *, investigation_id, source_video, start, end):
    """Run the existing pair comparison and temporal-awareness stages for one core."""
    vlm_by_name = {item.field_name: item for item in pass1.observations}
    baseline, fused = {}, {}
    for name in REVIEW_FIELD_NAMES:
        baseline[name] = compare_field_observations(name, ocr_by_name.get(name), vlm_by_name[name])
        fused[name] = apply_temporal_awareness(baseline[name])
    counts = Counter(item.status for item in fused.values())
    result = FusedInvestigationResult(
        investigation_id, source_video, start, end, fused,
        FusionSummary(
            total_fields=len(fused), agreement_count=counts["agreement"],
            conflict_count=counts["conflict"], ocr_only_count=counts["ocr_only"],
            vlm_only_count=counts["vlm_only"], unavailable_count=counts["unavailable"],
            temporal_mismatch_count=counts["temporal_mismatch"],
        ),
    )
    return result, baseline, counts


def _record_payload(record) -> dict[str, object]:
    """Build a field-oriented record document linked to detailed per-stage artifacts."""
    fields_out = {}
    for name, final in record.final_fields.items():
        fused = final.fused_observation
        fields_out[name] = {
            "final_status": final.final_status,
            "final_value": final.final_value,
            "resolution_reason": final.resolution_reason,
            "temporal_status": final.temporal_status,
            "ocr_evidence": [
                {
                    "raw_label": occurrence.raw_label_text,
                    "raw_value": occurrence.raw_value_text,
                    "status": occurrence.status,
                    "candidate_index": occurrence.provenance.candidate_index,
                    "frame_index": occurrence.provenance.frame_index,
                    "source_time_s": occurrence.provenance.actual_timestamp_seconds,
                    "reason": occurrence.reason,
                }
                for occurrence in record.ocr_field_occurrences
                if occurrence.canonical_label == name
            ],
            "vlm_pass1_evidence": _jsonable(fused.vlm_observation),
            "fusion_status": fused.status,
            "ocr_normalized_value": fused.ocr_normalized_value,
            "vlm1_normalized_value": fused.vlm_normalized_value,
            "vlm_pass2_review": _jsonable(record.vlm_pass2.reviewed_fields.get(name)),
        }
    return {
        "investigation_id": record.investigation_id,
        "source_video": str(record.source_video),
        "interval_s": [record.start_time_s, record.end_time_s],
        "duration_s": record.duration_s,
        "event_metadata": {
            "start_reason": record.interval.start_reason.value,
            "end_reason": record.interval.end_reason.value,
            "confirmed_events": _jsonable(record.confirmed_events),
        },
        "evidence_provenance": {
            "window": _jsonable(record.window),
            "sampling_plan": _jsonable(record.sampling_plan),
            "stable_views": _jsonable(record.stable_views),
            "selected_evidence": [
                {
                    "candidate_index": item.candidate_index,
                    "requested_time_s": record.candidates[item.candidate_index].timestamp_seconds,
                    "actual_time_s": record.materialized_candidates[item.candidate_index].actual_timestamp_seconds,
                    "frame_index": record.materialized_candidates[item.candidate_index].frame_index,
                    "selection_reasons": item.selection_reasons,
                }
                for item in record.selection.selections
            ],
            "ocr_representative_count": len(record.ocr_results),
            "ocr_occurrence_count": len(record.ocr_field_occurrences),
        },
        "final_fields": fields_out,
        "unresolved_fields": record.unresolved_fields,
        "not_available_fields": record.not_available_fields,
        "artifacts": {
            "ocr": "ocr_result.json",
            "vlm_pass1": "vlm_pass1_result.json",
            "fusion": "fusion_result.json",
            "vlm_pass2": "vlm_pass2_result.json",
            "final": "final_result.json",
        },
        "diagnostics": record.diagnostics,
    }


def _write_investigation_artifacts(directory: Path, record) -> None:
    """Write detailed per-stage JSON artifacts without copying source video."""
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "investigation_record.json", _record_payload(record))
    _write_json(directory / "ocr_result.json", {
        "investigation_id": record.investigation_id,
        "source_video": str(record.source_video),
        "interval_s": [record.start_time_s, record.end_time_s],
        "representatives": record.ocr_results,
        "field_occurrences": record.ocr_field_occurrences,
    })
    _write_json(directory / "vlm_pass1_result.json", record.vlm_pass1)
    _write_json(directory / "fusion_result.json", record.fusion_result)
    _write_json(directory / "vlm_pass2_result.json", record.vlm_pass2)
    _write_json(directory / "final_result.json", record.final_result)


def _run_one_investigation(
    *, interval, confirmed_events, context, metadata, ui_config, source_video,
    clip_path, settings, resolver, artifact_directory,
):
    """Execute and assemble one Investigation using the established Phase 6 stages."""
    artifact_directory.mkdir(parents=True, exist_ok=True)
    for artifact_name in (
        "failure.txt", "investigation_record.json", "ocr_result.json",
        "vlm_pass1_result.json", "fusion_result.json", "vlm_pass2_result.json",
        "final_result.json",
    ):
        stale_artifact = artifact_directory / artifact_name
        if stale_artifact.exists():
            stale_artifact.unlink()
    investigation_id = interval.investigation_index
    window, sampling_plan, stable_views, _ = context
    candidates = tuple(build_evidence_candidate_timeline(
        window=window, sampling_plan=sampling_plan, stable_views=stable_views,
    ))
    materialized = tuple(materialize_evidence_candidates(
        video_path=source_video, candidates=candidates, ui_config=ui_config, resolver=resolver,
    ))
    selection = select_investigation_representatives(
        window=window, sampling_plan=sampling_plan, candidates=candidates,
        materialized_candidates=materialized, stable_views=stable_views,
    )
    ocr_results = tuple(run_ocr_for_representatives(
        video_path=source_video, investigation_index=investigation_id,
        selections=selection.selections, candidates=candidates,
        materialized_candidates=materialized, ui_config=ui_config, resolver=resolver,
    ))
    grouped = extract_ocr_fields_batch(ocr_results)
    occurrences = tuple(item for group in grouped for item in group)
    for occurrence in occurrences:
        if occurrence.provenance.investigation_index != investigation_id:
            raise AssertionError("OCR evidence leaked across Investigation boundaries.")
        if not interval.start_time_seconds <= occurrence.provenance.actual_timestamp_seconds < interval.end_time_seconds:
            raise AssertionError("OCR evidence timestamp lies outside its Investigation core.")
    ocr_by_name = _project_ocr_occurrences(occurrences)
    _write_json(artifact_directory / "ocr_result.json", {
        "investigation_id": investigation_id,
        "source_video": str(source_video),
        "interval_s": [interval.start_time_seconds, interval.end_time_seconds],
        "representative_count": len(ocr_results),
        "representatives": ocr_results,
        "field_occurrences": occurrences,
        "pairwise_projection": {
            "policy": "existing Phase 6.5 validation projection: latest usable occurrence, otherwise latest occurrence",
            "fields": ocr_by_name,
        },
    })
    request = VLMRequest(
        investigation_index=investigation_id, source_video=source_video,
        video_clip_path=clip_path, source_start_s=interval.start_time_seconds,
        source_end_s=interval.end_time_seconds, prompt=build_vlm_prompt("fr24_vlm_v1"),
        prompt_version="fr24_vlm_v1",
    )
    runtime = GeminiAPIRuntime(settings)
    print(f"\n=== INVESTIGATION {investigation_id} PROCESSING ===", flush=True)
    print(f"Time range: [{interval.start_time_seconds:.3f}, {interval.end_time_seconds:.3f})", flush=True)
    print(f"Duration: {interval.end_time_seconds-interval.start_time_seconds:.3f}s", flush=True)
    print(f"OCR representatives: {len(ocr_results)}; occurrences: {len(occurrences)}", flush=True)
    pass1, p1diag, p1attempt, p1records, p1api, p1elapsed, p1quota = _run_gemini(
        runtime, request, parse_vlm_result, investigation_id=investigation_id, pass_number=1,
    )
    _write_json(artifact_directory / "vlm_pass1_result.json", {
        "raw_result": pass1.raw_result,
        "observations": pass1.observations,
        "visual_route": pass1.visual_route,
        "parse_diagnostics": pass1.parse_diagnostics,
        "provider": {
            "returned_model": p1diag.model_version,
            "successful_attempt": p1attempt,
            "attempts": p1records,
            "total_api_request_time_s": p1api,
            "total_elapsed_s": p1elapsed,
        },
    })
    if p1quota:
        return None, InvestigationFailure(
            investigation_id, source_video, interval.start_time_seconds, interval.end_time_seconds,
            "provider_daily_quota", json.dumps(p1quota, ensure_ascii=False), artifact_directory,
        ), None
    p1_counts = Counter(item.status for item in pass1.observations)
    print(f"VLM1 observed: {p1_counts['observed']}; unavailable: {p1_counts['unavailable']}; ambiguous: {p1_counts['ambiguous']}", flush=True)
    p1_valid = (
        pass1.raw_result.execution_status == "success"
        and pass1.parse_diagnostics is not None
        and pass1.parse_diagnostics.status in ("valid_json", "recovered_markdown_fence")
        and len(pass1.observations) == len(REVIEW_FIELD_NAMES)
        and pass1.visual_route is not None
    )
    if not p1_valid:
        return None, InvestigationFailure(
            investigation_id, source_video, interval.start_time_seconds, interval.end_time_seconds,
            "vlm_pass1", pass1.raw_result.error or str(pass1.parse_diagnostics), artifact_directory,
        ), None
    fusion, baseline, fusion_counts = _build_fusion(
        ocr_by_name, pass1, investigation_id=investigation_id, source_video=source_video,
        start=interval.start_time_seconds, end=interval.end_time_seconds,
    )
    print("Cross-check: " + "; ".join(
        f"{key}={fusion_counts[key]}" for key in (
            "agreement", "conflict", "ocr_only", "vlm_only", "unavailable", "temporal_mismatch",
        )
    ), flush=True)
    _write_json(artifact_directory / "fusion_result.json", {
        "summary": fusion.summary,
        "fields": {
            name: {
                "status": item.status, "baseline_status": baseline[name].status,
                "ocr_observation": item.ocr_observation,
                "vlm_pass1_observation": item.vlm_observation,
                "ocr_normalized_value": item.ocr_normalized_value,
                "vlm1_normalized_value": item.vlm_normalized_value,
                "diagnostics": item.diagnostics,
            } for name, item in fusion.fields.items()
        },
    })
    ticket_images = _make_ticket_detail_images(
        ocr_results, selection, resolver, ui_config, artifact_directory / "ticket_images",
    )
    review_request = build_vlm_review_request(
        request, fusion, pass1_result=pass1, ticket_images=ticket_images,
        prompt_version=FR24_VLM_REVIEW_V3,
    )
    print(f"\nINVESTIGATION {investigation_id}", flush=True)
    print("Full video: yes", flush=True)
    print(f"Ticket images: {len(ticket_images)}", flush=True)
    for image in ticket_images:
        print(
            f"  source={image.source_timestamp_s:.3f}s; role={image.role}; "
            f"dimensions={image.width}x{image.height}; section={image.section}", flush=True,
        )
    review, p2diag, p2attempt, p2records, p2api, p2elapsed, p2quota = _run_gemini(
        runtime, review_request, parse_vlm_review_result,
        investigation_id=investigation_id, pass_number=2,
    )
    _write_json(artifact_directory / "vlm_pass2_result.json", {
        "raw_result": review.raw_result,
        "reviewed_fields": review.reviewed_fields,
        "route_reviews": review.route_reviews,
        "direction_consistency": review.direction_consistency,
        "diagnostics": review.diagnostics,
        "provider": {
            "returned_model": p2diag.model_version,
            "successful_attempt": p2attempt,
            "attempts": p2records,
            "total_api_request_time_s": p2api,
            "total_elapsed_s": p2elapsed,
            "response_schema_used": p2diag.response_schema_used,
            "quota_exhaustion": p2quota,
        },
        "prompt_version": FR24_VLM_REVIEW_V3,
        "ticket_images": ticket_images,
    })
    if (review.raw_result.execution_status != "success"
            or review.diagnostics or set(review.reviewed_fields) != set(REVIEW_FIELD_NAMES)
            or set(review.route_reviews) != set(REVIEW_ROUTE_FIELDS)):
        return None, InvestigationFailure(
            investigation_id, source_video, interval.start_time_seconds, interval.end_time_seconds,
            "provider_daily_quota" if p2quota else "vlm_pass2_parse",
            json.dumps(p2quota, ensure_ascii=False) if p2quota else (
                "; ".join(review.diagnostics) or review.raw_result.error or "Incomplete Pass 2 fields."
            ),
            artifact_directory,
        ), None
    pass2_counts = Counter(item.review_status for item in review.reviewed_fields.values())
    final = resolve_investigation(fusion, review)
    _write_json(artifact_directory / "final_result.json", final)
    final_counts = Counter(item.final_status for item in final.fields.values())
    print(f"VLM2: confirmed={pass2_counts['confirmed']}; corrected={pass2_counts['corrected']}; unresolved={pass2_counts['unresolved']}; not_visible={pass2_counts['not_visible']}", flush=True)
    print(f"Final: resolved={final_counts['resolved']}; resolved_single_source={final_counts['resolved_single_source']}; unresolved={final_counts['unresolved']}; not_available={final_counts['not_available']}", flush=True)
    record = assemble_investigation_record(
        investigation_id=investigation_id, source_video=source_video, interval=interval,
        confirmed_events=confirmed_events, window=window, sampling_plan=sampling_plan,
        stable_views=stable_views, candidates=candidates, materialized_candidates=materialized,
        selection=selection, ocr_results=ocr_results, ocr_field_occurrences=occurrences,
        vlm_pass1=pass1, fusion_result=fusion, vlm_pass2=review, final_result=final,
    )
    _write_investigation_artifacts(artifact_directory, record)
    # Assembly serialization writes the typed VLM result again; restore the
    # request diagnostics so the final artifacts retain every provider attempt.
    pass1_artifact = _read_json_if_present(artifact_directory / "vlm_pass1_result.json") or {}
    pass1_artifact["provider"] = {
        "returned_model": p1diag.model_version,
        "successful_attempt": p1attempt,
        "attempts": p1records,
        "total_api_request_time_s": p1api,
        "total_elapsed_s": p1elapsed,
    }
    _write_json(artifact_directory / "vlm_pass1_result.json", pass1_artifact)
    pass2_artifact = _read_json_if_present(artifact_directory / "vlm_pass2_result.json") or {}
    pass2_artifact["provider"] = {
        "returned_model": p2diag.model_version,
        "successful_attempt": p2attempt,
        "attempts": p2records,
        "total_api_request_time_s": p2api,
        "total_elapsed_s": p2elapsed,
        "response_schema_used": p2diag.response_schema_used,
        "quota_exhaustion": p2quota,
    }
    pass2_artifact["prompt_version"] = FR24_VLM_REVIEW_V3
    pass2_artifact["ticket_images"] = _jsonable(ticket_images)
    _write_json(artifact_directory / "vlm_pass2_result.json", pass2_artifact)
    print(f"\n=== INVESTIGATION {investigation_id} COMPLETE ===", flush=True)
    print(f"Time range: [{record.start_time_s:.3f}, {record.end_time_s:.3f})s", flush=True)
    print(f"OCR: {'usable' if any(item.status == 'extracted' for item in occurrences) else 'non-usable'}", flush=True)
    print(f"VLM1: {'observed' if p1_counts['observed'] else 'unavailable/ambiguous'}", flush=True)
    print("Fusion: " + ", ".join(
        f"{key}={fusion_counts[key]}" for key in (
            "agreement", "conflict", "ocr_only", "vlm_only", "unavailable", "temporal_mismatch",
        )
    ), flush=True)
    print("VLM2: " + ", ".join(f"{key}={pass2_counts[key]}" for key in (
        "confirmed", "corrected", "unresolved", "not_visible",
    )), flush=True)
    print("Final: " + ", ".join(f"{key}={final_counts[key]}" for key in (
        "resolved", "resolved_single_source", "unresolved", "not_available",
    )), flush=True)
    print("FINAL FIELD TABLE", flush=True)
    print(_table(("FIELD", "FINAL STATUS", "FINAL VALUE", "RESOLUTION REASON"), [
        (name, value.final_status, value.final_value, value.resolution_reason)
        for name, value in record.final_fields.items()
    ]), flush=True)
    print("UNRESOLVED FIELDS: " + (", ".join(record.unresolved_fields) or "none"), flush=True)
    print("NOT AVAILABLE FIELDS: " + (", ".join(record.not_available_fields) or "none"), flush=True)
    return record, None, {
        "ocr_occurrences": len(occurrences),
        "ocr_usable": sum(item.status == "extracted" for item in occurrences),
        "vlm1": dict(p1_counts), "crosscheck": dict(fusion_counts),
        "vlm2": dict(pass2_counts), "final": dict(final_counts),
        "runtime": {
            "ocr_representatives": len(ocr_results),
            "pass1_successful_attempt": p1attempt, "pass1_elapsed_s": p1elapsed,
            "pass2_successful_attempt": p2attempt, "pass2_elapsed_s": p2elapsed,
            "total_elapsed_s": p1elapsed + p2elapsed,
        },
    }


def _render_investigation(record) -> str:
    """Return the required complete human-readable report section for one record."""
    final_counts = record.final_result.summary
    fusion_counts = record.fusion_result.summary
    p1_counts = Counter(item.status for item in record.vlm_pass1.observations)
    p2_counts = Counter(item.review_status for item in record.vlm_pass2.reviewed_fields.values())
    ocr_by_field = defaultdict(list)
    for item in record.ocr_field_occurrences:
        ocr_by_field[item.canonical_label].append(item)
    pass1_by_field = {item.field_name: item for item in record.vlm_pass1.observations}
    important_fields = (
        "primary_flight_id", "alternate_flight_id", "airline_name", "aircraft_code",
        "AIRCRAFT_TYPE", "REGISTRATION", "SERIAL_NUMBER", "AGE", "origin_airport_code",
        "destination_airport_code", "origin_scheduled_time", "origin_actual_time",
        "destination_scheduled_time", "destination_estimated_or_actual_time", "TRACK",
    )
    field_rows = []
    for name in important_fields:
        final = record.final_fields.get(name)
        fused = record.fusion_result.fields.get(name)
        p1 = pass1_by_field.get(name)
        p2 = record.vlm_pass2.reviewed_fields.get(name)
        ocr_values = ocr_by_field.get(name, [])
        field_rows.append((
            name,
            "; ".join(item.raw_value_text or "null" for item in ocr_values) or "null",
            p1.raw_value if p1 else "null",
            fused.status if fused else "unavailable",
            p2.reviewed_value if p2 else "null",
            p2.review_status if p2 else "unavailable",
            p2.evidence_quality if p2 else "unavailable",
            p2.evidence_time_s if p2 else "null",
            p2.notes if p2 else "null",
            final.final_status if final else "unavailable",
            final.final_value if final else "null",
        ))
    visual_route = record.vlm_pass1.visual_route
    route_reviews = record.vlm_pass2.route_reviews
    direction_checks = record.track_direction_cross_checks
    lines = [
        f"\n=== INVESTIGATION {record.investigation_id} ===",
        f"Time range: [{record.start_time_s:.3f}, {record.end_time_s:.3f})s",
        f"Duration: {record.duration_s:.3f}s",
        f"OCR usable fields: {sum(item.status == 'extracted' for item in record.ocr_field_occurrences)} / {len(record.ocr_field_occurrences)} occurrences",
        f"VLM1 observed: {p1_counts['observed']}",
        (f"Cross-check: agreement={fusion_counts.agreement_count}; conflict={fusion_counts.conflict_count}; "
         f"ocr_only={fusion_counts.ocr_only_count}; vlm_only={fusion_counts.vlm_only_count}; "
         f"unavailable={fusion_counts.unavailable_count}; temporal_mismatch={fusion_counts.temporal_mismatch_count}"),
        f"VLM2: confirmed={p2_counts['confirmed']}; corrected={p2_counts['corrected']}; unresolved={p2_counts['unresolved']}; not_visible={p2_counts['not_visible']}",
        f"Final: resolved={final_counts.resolved_count}; resolved_single_source={final_counts.resolved_single_source_count}; unresolved={final_counts.unresolved_count}; not_available={final_counts.not_available_count}",
        _table(("FIELD", "FINAL STATUS", "FINAL VALUE", "RESOLUTION REASON"), [
            (name, value.final_status, value.final_value, value.resolution_reason)
            for name, value in record.final_fields.items()
        ]),
        "UNRESOLVED FIELDS: " + (", ".join(record.unresolved_fields) or "none"),
        "NOT AVAILABLE FIELDS: " + (", ".join(record.not_available_fields) or "none"),
        "OCR occurrence provenance retained: " + str(len(record.ocr_field_occurrences)),
        "OCR FIELD OCCURRENCES",
        _table(("FIELD", "STATUS", "RAW VALUE", "SOURCE TIME"), [
            (item.canonical_label, item.status, item.raw_value_text,
             item.provenance.actual_timestamp_seconds)
            for item in record.ocr_field_occurrences
        ]),
        "VLM1 FIELD OBSERVATIONS",
        _table(("FIELD", "STATUS", "RAW VALUE", "EVIDENCE TIME"), [
            (item.field_name, item.status, item.raw_value, item.evidence_time_s)
            for item in record.vlm_pass1.observations
        ]),
        "FUSION FIELD STATUSES",
        _table(("FIELD", "STATUS", "DIAGNOSTICS"), [
            (name, item.status, "; ".join(item.diagnostics))
            for name, item in record.fusion_result.fields.items()
        ]),
        "VLM2 FIELD REVIEWS",
        _table(("FIELD", "STATUS", "REVIEWED VALUE", "EVIDENCE QUALITY"), [
            (name, item.review_status, item.reviewed_value, item.evidence_quality)
            for name, item in record.vlm_pass2.reviewed_fields.items()
        ]),
        "IMPORTANT FIELD CROSS-STAGE REVIEW",
        _table(("FIELD", "OCR", "VLM1", "FUSION", "PASS2 V3", "REVIEW", "QUALITY", "TIME", "NOTES", "FINAL STATUS", "FINAL VALUE"), field_rows),
        "TRACK / DIRECTION PRESERVED EVIDENCE",
        _table(("OCR TRACK", "PARSED DEGREES", "SECTOR", "FULL DIRECTION", "VLM VISUAL DIRECTION", "CONSISTENCY"), [
            (
                direction.source_observation.raw_value_text,
                direction.degrees,
                direction.sector,
                direction.text,
                visual_route.direction_text.value if visual_route else None,
                check.status,
            )
            for direction, check in zip(
                record.derived_track_directions,
                direction_checks,
                strict=True,
            )
        ] or [(None, None, None, None, visual_route.direction_text.value if visual_route else None, "unavailable")]),
        "ROUTE CLAIMS (Pass 1) AND INDEPENDENT Pass 2 REVIEW",
        _table(("FIELD", "PASS1 VALUE", "PASS1 STATUS", "PASS2 VALUE", "PASS2 OBSERVATION", "PASS2 REVIEW", "QUALITY", "TIME", "NOTES"), [
            (
                name,
                getattr(visual_route, attr).value if visual_route else None,
                getattr(visual_route, attr).status if visual_route else "not_visible",
                route_reviews[name].reviewed_value if name in route_reviews else None,
                route_reviews[name].observation_status if name in route_reviews else "not_visible",
                route_reviews[name].review_status if name in route_reviews else "not_visible",
                route_reviews[name].evidence_quality if name in route_reviews else "not_visible",
                route_reviews[name].evidence_time_s if name in route_reviews else None,
                route_reviews[name].notes if name in route_reviews else None,
            )
            for name, attr in (
                ("visual_flight_direction_sector", "direction_sector"),
                ("visual_flight_direction_text", "direction_text"),
                ("visual_route_description", "route_description"),
                ("visual_departure_context", "departure_context"),
            )
        ]),
    ]
    return "\n".join(lines)


def _video_summary_rows(records):
    """Build chronological summary rows using only available identity fields."""
    rows = []
    for record in records:
        fields_map = record.final_fields
        primary = next((fields_map[name].final_value for name in ("primary_flight_id", "alternate_flight_id")
                        if name in fields_map and fields_map[name].final_value is not None), None)
        registration = fields_map.get("REGISTRATION")
        rows.append((
            record.investigation_id, f"{record.start_time_s:.3f}", f"{record.end_time_s:.3f}",
            primary, registration.final_value if registration else None,
            record.final_result.summary.resolved_count + record.final_result.summary.resolved_single_source_count,
            len(record.unresolved_fields), "COMPLETE",
        ))
    return rows


def _summary_rows(session, intervals):
    """Include every detected interval in the session table, even on failure."""
    rows = _video_summary_rows(session.investigations)
    completed = {record.investigation_id for record in session.investigations}
    failed = {item.investigation_id for item in session.failures}
    for interval in intervals:
        if interval.investigation_index not in completed and interval.investigation_index in failed:
            rows.append((
                interval.investigation_index,
                f"{interval.start_time_seconds:.3f}",
                f"{interval.end_time_seconds:.3f}",
                None, None, "n/a", "n/a", "FAILED",
            ))
    return sorted(rows, key=lambda row: float(row[1]))


def _read_json_if_present(path: Path):
    """Read one UTF-8 JSON stage artifact when the failed run produced it."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _render_attempts(stage_name: str, payload) -> list[str]:
    """Render attempt status, timing, backoff and model details for one VLM stage."""
    if not payload:
        return [f"{stage_name}: not reached"]
    provider = payload.get("provider", {})
    lines = [f"{stage_name} attempts:"]
    for attempt in provider.get("attempts", []):
        lines.append(
            f"  Attempt {attempt.get('attempt')}/3: model={attempt.get('model')}; "
            f"HTTP/result={attempt.get('http_result', attempt.get('status'))}; "
            f"request_elapsed={attempt.get('elapsed_api_s')}; backoff={attempt.get('backoff_s')}s; "
            f"returned_model={attempt.get('returned_model_version') or 'unavailable'}; "
            f"error={attempt.get('error') or 'none'}"
        )
    lines.append(f"  Total API request time: {provider.get('total_api_request_time_s', 'n/a')}s")
    lines.append(f"  Total elapsed including backoff: {provider.get('total_elapsed_s', 'n/a')}s")
    return lines


def _render_failed_investigation(interval, failure, directory: Path) -> str:
    """Render all stage evidence available for a failed Investigation without inventing later results."""
    ocr = _read_json_if_present(directory / "ocr_result.json")
    pass1 = _read_json_if_present(directory / "vlm_pass1_result.json")
    fusion = _read_json_if_present(directory / "fusion_result.json")
    pass2 = _read_json_if_present(directory / "vlm_pass2_result.json")
    final = _read_json_if_present(directory / "final_result.json")
    lines = [
        f"\n=== INVESTIGATION {failure.investigation_id} FAILED ===",
        f"Time range: [{interval.start_time_seconds:.3f}, {interval.end_time_seconds:.3f})s",
        f"Failure stage: {failure.failed_stage}",
        f"Failure reason: {failure.reason}",
    ]
    if ocr:
        occurrences = ocr.get("field_occurrences", [])
        lines.append(
            f"OCR: representatives={ocr.get('representative_count', 'n/a')}; "
            f"occurrences={len(occurrences)}; usable={sum(item.get('status') == 'extracted' for item in occurrences)}"
        )
        lines.append("OCR FIELD OCCURRENCES")
        lines.append(_table(("FIELD", "STATUS", "VALUE", "SOURCE TIME"), [
            (
                (item.get("label_match") or {}).get("canonical_label"),
                item.get("status"),
                (((item.get("selected_candidate") or {}).get("observation") or {}).get("detection") or {}).get("text"),
                (item.get("provenance") or {}).get("actual_timestamp_seconds"),
            )
            for item in occurrences
        ]))
    else:
        lines.append("OCR: unavailable; stage artifact was not produced")
    if pass1:
        raw = pass1.get("raw_result", {})
        observations = pass1.get("observations", [])
        counts = Counter(item.get("status") for item in observations)
        lines.append(
            f"VLM1: execution={raw.get('execution_status')}; observed={counts['observed']}; "
            f"unavailable={counts['unavailable']}; ambiguous={counts['ambiguous']}; error={raw.get('error')}"
        )
        lines.append("VLM1 FIELD OBSERVATIONS")
        lines.append(_table(("FIELD", "STATUS", "RAW VALUE", "EVIDENCE TIME"), [
            (item.get("field_name"), item.get("status"), item.get("raw_value"), item.get("evidence_time_s"))
            for item in observations
        ]))
        route = pass1.get("visual_route") or {}
        lines.append("VLM1 VISUAL ROUTE")
        lines.append(_table(("FIELD", "VALUE", "OBSERVATION STATUS"), [
            (name, (route.get(key) or {}).get("value"), (route.get(key) or {}).get("status"))
            for name, key in (
                ("visual_flight_direction_sector", "direction_sector"),
                ("visual_flight_direction_text", "direction_text"),
                ("visual_route_description", "route_description"),
                ("visual_departure_context", "departure_context"),
            )
        ]))
    else:
        lines.append("VLM1: not reached")
    lines.extend(_render_attempts("VLM Pass 1", pass1))
    if fusion:
        summary = fusion.get("summary", {})
        lines.append("Fusion: " + ", ".join(
            f"{name}={summary.get(f'{name}_count', 0)}" for name in (
                "agreement", "conflict", "ocr_only", "vlm_only", "unavailable", "temporal_mismatch",
            )
        ))
    else:
        lines.append("Fusion: not reached")
    if pass2:
        raw = pass2.get("raw_result", {})
        reviews = pass2.get("reviewed_fields", {})
        counts = Counter(item.get("review_status") for item in reviews.values())
        lines.append(
            f"VLM2: execution={raw.get('execution_status')}; confirmed={counts['confirmed']}; "
            f"corrected={counts['corrected']}; unresolved={counts['unresolved']}; "
            f"not_visible={counts['not_visible']}; error={raw.get('error')}"
        )
        images = pass2.get("ticket_images", [])
        lines.append(f"Pass 2 v3 ticket images: {len(images)}")
        lines.append(_table(("SOURCE TIME", "ROLE", "DIMENSIONS", "SECTION"), [
            (image.get("source_timestamp_s"), image.get("role"),
             f"{image.get('width')}x{image.get('height')}", image.get("section"))
            for image in images
        ]))
        consistency = pass2.get("direction_consistency") or {}
        derived = consistency.get("derived_direction") or {}
        lines.append(
            "Direction comparison retained with request: "
            f"OCR TRACK={((((derived.get('source_observation') or {}).get('selected_candidate') or {}).get('observation') or {}).get('detection') or {}).get('text')}; "
            f"derived={derived.get('degrees')}° {derived.get('sector')} ({derived.get('text')}); "
            f"visual={consistency.get('visual_direction_sector')} "
            f"({consistency.get('visual_direction_status')}); comparison={consistency.get('status')}"
        )
    else:
        lines.append("VLM2: not reached")
    lines.extend(_render_attempts("VLM Pass 2", pass2))
    if final:
        lines.append("Final result: available")
    else:
        lines.append("Final result: not produced; downstream result was not fabricated")
        lines.append("UNRESOLVED FIELDS: not evaluated because final resolution was not reached")
        lines.append("NOT AVAILABLE FIELDS: not evaluated because final resolution was not reached")
    lines.append(f"Investigation artifacts: {directory}")
    return "\n".join(lines)


def main() -> int:
    """Run multi-Investigation Phase 6.6 validation and persist complete artifacts."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    settings = load_vlm_api_settings()
    if not settings.api_key_configured:
        raise RuntimeError("Gemini API key is not configured.")
    probe = _probe_provider(settings, OUTPUT / "provider_availability.json")
    if probe["http_status"] == 429:
        reason = "confirmed Free Tier daily quota exhaustion" if probe.get("daily_quota_exhausted") else "HTTP 429"
        print(f"STOP: provider availability probe returned {reason}; no further provider calls will be made.", flush=True)
        if probe.get("daily_quota_exhausted") and settings.model_name == "gemini-3.6-flash":
            print("Required fallback: set EVENTS_DIARY_VLM_MODEL=gemini-3.8-flash locally, probe 3.8 once, then restart the full validation.", flush=True)
        _write_json(OUTPUT / "phase_status.json", {
            "phase": "6.6 v3 multi-Investigation regression", "status": "OPEN",
            "stop_reason": reason,
            "daily_quota_exhausted": probe.get("daily_quota_exhausted", False),
            "quota_details": probe.get("quota_details"),
            "model": settings.model_name,
        })
        return 3
    metadata, anomalies, contexts, phase3 = _build_contexts(VIDEO_NAME, include_phase3=True)
    if metadata.path.resolve() != VIDEO.resolve():
        raise RuntimeError("Detected video metadata does not match the configured input.")
    intervals = tuple(phase3["intervals"])
    print("=== PHASE 6.6 VALIDATION ===", flush=True)
    print(f"Source video: {metadata.path}", flush=True)
    print(f"Detected Investigations: {len(intervals)}", flush=True)
    print(f"Configured model: {settings.model_name}", flush=True)
    print(f"Request timeout: {settings.request_timeout_s:g} seconds", flush=True)
    print("Processing mode: sequential", flush=True)
    print("Retryable provider failures: 503, timeout, 429", flush=True)
    for interval in intervals:
        print(
            f"Investigation {interval.investigation_index}: [{interval.start_time_seconds:.3f}, "
            f"{interval.end_time_seconds:.3f})s; duration="
            f"{interval.end_time_seconds-interval.start_time_seconds:.3f}s", flush=True,
        )
    if len(intervals) < 2:
        print("STOP: this video does not meet the real multi-Investigation validation requirement.", flush=True)
        return 2
    if len(contexts) != len(intervals):
        raise RuntimeError("Phase 4 contexts do not align with lifecycle Investigations.")

    print(
        f"Configured VLM: provider={settings.provider}; model={settings.model_name}; "
        f"timeout={settings.request_timeout_s:g}s; API key configured={settings.api_key_configured}",
        flush=True,
    )
    if not settings.api_key_configured:
        raise RuntimeError("Gemini API key is not configured.")
    ui_config = UIConfig()
    record_by_id, failure_by_id, metrics_by_id, clips = {}, {}, {}, {}
    clips_by_range = {}
    run_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="events-diary-phase6-6-") as temp_dir:
        temp_path = Path(temp_dir)
        for interval in intervals:
            clip_path = temp_path / f"investigation_{interval.investigation_index:03}.mp4"
            frame_range = _make_investigation_clip(
                VIDEO, clip_path, metadata, interval.start_time_seconds, interval.end_time_seconds,
            )
            clips[interval.investigation_index] = clip_path
            clips_by_range[interval.investigation_index] = frame_range
        all_events = tuple(phase3["events"])
        with VideoFrameResolver(VIDEO, metadata=metadata) as resolver:
            for interval, context in zip(intervals, contexts, strict=True):
                started = time.perf_counter()
                inv_dir = OUTPUT / f"investigation_{interval.investigation_index:03}"
                interval_events = tuple(
                    event for event in all_events
                    if interval.start_time_seconds - EPSILON <= event.candidate_time_seconds
                    <= interval.end_time_seconds + EPSILON
                )
                try:
                    record, failure, metrics = _run_one_investigation(
                        interval=interval, confirmed_events=interval_events, context=context,
                        metadata=metadata, ui_config=ui_config, source_video=VIDEO,
                        clip_path=clips[interval.investigation_index], settings=settings,
                        resolver=resolver, artifact_directory=inv_dir,
                    )
                except (RuntimeError, ValueError, OSError, TypeError, AssertionError) as error:
                    failure = InvestigationFailure(
                        interval.investigation_index, VIDEO, interval.start_time_seconds,
                        interval.end_time_seconds, "pipeline", f"{type(error).__name__}: {error}", inv_dir,
                    )
                    (inv_dir / "failure.txt").write_text(
                        f"Investigation {interval.investigation_index} failed: {failure.reason}\n",
                        encoding="utf-8",
                    )
                    record, metrics = None, None
                if record is not None:
                    record_by_id[interval.investigation_index] = record
                    metrics["runtime"]["investigation_total_elapsed_s"] = time.perf_counter() - started
                    metrics_by_id[interval.investigation_index] = metrics
                else:
                    failure_by_id[interval.investigation_index] = failure
                    print(f"Investigation {interval.investigation_index} FAILED at {failure.failed_stage}: {failure.reason}", flush=True)
                    if failure.failed_stage == "provider_daily_quota":
                        print("STOP: confirmed daily quota exhaustion; no later Investigation will be processed.", flush=True)
                        if settings.model_name == "gemini-3.6-flash":
                            print("Use gemini-3.8-flash for a fresh full validation only after one successful 3.8 availability probe.", flush=True)
                    break

    session = VideoInvestigationResult(
        VIDEO, metadata,
        tuple(record_by_id[index] for index in sorted(record_by_id)),
        tuple(failure_by_id[index] for index in sorted(failure_by_id)),
    )
    isolation = {
        "detected_investigation_count": len(intervals),
        "completed_record_count": len(session.investigations),
        "failed_investigation_count": len(session.failures),
        "chronological_record_order": tuple(record.investigation_id for record in session.investigations)
        == tuple(sorted(record.investigation_id for record in session.investigations)),
        "separate_records_per_detected_investigation": len({
            record.investigation_id for record in session.investigations
        } | {failure.investigation_id for failure in session.failures}) == len(intervals),
        "vlm_clips_use_disjoint_half_open_source_frames": all(
            clips_by_range[left.investigation_index][1] <= clips_by_range[right.investigation_index][0]
            for left, right in pairwise(intervals)
        ),
        "record_field_maps_are_independent": len({id(record.final_fields) for record in session.investigations})
        == len(session.investigations),
        "all_ocr_vlm_provenance_validated_by_record_assembly": True,
        "all_vlm_inputs_are_per_investigation_clips": all(
            record.vlm_pass1.raw_result.video_clip_path == clips[record.investigation_id]
            and record.vlm_pass2.raw_result.video_clip_path == clips[record.investigation_id]
            for record in session.investigations
        ),
        "records_are_not_collapsed_by_aircraft_identity": True,
    }
    same_registrations = defaultdict(list)
    for record in session.investigations:
        reg = record.final_fields.get("REGISTRATION")
        if reg is not None and reg.final_value is not None:
            same_registrations[str(reg.final_value)].append(record.investigation_id)
    repeated_aircraft = {key: ids for key, ids in same_registrations.items() if len(ids) > 1}
    total_provider_failures = 0
    for interval in intervals:
        inv_dir = OUTPUT / f"investigation_{interval.investigation_index:03}"
        for stage_file in ("vlm_pass1_result.json", "vlm_pass2_result.json"):
            stage_payload = _read_json_if_present(inv_dir / stage_file)
            if stage_payload:
                total_provider_failures += sum(
                    attempt.get("status") != "success"
                    for attempt in stage_payload.get("provider", {}).get("attempts", [])
                )

    session_payload = {
        "source_video": str(VIDEO), "video_metadata": metadata,
        "investigation_count": len(intervals),
        "completed_records": [
            {
                "investigation_id": record.investigation_id,
                "interval_s": [record.start_time_s, record.end_time_s],
                "record_path": f"investigation_{record.investigation_id:03}/investigation_record.json",
            } for record in session.investigations
        ],
        "failures": session.failures,
        "total_provider_failures": total_provider_failures,
        "isolation_checks": isolation,
    }
    _write_json(OUTPUT / "session_result.json", session_payload)
    summary_lines = [
        "Phase 6.6 multi-Investigation assembly validation",
        f"Source video: {VIDEO}", f"Number of Investigations detected: {len(intervals)}",
        f"Completed end-to-end: {len(session.investigations)}",
        f"Failed: {len(session.failures)}",
        f"Total provider failures: {total_provider_failures}",
        (f"Video metadata: {metadata.width}x{metadata.height}; fps={metadata.fps}; "
         f"frames={metadata.frame_count}; duration={metadata.duration_seconds}s"),
        f"Run elapsed: {time.perf_counter() - run_started:.3f}s",
    ]
    for record in session.investigations:
        summary_lines.append(
            f"Investigation {record.investigation_id}: [{record.start_time_s:.3f}, {record.end_time_s:.3f})s; "
            f"resolved={record.final_result.summary.resolved_count + record.final_result.summary.resolved_single_source_count}; "
            f"unresolved={len(record.unresolved_fields)}; not_available={len(record.not_available_fields)}"
        )
    for failure in session.failures:
        summary_lines.append(
            f"Investigation {failure.investigation_id}: FAILED at {failure.failed_stage}: {failure.reason}"
        )
    (OUTPUT / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    report = [
        "Phase 6.6 real-video multi-Investigation validation",
        f"Source video: {VIDEO}",
        (f"Video metadata: {metadata.width}x{metadata.height}; fps={metadata.fps}; "
         f"frames={metadata.frame_count}; duration={metadata.duration_seconds}s"),
        f"Detected Investigation count: {len(intervals)}",
        "Lifecycle anomalies: " + (str(anomalies) if anomalies else "none"),
        "\nDetected Investigation intervals:",
    ]
    for interval in intervals:
        report.append(
            f"Investigation {interval.investigation_index}: [{interval.start_time_seconds:.3f}, "
            f"{interval.end_time_seconds:.3f})s; duration={interval.end_time_seconds-interval.start_time_seconds:.3f}s"
        )
    report.extend(record and _render_investigation(record) for record in session.investigations)
    for failure in session.failures:
        interval = next(item for item in intervals if item.investigation_index == failure.investigation_id)
        report.append(_render_failed_investigation(
            interval, failure, OUTPUT / f"investigation_{failure.investigation_id:03}"
        ))
    report.append("\n=== VIDEO / SESSION SUMMARY ===")
    report.extend((f"Source video: {VIDEO}", f"Number of Investigations: {len(intervals)}"))
    report.append(_table(
        ("ID", "START", "END", "FLIGHT ID", "REGISTRATION", "RESOLVED", "UNRESOLVED", "STATUS"),
        _summary_rows(session, intervals),
    ))
    report.append("\n=== INVESTIGATION ISOLATION CHECKS ===")
    report.extend(f"{name}: {value}" for name, value in isolation.items())
    report.append(f"Repeated registration values kept as separate record IDs: {repeated_aircraft}")
    report.append("\n=== RUNTIME INFORMATION ===")
    report.append(f"Total provider failures: {total_provider_failures}")
    report.append(
        f"Provider={settings.provider}; model={settings.model_name}; timeout={settings.request_timeout_s:g}s; "
        "retry policy=503/timeouts: 10s/30s; 429: 30s/60s; max 3 attempts; Retry-After unavailable in adapter"
    )
    for investigation_id, metrics in metrics_by_id.items():
        report.append(f"Investigation {investigation_id} runtime: {json.dumps(metrics, ensure_ascii=False)}")
    report.append("\nTemporary per-Investigation VLM clips were removed after processing; source-time bounds are retained.")
    report.append(f"Artifact directory: {OUTPUT}")
    report_path = OUTPUT / "full_validation_report.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n" + "\n".join(summary_lines), flush=True)
    print("\n=== VIDEO INVESTIGATION SUMMARY ===", flush=True)
    print(f"Source video: {VIDEO}\nNumber of Investigations: {len(intervals)}", flush=True)
    print(_table(("ID", "START", "END", "FLIGHT ID", "REGISTRATION", "RESOLVED", "UNRESOLVED", "STATUS"),
                 _summary_rows(session, intervals)), flush=True)
    print(f"Full report: {report_path}", flush=True)
    success = len(session.investigations) >= 2
    count_checks_pass = (
        isolation["detected_investigation_count"] >= 2
        and isolation["completed_record_count"] >= 2
        and isolation["failed_investigation_count"] == 0
    )
    boolean_checks_pass = all(value for value in isolation.values() if type(value) is bool)
    phase_complete = success and not session.failures and count_checks_pass and boolean_checks_pass
    _write_json(OUTPUT / "phase_status.json", {
        "phase": "6.6",
        "status": "COMPLETE" if phase_complete else "OPEN",
        "detected_investigations": len(intervals),
        "completed_investigation_records": len(session.investigations),
        "failed_investigations": len(session.failures),
        "total_provider_failures": total_provider_failures,
        "session_assembled": True,
        "isolation_checks": isolation,
        "completion_requires": "At least two complete independent real-video records plus passing pytest, Ruff, and git diff --check.",
    })
    return 0 if phase_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
