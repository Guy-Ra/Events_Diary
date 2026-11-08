"""Application entry point and Phase 9 video-to-Investigation orchestrator.

The video and Investigation pipeline is composed from existing production APIs.
Imports remain side-effect-free, all understanding modes share preparation, and
mode-specific OCR/VLM processing ends at the in-memory VideoInvestigationResult.
One shared mode-neutral back half then builds an in-memory CanonicalSession,
renders both diary strings, and persists only those final UTF-8 outputs. Image
frames and temporary VLM clips/crops are transient.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from events_diary.config import (
    AppConfig,
    RuntimeConfig,
    UnderstandingMode,
    load_runtime_config,
)
from events_diary.logging_config import configure_logging
from events_diary.runtime import get_runtime_profile

if TYPE_CHECKING:
    from events_diary.adaptive_sampling import AdaptiveSamplingPlan
    from events_diary.evidence_candidate_timeline import EvidenceCandidate
    from events_diary.frame_materialization import MaterializedEvidenceCandidate
    from events_diary.investigation_lifecycle import InvestigationInterval
    from events_diary.investigation_record import VideoInvestigationResult
    from events_diary.investigation_signals import (
        InvestigationPresenceEvidence,
        InvestigationSignals,
    )
    from events_diary.investigation_temporal import (
        CandidateTimelineEntry,
        ConfirmedInvestigationEvent,
    )
    from events_diary.investigation_window import InvestigationWindow
    from events_diary.representative_selection import InvestigationSelection
    from events_diary.ticket_scroll import TicketScrollState
    from events_diary.ticket_stable_views import TicketStableView
    from events_diary.video import VideoMetadata

logger = logging.getLogger(__name__)

# MP4 is the only source-video format used in the repository's completed
# end-to-end validations; do not claim unvalidated decoder formats as MVP input.
SUPPORTED_VIDEO_SUFFIXES = frozenset({".mp4"})
OPERATIONAL_OUTPUT_FILENAME = "operational_diary.txt"
DETAILED_OUTPUT_FILENAME = "detailed_diary.txt"
VLM_RESULTS_FILENAME = "vlm_run_results.json"
UNDERSTANDING_MODE_LABELS = {
    UnderstandingMode.OCR_ONLY: "OCR only",
    UnderstandingMode.VLM_ONLY: "VLM only",
    UnderstandingMode.OCR_VLM: "OCR + VLM",
}


class UnderstandingModeReadiness(str, Enum):
    """Report whether current Phase 6 contracts can represent a selected mode."""

    SUPPORTED_NOW = "supported_now"
    REQUIRES_PHASE_9_3_ADAPTATION = "requires_phase_9_3_adaptation"


MODE_READINESS = {
    UnderstandingMode.OCR_ONLY: UnderstandingModeReadiness.SUPPORTED_NOW,
    UnderstandingMode.VLM_ONLY: UnderstandingModeReadiness.SUPPORTED_NOW,
    UnderstandingMode.OCR_VLM: UnderstandingModeReadiness.SUPPORTED_NOW,
}
_MODE_FEATURES = {
    UnderstandingMode.OCR_ONLY: (True, False, False),
    UnderstandingMode.VLM_ONLY: (False, True, False),
    UnderstandingMode.OCR_VLM: (True, True, True),
}


class InputVideoSelectionError(ValueError):
    """Raised when the single-video MVP input policy is not satisfied."""


class UnsupportedUnderstandingModeError(ValueError):
    """Raised when current Phase 6 record contracts cannot represent a mode."""


class PipelineNotReadyError(RuntimeError):
    """Retained for callers that need to identify a disconnected later stage."""


class IncompleteVideoRunError(RuntimeError):
    """Raised when one or more required Investigations did not complete."""


@dataclass(frozen=True)
class PipelineRunPlan:
    """Resolved, side-effect-free plan for one input video and two diary files."""

    video_path: Path
    understanding_mode: UnderstandingMode
    operational_output_path: Path
    detailed_output_path: Path


@dataclass(frozen=True)
class InvestigationDetectionResult:
    """Transient pair-level records needed to construct evidence per interval."""

    presences: tuple[InvestigationPresenceEvidence, ...]
    signals: tuple[InvestigationSignals, ...]
    timeline: tuple[CandidateTimelineEntry, ...]
    scroll_states: tuple[TicketScrollState, ...]
    confirmed_events: tuple[ConfirmedInvestigationEvent, ...]
    intervals: tuple[InvestigationInterval, ...]


@dataclass(frozen=True)
class PreparedInvestigation:
    """One lifecycle interval and its selected, scalar-only evidence references.

    Candidate/materialization sequences remain Investigation-local so their
    indexes retain the identity expected by existing OCR and later VLM APIs.
    No source-frame image or copied video payload is retained.
    """

    investigation_id: int
    interval: InvestigationInterval
    window: InvestigationWindow
    sampling_plan: AdaptiveSamplingPlan
    stable_views: tuple[TicketStableView, ...]
    evidence_candidates: tuple[EvidenceCandidate, ...]
    materialized_candidates: tuple[MaterializedEvidenceCandidate, ...]
    representative_selection: InvestigationSelection
    confirmed_events: tuple[ConfirmedInvestigationEvent, ...] = ()


@dataclass(frozen=True)
class VideoPipelineResult:
    """Completed video preparation boundary for Phase 9.3 understanding modes."""

    metadata: VideoMetadata
    investigations: tuple[PreparedInvestigation, ...]


@dataclass(frozen=True)
class PipelineOutput:
    """Paths and overall status for one diary pipeline execution."""

    operational_diary_path: Path
    detailed_diary_path: Path
    run_status: str = "completed"
    vlm_results_path: Path | None = None


def finalize_pipeline_result(
    video_result: VideoInvestigationResult,
    output_dir: Path,
    understanding_mode: UnderstandingMode,
) -> PipelineOutput:
    """Canonicalize results, represent VLM failures, and write UTF-8 outputs."""
    from dataclasses import replace

    from events_diary.canonical_events import (
        CanonicalDiaryFailure,
        build_canonical_session,
    )
    from events_diary.detailed_diary import render_detailed_diary
    from events_diary.operational_diary import render_operational_diary

    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be pathlib.Path.")
    if not isinstance(understanding_mode, UnderstandingMode):
        raise TypeError("understanding_mode must be UnderstandingMode.")
    if video_result.failures and understanding_mode is not UnderstandingMode.VLM_ONLY:
        _reject_failed_investigations(video_result, understanding_mode)
    failures = tuple(CanonicalDiaryFailure(
        item.investigation_id, item.start_time_s, item.end_time_s,
        _safe_failure_reason(item.reason),
    ) for item in video_result.failures)
    session_input = replace(video_result, failures=()) if video_result.failures else video_result
    session = build_canonical_session(session_input)
    operational_body = (
        render_operational_diary(session, failures=failures)
        if failures else render_operational_diary(session)
    )
    detailed_body = (
        render_detailed_diary(session, failures=failures)
        if failures else render_detailed_diary(session)
    )
    run_status = (
        "partial" if video_result.investigations and video_result.failures
        else "failed" if video_result.failures
        else "completed"
    )
    header = f"Understanding mode: {UNDERSTANDING_MODE_LABELS[understanding_mode]}"
    if understanding_mode is UnderstandingMode.VLM_ONLY:
        header = f"Run status: {run_status}\n{header}"
    operational_text = header + (f"\n\n{operational_body}" if operational_body else "")
    detailed_text = header + (f"\n\n{detailed_body}" if detailed_body else "")

    output_dir.mkdir(parents=True, exist_ok=True)
    operational_path = output_dir / OPERATIONAL_OUTPUT_FILENAME
    detailed_path = output_dir / DETAILED_OUTPUT_FILENAME
    vlm_results_path = None
    writes = [(operational_path, operational_text), (detailed_path, detailed_text)]
    if understanding_mode is UnderstandingMode.VLM_ONLY:
        vlm_results_path = output_dir / VLM_RESULTS_FILENAME
        writes.append((vlm_results_path, _vlm_results_json(video_result, run_status)))
    temporary_paths = []
    try:
        for target, content in writes:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output_dir,
                prefix=f".{target.name}.", suffix=".tmp", delete=False,
            ) as handle:
                handle.write(content)
                temporary = Path(handle.name)
            temporary_paths.append((temporary, target))
        for temporary, target in temporary_paths:
            temporary.replace(target)
    finally:
        for temporary, _ in temporary_paths:
            temporary.unlink(missing_ok=True)
    return PipelineOutput(operational_path, detailed_path, run_status, vlm_results_path)


def _vlm_results_json(video_result, run_status: str) -> str:
    """Serialize successful structured VLM evidence and sanitized failure diagnostics."""
    successful = []
    for record in video_result.investigations:
        result = record.vlm_pass1
        if result is None:
            continue
        raw = result.raw_result
        selected = []
        selection_record = getattr(record, "selection", None)
        materialized_candidates = getattr(record, "materialized_candidates", ())
        for selection in selection_record.selections if selection_record is not None else ():
            index = selection.candidate_index
            if index >= len(materialized_candidates):
                continue
            item = materialized_candidates[index]
            selected.append({
                "frame_index": item.frame_index,
                "source_timestamp_s": item.actual_timestamp_seconds,
                "selection_reasons": list(selection.selection_reasons),
            })
        route = result.visual_route
        successful.append({
            "investigation_id": record.investigation_id,
            "source_video": str(record.source_video),
            "interval": {"start_s": record.start_time_s, "end_s": record.end_time_s},
            "model": raw.model_name,
            "model_revision": raw.model_revision,
            "prompt_version": raw.prompt_version,
            "selected_evidence": selected,
            "fields": [
                {"name": item.field_name, "value": item.raw_value, "status": item.status,
                 "evidence_time_s": item.evidence_time_s}
                for item in result.observations
            ],
            "visual_route": None if route is None else {
                name: {"value": claim.value, "status": claim.status}
                for name, claim in (
                    ("direction_sector", route.direction_sector),
                    ("direction_text", route.direction_text),
                    ("route_description", route.route_description),
                    ("departure_context", route.departure_context),
                )
            },
        })
    failed = []
    for item in video_result.failures:
        diagnostics = None
        if item.provider_diagnostics_json:
            try:
                diagnostics = json.loads(item.provider_diagnostics_json)
            except json.JSONDecodeError:
                diagnostics = {"diagnostic_parse_error": "provider diagnostics were not valid JSON"}
        failed.append({
            "investigation_id": item.investigation_id,
            "source_video": str(item.source_video),
            "interval": {"start_s": item.start_time_s, "end_s": item.end_time_s},
            "failed_stage": item.failed_stage,
            "error_type": item.error_type,
            "reason": _safe_failure_reason(item.reason),
            "provider_diagnostics": diagnostics,
        })
    return json.dumps({
        "run_status": run_status,
        "source_video": str(video_result.source_video),
        "investigation_count": video_result.investigation_count,
        "successful_investigations": successful,
        "failed_investigations": failed,
    }, ensure_ascii=False, indent=2) + "\n"


def _safe_failure_reason(reason: str) -> str:
    """Return a short failure reason without credentials or provider payloads."""
    text = " ".join(reason.split())
    if text.startswith(("{", "[")):
        return "Provider or runtime returned a structured error response."
    text = re.sub(r"(?i)(x-goog-api-key|api[_ -]?key|authorization)\s*[:=]\s*\S+", r"\1=[redacted]", text)
    text = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[redacted]", text)
    text = re.sub(r"\bBearer\s+\S+", "Bearer [redacted]", text, flags=re.IGNORECASE)
    return text[:240] or "No concise reason was supplied."


def _reject_failed_investigations(video_result, understanding_mode: UnderstandingMode) -> None:
    """Log every failed Investigation, then block canonicalization and output."""
    failures = video_result.failures
    if not failures:
        return
    logger.error(
        "Cannot produce complete diaries: %d Investigation(s) failed in %s mode.",
        len(failures), understanding_mode.value,
    )
    for failure in failures:
        logger.error(
            "Investigation %s [%.3f, %.3f) failed | mode=%s | stage=%s | error_type=%s | reason=%s",
            failure.investigation_id, failure.start_time_s, failure.end_time_s,
            understanding_mode.value, failure.failed_stage,
            failure.error_type or "UnknownError", _safe_failure_reason(failure.reason),
        )
    identities = ", ".join(str(item.investigation_id) for item in failures)
    raise IncompleteVideoRunError(
        f"Cannot produce complete diaries: {len(failures)} Investigation(s) failed "
        f"(IDs: {identities}). See the per-Investigation failure details above."
    )


def _log_ocr_only_fields(video_result) -> None:
    """Log meaningful resolved OCR-only diary fields before canonical rendering."""
    fields = (
        ("primary_id", "primary_flight_id"),
        ("alternate_id", "alternate_flight_id"),
        ("airline", "airline_name"),
        ("origin", "origin_airport_code"),
        ("destination", "destination_airport_code"),
        ("scheduled_departure", "origin_scheduled_time"),
        ("actual_departure", "origin_actual_time"),
        ("scheduled_arrival", "destination_scheduled_time"),
        ("estimated_or_actual_arrival", "destination_estimated_or_actual_time"),
        ("registration", "REGISTRATION"),
        ("aircraft_code", "aircraft_code"),
        ("aircraft_type", "AIRCRAFT_TYPE"),
        ("TRACK", "TRACK"),
    )
    for record in video_result.investigations:
        final_fields = record.final_result.fields
        values = {
            label: (
                final_fields[name].final_value
                if name in final_fields and final_fields[name].final_status in ("resolved", "resolved_single_source")
                else None
            )
            for label, name in fields
        }
        consistency = record.final_result.direction_consistency
        values["direction"] = (
            consistency.derived_direction.text
            if consistency is not None and consistency.derived_direction is not None else None
        )
        logger.info(
            "OCR-only final fields | investigation_id=%s | interval=[%.3f, %.3f) | fields=%s",
            record.investigation_id, record.start_time_s, record.end_time_s, values,
        )


def select_input_video(input_dir: Path) -> Path:
    """Select the only immediate MP4 file or fail with a clear input error."""
    if not isinstance(input_dir, Path):
        raise TypeError("input_dir must be pathlib.Path.")
    if not input_dir.exists():
        raise FileNotFoundError(f"MVP input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"MVP input path is not a directory: {input_dir}")
    videos = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )
    if not videos:
        raise InputVideoSelectionError(
            f"No supported video found in {input_dir}; place exactly one .mp4 file there.",
        )
    if len(videos) > 1:
        names = ", ".join(path.name for path in videos)
        raise InputVideoSelectionError(
            f"Expected exactly one .mp4 video in {input_dir}; found {len(videos)}: {names}.",
        )
    return videos[0].resolve()


def build_pipeline_run_plan(config: RuntimeConfig) -> PipelineRunPlan:
    """Resolve video and fixed output paths without running processing stages."""
    if not isinstance(config, RuntimeConfig):
        raise TypeError("config must be RuntimeConfig.")
    video_path = select_input_video(config.input_dir)
    return PipelineRunPlan(
        video_path=video_path,
        understanding_mode=config.understanding_mode,
        operational_output_path=config.output_dir / OPERATIONAL_OUTPUT_FILENAME,
        detailed_output_path=config.output_dir / DETAILED_OUTPUT_FILENAME,
    )


def detect_investigations(
    metadata: VideoMetadata, ui_config=None,
) -> InvestigationDetectionResult:
    """Run existing sampled-frame, pair-signal, confirmation and lifecycle APIs.

    Source frames are consumed in order and released with this helper's local
    sample collection. Only the immutable measurements and temporal records
    needed by evidence preparation survive its return.
    """
    from events_diary.config import UIConfig
    from events_diary.investigation_detection import detect_investigation_candidate
    from events_diary.investigation_lifecycle import build_investigation_lifecycle
    from events_diary.investigation_signals import (
        analyze_investigation_pair,
        estimate_investigation_presence,
        is_investigation_present,
    )
    from events_diary.investigation_temporal import (
        CandidateTimelineEntry,
        bridge_transient_absence,
        confirm_investigation_candidates,
    )
    from events_diary.ticket_scroll import analyze_ticket_scroll_sequence
    from events_diary.video import sample_video

    ui_config = UIConfig() if ui_config is None else ui_config
    samples = sample_video(metadata.path, metadata.fps)
    if len(samples) != metadata.frame_count:
        raise ValueError(
            f"Sampling returned {len(samples)} of {metadata.frame_count} reported frames "
            f"from {metadata.path}.",
        )
    if any(sample.frame_index != index for index, sample in enumerate(samples)):
        raise ValueError(f"Video sampling did not return contiguous source frames: {metadata.path}")
    if len(samples) < 2:
        raise ValueError(f"At least two sampled frames are required: {metadata.path}")

    presences = tuple(estimate_investigation_presence(sample.image, ui_config) for sample in samples)
    signals = tuple(
        analyze_investigation_pair(samples[index - 1].image, sample.image, ui_config)
        for index, sample in enumerate(samples)
        if index > 0
    )
    del samples

    effective_presence = bridge_transient_absence(tuple(
        (index / metadata.fps, is_investigation_present(evidence))
        for index, evidence in enumerate(presences)
    ))
    timeline = tuple(
        CandidateTimelineEntry(
            pair_index=index,
            time_before_seconds=index / metadata.fps,
            time_after_seconds=(index + 1) / metadata.fps,
            present_before=effective_presence[index],
            present_after=effective_presence[index + 1],
            candidate=detect_investigation_candidate(
                presences[index],
                presences[index + 1],
                signals[index],
                effective_presence_before=effective_presence[index],
                effective_presence_after=effective_presence[index + 1],
            ),
        )
        for index in range(len(signals))
    )
    scroll_states = tuple(analyze_ticket_scroll_sequence(
        tuple(signal.ticket_scroll_evidence for signal in signals),
    ))
    confirmed_events, _ = confirm_investigation_candidates(timeline)
    intervals, anomalies = build_investigation_lifecycle(
        initial_present=effective_presence[0],
        video_start_time_seconds=0.0,
        video_end_time_seconds=metadata.duration_seconds,
        confirmed_events=confirmed_events,
    )
    if anomalies:
        logger.warning("Investigation lifecycle reported %d anomaly/anomalies.", len(anomalies))
    return InvestigationDetectionResult(
        presences=presences,
        signals=signals,
        timeline=timeline,
        scroll_states=scroll_states,
        confirmed_events=tuple(confirmed_events),
        intervals=tuple(intervals),
    )


def prepare_investigation_evidence(
    metadata: VideoMetadata, detection: InvestigationDetectionResult, ui_config=None,
) -> tuple[PreparedInvestigation, ...]:
    """Build windows through representative references for each detected interval.

    Existing Phase 4/5 APIs own all calculations. Each interval is processed
    separately, keeping repeated Investigations distinct and candidate indexes
    local to their original lifecycle identity. The frame resolver and its cache
    are closed after each interval; only scalar materialization records survive.
    """
    from events_diary.adaptive_sampling import (
        SamplingSignalEntry,
        build_adaptive_sampling_plan,
    )
    from events_diary.config import UIConfig
    from events_diary.evidence_candidate_timeline import (
        build_evidence_candidate_timeline,
    )
    from events_diary.frame_materialization import (
        VideoFrameResolver,
        materialize_evidence_candidates,
    )
    from events_diary.investigation_window import build_investigation_windows
    from events_diary.representative_selection import (
        select_investigation_representatives,
    )
    from events_diary.ticket_stable_views import (
        TicketStableViewEntry,
        detect_ticket_stable_views,
    )

    ui_config = UIConfig() if ui_config is None else ui_config
    windows = build_investigation_windows(
        detection.intervals,
        video_start_time_seconds=0.0,
        video_end_time_seconds=metadata.duration_seconds,
    )
    prepared = []
    for interval, window in zip(detection.intervals, windows, strict=True):
        core = tuple(
            entry for entry in detection.timeline
            if interval.start_time_seconds <= entry.time_before_seconds
            and entry.time_after_seconds <= interval.end_time_seconds
        )
        stable_entries = tuple(
            TicketStableViewEntry(
                entry.pair_index,
                entry.time_before_seconds,
                entry.time_after_seconds,
                entry.present_before and entry.present_after,
                detection.signals[entry.pair_index].ticket_scroll_detected,
                detection.signals[entry.pair_index].ticket_body_change,
                detection.scroll_states[entry.pair_index],
            )
            for entry in core
        )
        window_entries = tuple(
            entry for entry in detection.timeline
            if window.window_start_time_seconds <= entry.time_before_seconds
            and entry.time_after_seconds <= window.window_end_time_seconds
        )
        sampling_entries = tuple(
            SamplingSignalEntry(
                entry.pair_index,
                entry.time_before_seconds,
                entry.time_after_seconds,
                detection.signals[entry.pair_index],
            )
            for entry in window_entries
        )
        sampling_plan = build_adaptive_sampling_plan(
            window=window,
            signal_entries=sampling_entries,
            source_fps=metadata.fps,
        )
        stable_views = detect_ticket_stable_views(
            investigation_index=interval.investigation_index,
            entries=stable_entries,
            investigation_start_time_seconds=interval.start_time_seconds,
            investigation_end_time_seconds=interval.end_time_seconds,
        )
        candidates = build_evidence_candidate_timeline(
            window=window,
            sampling_plan=sampling_plan,
            stable_views=stable_views,
        )
        with VideoFrameResolver(metadata.path, metadata=metadata) as resolver:
            materialized = materialize_evidence_candidates(
                video_path=metadata.path,
                candidates=candidates,
                ui_config=ui_config,
                metadata=metadata,
                resolver=resolver,
            )
        selection = select_investigation_representatives(
            window=window,
            sampling_plan=sampling_plan,
            candidates=candidates,
            materialized_candidates=materialized,
            stable_views=stable_views,
        )
        prepared.append(PreparedInvestigation(
            investigation_id=interval.investigation_index,
            interval=interval,
            window=window,
            sampling_plan=sampling_plan,
            stable_views=tuple(stable_views),
            evidence_candidates=tuple(candidates),
            materialized_candidates=tuple(materialized),
            representative_selection=selection,
            confirmed_events=tuple(
                event for event in detection.confirmed_events
                if interval.start_time_seconds - 1e-9 <= event.candidate_time_seconds
                <= interval.end_time_seconds + 1e-9
            ),
        ))
    return tuple(prepared)


def prepare_video_pipeline(config: RuntimeConfig) -> VideoPipelineResult:
    """Prepare the one configured video to the ready-for-understanding boundary."""
    from events_diary.video import read_video_metadata

    plan = build_pipeline_run_plan(config)
    metadata = read_video_metadata(plan.video_path)
    detection = detect_investigations(metadata)
    investigations = prepare_investigation_evidence(metadata, detection)
    return VideoPipelineResult(metadata, investigations)


class _UnderstandingStageError(RuntimeError):
    """Identify which required branch failed while retaining the original cause."""

    def __init__(self, stage: str, cause: Exception, *, provider_diagnostics_json: str | None = None):
        super().__init__(str(cause) or type(cause).__name__)
        self.stage = stage
        self.cause = cause
        self.provider_diagnostics_json = provider_diagnostics_json


def _make_continuous_clip(metadata, investigation, destination: Path) -> None:
    """Write one temporary half-open continuous Investigation clip from source frames."""
    import cv2

    first = round(investigation.interval.start_time_seconds * metadata.fps)
    end = round(investigation.interval.end_time_seconds * metadata.fps)
    if end <= first:
        raise ValueError("Investigation interval contains no source frames.")
    capture = cv2.VideoCapture(str(metadata.path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not decode source video: {metadata.path}")
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create temporary Investigation clip.")
    written = 0
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        for _ in range(first, end):
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        capture.release()
    if written != end - first:
        raise RuntimeError(f"Investigation clip decode was incomplete ({written}/{end-first} frames).")


def _run_pass1(runtime, metadata, investigation, clip_path, *, ticket_images=()):
    """Run and validate the existing full-clip VLM Pass 1 parser contract."""
    from events_diary.vlm import VLMRequest
    from events_diary.vlm_parsing import parse_vlm_result
    from events_diary.vlm_prompts import FR24_VLM_V1, build_vlm_prompt

    interval = investigation.interval
    prompt = build_vlm_prompt(FR24_VLM_V1)
    if ticket_images:
        prompt += (
            "\n\nYou are analyzing this one Investigation: the continuous clip and every ticket "
            "image belong to the same Investigation and aircraft. Use the continuous video for "
            "scrolling, aircraft/map movement, historical trail, and visual route context. Use "
            "the clearest supplied ticket crop first for small ticket text, including flight IDs, "
            "airline, origin/destination, all four times, registration, aircraft code/type, and "
            "ticket TRACK. Copy only what is visibly supported; never guess or replace ticket "
            "text with outside aviation knowledge. In particular, never infer registration or "
            "aircraft variant/code from appearance, airline, or flight identity, and never guess "
            "times. Preserve displayed times as written. Return null when a value is unreadable "
            "or not visible; use N/A only when that exact field visibly says N/A. Keep ticket "
            "TRACK separate from map direction. Report visual direction, highlighted/historical "
            "route, geographic movement, and departure context separately from ticket facts; "
            "do not use map interpretation to fill origin, destination, TRACK, registration, or "
            "aircraft type."
        )
    request = VLMRequest(
        investigation_index=investigation.investigation_id, source_video=metadata.path,
        video_clip_path=clip_path, source_start_s=interval.start_time_seconds,
        source_end_s=interval.end_time_seconds, prompt=prompt,
        prompt_version=FR24_VLM_V1, ticket_images=tuple(ticket_images),
    )
    result = parse_vlm_result(request, runtime.run(request))
    diagnostics = result.parse_diagnostics
    if (result.raw_result.execution_status != "success" or diagnostics is None
            or diagnostics.status not in ("valid_json", "recovered_markdown_fence")
            or result.visual_route is None or len(result.observations) != 36):
        cause = RuntimeError(
            result.raw_result.error or (diagnostics.error if diagnostics else None)
            or "VLM Pass 1 did not produce the complete validated response."
        )
        if result.raw_result.execution_status != "success":
            raise _UnderstandingStageError(
                "vlm_pass1", cause,
                provider_diagnostics_json=result.raw_result.provider_diagnostics_json,
            )
        raise cause
    return request, result


def _vlm_ticket_images(prepared, metadata, ui_config, directory):
    """Materialize up to two selected full-ticket crops for VLM-only Pass 1."""
    import cv2

    from events_diary.frame_materialization import VideoFrameResolver
    from events_diary.ticket_ocr import ticket_ocr_roi
    from events_diary.vlm import VLMImageInput

    selections = prepared.representative_selection.selections
    materialized = prepared.materialized_candidates
    candidates = prepared.evidence_candidates
    eligible = []
    for selection in selections:
        reasons = set(selection.selection_reasons)
        if "stable_ticket" not in reasons and "post_scroll_stable" not in reasons:
            continue
        index = selection.candidate_index
        if not 0 <= index < len(materialized) or index >= len(candidates):
            continue
        item = materialized[index]
        if (item.resolution_status not in ("resolved", "video_end_adjusted")
                or item.frame_index is None or item.actual_timestamp_seconds is None):
            continue
        eligible.append((selection, item, candidates[index]))

    upper = next(
        (item for item in eligible
         if "stable_ticket" in item[0].selection_reasons
         and "post_scroll_stable" not in item[0].selection_reasons),
        next((item for item in eligible if "stable_ticket" in item[0].selection_reasons), None),
    )
    lower = next(
        (item for item in reversed(eligible) if "post_scroll_stable" in item[0].selection_reasons),
        None,
    )
    chosen = []
    for role_item in (upper, lower):
        if role_item is not None and all(
            role_item[1].frame_index != prior[1].frame_index for prior in chosen
        ):
            chosen.append(role_item)
    if not chosen:
        logger.info(
            "VLM-only Investigation %s has no usable selected ticket representatives; sending video only.",
            prepared.investigation_id,
        )
        return ()

    images = []
    with VideoFrameResolver(metadata.path, metadata=metadata) as resolver:
        for selection, item, _candidate in chosen:
            frame = resolver.resolve(item.actual_timestamp_seconds)
            if frame.frame_index != item.frame_index or frame.image is None:
                continue
            crop = ticket_ocr_roi(
                ui_config, frame.image.shape[1], frame.image.shape[0],
            ).crop(frame.image)
            crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            role = (
                "post_scroll_ticket_details"
                if "post_scroll_stable" in selection.selection_reasons
                else "upper_ticket_header_and_route_times"
            )
            section = (
                "lower ticket fields visible after scrolling"
                if role == "post_scroll_ticket_details"
                else "ticket header, flight identity, route, and times"
            )
            path = directory / f"ticket_{item.frame_index:06d}_{item.actual_timestamp_seconds:.3f}s.png"
            ok, encoded = cv2.imencode(".png", crop)
            if not ok:
                continue
            path.write_bytes(encoded.tobytes())
            images.append(VLMImageInput(path, item.actual_timestamp_seconds, role, section))
    logger.info(
        "VLM-only Investigation %s selected %d ticket crop(s): %s",
        prepared.investigation_id, len(images),
        [{"time_s": image.source_timestamp_s, "role": image.role, "section": image.section}
         for image in images],
    )
    return tuple(images)


def _log_vlm_debug_summary(video_result) -> None:
    """Log parsed Pass 1 values and statuses transiently for one diagnostic run."""
    for record in video_result.investigations:
        result = record.vlm_pass1
        if result is None:
            continue
        fields = {
            item.field_name: {"value": item.raw_value, "status": item.status}
            for item in result.observations
        }
        route = result.visual_route
        route_summary = None if route is None else {
            name: {"value": claim.value, "status": claim.status}
            for name, claim in (
                ("visual_flight_direction_sector", route.direction_sector),
                ("visual_flight_direction_text", route.direction_text),
                ("visual_route_description", route.route_description),
                ("visual_departure_context", route.departure_context),
            )
        }
        logger.info(
            "VLM-only Pass 1 in-memory summary Investigation %s: %s",
            record.investigation_id,
            json.dumps({"ticket_fields": fields, "visual_route": route_summary}, ensure_ascii=False),
        )


def _run_ocr(prepared, metadata, engine):
    """Run the existing selected-representative OCR and spatial extraction path."""
    from events_diary.config import UIConfig
    from events_diary.frame_materialization import VideoFrameResolver
    from events_diary.ocr_fields import extract_ocr_fields_batch
    from events_diary.ticket_ocr import run_ocr_for_representatives

    ui_config = UIConfig()
    with VideoFrameResolver(metadata.path, metadata=metadata) as resolver:
        results = run_ocr_for_representatives(
            video_path=metadata.path, investigation_index=prepared.investigation_id,
            selections=prepared.representative_selection.selections,
            candidates=prepared.evidence_candidates,
            materialized_candidates=prepared.materialized_candidates,
            ui_config=ui_config, resolver=resolver, engine=engine,
        )
    occurrences = tuple(item for batch in extract_ocr_fields_batch(results) for item in batch)
    return results, occurrences


def _ticket_detail_images(prepared, ocr_results, metadata, ui_config, directory):
    """Create temporary 2x crops from the exact OCR-selected ticket representatives."""
    import cv2

    from events_diary.frame_materialization import VideoFrameResolver
    from events_diary.ticket_ocr import ticket_ocr_roi
    from events_diary.vlm_review import TicketDetailImage

    selection_by_index = {
        item.candidate_index: item for item in prepared.representative_selection.selections
    }
    seen, images = set(), []
    with VideoFrameResolver(metadata.path, metadata=metadata) as resolver:
        for result in ocr_results:
            if result.frame_index in seen:
                continue
            seen.add(result.frame_index)
            selected = selection_by_index[result.candidate_index]
            role = (
                "post_scroll_stable" if "post_scroll_stable" in selected.selection_reasons
                else "stable_ticket"
            )
            section = "lower scrolled ticket details" if role == "post_scroll_stable" else "upper ticket and flight identity"
            frame = resolver.resolve(result.actual_timestamp_seconds)
            if frame.frame_index != result.frame_index or frame.image is None:
                raise RuntimeError("Pass 2 frame differs from its selected OCR representative.")
            crop = ticket_ocr_roi(ui_config, frame.image.shape[1], frame.image.shape[0]).crop(frame.image)
            crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            path = directory / f"ticket_{result.frame_index:06d}_{result.actual_timestamp_seconds:.3f}s.png"
            ok, encoded = cv2.imencode(".png", crop)
            if not ok:
                raise RuntimeError("Could not encode selected ticket crop for Pass 2.")
            path.write_bytes(encoded.tobytes())
            images.append(TicketDetailImage(
                path, result.actual_timestamp_seconds, role, crop.shape[1], crop.shape[0], section,
            ))
    if not images:
        raise RuntimeError("Phase 5 selection supplied no eligible ticket images for Pass 2 v3.")
    return tuple(images)


def _understand_one(prepared, metadata, mode, *, ocr_engine=None, vlm_runtime=None):
    """Execute the selected mode for one prepared Investigation and assemble its record."""
    import tempfile

    from events_diary.config import UIConfig
    from events_diary.fusion import (
        FusedInvestigationResult,
        apply_temporal_awareness,
        compare_field_observations,
        resolve_investigation,
        resolve_single_source_investigation,
    )
    from events_diary.investigation_record import assemble_investigation_record
    from events_diary.vlm_review import (
        FR24_VLM_REVIEW_V3,
        REVIEW_FIELD_NAMES,
        REVIEW_ROUTE_FIELDS,
        build_vlm_review_request,
        parse_vlm_review_result,
    )

    interval = prepared.interval
    runs_ocr, runs_vlm, runs_fusion = _MODE_FEATURES[mode]
    base = {
        "investigation_id": prepared.investigation_id,
        "source_video": metadata.path,
        "interval": interval,
        "confirmed_events": prepared.confirmed_events,
        "window": prepared.window,
        "sampling_plan": prepared.sampling_plan,
        "stable_views": prepared.stable_views,
        "candidates": prepared.evidence_candidates,
        "materialized_candidates": prepared.materialized_candidates,
        "selection": prepared.representative_selection,
    }
    ocr_results, occurrences = (), ()
    record_occurrences = ()
    pass1 = fusion = review = None
    try:
        if runs_ocr:
            try:
                ocr_results, occurrences = _run_ocr(prepared, metadata, ocr_engine)
                record_occurrences = occurrences
                if mode is UnderstandingMode.OCR_ONLY:
                    from events_diary.ocr_upper_ticket import (
                        canonical_ocr_observations,
                        extract_fr24_tickets_batch,
                    )

                    tickets = extract_fr24_tickets_batch(ocr_results)
                    upper_occurrences = tuple(
                        item for ticket in tickets for item in canonical_ocr_observations(ticket)
                    )
                    record_occurrences = (*occurrences, *upper_occurrences)
            except Exception as error:
                raise _UnderstandingStageError("ocr", error) from error
        if not runs_vlm:
            single_source_occurrences = tuple(
                item for item in occurrences
                if item.canonical_label not in ("SCHEDULED", "ACTUAL", "ESTIMATED")
            )
            if mode is UnderstandingMode.OCR_ONLY:
                single_source_occurrences = (*single_source_occurrences, *upper_occurrences)
            final = resolve_single_source_investigation(
                investigation_id=prepared.investigation_id, source_video=metadata.path,
                start_time_s=interval.start_time_seconds, end_time_s=interval.end_time_seconds,
                observations=single_source_occurrences,
            )
        else:
            if vlm_runtime is None:
                raise _UnderstandingStageError("vlm_pass1", RuntimeError("VLM runtime is not configured."))
            with tempfile.TemporaryDirectory(prefix="events-diary-investigation-") as temporary:
                directory = Path(temporary)
                clip_path = directory / "investigation.mp4"
                try:
                    _make_continuous_clip(metadata, prepared, clip_path)
                    ticket_images = ()
                    if mode is UnderstandingMode.VLM_ONLY:
                        ticket_images = _vlm_ticket_images(
                            prepared, metadata, UIConfig(), directory,
                        )
                    request, pass1 = _run_pass1(
                        vlm_runtime, metadata, prepared, clip_path,
                        ticket_images=ticket_images,
                    )
                except _UnderstandingStageError:
                    raise
                except Exception as error:
                    raise _UnderstandingStageError("vlm_pass1", error) from error
                if not runs_fusion:
                    final = resolve_single_source_investigation(
                        investigation_id=prepared.investigation_id, source_video=metadata.path,
                        start_time_s=interval.start_time_seconds, end_time_s=interval.end_time_seconds,
                        observations=pass1.observations,
                    )
                else:
                    try:
                        vlm_by_name = {item.field_name: item for item in pass1.observations}
                        ocr_by_name = {}
                        for item in occurrences:
                            name = item.canonical_label
                            previous = ocr_by_name.get(name)
                            item_key = (item.status == "extracted", item.provenance.actual_timestamp_seconds,
                                        item.provenance.candidate_index)
                            previous_key = (
                                (previous.status == "extracted", previous.provenance.actual_timestamp_seconds,
                                 previous.provenance.candidate_index)
                                if previous is not None else None
                            )
                            if previous_key is None or item_key > previous_key:
                                ocr_by_name[name] = item
                        field_names = dict.fromkeys((*ocr_by_name, *vlm_by_name))
                        fused_fields = {
                            name: apply_temporal_awareness(compare_field_observations(
                                name, ocr_by_name.get(name), vlm_by_name.get(name),
                            ))
                            for name in field_names
                        }
                        fusion = FusedInvestigationResult(
                            prepared.investigation_id, metadata.path,
                            interval.start_time_seconds, interval.end_time_seconds, fused_fields,
                        )
                    except Exception as error:
                        raise _UnderstandingStageError("fusion", error) from error
                    try:
                        ticket_images = _ticket_detail_images(
                            prepared, ocr_results, metadata, UIConfig(), directory,
                        )
                        review_request = build_vlm_review_request(
                            request, fusion, pass1_result=pass1, ticket_images=ticket_images,
                            prompt_version=FR24_VLM_REVIEW_V3,
                        )
                        review = parse_vlm_review_result(review_request, vlm_runtime.run(review_request))
                        if (review.raw_result.execution_status != "success" or review.diagnostics
                                or set(review.reviewed_fields) != set(REVIEW_FIELD_NAMES)
                                or set(review.route_reviews) != set(REVIEW_ROUTE_FIELDS)):
                            raise RuntimeError(
                                review.raw_result.error or "; ".join(review.diagnostics)
                                or "VLM Pass 2 did not produce the complete validated response."
                            )
                    except Exception as error:
                        raise _UnderstandingStageError("vlm_pass2", error) from error
                    try:
                        final = resolve_investigation(fusion, review)
                    except Exception as error:
                        raise _UnderstandingStageError("resolution", error) from error
        return assemble_investigation_record(
            **base, ocr_results=ocr_results, ocr_field_occurrences=record_occurrences,
            vlm_pass1=pass1, fusion_result=fusion, vlm_pass2=review, final_result=final,
        )
    except _UnderstandingStageError:
        raise
    except Exception as error:
        stage = "ocr" if runs_ocr and not runs_vlm else "understanding"
        raise _UnderstandingStageError(stage, error) from error


def run_understanding(prepared: VideoPipelineResult, mode: UnderstandingMode, *, vlm_settings=None):
    """Run one selected understanding branch and return the common video result."""
    from events_diary.investigation_record import (
        InvestigationFailure,
        VideoInvestigationResult,
    )

    if not isinstance(prepared, VideoPipelineResult):
        raise TypeError("prepared must be VideoPipelineResult.")
    if not isinstance(mode, UnderstandingMode):
        raise TypeError("mode must be UnderstandingMode.")
    runs_ocr, runs_vlm, _ = _MODE_FEATURES[mode]
    if not prepared.investigations:
        return VideoInvestigationResult(prepared.metadata.path, prepared.metadata, ())
    ocr_engine = None
    if runs_ocr:
        try:
            from events_diary.ticket_ocr import RapidOCREngine
            ocr_engine = RapidOCREngine()
        except Exception as error:  # noqa: BLE001 - surface initialization errors as Investigation failures
            return VideoInvestigationResult(
                prepared.metadata.path, prepared.metadata, (), tuple(
                    InvestigationFailure(
                        item.investigation_id, prepared.metadata.path,
                        item.interval.start_time_seconds, item.interval.end_time_seconds,
                        "ocr_initialization", str(error) or type(error).__name__,
                        error_type=type(error).__name__,
                    ) for item in prepared.investigations
                ),
            )
    vlm_runtime = None
    if runs_vlm:
        if vlm_settings is None:
            raise ValueError("VLM mode requires centralized VLM API settings.")
        from events_diary.gemini_api import GeminiAPIRuntime
        try:
            vlm_runtime = GeminiAPIRuntime(vlm_settings)
        except Exception as error:  # noqa: BLE001 - surface initialization errors as Investigation failures
            return VideoInvestigationResult(
                prepared.metadata.path, prepared.metadata, (), tuple(
                    InvestigationFailure(
                        item.investigation_id, prepared.metadata.path,
                        item.interval.start_time_seconds, item.interval.end_time_seconds,
                        "vlm_initialization", str(error) or type(error).__name__,
                        error_type=type(error).__name__,
                    ) for item in prepared.investigations
                ),
            )
    records, failures = [], []
    for investigation in prepared.investigations:
        try:
            records.append(_understand_one(
                investigation, prepared.metadata, mode,
                ocr_engine=ocr_engine, vlm_runtime=vlm_runtime,
            ))
        except Exception as error:  # noqa: BLE001 - preserve required branch failures per Investigation
            stage = error.stage if isinstance(error, _UnderstandingStageError) else "understanding"
            cause = (
                error.__cause__ or error.cause
                if isinstance(error, _UnderstandingStageError) else error
            )
            failures.append(InvestigationFailure(
                investigation.investigation_id, prepared.metadata.path,
                investigation.interval.start_time_seconds, investigation.interval.end_time_seconds,
                stage, str(error) or type(error).__name__,
                error_type=type(cause).__name__,
                provider_diagnostics_json=(
                    error.provider_diagnostics_json
                    if isinstance(error, _UnderstandingStageError) else None
                ),
            ))
    return VideoInvestigationResult(prepared.metadata.path, prepared.metadata, tuple(records), tuple(failures))


def run_pipeline(config: RuntimeConfig) -> PipelineOutput:
    """Run the configured mode and write both final diaries to its output folder."""
    prepared = prepare_video_pipeline(config)
    video_result = run_understanding(
        prepared, config.understanding_mode, vlm_settings=config.vlm_api_settings,
    )
    if config.understanding_mode is not UnderstandingMode.VLM_ONLY:
        _reject_failed_investigations(video_result, config.understanding_mode)
    if config.understanding_mode is UnderstandingMode.OCR_ONLY:
        _log_ocr_only_fields(video_result)
    if (config.understanding_mode is UnderstandingMode.VLM_ONLY
            and os.environ.get("EVENTS_DIARY_VLM_DEBUG_SUMMARY") == "1"):
        _log_vlm_debug_summary(video_result)
    output = finalize_pipeline_result(
        video_result, config.output_dir, config.understanding_mode,
    )
    if isinstance(output, PipelineOutput):
        logger.info(
            "Video run finished | status=%s | operational_diary=%s | detailed_diary=%s%s",
            output.run_status, output.operational_diary_path, output.detailed_diary_path,
            f" | VLM diagnostics={output.vlm_results_path}" if output.vlm_results_path else "",
        )
    return output


def main(config: RuntimeConfig | None = None) -> PipelineOutput:
    """Initialize diagnostics and run the configured video-to-diary pipeline."""
    app_config = AppConfig()
    configure_logging(app_config)
    runtime = get_runtime_profile()
    logger.info("Events Diary initialized successfully.")
    logger.info(
        "Runtime: %s | Python %s | %s",
        runtime.operating_system,
        runtime.python_version,
        runtime.machine,
    )
    runtime_config = config if config is not None else load_runtime_config()
    return run_pipeline(runtime_config)
