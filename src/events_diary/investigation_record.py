"""Immutable assembly contracts for complete per-Investigation evidence records.

This module only groups results already produced by Phases 3–6. It does not
detect Investigations, decode video, select evidence, run OCR/VLM, fuse values,
or resolve fields. Each record retains the existing typed results and selected
evidence references; repeated OCR occurrences remain available alongside the
authoritative Phase 6.5 final field map. A video result preserves one record per
chronological Investigation, including repeated inspections of the same aircraft.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real
from pathlib import Path

from events_diary.adaptive_sampling import AdaptiveSamplingPlan
from events_diary.evidence_candidate_timeline import EvidenceCandidate
from events_diary.frame_materialization import MaterializedEvidenceCandidate
from events_diary.fusion import FinalInvestigationResult, FusedInvestigationResult
from events_diary.investigation_lifecycle import InvestigationInterval
from events_diary.investigation_temporal import ConfirmedInvestigationEvent
from events_diary.investigation_window import InvestigationWindow
from events_diary.ocr_fields import OCRFieldObservation
from events_diary.representative_selection import InvestigationSelection
from events_diary.ticket_ocr import OCRTicketResult
from events_diary.ticket_stable_views import TicketStableView
from events_diary.video import VideoMetadata
from events_diary.vlm import VLMInvestigationResult
from events_diary.vlm_review import VLMReviewResult

_TIME_EPSILON = 1e-9


def _inside_interval(value: float, start: float, end: float) -> bool:
    """Return whether a physical evidence time belongs to a half-open core."""
    return start - _TIME_EPSILON <= value < end


@dataclass(frozen=True)
class InvestigationRecord:
    """One Investigation's bounds, evidence provenance, and authoritative result.

    Existing models are nested rather than copied into duplicate schemas. The
    candidate/materialization sequences use Investigation-local indexes referenced
    by ``selection``. All OCR occurrences are retained, even when multiple values
    for a dynamic field appear; ``final_fields`` comes only from final resolution.
    """

    investigation_id: int
    source_video: Path
    interval: InvestigationInterval
    confirmed_events: tuple[ConfirmedInvestigationEvent, ...]
    window: InvestigationWindow
    sampling_plan: AdaptiveSamplingPlan
    stable_views: tuple[TicketStableView, ...]
    candidates: tuple[EvidenceCandidate, ...]
    materialized_candidates: tuple[MaterializedEvidenceCandidate, ...]
    selection: InvestigationSelection
    ocr_results: tuple[OCRTicketResult, ...]
    ocr_field_occurrences: tuple[OCRFieldObservation, ...]
    vlm_pass1: VLMInvestigationResult
    fusion_result: FusedInvestigationResult
    vlm_pass2: VLMReviewResult
    final_result: FinalInvestigationResult
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject mixed provenance and freeze all sequence fields."""
        if isinstance(self.investigation_id, bool) or not isinstance(self.investigation_id, int):
            raise TypeError("investigation_id must be an integer, not bool.")
        if self.investigation_id < 0:
            raise ValueError("investigation_id must be nonnegative.")
        if not isinstance(self.source_video, Path):
            raise TypeError("source_video must be pathlib.Path.")
        if not isinstance(self.interval, InvestigationInterval):
            raise TypeError("interval must be InvestigationInterval.")
        if self.interval.investigation_index != self.investigation_id:
            raise ValueError("investigation_id must match the lifecycle interval index.")
        start, end = self.interval.start_time_seconds, self.interval.end_time_seconds
        for name, value in (("start", start), ("end", end)):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Investigation {name} must be a real number, not bool.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Investigation {name} must be finite and nonnegative.")
        if end < start:
            raise ValueError("Investigation interval bounds must be finite and ordered.")
        sequence_types = {
            "confirmed_events": ConfirmedInvestigationEvent,
            "stable_views": TicketStableView,
            "candidates": EvidenceCandidate,
            "materialized_candidates": MaterializedEvidenceCandidate,
            "ocr_results": OCRTicketResult,
            "ocr_field_occurrences": OCRFieldObservation,
        }
        for name, expected_type in sequence_types.items():
            values = tuple(getattr(self, name))
            if any(not isinstance(value, expected_type) for value in values):
                raise TypeError(f"{name} must contain {expected_type.__name__} records.")
            object.__setattr__(self, name, values)
        if not isinstance(self.window, InvestigationWindow):
            raise TypeError("window must be InvestigationWindow.")
        if not isinstance(self.sampling_plan, AdaptiveSamplingPlan):
            raise TypeError("sampling_plan must be AdaptiveSamplingPlan.")
        if not isinstance(self.selection, InvestigationSelection):
            raise TypeError("selection must be InvestigationSelection.")
        if not isinstance(self.vlm_pass1, VLMInvestigationResult):
            raise TypeError("vlm_pass1 must be VLMInvestigationResult.")
        if not isinstance(self.fusion_result, FusedInvestigationResult):
            raise TypeError("fusion_result must be FusedInvestigationResult.")
        if not isinstance(self.vlm_pass2, VLMReviewResult):
            raise TypeError("vlm_pass2 must be VLMReviewResult.")
        if not isinstance(self.final_result, FinalInvestigationResult):
            raise TypeError("final_result must be FinalInvestigationResult.")
        if self.selection.investigation_index != self.investigation_id:
            raise ValueError("selection must belong to this Investigation.")
        if self.window.investigation_index != self.investigation_id:
            raise ValueError("window must belong to this Investigation.")
        if (self.window.investigation_start_time_seconds != start
                or self.window.investigation_end_time_seconds != end):
            raise ValueError("window core bounds must match the lifecycle interval.")
        if len(self.candidates) != len(self.materialized_candidates):
            raise ValueError("candidate and materialization sequences must align.")
        for candidate, materialized in zip(self.candidates, self.materialized_candidates, strict=True):
            if candidate.investigation_index != self.investigation_id:
                raise ValueError("candidate belongs to a different Investigation.")
            if materialized.evidence_candidate != candidate:
                raise ValueError("materialized candidate must retain its matching candidate.")
        for chosen in self.selection.selections:
            if not 0 <= chosen.candidate_index < len(self.candidates):
                raise ValueError("selection candidate index is outside its Investigation-local sequence.")
        self._validate_source_result(self.vlm_pass1, start, end, "VLM Pass 1")
        self._validate_source_result(self.vlm_pass2, start, end, "VLM Pass 2")
        for item in self.ocr_results:
            if item.investigation_index != self.investigation_id:
                raise ValueError("OCR result belongs to a different Investigation.")
            if not _inside_interval(item.actual_timestamp_seconds, start, end):
                raise ValueError("OCR frame timestamp must lie inside this Investigation.")
        for item in self.ocr_field_occurrences:
            if item.provenance.investigation_index != self.investigation_id:
                raise ValueError("OCR field occurrence belongs to a different Investigation.")
            if not _inside_interval(item.provenance.actual_timestamp_seconds, start, end):
                raise ValueError("OCR field timestamp must lie inside this Investigation.")
        for result_name, result in (
            ("fusion", self.fusion_result), ("final resolution", self.final_result),
        ):
            if (result.investigation_id != self.investigation_id
                    or result.source_video != self.source_video
                    or result.start_time_s != start or result.end_time_s != end):
                raise ValueError(f"{result_name} provenance must match this Investigation.")
        if set(self.final_result.fields) != set(self.fusion_result.fields):
            raise ValueError("final field names must match the fusion field names.")
        messages = tuple(self.diagnostics)
        if any(not isinstance(message, str) for message in messages):
            raise TypeError("diagnostics must contain strings.")
        object.__setattr__(self, "diagnostics", messages)

    def _validate_source_result(self, result, start: float, end: float, name: str) -> None:
        """Ensure a nested parsed VLM result belongs to this video interval."""
        raw = result.raw_result
        if (raw.investigation_index != self.investigation_id
                or raw.source_video != self.source_video
                or raw.source_start_s != start or raw.source_end_s != end):
            raise ValueError(f"{name} provenance must match this Investigation.")

    @property
    def start_time_s(self) -> float:
        """Return the original-video Investigation start time."""
        return self.interval.start_time_seconds

    @property
    def end_time_s(self) -> float:
        """Return the original-video Investigation end time."""
        return self.interval.end_time_seconds

    @property
    def duration_s(self) -> float:
        """Return the half-open Investigation core duration in seconds."""
        return self.end_time_s - self.start_time_s

    @property
    def final_fields(self) -> Mapping[str, object]:
        """Expose the immutable authoritative field-to-final-result mapping."""
        return self.final_result.fields

    @property
    def unresolved_fields(self) -> tuple[str, ...]:
        """List fields the Phase 6.5 resolver could not settle."""
        return tuple(name for name, item in self.final_fields.items() if item.final_status == "unresolved")

    @property
    def not_available_fields(self) -> tuple[str, ...]:
        """List fields with no available evidence according to final resolution."""
        return tuple(name for name, item in self.final_fields.items() if item.final_status == "not_available")


def assemble_investigation_record(
    *, investigation_id: int,
    source_video: Path,
    interval: InvestigationInterval,
    confirmed_events: Sequence[ConfirmedInvestigationEvent],
    window: InvestigationWindow,
    sampling_plan: AdaptiveSamplingPlan,
    stable_views: Sequence[TicketStableView],
    candidates: Sequence[EvidenceCandidate],
    materialized_candidates: Sequence[MaterializedEvidenceCandidate],
    selection: InvestigationSelection,
    ocr_results: Sequence[OCRTicketResult],
    ocr_field_occurrences: Sequence[OCRFieldObservation],
    vlm_pass1: VLMInvestigationResult,
    fusion_result: FusedInvestigationResult,
    vlm_pass2: VLMReviewResult,
    final_result: FinalInvestigationResult,
    diagnostics: Sequence[str] = (),
) -> InvestigationRecord:
    """Assemble already-produced Investigation evidence without running stages."""
    return InvestigationRecord(
        investigation_id=investigation_id, source_video=source_video, interval=interval,
        confirmed_events=tuple(confirmed_events), window=window, sampling_plan=sampling_plan,
        stable_views=tuple(stable_views), candidates=tuple(candidates),
        materialized_candidates=tuple(materialized_candidates), selection=selection,
        ocr_results=tuple(ocr_results), ocr_field_occurrences=tuple(ocr_field_occurrences),
        vlm_pass1=vlm_pass1, fusion_result=fusion_result, vlm_pass2=vlm_pass2,
        final_result=final_result, diagnostics=tuple(diagnostics),
    )


@dataclass(frozen=True)
class InvestigationFailure:
    """An explicitly incomplete Investigation with its interval and stopping stage."""

    investigation_id: int
    source_video: Path
    start_time_s: float
    end_time_s: float
    failed_stage: str
    reason: str
    artifact_directory: Path

    def __post_init__(self) -> None:
        """Validate failure identity and original-video time bounds."""
        if isinstance(self.investigation_id, bool) or not isinstance(self.investigation_id, int):
            raise TypeError("investigation_id must be an integer, not bool.")
        if self.investigation_id < 0:
            raise ValueError("investigation_id must be nonnegative.")
        if not isinstance(self.source_video, Path) or not isinstance(self.artifact_directory, Path):
            raise TypeError("source_video and artifact_directory must be pathlib.Path.")
        for name, value in (("start_time_s", self.start_time_s), ("end_time_s", self.end_time_s)):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number, not bool.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.end_time_s < self.start_time_s:
            raise ValueError("end_time_s must be greater than or equal to start_time_s.")
        if not isinstance(self.failed_stage, str) or not self.failed_stage.strip():
            raise ValueError("failed_stage must be a nonempty string.")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a nonempty string.")


@dataclass(frozen=True)
class VideoInvestigationResult:
    """One source video's ordered independent Investigation records."""

    source_video: Path
    video_metadata: VideoMetadata
    investigations: tuple[InvestigationRecord, ...]
    failures: tuple[InvestigationFailure, ...] = ()

    def __post_init__(self) -> None:
        """Freeze records and verify common source identity and chronological order."""
        if not isinstance(self.source_video, Path):
            raise TypeError("source_video must be pathlib.Path.")
        if not isinstance(self.video_metadata, VideoMetadata):
            raise TypeError("video_metadata must be VideoMetadata.")
        if self.video_metadata.path.resolve() != self.source_video.resolve():
            raise ValueError("video metadata must describe source_video.")
        records = tuple(self.investigations)
        failures = tuple(self.failures)
        if any(not isinstance(item, InvestigationRecord) for item in records):
            raise TypeError("investigations must contain InvestigationRecord values.")
        if any(not isinstance(item, InvestigationFailure) for item in failures):
            raise TypeError("failures must contain InvestigationFailure values.")
        if any(item.source_video.resolve() != self.source_video.resolve() for item in records):
            raise ValueError("all Investigation records must share source_video.")
        if any(item.source_video.resolve() != self.source_video.resolve() for item in failures):
            raise ValueError("all Investigation failures must share source_video.")
        identities = [item.investigation_id for item in (*records, *failures)]
        if len(set(identities)) != len(identities):
            raise ValueError("Investigation records must remain distinct and have unique IDs.")
        for previous, current in pairwise(records):
            if current.start_time_s < previous.start_time_s:
                raise ValueError("Investigation records must be ordered by start time.")
        object.__setattr__(self, "investigations", records)
        object.__setattr__(self, "failures", failures)

    @property
    def investigation_count(self) -> int:
        """Return the number of independent Investigation records."""
        return len(self.investigations) + len(self.failures)
