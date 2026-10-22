"""Immutable OCR/VLM fusion contracts, normalization, comparison, and timing.

Normalization here removes only field-appropriate presentation differences. It
does not infer or repair values. Comparison applies a fixed availability/equality
truth table. A separate temporal layer marks unaligned dynamic-field conflicts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from events_diary.ocr_fields import OCRFieldObservation
from events_diary.vlm import VLMFieldObservation

if TYPE_CHECKING:
    from events_diary.vlm_review import VLMReviewObservation, VLMReviewResult

FusionStatus = Literal[
    "agreement",
    "conflict",
    "ocr_only",
    "vlm_only",
    "unavailable",
    "temporal_mismatch",
]

_FUSION_STATUSES = (
    "agreement",
    "conflict",
    "ocr_only",
    "vlm_only",
    "unavailable",
    "temporal_mismatch",
)
_SUMMARY_COUNTS = (
    "total_fields",
    "agreement_count",
    "conflict_count",
    "ocr_only_count",
    "vlm_only_count",
    "unavailable_count",
    "temporal_mismatch_count",
)
_FINAL_STATUSES = ("resolved", "resolved_single_source", "unresolved", "not_available")

# The families cover current OCR canonical labels and VLM parser field names.
# Unknown names use conservative whitespace cleanup and preserve letter case.
IDENTIFIER_FIELDS = frozenset({
    "REGISTRATION", "SERIAL_NUMBER", "ICAO_ADDRESS", "SQUAWK",
    "primary_flight_id", "alternate_flight_id", "aircraft_code",
    "origin_airport_code", "destination_airport_code",
})
NUMERIC_FIELDS = frozenset({
    "BAROMETRIC_ALTITUDE", "VERTICAL_SPEED", "GPS_ALTITUDE", "GROUND_SPEED",
    "TRUE_AIRSPEED", "INDICATED_AIRSPEED", "MACH", "TEMPERATURE", "LATITUDE",
    "LONGITUDE",
})
HEADING_FIELDS = frozenset({"TRACK"})
BOOLEAN_FIELDS = frozenset({"scroll_observed", "same_aircraft_throughout"})
TIME_FIELDS = frozenset({
    "SCHEDULED", "ACTUAL", "ESTIMATED",
    "origin_scheduled_time", "origin_actual_time", "destination_scheduled_time",
    "destination_estimated_or_actual_time",
})
TEXT_FIELDS = frozenset({
    "COUNTRY_OF_REGISTRATION", "AGE", "AIRCRAFT_CATEGORY", "AIRCRAFT_TYPE",
    "WIND", "FIR_UIR",
    "airline_name", "origin_city", "origin_timezone_text", "destination_city",
    "destination_timezone_text",
})
FIELD_FAMILIES = MappingProxyType({
    "identifier": IDENTIFIER_FIELDS,
    "numeric": NUMERIC_FIELDS,
    "heading": HEADING_FIELDS,
    "boolean": BOOLEAN_FIELDS,
    "text": TEXT_FIELDS,
    "time": TIME_FIELDS,
})

STATIC_FIELDS = frozenset({
    "REGISTRATION", "SERIAL_NUMBER", "ICAO_ADDRESS", "COUNTRY_OF_REGISTRATION",
    "AGE", "AIRCRAFT_CATEGORY", "AIRCRAFT_TYPE", "SCHEDULED",
    "airline_name", "primary_flight_id", "alternate_flight_id", "aircraft_code",
    "origin_airport_code", "origin_city", "origin_timezone_text",
    "destination_airport_code", "destination_city", "destination_timezone_text",
    "origin_scheduled_time", "destination_scheduled_time", "same_aircraft_throughout",
})
# SQUAWK is operational and can change during a flight, so it is treated as
# dynamic within an Investigation rather than as aircraft identity.
DYNAMIC_FIELDS = frozenset({
    "SQUAWK", "BAROMETRIC_ALTITUDE", "VERTICAL_SPEED", "GPS_ALTITUDE", "TRACK",
    "GROUND_SPEED", "TRUE_AIRSPEED", "INDICATED_AIRSPEED", "MACH", "WIND",
    "TEMPERATURE", "FIR_UIR", "LATITUDE", "LONGITUDE", "ACTUAL", "ESTIMATED",
    "origin_actual_time", "destination_estimated_or_actual_time", "scroll_observed",
})


def _diagnostics(name: str, values: tuple[str, ...]) -> None:
    """Validate that a diagnostics collection contains only readable strings."""
    if any(not isinstance(value, str) for value in values):
        raise TypeError(f"{name} must contain strings.")


def _clean_whitespace(value: str) -> str:
    """Trim and collapse whitespace without changing text or punctuation."""
    return " ".join(value.split())


def normalize_field_value(field_name: str, raw_value: str | bool | None) -> str | bool | None:
    """Return a deterministic comparison form without changing source evidence.

    Identifiers are uppercased, track values lose degree-symbol formatting,
    explicit boolean strings become booleans, and other strings receive only
    whitespace cleanup. Unknown fields follow that same conservative text rule.
    """
    if not isinstance(field_name, str) or not field_name.strip():
        raise ValueError("field_name must be a nonempty string.")
    if raw_value is None or isinstance(raw_value, bool):
        return raw_value
    if not isinstance(raw_value, str):
        raise TypeError("raw_value must be a string, bool, or None.")

    cleaned = _clean_whitespace(raw_value)
    if field_name in BOOLEAN_FIELDS:
        explicit = cleaned.casefold()
        if explicit == "true":
            return True
        if explicit == "false":
            return False
        return cleaned
    if field_name in IDENTIFIER_FIELDS:
        return cleaned.upper()
    if field_name in HEADING_FIELDS:
        return _clean_whitespace(cleaned.replace("ֲ°", "").replace("°", ""))
    return cleaned


def compare_field_observations(
    field_name: str,
    ocr_observation: OCRFieldObservation | None,
    vlm_observation: VLMFieldObservation | None,
) -> FusedFieldObservation:
    """Compare one OCR/VLM field using availability and normalized equality only.

    OCR is available only when extraction selected a candidate; VLM is available
    only for ``observed`` status with a non-None raw value. Ambiguous and
    unavailable source records are preserved but excluded from comparison.
    """
    if not isinstance(field_name, str) or not field_name.strip():
        raise ValueError("field_name must be a nonempty string.")
    if ocr_observation is not None and not isinstance(
        ocr_observation, OCRFieldObservation
    ):
        raise TypeError("ocr_observation must be OCRFieldObservation or None.")
    if vlm_observation is not None and not isinstance(
        vlm_observation, VLMFieldObservation
    ):
        raise TypeError("vlm_observation must be VLMFieldObservation or None.")

    ocr_available = (
        ocr_observation is not None
        and ocr_observation.status == "extracted"
        and ocr_observation.selected_candidate is not None
    )
    vlm_available = (
        vlm_observation is not None
        and vlm_observation.status == "observed"
        and vlm_observation.raw_value is not None
    )
    ocr_value = (
        normalize_field_value(field_name, ocr_observation.raw_value_text)
        if ocr_available
        else None
    )
    vlm_value = (
        normalize_field_value(field_name, vlm_observation.raw_value)
        if vlm_available
        else None
    )

    if ocr_available and vlm_available:
        if ocr_value == vlm_value:
            status: FusionStatus = "agreement"
            resolved_value = ocr_value
            diagnostic = "values match after normalization"
        else:
            status = "conflict"
            resolved_value = None
            diagnostic = "values differ after normalization"
    elif ocr_available:
        status = "ocr_only"
        resolved_value = ocr_value
        diagnostic = "only OCR produced a usable value"
    elif vlm_available:
        status = "vlm_only"
        resolved_value = vlm_value
        diagnostic = "only VLM produced a usable value"
    else:
        status = "unavailable"
        resolved_value = None
        diagnostic = "neither source produced a usable value"

    return FusedFieldObservation(
        field_name=field_name,
        status=status,
        ocr_observation=ocr_observation,
        vlm_observation=vlm_observation,
        ocr_normalized_value=ocr_value,
        vlm_normalized_value=vlm_value,
        resolved_value=resolved_value,
        diagnostics=(diagnostic,),
    )


@dataclass(frozen=True)
class FusedFieldObservation:
    """Fusion data for one field, retaining original source records unchanged."""

    field_name: str
    status: FusionStatus
    ocr_observation: OCRFieldObservation | None = None
    vlm_observation: VLMFieldObservation | None = None
    ocr_normalized_value: str | bool | None = None
    vlm_normalized_value: str | bool | None = None
    resolved_value: str | bool | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate contract types and freeze diagnostic messages."""
        if not isinstance(self.field_name, str) or not self.field_name.strip():
            raise ValueError("field_name must be a nonempty string.")
        if self.status not in _FUSION_STATUSES:
            raise ValueError("Unsupported fusion status.")
        if self.ocr_observation is not None and not isinstance(
            self.ocr_observation, OCRFieldObservation
        ):
            raise TypeError("ocr_observation must be OCRFieldObservation or None.")
        if self.vlm_observation is not None and not isinstance(
            self.vlm_observation, VLMFieldObservation
        ):
            raise TypeError("vlm_observation must be VLMFieldObservation or None.")
        for name in ("ocr_normalized_value", "vlm_normalized_value", "resolved_value"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, (str, bool)):
                raise TypeError(f"{name} must be a string, bool, or None.")
        messages = tuple(self.diagnostics)
        _diagnostics("diagnostics", messages)
        object.__setattr__(self, "diagnostics", messages)


def is_temporally_comparable(
    field_name: str,
    ocr_observation: OCRFieldObservation | None,
    vlm_observation: VLMFieldObservation | None,
) -> bool:
    """Return whether a dynamic-field pair has matching source point times.

    OCR contributes its physical frame timestamp. A VLM clip interval alone is
    not field-specific timing evidence; the VLM observation must carry an
    ``evidence_time_s`` exactly equal to the OCR source timestamp.
    """
    if not isinstance(field_name, str) or not field_name.strip():
        raise ValueError("field_name must be a nonempty string.")
    if field_name not in DYNAMIC_FIELDS:
        return True
    if (
        ocr_observation is None
        or ocr_observation.status != "extracted"
        or ocr_observation.selected_candidate is None
        or vlm_observation is None
        or vlm_observation.status != "observed"
        or vlm_observation.raw_value is None
        or vlm_observation.evidence_time_s is None
    ):
        return False
    ocr_time = ocr_observation.provenance.actual_timestamp_seconds
    return (
        isinstance(ocr_time, Real)
        and not isinstance(ocr_time, bool)
        and math.isfinite(ocr_time)
        and ocr_time == vlm_observation.evidence_time_s
    )


def apply_temporal_awareness(
    fused_observation: FusedFieldObservation,
) -> FusedFieldObservation:
    """Mark unaligned dynamic-field conflicts without resolving either source."""
    if not isinstance(fused_observation, FusedFieldObservation):
        raise TypeError("fused_observation must be FusedFieldObservation.")
    if fused_observation.status != "conflict":
        return fused_observation
    if fused_observation.field_name not in DYNAMIC_FIELDS:
        return fused_observation
    if is_temporally_comparable(
        fused_observation.field_name,
        fused_observation.ocr_observation,
        fused_observation.vlm_observation,
    ):
        return fused_observation
    return replace(
        fused_observation,
        status="temporal_mismatch",
        resolved_value=None,
        diagnostics=(
            *fused_observation.diagnostics,
            "dynamic field values differ at non-comparable source times",
        ),
    )


@dataclass(frozen=True)
class FusionSummary:
    """Counts of fields by the supported fusion status labels."""

    total_fields: int
    agreement_count: int = 0
    conflict_count: int = 0
    ocr_only_count: int = 0
    vlm_only_count: int = 0
    unavailable_count: int = 0
    temporal_mismatch_count: int = 0

    def __post_init__(self) -> None:
        """Require nonnegative integer counts, excluding boolean values."""
        for name in _SUMMARY_COUNTS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, not bool.")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")


@dataclass(frozen=True)
class FusedInvestigationResult:
    """Immutable field mapping and original-video bounds for one Investigation."""

    investigation_id: str | int
    source_video: Path
    start_time_s: float
    end_time_s: float
    fields: Mapping[str, FusedFieldObservation]
    summary: FusionSummary | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate Investigation bounds and expose a read-only field mapping."""
        if isinstance(self.investigation_id, bool) or not isinstance(
            self.investigation_id, (str, int)
        ):
            raise TypeError("investigation_id must be a string or integer, not bool.")
        if isinstance(self.investigation_id, str) and not self.investigation_id.strip():
            raise ValueError("investigation_id must not be empty.")
        if not isinstance(self.source_video, Path):
            raise TypeError("source_video must be pathlib.Path.")
        for name in ("start_time_s", "end_time_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number, not bool.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.end_time_s < self.start_time_s:
            raise ValueError("end_time_s must be greater than or equal to start_time_s.")
        if not isinstance(self.fields, Mapping):
            raise TypeError("fields must be a mapping of names to fused observations.")
        copied_fields = dict(self.fields)
        for name, observation in copied_fields.items():
            if not isinstance(name, str) or not isinstance(
                observation, FusedFieldObservation
            ):
                raise TypeError("fields must map strings to FusedFieldObservation records.")
            if name != observation.field_name:
                raise ValueError("Each field mapping key must match field_name.")
        if self.summary is not None:
            if not isinstance(self.summary, FusionSummary):
                raise TypeError("summary must be FusionSummary or None.")
            if self.summary.total_fields != len(copied_fields):
                raise ValueError("summary.total_fields must match the number of fields.")
        messages = tuple(self.diagnostics)
        _diagnostics("diagnostics", messages)
        object.__setattr__(self, "fields", MappingProxyType(copied_fields))
        object.__setattr__(self, "diagnostics", messages)


@dataclass(frozen=True)
class FinalFieldResult:
    """Explainable final resolution while retaining all source observations."""

    field_name: str
    final_status: Literal["resolved", "resolved_single_source", "unresolved", "not_available"]
    final_value: str | bool | None
    fused_observation: FusedFieldObservation
    vlm_review_observation: VLMReviewObservation | None
    resolution_reason: str
    temporal_status: str | None = None

    def __post_init__(self) -> None:
        """Validate final status, value, provenance links, and concise reason."""
        from events_diary.vlm_review import VLMReviewObservation

        if not isinstance(self.field_name, str) or not self.field_name.strip():
            raise ValueError("field_name must be a nonempty string.")
        if self.final_status not in _FINAL_STATUSES:
            raise ValueError("Unsupported final_status.")
        if self.final_value is not None and not isinstance(self.final_value, (str, bool)):
            raise TypeError("final_value must be a string, bool, or None.")
        if self.final_status in ("unresolved", "not_available") and self.final_value is not None:
            raise ValueError("unresolved and not_available results must have no final_value.")
        if self.final_status in ("resolved", "resolved_single_source") and self.final_value is None:
            raise ValueError("resolved results must have a final_value.")
        if not isinstance(self.fused_observation, FusedFieldObservation):
            raise TypeError("fused_observation must be FusedFieldObservation.")
        if self.field_name != self.fused_observation.field_name:
            raise ValueError("field_name must match fused_observation.field_name.")
        if self.vlm_review_observation is not None and not isinstance(
            self.vlm_review_observation, VLMReviewObservation
        ):
            raise TypeError("vlm_review_observation must be VLMReviewObservation or None.")
        if self.vlm_review_observation is not None and (
            self.vlm_review_observation.field_name != self.field_name
        ):
            raise ValueError("review field_name must match the fused field.")
        if not isinstance(self.resolution_reason, str) or not self.resolution_reason.strip():
            raise ValueError("resolution_reason must be a nonempty string.")
        if self.temporal_status is not None and not isinstance(self.temporal_status, str):
            raise TypeError("temporal_status must be a string or None.")


@dataclass(frozen=True)
class FinalInvestigationSummary:
    """Counts final field outcomes without adding confidence scores."""

    total_fields: int
    resolved_count: int
    resolved_single_source_count: int
    unresolved_count: int
    not_available_count: int

    def __post_init__(self) -> None:
        """Require nonnegative integer counts whose categories sum to total."""
        counts = (
            self.total_fields, self.resolved_count, self.resolved_single_source_count,
            self.unresolved_count, self.not_available_count,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise TypeError("Final summary counts must be integers, not bool.")
        if any(value < 0 for value in counts):
            raise ValueError("Final summary counts must be nonnegative.")
        if sum(counts[1:]) != self.total_fields:
            raise ValueError("Final summary status counts must sum to total_fields.")


@dataclass(frozen=True)
class FinalInvestigationResult:
    """Immutable final field map retaining original Investigation provenance."""

    investigation_id: str | int
    source_video: Path
    start_time_s: float
    end_time_s: float
    fields: Mapping[str, FinalFieldResult]
    summary: FinalInvestigationSummary
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate field-map consistency and freeze the mapping and diagnostics."""
        if isinstance(self.investigation_id, bool) or not isinstance(
            self.investigation_id, (str, int)
        ):
            raise TypeError("investigation_id must be a string or integer, not bool.")
        if isinstance(self.investigation_id, str) and not self.investigation_id.strip():
            raise ValueError("investigation_id must not be empty.")
        if not isinstance(self.source_video, Path):
            raise TypeError("source_video must be pathlib.Path.")
        for name in ("start_time_s", "end_time_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number, not bool.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.end_time_s < self.start_time_s:
            raise ValueError("end_time_s must be greater than or equal to start_time_s.")
        if not isinstance(self.fields, Mapping):
            raise TypeError("fields must be a mapping of names to final field results.")
        copied = dict(self.fields)
        if any(
            not isinstance(name, str) or not isinstance(result, FinalFieldResult)
            or name != result.field_name for name, result in copied.items()
        ):
            raise ValueError("fields keys must match FinalFieldResult field names.")
        if not isinstance(self.summary, FinalInvestigationSummary):
            raise TypeError("summary must be FinalInvestigationSummary.")
        if self.summary.total_fields != len(copied):
            raise ValueError("summary.total_fields must match the number of fields.")
        messages = tuple(self.diagnostics)
        _diagnostics("diagnostics", messages)
        object.__setattr__(self, "fields", MappingProxyType(copied))
        object.__setattr__(self, "diagnostics", messages)


def resolve_field(
    fused_observation: FusedFieldObservation,
    vlm_review_observation: VLMReviewObservation | None,
) -> FinalFieldResult:
    """Resolve one fused field conservatively using source pattern and Pass 2 evidence."""
    from events_diary.vlm_review import VLMReviewObservation

    if not isinstance(fused_observation, FusedFieldObservation):
        raise TypeError("fused_observation must be FusedFieldObservation.")
    if vlm_review_observation is not None and not isinstance(
        vlm_review_observation, VLMReviewObservation
    ):
        raise TypeError("vlm_review_observation must be VLMReviewObservation or None.")
    field = fused_observation
    review = vlm_review_observation
    reviewed = (
        normalize_field_value(field.field_name, review.reviewed_value)
        if review is not None else None
    )
    clear_usable = (
        review is not None and reviewed is not None
        and review.evidence_quality == "clear"
        and review.review_status in ("confirmed", "corrected")
    )

    def result(status, value, reason, temporal_status=None):
        """Construct one result while consistently retaining both evidence layers."""
        return FinalFieldResult(
            field_name=field.field_name, final_status=status, final_value=value,
            fused_observation=field, vlm_review_observation=review,
            resolution_reason=reason, temporal_status=temporal_status,
        )

    status = field.status
    if status == "agreement":
        common = field.ocr_normalized_value
        if common is None or common != field.vlm_normalized_value:
            return result("unresolved", None, "agreement lacks two matching usable values")
        if review is None or review.review_status in ("unresolved", "not_visible"):
            return result("resolved", common, "independent OCR and VLM1 agreement; Pass 2 did not reconfirm")
        if clear_usable and reviewed == common:
            return result("resolved", common, "independent OCR and VLM1 agreement confirmed by clear VLM review")
        return result("unresolved", None, "Pass 2 challenged or could not clearly confirm prior agreement")

    if status == "conflict":
        candidates = (field.ocr_normalized_value, field.vlm_normalized_value)
        matched = [candidate for candidate in candidates if candidate is not None and candidate == reviewed]
        if clear_usable and len(matched) == 1:
            source = "OCR" if matched[0] == candidates[0] else "VLM1"
            return result("resolved", matched[0], f"clear VLM review confirmed the {source} candidate in conflict")
        return result("unresolved", None, "conflict remained unresolved after review")

    if status in ("ocr_only", "vlm_only"):
        candidate = field.ocr_normalized_value if status == "ocr_only" else field.vlm_normalized_value
        if clear_usable and reviewed == candidate and candidate is not None:
            source = "OCR" if status == "ocr_only" else "VLM1"
            return result("resolved_single_source", candidate, f"{source}-only value clearly confirmed by VLM review")
        return result("unresolved", None, "single-source value was not clearly confirmed by Pass 2")

    if status == "unavailable":
        if clear_usable:
            return result("resolved_single_source", reviewed, "final value comes only from clear Pass 2 review evidence")
        return result("not_available", None, "no source produced usable evidence")

    if status == "temporal_mismatch":
        ocr = field.ocr_observation
        ocr_time = (
            ocr.provenance.actual_timestamp_seconds
            if ocr is not None and ocr.status == "extracted" and ocr.selected_candidate is not None
            else None
        )
        time_matches = (
            review is not None and review.evidence_time_s is not None
            and isinstance(ocr_time, Real) and not isinstance(ocr_time, bool)
            and math.isfinite(ocr_time) and review.evidence_time_s == ocr_time
        )
        if clear_usable and time_matches:
            if reviewed == field.ocr_normalized_value:
                return result("resolved", reviewed, "clear Pass 2 review matches OCR at the comparable source time", "comparable")
            vlm_time = field.vlm_observation.evidence_time_s if field.vlm_observation is not None else None
            if reviewed == field.vlm_normalized_value and vlm_time == ocr_time:
                return result("resolved", reviewed, "clear Pass 2 review matches VLM1 at the comparable source time", "comparable")
        reason = "temporal mismatch lacks comparable Pass 2 timing" if not time_matches else "temporal mismatch remained unresolved after review"
        return result("unresolved", None, reason, "non_comparable" if not time_matches else "comparable_but_unresolved")

    raise ValueError(f"Unsupported fusion status: {status!r}.")


def resolve_investigation(
    fused_result: FusedInvestigationResult,
    review_result: VLMReviewResult | None = None,
) -> FinalInvestigationResult:
    """Resolve every fused field and summarize outcomes without orchestration."""
    from events_diary.vlm_review import VLMReviewResult

    if not isinstance(fused_result, FusedInvestigationResult):
        raise TypeError("fused_result must be FusedInvestigationResult.")
    if review_result is not None and not isinstance(review_result, VLMReviewResult):
        raise TypeError("review_result must be VLMReviewResult or None.")
    if review_result is not None and (
        review_result.investigation_id != fused_result.investigation_id
        or review_result.source_video != fused_result.source_video
        or review_result.source_start_s != fused_result.start_time_s
        or review_result.source_end_s != fused_result.end_time_s
    ):
        raise ValueError("Pass 2 and fused Investigation provenance must match.")
    reviews = review_result.reviewed_fields if review_result is not None else {}
    fields = {
        name: resolve_field(observation, reviews.get(name))
        for name, observation in fused_result.fields.items()
    }
    counts = {name: 0 for name in _FINAL_STATUSES}
    for field_result in fields.values():
        counts[field_result.final_status] += 1
    summary = FinalInvestigationSummary(
        total_fields=len(fields), resolved_count=counts["resolved"],
        resolved_single_source_count=counts["resolved_single_source"],
        unresolved_count=counts["unresolved"], not_available_count=counts["not_available"],
    )
    diagnostics = (*fused_result.diagnostics, *(review_result.diagnostics if review_result else ()))
    return FinalInvestigationResult(
        investigation_id=fused_result.investigation_id,
        source_video=fused_result.source_video,
        start_time_s=fused_result.start_time_s,
        end_time_s=fused_result.end_time_s,
        fields=fields,
        summary=summary,
        diagnostics=diagnostics,
    )
