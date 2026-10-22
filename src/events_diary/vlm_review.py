"""Full-Investigation VLM review context, request, output and parser contracts.

Pass 2 receives one continuous clip and a compact serialization of existing
OCR, Pass 1, and fusion evidence. It preserves reviewed text verbatim and does
not decide final field values.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from events_diary.track_direction import (
    DirectionConsistency,
    compare_track_direction,
    derive_track_direction,
)
from events_diary.vlm import (
    RawVLMResult,
    VLMInvestigationResult,
    VLMRequest,
    VLMVisualRouteObservation,
)
from events_diary.vlm_parsing import VLM_FIELD_MAPPING
from events_diary.vlm_prompts import VLM_BOOLEAN_FIELDS

if TYPE_CHECKING:
    from events_diary.fusion import FusedInvestigationResult

FR24_VLM_REVIEW_V1 = "fr24_vlm_review_v1"
REVIEW_STATUSES = ("confirmed", "corrected", "unresolved", "not_visible")
EVIDENCE_QUALITIES = ("clear", "partial", "weak", "not_visible")
HIGH_PRIORITY_STATUSES = frozenset({
    "conflict", "temporal_mismatch", "ocr_only", "vlm_only", "unavailable",
})
REVIEW_FIELD_NAMES = tuple(VLM_FIELD_MAPPING.values())
_REVIEW_FIELD_NAME_SET = frozenset(REVIEW_FIELD_NAMES)
REVIEW_ROUTE_FIELDS = (
    "visual_flight_direction_sector", "visual_flight_direction_text",
    "visual_route_description", "visual_departure_context",
)


@dataclass(frozen=True)
class VLMReviewFieldContext:
    """Serializable source and fusion evidence supplied for one review field."""

    field_name: str
    fusion_status: str
    ocr_raw_value: str | None
    ocr_available: bool
    vlm_pass1_raw_value: str | bool | None
    vlm_pass1_available: bool
    ocr_normalized_value: str | bool | None
    vlm_normalized_value: str | bool | None
    ocr_timestamp_s: float | None = None
    vlm_source_start_s: float | None = None
    vlm_source_end_s: float | None = None
    vlm_evidence_time_s: float | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate lightweight context values and freeze diagnostics."""
        if self.field_name not in _REVIEW_FIELD_NAME_SET:
            raise ValueError("field_name must be a current canonical VLM field.")
        if not isinstance(self.fusion_status, str) or not self.fusion_status:
            raise ValueError("fusion_status must be a nonempty string.")
        for name in ("ocr_available", "vlm_pass1_available"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool.")
        if self.ocr_raw_value is not None and not isinstance(self.ocr_raw_value, str):
            raise TypeError("ocr_raw_value must be a string or None.")
        for name in ("vlm_pass1_raw_value", "ocr_normalized_value", "vlm_normalized_value"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, (str, bool)):
                raise TypeError(f"{name} must be a string, bool, or None.")
        for name in (
            "ocr_timestamp_s", "vlm_source_start_s", "vlm_source_end_s", "vlm_evidence_time_s"
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite nonnegative seconds or None.")
        messages = tuple(self.diagnostics)
        if any(not isinstance(message, str) for message in messages):
            raise TypeError("diagnostics must contain strings.")
        object.__setattr__(self, "diagnostics", messages)

    @property
    def priority(self) -> Literal["high", "normal"]:
        """Return explicit review priority while retaining agreement review."""
        return "high" if self.fusion_status in HIGH_PRIORITY_STATUSES else "normal"


@dataclass(frozen=True)
class VLMReviewContext:
    """Compact immutable full-Investigation review context with all canonical fields."""

    investigation_id: str | int
    source_video: Path
    source_start_s: float
    source_end_s: float
    fields: Mapping[str, VLMReviewFieldContext]
    visual_route: VLMVisualRouteObservation | None = None
    direction_consistency: DirectionConsistency | None = None

    def __post_init__(self) -> None:
        """Freeze the field map and validate its keys and source interval."""
        if isinstance(self.investigation_id, bool) or not isinstance(
            self.investigation_id, (str, int)
        ):
            raise TypeError("investigation_id must be a string or integer, not bool.")
        if not isinstance(self.source_video, Path):
            raise TypeError("source_video must be pathlib.Path.")
        for name in ("source_start_s", "source_end_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number, not bool.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.source_end_s <= self.source_start_s:
            raise ValueError("source_end_s must be greater than source_start_s.")
        fields = dict(self.fields)
        if set(fields) != _REVIEW_FIELD_NAME_SET:
            raise ValueError("Review context must contain every canonical VLM field exactly once.")
        if any(
            not isinstance(item, VLMReviewFieldContext) or key != item.field_name
            for key, item in fields.items()
        ):
            raise ValueError("Review context keys must match their field_name.")
        if self.visual_route is not None and not isinstance(self.visual_route, VLMVisualRouteObservation):
            raise TypeError("visual_route must be VLMVisualRouteObservation or None.")
        object.__setattr__(self, "fields", MappingProxyType(fields))


def build_vlm_review_context(
    fused_result: FusedInvestigationResult,
    *,
    pass1_result=None,
) -> VLMReviewContext:
    """Build clean serializable context from a fusion result, without source copies."""
    required = (
        "investigation_id", "source_video", "start_time_s", "end_time_s", "fields"
    )
    if any(not hasattr(fused_result, name) for name in required):
        raise TypeError("fused_result must be a FusedInvestigationResult-like record.")
    if pass1_result is not None and not isinstance(pass1_result, VLMInvestigationResult):
        raise TypeError("pass1_result must be VLMInvestigationResult or None.")
    if pass1_result is not None and (
        pass1_result.investigation_index != fused_result.investigation_id
        or pass1_result.raw_result.source_video != fused_result.source_video
        or pass1_result.source_start_s != fused_result.start_time_s
        or pass1_result.source_end_s != fused_result.end_time_s
    ):
        raise ValueError("Pass 1 and fusion provenance must match for review context.")
    contexts: dict[str, VLMReviewFieldContext] = {}
    for name in REVIEW_FIELD_NAMES:
        fused = fused_result.fields.get(name)
        if fused is None:
            contexts[name] = VLMReviewFieldContext(
                field_name=name,
                fusion_status="unavailable",
                ocr_raw_value=None,
                ocr_available=False,
                vlm_pass1_raw_value=None,
                vlm_pass1_available=False,
                ocr_normalized_value=None,
                vlm_normalized_value=None,
            )
            continue
        ocr = fused.ocr_observation
        vlm = fused.vlm_observation
        ocr_available = (
            ocr is not None and ocr.status == "extracted" and ocr.selected_candidate is not None
        )
        vlm_available = vlm is not None and vlm.status == "observed" and vlm.raw_value is not None
        contexts[name] = VLMReviewFieldContext(
            field_name=name,
            fusion_status=fused.status,
            ocr_raw_value=ocr.raw_value_text if ocr is not None else None,
            ocr_available=ocr_available,
            vlm_pass1_raw_value=vlm.raw_value if vlm is not None else None,
            vlm_pass1_available=vlm_available,
            ocr_normalized_value=fused.ocr_normalized_value,
            vlm_normalized_value=fused.vlm_normalized_value,
            ocr_timestamp_s=(
                ocr.provenance.actual_timestamp_seconds if ocr is not None else None
            ),
            vlm_source_start_s=vlm.source_start_s if vlm is not None else None,
            vlm_source_end_s=vlm.source_end_s if vlm is not None else None,
            vlm_evidence_time_s=vlm.evidence_time_s if vlm is not None else None,
            diagnostics=fused.diagnostics,
        )
    visual_route = pass1_result.visual_route if pass1_result is not None else None
    track_fused = fused_result.fields.get("TRACK")
    derived = derive_track_direction(track_fused.ocr_observation) if (
        track_fused is not None and track_fused.ocr_observation is not None
    ) else None
    visual_sector = (
        visual_route.direction_sector.value
        if visual_route is not None and visual_route.direction_sector.status in ("observed", "inferred")
        else None
    )
    visual_status = (
        visual_route.direction_sector.status if visual_route is not None else "not_visible"
    )
    direction_consistency = compare_track_direction(
        derived, visual_sector, visual_direction_status=visual_status
    )
    return VLMReviewContext(
        investigation_id=fused_result.investigation_id,
        source_video=fused_result.source_video,
        source_start_s=fused_result.start_time_s,
        source_end_s=fused_result.end_time_s,
        fields=contexts,
        visual_route=visual_route,
        direction_consistency=direction_consistency,
    )


def _context_payload(context: VLMReviewContext) -> dict[str, object]:
    """Convert review context to JSON-safe values without exposing model objects."""
    route = context.visual_route
    consistency = context.direction_consistency
    return {
        "investigation_id": context.investigation_id,
        "source_video": str(context.source_video),
        "source_interval_s": [context.source_start_s, context.source_end_s],
        "fields": {
            name: {
                "priority": item.priority,
                "fusion_status": item.fusion_status,
                "ocr_raw_value": item.ocr_raw_value,
                "ocr_available": item.ocr_available,
                "vlm_pass1_raw_value": item.vlm_pass1_raw_value,
                "vlm_pass1_available": item.vlm_pass1_available,
                "ocr_normalized_value": item.ocr_normalized_value,
                "vlm_normalized_value": item.vlm_normalized_value,
                "ocr_timestamp_s": item.ocr_timestamp_s,
                "vlm_source_interval_s": (
                    [item.vlm_source_start_s, item.vlm_source_end_s]
                    if item.vlm_source_start_s is not None
                    and item.vlm_source_end_s is not None
                    else None
                ),
                "vlm_evidence_time_s": item.vlm_evidence_time_s,
                "diagnostics": item.diagnostics,
            }
            for name, item in context.fields.items()
        },
        "visual_route": ({
            "visual_flight_direction_sector": {"value": route.direction_sector.value, "status": route.direction_sector.status},
            "visual_flight_direction_text": {"value": route.direction_text.value, "status": route.direction_text.status},
            "visual_route_description": {"value": route.route_description.value, "status": route.route_description.status},
            "visual_departure_context": {"value": route.departure_context.value, "status": route.departure_context.status},
        } if route is not None else None),
        "track_direction_cross_check": ({
            "status": consistency.status,
            "derived_track_direction": consistency.derived_direction.text if consistency.derived_direction else None,
            "derived_track_sector": consistency.derived_direction.sector if consistency.derived_direction else None,
            "visual_direction_sector": consistency.visual_direction_sector,
            "visual_direction_status": consistency.visual_direction_status,
            "instruction": "Compare the two preserved sources; do not overwrite either source or choose a winner.",
        } if consistency is not None else {"status": "unavailable"}),
    }


def build_vlm_review_prompt(context: VLMReviewContext) -> str:
    """Build the dedicated full-video review prompt with explicit priority guidance."""
    if not isinstance(context, VLMReviewContext):
        raise TypeError("context must be VLMReviewContext.")
    temporal_guidance = []
    for name, item in context.fields.items():
        if item.fusion_status == "temporal_mismatch" and item.ocr_timestamp_s is not None:
            temporal_guidance.append(
                f"For {name}, inspect around original-video time "
                f"{item.ocr_timestamp_s:g}s and report what is visible there."
            )
    timing = "\n".join(temporal_guidance) or "No field-specific OCR timestamp was supplied."
    return f"""Review the complete continuous Investigation video again. Inspect every field in the supplied review context; do not skip agreement fields. HIGH priority fields are marked high and include conflicts, temporal mismatches, one-source values, and unavailable values. NORMAL priority agreement fields still require an independent visual review.

Do not assume OCR or VLM Pass 1 is correct. Treat both values only as hypotheses to verify against the video. Pay particular attention to conflicting, mismatched, and missing values. Do not vote, blindly repeat prior values, infer hidden information, repair text, or normalize it. Preserve visible source text faithfully. Use confirmed when the video supports a prior value, corrected when a different value is directly visible, unresolved when the evidence cannot distinguish candidates, and not_visible when the field cannot be reviewed. A corrected value is a review observation, not a final system decision.

For temporal_mismatch fields, inspect around the supplied OCR point time when available. Do not invent a timestamp. {timing}

Review context JSON:
{json.dumps(_context_payload(context), ensure_ascii=False, separators=(',', ':'))}

Review the map separately from ticket fields. Recheck each visual route claim and the deterministic TRACK-direction comparison against the full video. Preserve observed versus inferred versus not_visible for route claims, and report direction consistency without selecting either source as authoritative. Never place map-derived context in ticket origin/destination fields.

Return one JSON object with a `fields` object keyed by all canonical ticket field names shown in the context, and a separate `visual_route` object keyed by the four visual route field names. Each review item must contain reviewed_value, review_status, evidence_quality, evidence_time_s, and notes. Use confirmed, corrected, unresolved, or not_visible for review_status. Use clear, partial, weak, or not_visible for evidence_quality. Values may be null when unresolved or not visible. Review every ticket and visual route field."""


@dataclass(frozen=True)
class VLMReviewRequest:
    """One Pass 2 request that reuses the complete continuous Investigation clip."""

    investigation_index: int
    source_video: Path
    video_clip_path: Path
    source_start_s: float
    source_end_s: float
    context: VLMReviewContext
    prompt: str
    prompt_version: str = FR24_VLM_REVIEW_V1

    def __post_init__(self) -> None:
        """Validate full-clip provenance and ensure prompt/context identities match."""
        if not isinstance(self.investigation_index, int) or isinstance(self.investigation_index, bool):
            raise TypeError("investigation_index must be an integer, not bool.")
        if self.investigation_index < 0:
            raise ValueError("investigation_index must be nonnegative.")
        if not isinstance(self.source_video, Path) or not isinstance(self.video_clip_path, Path):
            raise TypeError("source_video and video_clip_path must be pathlib.Path.")
        for name in ("source_start_s", "source_end_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number, not bool.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.source_end_s <= self.source_start_s:
            raise ValueError("source_end_s must be greater than source_start_s.")
        if not isinstance(self.context, VLMReviewContext):
            raise TypeError("context must be VLMReviewContext.")
        if (
            self.source_video != self.context.source_video
            or self.source_start_s != self.context.source_start_s
            or self.source_end_s != self.context.source_end_s
            or self.investigation_index != self.context.investigation_id
        ):
            raise ValueError("Request provenance must match its review context.")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a nonempty string.")
        if self.prompt_version != FR24_VLM_REVIEW_V1:
            raise ValueError("Unsupported VLM review prompt version.")


def build_vlm_review_request(
    base_request: VLMRequest,
    fused_result: FusedInvestigationResult,
    *,
    pass1_result: VLMInvestigationResult,
) -> VLMReviewRequest:
    """Create Pass 2 request from Pass 1's complete clip and cross-check result."""
    if not isinstance(base_request, VLMRequest):
        raise TypeError("base_request must be VLMRequest.")
    if not isinstance(pass1_result, VLMInvestigationResult):
        raise TypeError("pass1_result must be VLMInvestigationResult.")
    context = build_vlm_review_context(fused_result, pass1_result=pass1_result)
    if (
        context.investigation_id != base_request.investigation_index
        or context.source_video != base_request.source_video
        or context.source_start_s != base_request.source_start_s
        or context.source_end_s != base_request.source_end_s
    ):
        raise ValueError("Base VLM request and fusion result provenance must match.")
    return VLMReviewRequest(
        investigation_index=base_request.investigation_index,
        source_video=base_request.source_video,
        video_clip_path=base_request.video_clip_path,
        source_start_s=base_request.source_start_s,
        source_end_s=base_request.source_end_s,
        context=context,
        prompt=build_vlm_review_prompt(context),
    )


@dataclass(frozen=True)
class VLMReviewObservation:
    """One verbatim Pass 2 field report, not a final resolved value."""

    field_name: str
    reviewed_value: str | bool | None
    review_status: Literal["confirmed", "corrected", "unresolved", "not_visible"]
    evidence_quality: Literal["clear", "partial", "weak", "not_visible"]
    evidence_time_s: float | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        """Validate review literals while preserving returned field text exactly."""
        if self.field_name not in _REVIEW_FIELD_NAME_SET:
            raise ValueError("field_name must be a current canonical VLM field.")
        if self.review_status not in REVIEW_STATUSES:
            raise ValueError("Unsupported review_status.")
        if self.evidence_quality not in EVIDENCE_QUALITIES:
            raise ValueError("Unsupported evidence_quality.")
        if self.reviewed_value is not None and not isinstance(
            self.reviewed_value, (str, bool)
        ):
            raise TypeError("reviewed_value must be a string, bool, or None.")
        is_boolean_field = self.field_name in VLM_BOOLEAN_FIELDS
        if self.reviewed_value is not None and (
            isinstance(self.reviewed_value, bool) != is_boolean_field
        ):
            raise TypeError("reviewed_value type does not match the canonical field.")
        if self.evidence_time_s is not None and (
            isinstance(self.evidence_time_s, bool)
            or not isinstance(self.evidence_time_s, Real)
            or not math.isfinite(self.evidence_time_s)
            or self.evidence_time_s < 0
        ):
            raise ValueError("evidence_time_s must be finite nonnegative seconds or None.")
        if self.notes is not None and not isinstance(self.notes, str):
            raise TypeError("notes must be a string or None.")


@dataclass(frozen=True)
class VLMRouteReviewObservation:
    """Pass 2 review of one map-derived claim, kept outside ticket fields."""

    field_name: str
    reviewed_value: str | None
    review_status: Literal["confirmed", "corrected", "unresolved", "not_visible"]
    evidence_quality: Literal["clear", "partial", "weak", "not_visible"]
    observation_status: Literal["observed", "inferred", "not_visible"]

    def __post_init__(self) -> None:
        """Validate route review identity and preserve evidence-status distinctions."""
        if self.field_name not in REVIEW_ROUTE_FIELDS:
            raise ValueError("field_name must identify a visual route field.")
        if self.review_status not in REVIEW_STATUSES or self.evidence_quality not in EVIDENCE_QUALITIES:
            raise ValueError("Unsupported route review status or evidence quality.")
        if self.observation_status not in ("observed", "inferred", "not_visible"):
            raise ValueError("Unsupported route observation status.")
        if self.reviewed_value is not None and (
            not isinstance(self.reviewed_value, str) or not self.reviewed_value.strip()
        ):
            raise ValueError("reviewed_value must be nonempty text or None.")
        if self.observation_status == "not_visible" and self.reviewed_value is not None:
            raise ValueError("not_visible route review must have a null reviewed_value.")


@dataclass(frozen=True)
class VLMReviewResult:
    """Pass 2 parsed fields and diagnostics with provider provenance retained."""

    raw_result: RawVLMResult
    reviewed_fields: Mapping[str, VLMReviewObservation]
    diagnostics: tuple[str, ...] = ()
    route_reviews: Mapping[str, VLMRouteReviewObservation] = field(default_factory=dict)
    direction_consistency: DirectionConsistency | None = None

    def __post_init__(self) -> None:
        """Freeze and validate review fields while retaining raw runtime output."""
        if not isinstance(self.raw_result, RawVLMResult):
            raise TypeError("raw_result must be RawVLMResult.")
        fields = dict(self.reviewed_fields)
        routes = dict(self.route_reviews)
        if self.direction_consistency is not None and not isinstance(
            self.direction_consistency, DirectionConsistency
        ):
            raise TypeError("direction_consistency must be DirectionConsistency or None.")
        if any(
            not isinstance(item, VLMReviewObservation) or key != item.field_name
            for key, item in fields.items()
        ):
            raise ValueError("reviewed_fields keys must match VLMReviewObservation field names.")
        if self.raw_result.execution_status != "success" and fields:
            raise ValueError("Failed Pass 2 execution must not contain reviewed fields.")
        if any(not isinstance(item, VLMRouteReviewObservation) or key != item.field_name
               for key, item in routes.items()):
            raise ValueError("route_reviews keys must match VLMRouteReviewObservation names.")
        if self.raw_result.execution_status != "success" and routes:
            raise ValueError("Failed Pass 2 execution must not contain route reviews.")
        messages = tuple(self.diagnostics)
        if any(not isinstance(message, str) for message in messages):
            raise TypeError("diagnostics must contain strings.")
        object.__setattr__(self, "reviewed_fields", MappingProxyType(fields))
        object.__setattr__(self, "route_reviews", MappingProxyType(routes))
        object.__setattr__(self, "diagnostics", messages)

    @property
    def investigation_id(self) -> int:
        """Expose Investigation identity from the common raw VLM result."""
        return self.raw_result.investigation_index

    @property
    def source_video(self) -> Path:
        """Expose the original recording path."""
        return self.raw_result.source_video

    @property
    def source_start_s(self) -> float:
        """Expose the original-video interval start."""
        return self.raw_result.source_start_s

    @property
    def source_end_s(self) -> float:
        """Expose the original-video interval end."""
        return self.raw_result.source_end_s

    @property
    def model_name(self) -> str:
        """Expose configured provider model name."""
        return self.raw_result.model_name

    @property
    def model_revision(self) -> str | None:
        """Expose provider-reported model revision."""
        return self.raw_result.model_revision

    @property
    def prompt_version(self) -> str:
        """Expose the dedicated Pass 2 prompt version."""
        return self.raw_result.prompt_version

    @property
    def execution_status(self) -> str:
        """Expose provider execution status without duplicating it."""
        return self.raw_result.execution_status


def parse_vlm_review_result(
    request: VLMReviewRequest,
    raw_result: RawVLMResult,
) -> VLMReviewResult:
    """Parse Pass 2 JSON while preserving every returned value verbatim."""
    if not isinstance(request, VLMReviewRequest):
        raise TypeError("request must be VLMReviewRequest.")
    if not isinstance(raw_result, RawVLMResult):
        raise TypeError("raw_result must be RawVLMResult.")
    if (
        request.investigation_index != raw_result.investigation_index
        or request.source_video != raw_result.source_video
        or request.video_clip_path != raw_result.video_clip_path
        or request.source_start_s != raw_result.source_start_s
        or request.source_end_s != raw_result.source_end_s
        or request.prompt_version != raw_result.prompt_version
    ):
        raise ValueError("Review request and raw result provenance must match.")
    if raw_result.execution_status != "success":
        return VLMReviewResult(
            raw_result, {}, ("Pass 2 execution did not succeed.",),
            direction_consistency=request.context.direction_consistency,
        )
    try:
        payload = json.loads(raw_result.raw_text or "")
    except (json.JSONDecodeError, TypeError):
        return VLMReviewResult(
            raw_result, {}, ("Pass 2 response is not valid JSON.",),
            direction_consistency=request.context.direction_consistency,
        )
    if not isinstance(payload, dict) or not isinstance(payload.get("fields"), dict):
        return VLMReviewResult(
            raw_result, {}, ("Pass 2 response must contain a fields object.",),
            direction_consistency=request.context.direction_consistency,
        )

    fields: dict[str, VLMReviewObservation] = {}
    diagnostics: list[str] = []
    payload_fields = payload["fields"]
    for unknown in sorted(set(payload_fields) - _REVIEW_FIELD_NAME_SET):
        diagnostics.append(f"Ignored unknown review field: {unknown}.")
    for name in REVIEW_FIELD_NAMES:
        raw_field = payload_fields.get(name)
        if raw_field is None:
            diagnostics.append(f"Missing review field: {name}.")
            continue
        if not isinstance(raw_field, dict):
            diagnostics.append(f"Invalid review field shape: {name}.")
            continue
        try:
            observation = VLMReviewObservation(
                field_name=name,
                reviewed_value=raw_field.get("reviewed_value"),
                review_status=raw_field.get("review_status"),
                evidence_quality=raw_field.get("evidence_quality"),
                evidence_time_s=raw_field.get("evidence_time_s"),
                notes=raw_field.get("notes"),
            )
            if observation.evidence_time_s is not None and not (
                request.source_start_s
                <= observation.evidence_time_s
                <= request.source_end_s
            ):
                raise ValueError("evidence_time_s outside request source interval")
            fields[name] = observation
        except (TypeError, ValueError):
            diagnostics.append(f"Invalid review field values: {name}.")
    routes: dict[str, VLMRouteReviewObservation] = {}
    route_payload = payload.get("visual_route")
    if not isinstance(route_payload, dict):
        diagnostics.append("Pass 2 response must contain a visual_route object.")
    else:
        for name in REVIEW_ROUTE_FIELDS:
            item = route_payload.get(name)
            if not isinstance(item, dict):
                diagnostics.append(f"Missing or invalid visual route review: {name}.")
                continue
            try:
                observation = VLMRouteReviewObservation(
                    field_name=name,
                    reviewed_value=item.get("reviewed_value"),
                    review_status=item.get("review_status"),
                    evidence_quality=item.get("evidence_quality"),
                    observation_status=item.get("observation_status"),
                )
                routes[name] = observation
            except (TypeError, ValueError):
                diagnostics.append(f"Invalid visual route review values: {name}.")
    return VLMReviewResult(
        raw_result, fields, tuple(diagnostics), routes,
        request.context.direction_consistency,
    )
