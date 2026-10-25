"""Immutable Phase 7 canonical event and session contracts.

This module converts final Phase 6 Investigation records into the small set of
human-useful Events Diary fields. It does not inspect evidence, repeat fusion,
call OCR or Gemini, infer missing values, or expose Phase 6 diagnostics. The
canonical session is serializable source data for future deterministic
renderers; no renderer or file output is implemented here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields, is_dataclass
from numbers import Real
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from events_diary.investigation_record import (
        InvestigationRecord,
        VideoInvestigationResult,
    )

SCHEMA_VERSION = "1.0"
_CANONICAL_VALUE_PATHS = (
    "flight.primary_id", "flight.alternate_id", "flight.airline",
    "route.origin_airport_code", "route.destination_airport_code",
    "route.scheduled_departure", "route.actual_departure",
    "route.scheduled_arrival", "route.estimated_or_actual_arrival",
    "aircraft.registration", "aircraft.code", "aircraft.type",
    "movement.track_degrees", "movement.direction",
    "movement.visual_flight_direction", "movement.visual_route_description",
    "movement.visual_departure_context",
)
_QUALITY_PATHS = frozenset(_CANONICAL_VALUE_PATHS)
CANONICAL_RENDERER_FIELD_PATHS = frozenset((
    *_CANONICAL_VALUE_PATHS,
    "event_time_s", "investigation_id", "start_time_s", "end_time_s",
    "data_quality.unresolved_fields", "data_quality.not_available_fields",
))


@dataclass(frozen=True)
class CanonicalFieldProfile:
    """Ordered canonical field paths selected for one future diary profile."""

    name: str
    field_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject invalid, duplicate, or noncanonical paths in the profile."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("profile name must be nonempty text.")
        if not isinstance(self.field_paths, tuple):
            raise TypeError("field_paths must be an ordered tuple.")
        if any(not isinstance(path, str) for path in self.field_paths):
            raise TypeError("field_paths must contain strings.")
        if len(set(self.field_paths)) != len(self.field_paths):
            raise ValueError("field_paths must not contain duplicates.")
        invalid = tuple(
            path for path in self.field_paths
            if path not in CANONICAL_RENDERER_FIELD_PATHS
        )
        if invalid:
            raise ValueError(f"profile contains noncanonical paths: {invalid!r}.")


OPERATIONAL_DIARY_PROFILE = CanonicalFieldProfile(
    name="operational",
    field_paths=(
        "event_time_s",
        "flight.primary_id", "flight.alternate_id", "flight.airline",
        "route.origin_airport_code", "route.destination_airport_code",
        "route.scheduled_departure", "route.actual_departure",
        "route.scheduled_arrival", "route.estimated_or_actual_arrival",
        "aircraft.registration", "aircraft.code", "aircraft.type",
        "movement.direction",
    ),
)

DETAILED_DIARY_PROFILE = CanonicalFieldProfile(
    name="detailed",
    field_paths=(
        *OPERATIONAL_DIARY_PROFILE.field_paths,
        "investigation_id", "start_time_s", "end_time_s",
        "movement.track_degrees", "movement.visual_flight_direction",
        "movement.visual_route_description", "movement.visual_departure_context",
        "data_quality.unresolved_fields", "data_quality.not_available_fields",
    ),
)

if not set(OPERATIONAL_DIARY_PROFILE.field_paths).issubset(DETAILED_DIARY_PROFILE.field_paths):
    raise RuntimeError("Detailed diary profile must contain the Operational profile.")


def _text(name: str, value: str | None) -> None:
    """Validate optional nonempty text without normalizing its content."""
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{name} must be nonempty text or None.")


def _time(name: str, value: float) -> None:
    """Validate finite nonnegative seconds while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not bool.")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")


def _identifier(name: str, value: int) -> None:
    """Validate a nonnegative integer identity, excluding boolean values."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")


def _json_value(value):
    """Convert nested immutable contract values to JSON-ready dicts and lists."""
    if is_dataclass(value):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class CanonicalFlight:
    """Flight identifiers and airline name exposed by one diary event."""

    primary_id: str | None = None
    alternate_id: str | None = None
    airline: str | None = None

    def __post_init__(self) -> None:
        """Validate the optional canonical flight values."""
        for name in ("primary_id", "alternate_id", "airline"):
            _text(name, getattr(self, name))


@dataclass(frozen=True)
class CanonicalRoute:
    """Ticket-derived route codes and displayed departure/arrival times."""

    origin_airport_code: str | None = None
    destination_airport_code: str | None = None
    scheduled_departure: str | None = None
    actual_departure: str | None = None
    scheduled_arrival: str | None = None
    estimated_or_actual_arrival: str | None = None

    def __post_init__(self) -> None:
        """Validate optional canonical route values."""
        for item in fields(self):
            _text(item.name, getattr(self, item.name))


@dataclass(frozen=True)
class CanonicalAircraft:
    """Primary aircraft identity fields selected for the Events Diary."""

    registration: str | None = None
    code: str | None = None
    type: str | None = None

    def __post_init__(self) -> None:
        """Validate optional canonical aircraft values."""
        for name in ("registration", "code", "type"):
            _text(name, getattr(self, name))


@dataclass(frozen=True)
class CanonicalMovement:
    """Ticket-derived TRACK and selected semantic visual route context."""

    track_degrees: float | None = None
    direction: str | None = None
    visual_flight_direction: str | None = None
    visual_route_description: str | None = None
    visual_departure_context: str | None = None

    def __post_init__(self) -> None:
        """Validate a finite bearing and optional movement descriptions."""
        if self.track_degrees is not None:
            if isinstance(self.track_degrees, bool) or not isinstance(self.track_degrees, Real):
                raise TypeError("track_degrees must be numeric or None.")
            if not math.isfinite(self.track_degrees) or not 0 <= self.track_degrees <= 360:
                raise ValueError("track_degrees must be finite and within [0, 360].")
        for name in (
            "direction", "visual_flight_direction", "visual_route_description",
            "visual_departure_context",
        ):
            _text(name, getattr(self, name))


@dataclass(frozen=True)
class CanonicalDataQuality:
    """Separate canonical paths whose values are unresolved or unavailable."""

    unresolved_fields: tuple[str, ...] = ()
    not_available_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze, validate, sort, and keep quality path lists disjoint."""
        unresolved = tuple(self.unresolved_fields)
        unavailable = tuple(self.not_available_fields)
        for name, values in (
            ("unresolved_fields", unresolved),
            ("not_available_fields", unavailable),
        ):
            if any(not isinstance(value, str) for value in values):
                raise TypeError(f"{name} must contain canonical field paths.")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicate paths.")
            if any(value not in _QUALITY_PATHS for value in values):
                raise ValueError(f"{name} contains an unknown canonical field path.")
        if set(unresolved) & set(unavailable):
            raise ValueError("A canonical field path cannot be both unresolved and unavailable.")
        object.__setattr__(self, "unresolved_fields", tuple(sorted(unresolved)))
        object.__setattr__(self, "not_available_fields", tuple(sorted(unavailable)))


@dataclass(frozen=True)
class CanonicalEvent:
    """One Investigation represented as a clean, immutable diary event."""

    event_id: int
    investigation_id: int
    event_time_s: float
    start_time_s: float
    end_time_s: float
    flight: CanonicalFlight = field(default_factory=CanonicalFlight)
    route: CanonicalRoute = field(default_factory=CanonicalRoute)
    aircraft: CanonicalAircraft = field(default_factory=CanonicalAircraft)
    movement: CanonicalMovement = field(default_factory=CanonicalMovement)
    data_quality: CanonicalDataQuality = field(default_factory=CanonicalDataQuality)

    def __post_init__(self) -> None:
        """Validate identity, event times, and required nested value groups."""
        _identifier("event_id", self.event_id)
        _identifier("investigation_id", self.investigation_id)
        for name in ("event_time_s", "start_time_s", "end_time_s"):
            _time(name, getattr(self, name))
        if self.end_time_s < self.start_time_s:
            raise ValueError("end_time_s must not precede start_time_s.")
        if not self.start_time_s <= self.event_time_s <= self.end_time_s:
            raise ValueError("event_time_s must lie within the Investigation interval.")
        for name, expected in (
            ("flight", CanonicalFlight), ("route", CanonicalRoute),
            ("aircraft", CanonicalAircraft), ("movement", CanonicalMovement),
            ("data_quality", CanonicalDataQuality),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}.")
        values = _event_field_values(self)
        unresolved = set(self.data_quality.unresolved_fields)
        unavailable = set(self.data_quality.not_available_fields)
        for path, value in values.items():
            if value is not None and (path in unresolved or path in unavailable):
                raise ValueError(f"Non-null canonical field {path!r} cannot have a quality status.")
        for path in unresolved | unavailable:
            if values[path] is not None:
                raise ValueError(f"Canonical field {path!r} must be null when quality-labeled.")

    def to_dict(self) -> dict[str, object]:
        """Return the approved nested shape with JSON arrays for quality paths."""
        return _json_value(self)

    def field_state(self, canonical_path: str) -> Literal[
        "value", "unresolved", "not_available", "absent",
    ]:
        """Return the explicit missing-data state for one canonical field path."""
        if canonical_path not in _QUALITY_PATHS:
            raise ValueError("canonical_path must identify a known canonical field.")
        value = _event_field_values(self)[canonical_path]
        if value is not None:
            return "value"
        if canonical_path in self.data_quality.unresolved_fields:
            return "unresolved"
        if canonical_path in self.data_quality.not_available_fields:
            return "not_available"
        return "absent"


@dataclass(frozen=True)
class CanonicalSourceVideo:
    """Logical source-video identifier and duration, without filesystem access."""

    name: str
    duration_s: float

    def __post_init__(self) -> None:
        """Require a readable logical identifier and finite duration."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("source video name must be nonempty text.")
        _time("duration_s", self.duration_s)


@dataclass(frozen=True)
class CanonicalSession:
    """Versioned source-of-truth session containing independent canonical events."""

    source_video: CanonicalSourceVideo
    events: tuple[CanonicalEvent, ...]
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    events_count: int = field(init=False)

    def __post_init__(self) -> None:
        """Sort events deterministically and derive the count from the collection."""
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}.")
        if not isinstance(self.source_video, CanonicalSourceVideo):
            raise TypeError("source_video must be CanonicalSourceVideo.")
        events = tuple(self.events)
        if any(not isinstance(item, CanonicalEvent) for item in events):
            raise TypeError("events must contain CanonicalEvent records.")
        if len({item.event_id for item in events}) != len(events):
            raise ValueError("event_id values must be unique within a session.")
        if len({item.investigation_id for item in events}) != len(events):
            raise ValueError("each Investigation must map to exactly one CanonicalEvent.")
        events = tuple(sorted(events, key=lambda item: (item.event_time_s, item.investigation_id)))
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "events_count", len(events))

    def to_dict(self) -> dict[str, object]:
        """Return the external session shape as JSON-compatible data."""
        return {
            "schema_version": self.schema_version,
            "source_video": _json_value(self.source_video),
            "events_count": self.events_count,
            "events": [_json_value(event) for event in self.events],
        }

    def to_json(self, **kwargs) -> str:
        """Serialize the session contract to JSON text."""
        return json.dumps(self.to_dict(), **kwargs)


_FINAL_FIELD_PATHS = (
    ("flight.primary_id", "primary_flight_id"),
    ("flight.alternate_id", "alternate_flight_id"),
    ("flight.airline", "airline_name"),
    ("route.origin_airport_code", "origin_airport_code"),
    ("route.destination_airport_code", "destination_airport_code"),
    ("route.scheduled_departure", "origin_scheduled_time"),
    ("route.actual_departure", "origin_actual_time"),
    ("route.scheduled_arrival", "destination_scheduled_time"),
    ("route.estimated_or_actual_arrival", "destination_estimated_or_actual_time"),
    ("aircraft.registration", "REGISTRATION"),
    ("aircraft.code", "aircraft_code"),
    ("aircraft.type", "AIRCRAFT_TYPE"),
    ("movement.track_degrees", "TRACK"),
)


def _track_number(value: str | bool | None) -> float | None:
    """Convert a resolved Phase 6 numeric TRACK string without inferring a value."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and 0 <= number <= 360 else None


def _put_final_field(record, source_name: str, canonical_path: str, values: dict, quality: dict) -> None:
    """Copy one Phase 6 final status/value into its canonical path and quality list."""
    result = record.final_result.fields.get(source_name)
    if result is None:
        # Absence from the Phase 6 result map carries no explicit final status.
        # Keep the canonical value null without fabricating a quality label.
        return
    elif result.final_status in ("resolved", "resolved_single_source"):
        value = result.final_value
        if canonical_path == "movement.track_degrees":
            value = _track_number(value)
            if value is None:
                quality["unresolved_fields"].append(canonical_path)
        values[canonical_path] = value
    elif result.final_status == "unresolved":
        quality["unresolved_fields"].append(canonical_path)
    else:
        quality["not_available_fields"].append(canonical_path)


def _put_visual_claim(claim, canonical_path: str, values: dict, quality: dict) -> None:
    """Propagate a final Phase 6 visual review, preserving explicit missing states."""
    if claim is None:
        return
    if claim.review_status == "unresolved":
        quality["unresolved_fields"].append(canonical_path)
    elif claim.observation_status == "not_visible" or claim.review_status == "not_visible":
        quality["not_available_fields"].append(canonical_path)
    elif claim.review_status in ("confirmed", "corrected") and claim.reviewed_value is not None:
        values[canonical_path] = claim.reviewed_value
    else:
        quality["unresolved_fields"].append(canonical_path)


def _event_field_values(event: CanonicalEvent) -> dict[str, object]:
    """Return every canonical path and its stored value for invariant checks."""
    return {
        "flight.primary_id": event.flight.primary_id,
        "flight.alternate_id": event.flight.alternate_id,
        "flight.airline": event.flight.airline,
        "route.origin_airport_code": event.route.origin_airport_code,
        "route.destination_airport_code": event.route.destination_airport_code,
        "route.scheduled_departure": event.route.scheduled_departure,
        "route.actual_departure": event.route.actual_departure,
        "route.scheduled_arrival": event.route.scheduled_arrival,
        "route.estimated_or_actual_arrival": event.route.estimated_or_actual_arrival,
        "aircraft.registration": event.aircraft.registration,
        "aircraft.code": event.aircraft.code,
        "aircraft.type": event.aircraft.type,
        "movement.track_degrees": event.movement.track_degrees,
        "movement.direction": event.movement.direction,
        "movement.visual_flight_direction": event.movement.visual_flight_direction,
        "movement.visual_route_description": event.movement.visual_route_description,
        "movement.visual_departure_context": event.movement.visual_departure_context,
    }


def _put_visual_reviews(reviews, values: dict, quality: dict) -> None:
    """Map final Pass 2 visual reviews, falling back from text to reviewed sector."""
    direction_reviews = (
        reviews.get("visual_flight_direction_text"),
        reviews.get("visual_flight_direction_sector"),
    )
    usable_direction = next(
        (item for item in direction_reviews
         if item is not None and item.review_status in ("confirmed", "corrected")
         and item.observation_status != "not_visible" and item.reviewed_value is not None),
        None,
    )
    if usable_direction is not None:
        values["movement.visual_flight_direction"] = usable_direction.reviewed_value
    elif any(item is not None and item.review_status == "unresolved" for item in direction_reviews):
        quality["unresolved_fields"].append("movement.visual_flight_direction")
    elif any(item is not None for item in direction_reviews):
        quality["not_available_fields"].append("movement.visual_flight_direction")

    for source_name, canonical_path in (
        ("visual_route_description", "movement.visual_route_description"),
        ("visual_departure_context", "movement.visual_departure_context"),
    ):
        _put_visual_claim(reviews.get(source_name), canonical_path, values, quality)


def build_canonical_event(
    investigation_record: InvestigationRecord,
    event_id: int,
) -> CanonicalEvent:
    """Map one final Phase 6 record to one canonical event without re-resolution.

    Ticket field values and their unresolved/not-available status come directly
    from ``FinalInvestigationResult``. Visual route fields use final Pass 2 review
    observations when present; Pass 1 claims are not treated as final values.
    No OCR/VLM execution or evidence arbitration occurs.
    """
    from events_diary.investigation_record import InvestigationRecord

    if not isinstance(investigation_record, InvestigationRecord):
        raise TypeError("investigation_record must be InvestigationRecord.")
    _identifier("event_id", event_id)
    values: dict[str, object] = {}
    quality: dict[str, list[str]] = {"unresolved_fields": [], "not_available_fields": []}
    for canonical_path, source_name in _FINAL_FIELD_PATHS:
        _put_final_field(investigation_record, source_name, canonical_path, values, quality)

    consistency = investigation_record.final_result.direction_consistency
    track_status = investigation_record.final_result.fields.get("TRACK")
    if track_status is None:
        pass
    elif track_status.final_status == "not_available":
        quality["not_available_fields"].append("movement.direction")
    elif track_status.final_status == "unresolved":
        quality["unresolved_fields"].append("movement.direction")
    elif consistency is not None and consistency.derived_direction is not None:
        values["movement.direction"] = consistency.derived_direction.text
    _put_visual_reviews(investigation_record.vlm_pass2.route_reviews, values, quality)

    group_values = lambda prefix: {
        name: values.get(f"{prefix}.{name}")
        for name in {
            "flight": ("primary_id", "alternate_id", "airline"),
            "route": (
                "origin_airport_code", "destination_airport_code", "scheduled_departure",
                "actual_departure", "scheduled_arrival", "estimated_or_actual_arrival",
            ),
            "aircraft": ("registration", "code", "type"),
            "movement": (
                "track_degrees", "direction", "visual_flight_direction",
                "visual_route_description", "visual_departure_context",
            ),
        }[prefix]
    }
    return CanonicalEvent(
        event_id=event_id,
        investigation_id=investigation_record.investigation_id,
        event_time_s=investigation_record.start_time_s,
        start_time_s=investigation_record.start_time_s,
        end_time_s=investigation_record.end_time_s,
        flight=CanonicalFlight(**group_values("flight")),
        route=CanonicalRoute(**group_values("route")),
        aircraft=CanonicalAircraft(**group_values("aircraft")),
        movement=CanonicalMovement(**group_values("movement")),
        data_quality=CanonicalDataQuality(**{
            name: tuple(paths) for name, paths in quality.items()
        }),
    )


def build_canonical_session(
    video_result: VideoInvestigationResult,
) -> CanonicalSession:
    """Aggregate one Phase 6 video's records into one canonical session.

    Every input record becomes one event. Records are sorted by Investigation
    start time and ID before sequential event IDs are assigned. The source name
    and duration come directly from authoritative Phase 6 video metadata.
    Empty record collections produce a valid empty session.
    """
    from events_diary.investigation_record import VideoInvestigationResult

    if not isinstance(video_result, VideoInvestigationResult):
        raise TypeError("video_result must be VideoInvestigationResult.")
    if video_result.failures:
        raise ValueError("Cannot aggregate a video result with failed Investigations.")
    records = video_result.investigations
    investigation_ids = [record.investigation_id for record in records]
    if len(set(investigation_ids)) != len(investigation_ids):
        raise ValueError("Investigation IDs must be unique within one video session.")
    ordered_records = sorted(
        records, key=lambda record: (record.start_time_s, record.investigation_id),
    )
    events = tuple(
        build_canonical_event(record, event_id=event_id)
        for event_id, record in enumerate(ordered_records)
    )
    return CanonicalSession(
        source_video=CanonicalSourceVideo(
            video_result.video_metadata.path.name,
            video_result.video_metadata.duration_seconds,
        ),
        events=events,
    )
