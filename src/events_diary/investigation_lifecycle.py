"""Phase 3.4 offline lifecycle interpretation of confirmed semantic events.

Transform initial presence, ordered confirmed events, and video bounds into
Investigation intervals using only NO_INVESTIGATION and ACTIVE_INVESTIGATION.
Startup presence starts an interval without inventing an OPEN. Visual candidate
semantics can differ from lifecycle semantics: NO + SWITCH intentionally starts
an interval, while ACTIVE + OPEN replaces the current interval. ACTIVE + SWITCH
also replaces it at exactly the same boundary, without an artificial gap.

All boundaries use candidate_time_seconds, never the later confirmed_at_seconds.
An interval still active at video end closes with VIDEO_END. NO + CLOSE leaves
state unchanged and produces a diagnostic anomaly, not a fatal error or interval.
Malformed kinds/timestamps/order are rejected rather than silently corrected.
Equal event times preserve input order and may produce zero-duration intervals.

This module consumes existing ConfirmedInvestigationEvent records only. It does
not read video, perform CV, estimate presence/signals, detect or confirm candidates,
assign identity, select frames, extract content, or implement Phase 4. Intervals
are semantic offline boundaries, not evidence that aircraft identity is known.
Anomaly event_index is the zero-based position in the supplied event sequence,
not its candidate pair index. Inputs are not mutated or sorted.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Real

from events_diary.investigation_detection import InvestigationCandidateKind
from events_diary.investigation_temporal import ConfirmedInvestigationEvent


class InvestigationLifecycleState(Enum):
    """The lifecycle has either no Investigation or one active Investigation."""

    NO_INVESTIGATION = "no_investigation"
    ACTIVE_INVESTIGATION = "active_investigation"


class InvestigationStartReason(Enum):
    """Why an interval starts: startup presence, confirmed OPEN, or SWITCH."""

    VIDEO_STARTED_ACTIVE = "video_started_active"
    OPEN = "open"
    SWITCH = "switch"


class InvestigationEndReason(Enum):
    """Why an interval ends: CLOSE, replacement by SWITCH/OPEN, or video end."""

    CLOSE = "close"
    SWITCH = "switch"
    OPEN_REPLACEMENT = "open_replacement"
    VIDEO_END = "video_end"


@dataclass(frozen=True)
class InvestigationInterval:
    """One ordered immutable Investigation interval with semantic boundary reasons.

    investigation_index starts at zero and follows chronological output order.
    start_time_seconds <= end_time_seconds; replacements share a boundary and
    zero duration is allowed. No aircraft, content, confidence, or frames exist.
    """

    investigation_index: int
    start_time_seconds: float
    end_time_seconds: float
    start_reason: InvestigationStartReason
    end_reason: InvestigationEndReason


@dataclass(frozen=True)
class InvestigationLifecycleAnomaly:
    """Diagnostic for a tolerated semantic inconsistency, not an event/interval.

    event_index is the input sequence position. event_kind/time identify the
    confirmed event; state_before and deterministic reason explain the anomaly.
    """

    event_index: int
    event_kind: InvestigationCandidateKind
    event_time_seconds: float
    state_before: InvestigationLifecycleState
    reason: str


def _validate_time(name: str, value: float) -> None:
    """Reject boolean/non-real timestamps with TypeError and non-finite values with ValueError."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not a boolean.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def build_investigation_lifecycle(
    *, initial_present: bool, video_start_time_seconds: float, video_end_time_seconds: float,
    confirmed_events: Sequence[ConfirmedInvestigationEvent],
) -> tuple[list[InvestigationInterval], list[InvestigationLifecycleAnomaly]]:
    """Interpret confirmed events into chronological intervals and anomalies.

    Start active at video_start when initial_present is True, without an OPEN
    event. NO + OPEN/SWITCH starts an interval; NO + CLOSE records an anomaly.
    ACTIVE + CLOSE ends the interval; ACTIVE + SWITCH or OPEN replaces it at the
    same candidate timestamp, with end reason SWITCH or OPEN_REPLACEMENT.
    Close any final active interval at video_end with VIDEO_END.

    Validate finite real video bounds (nonnegative start, end >= start), a boolean
    initial presence, and existing confirmed-event records with finite in-bounds
    candidate times and OPEN/SWITCH/CLOSE kinds. Types raise TypeError; invalid
    values/order raise ValueError. Equal event times preserve input order.
    confirmed_at_seconds is confirmation metadata only, never a boundary.
    Return frozen interval/anomaly records without sorting or mutating input.
    No CV, confirmation, identity, or content extraction is performed.
    """
    if not isinstance(initial_present, bool):
        raise TypeError("initial_present must be a boolean.")
    _validate_time("video_start_time_seconds", video_start_time_seconds)
    _validate_time("video_end_time_seconds", video_end_time_seconds)
    if video_start_time_seconds < 0:
        raise ValueError("video_start_time_seconds must be non-negative.")
    if video_end_time_seconds < video_start_time_seconds:
        raise ValueError("video_end_time_seconds must be >= video_start_time_seconds.")
    if not isinstance(confirmed_events, Sequence):
        raise TypeError("confirmed_events must be a Sequence of ConfirmedInvestigationEvent.")
    previous_time = video_start_time_seconds
    for event in confirmed_events:
        if not isinstance(event, ConfirmedInvestigationEvent):
            raise TypeError("confirmed_events must contain ConfirmedInvestigationEvent records.")
        if not isinstance(event.kind, InvestigationCandidateKind):
            raise TypeError("event kind must be InvestigationCandidateKind.")
        if event.kind is InvestigationCandidateKind.NONE:
            raise ValueError("NONE cannot be a confirmed lifecycle event.")
        _validate_time("candidate_time_seconds", event.candidate_time_seconds)
        if not video_start_time_seconds <= event.candidate_time_seconds <= video_end_time_seconds:
            raise ValueError("candidate event time must lie inside the video bounds.")
        if event.candidate_time_seconds < previous_time:
            raise ValueError("confirmed events must be chronological by candidate time.")
        previous_time = event.candidate_time_seconds

    state = (InvestigationLifecycleState.ACTIVE_INVESTIGATION if initial_present
             else InvestigationLifecycleState.NO_INVESTIGATION)
    start_time = video_start_time_seconds if initial_present else None
    start_reason = InvestigationStartReason.VIDEO_STARTED_ACTIVE if initial_present else None
    intervals, anomalies = [], []
    for index, event in enumerate(confirmed_events):
        time = event.candidate_time_seconds
        if event.kind is InvestigationCandidateKind.CLOSE:
            if state is InvestigationLifecycleState.NO_INVESTIGATION:
                anomalies.append(InvestigationLifecycleAnomaly(
                    index, event.kind, time, state,
                    "confirmed CLOSE received while no Investigation was active"))
            else:
                intervals.append(InvestigationInterval(
                    len(intervals), start_time, time, start_reason, InvestigationEndReason.CLOSE))
                state = InvestigationLifecycleState.NO_INVESTIGATION
                start_time, start_reason = None, None
            continue
        if state is InvestigationLifecycleState.ACTIVE_INVESTIGATION:
            end_reason = (InvestigationEndReason.SWITCH
                          if event.kind is InvestigationCandidateKind.SWITCH
                          else InvestigationEndReason.OPEN_REPLACEMENT)
            intervals.append(InvestigationInterval(len(intervals), start_time, time,
                                                   start_reason, end_reason))
        start_time = time
        start_reason = (InvestigationStartReason.SWITCH
                        if event.kind is InvestigationCandidateKind.SWITCH
                        else InvestigationStartReason.OPEN)
        state = InvestigationLifecycleState.ACTIVE_INVESTIGATION
    if state is InvestigationLifecycleState.ACTIVE_INVESTIGATION:
        intervals.append(InvestigationInterval(len(intervals), start_time, video_end_time_seconds,
                                               start_reason, InvestigationEndReason.VIDEO_END))
    return intervals, anomalies
