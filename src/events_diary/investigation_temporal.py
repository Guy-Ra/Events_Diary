"""Phase 3.3 offline temporal confirmation of pair-level candidate hypotheses.

Consume semantic consecutive-pair entries only: no video, CV, signal aggregation,
or candidate detection occurs here. OPEN/SWITCH expect presence True; CLOSE expects
False. NONE entries are neutral, and identical candidates need not repeat. ANY
later non-NONE candidate through the inclusive look-ahead endpoint supersedes an
earlier one, even the same kind; no equivalent-candidate merging is attempted.

The provisional 0.7-second horizon comes from one accidental OPEN replaced by a
SWITCH after 0.5 seconds. It is time-based, not a frame-count rule. Confirmation
requires timestamp coverage through the horizon and consistent sampled presence.
candidate_time_seconds is the original after-frame event-time estimate;
confirmed_at_seconds is candidate time plus the horizon, never a replacement
event timestamp. A short ending yields INSUFFICIENT_FUTURE_CONTEXT, not a shortened
window. Superseding evidence can be decisive even before full coverage exists.

Unassociated presence reversal with full coverage has no specified status in this
version and raises ValueError explicitly for review. No generic REJECTED status
or exception suppression is invented. An inconsistent candidate's own after-state
is also invalid input. Sub-picosecond timestamp arithmetic differences use a
1e-12-second absolute tolerance at boundaries, not frame rounding or smoothing.

This is offline look-ahead: outputs are confirmed hypotheses, not Investigation
lifecycle state, intervals, identity, or startup-active initialization. Sampling
cannot establish unsampled behavior. Chronological consecutive-pair coverage and
shared-frame presence consistency are validated; inputs are never mutated.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real

from events_diary.investigation_detection import (
    InvestigationCandidate,
    InvestigationCandidateKind,
)

CONFIRMATION_WINDOW_SECONDS = 0.7
_TIME_EPSILON = 1e-12


@dataclass(frozen=True)
class CandidateTimelineEntry:
    """One consecutive pair's semantic presence/candidate record, without raw CV.

    pair_index is a nonnegative identifier. Times are finite seconds in source
    order; the shared after/before frame between entries must agree in time and
    presence. The confirmation API validates records before interpretation.
    """

    pair_index: int
    time_before_seconds: float
    time_after_seconds: float
    present_before: bool
    present_after: bool
    candidate: InvestigationCandidate


class ConfirmationStatus(Enum):
    """Confirmation, actual supersession, or missing full look-ahead coverage."""

    CONFIRMED = "confirmed"
    SUPERSEDED = "superseded"
    INSUFFICIENT_FUTURE_CONTEXT = "insufficient_future_context"


@dataclass(frozen=True)
class CandidateConfirmationResult:
    """Outcome for one non-NONE candidate, retaining its original event time.

    confirmed_at_seconds exists only for CONFIRMED. superseded_by_pair_index exists
    only for SUPERSEDED and identifies the first later candidate in the window.
    Neither field carries identity, confidence, or lifecycle state.
    """

    candidate_pair_index: int
    candidate_kind: InvestigationCandidateKind
    candidate_time_seconds: float
    status: ConfirmationStatus
    confirmed_at_seconds: float | None
    superseded_by_pair_index: int | None


@dataclass(frozen=True)
class ConfirmedInvestigationEvent:
    """Confirmed OPEN/SWITCH/CLOSE hypothesis with event and confirmation times.

    NONE cannot be an event. These fields do not define active Investigation
    intervals or lifecycle state; the original candidate time remains the estimate.
    """

    kind: InvestigationCandidateKind
    candidate_pair_index: int
    candidate_time_seconds: float
    confirmed_at_seconds: float

    def __post_init__(self):
        """Reject NONE or unsupported kinds when constructing a confirmed event."""
        if not isinstance(self.kind, InvestigationCandidateKind):
            raise TypeError("kind must be InvestigationCandidateKind.")
        if self.kind is InvestigationCandidateKind.NONE:
            raise ValueError("NONE cannot become a confirmed event.")


def _at_or_before(value: float, limit: float) -> bool:
    """Compare inclusive time boundaries with floating-point arithmetic tolerance."""
    return value <= limit or math.isclose(value, limit, rel_tol=0, abs_tol=_TIME_EPSILON)


def _validate_entries(entries: Sequence[CandidateTimelineEntry]) -> None:
    """Validate chronological consecutive-pair coverage and shared-frame semantics.

    TypeError rejects wrong records/types; ValueError rejects negative/duplicate
    identifiers, non-finite/reversed times, coverage gaps/overlap, shared-frame
    disagreement, or candidates with an inconsistent own after-presence.
    """
    previous = None
    for entry in entries:
        if not isinstance(entry, CandidateTimelineEntry):
            raise TypeError("entries must contain CandidateTimelineEntry records.")
        if isinstance(entry.pair_index, bool) or not isinstance(entry.pair_index, Integral):
            raise TypeError("pair_index must be an integer, not a boolean.")
        if entry.pair_index < 0:
            raise ValueError("pair_index must be non-negative.")
        for value in (entry.time_before_seconds, entry.time_after_seconds):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("timestamps must be real numbers, not booleans.")
            if not math.isfinite(value):
                raise ValueError("timestamps must be finite.")
        if entry.time_before_seconds > entry.time_after_seconds:
            raise ValueError("time_before_seconds must not exceed time_after_seconds.")
        if not isinstance(entry.present_before, bool) or not isinstance(entry.present_after, bool):
            raise TypeError("presence values must be booleans.")
        if (not isinstance(entry.candidate, InvestigationCandidate)
                or not isinstance(entry.candidate.kind, InvestigationCandidateKind)):
            raise TypeError("candidate must contain an InvestigationCandidateKind.")
        if entry.candidate.kind is not InvestigationCandidateKind.NONE:
            expected = entry.candidate.kind is not InvestigationCandidateKind.CLOSE
            if entry.present_after != expected:
                raise ValueError("candidate own after-presence contradicts its expected state.")
        if previous is not None:
            if entry.pair_index <= previous.pair_index:
                raise ValueError("pair_index must strictly increase.")
            if (entry.time_before_seconds < previous.time_before_seconds
                    or entry.time_after_seconds < previous.time_after_seconds):
                raise ValueError("entries must be chronological.")
            if not math.isclose(entry.time_before_seconds, previous.time_after_seconds,
                                rel_tol=0, abs_tol=_TIME_EPSILON):
                raise ValueError("entries must cover consecutive pairs without gaps or overlap.")
            if entry.present_before != previous.present_after:
                raise ValueError("shared-frame presence must agree between adjacent entries.")
        previous = entry


def confirm_investigation_candidates(
    entries: Sequence[CandidateTimelineEntry], *,
    confirmation_window_seconds: float = CONFIRMATION_WINDOW_SECONDS,
) -> tuple[list[ConfirmedInvestigationEvent], list[CandidateConfirmationResult]]:
    """Confirm candidates through an inclusive seconds-based offline horizon.

    Return chronological events and one result per non-NONE candidate. No input
    is mutated. Any later non-NONE within the horizon supersedes the candidate;
    the first such pair is identified. Otherwise full coverage and expected
    sampled after-presence are required. confirmed_at is exactly T + window.

    Threshold types raise TypeError (bool/non-real); non-finite/nonpositive windows
    raise ValueError. Invalid timeline semantics also raise TypeError/ValueError.
    If full coverage exists but presence reverses without an associated candidate,
    raise ValueError describing the unsupported case rather than inventing status.
    An early ending without supersession returns INSUFFICIENT_FUTURE_CONTEXT.
    No video/CV, temporal candidate generation, lifecycle, or merging occurs.
    """
    if (isinstance(confirmation_window_seconds, bool)
            or not isinstance(confirmation_window_seconds, Real)):
        raise TypeError("confirmation_window_seconds must be a real number, not a boolean.")
    if not math.isfinite(confirmation_window_seconds) or confirmation_window_seconds <= 0:
        raise ValueError("confirmation_window_seconds must be finite and >0.")
    if not isinstance(entries, Sequence):
        raise TypeError("entries must be a Sequence of CandidateTimelineEntry.")
    _validate_entries(entries)
    events, results = [], []
    for index, entry in enumerate(entries):
        kind = entry.candidate.kind
        if kind is InvestigationCandidateKind.NONE:
            continue
        time = entry.time_after_seconds
        horizon = time + confirmation_window_seconds
        if not math.isfinite(horizon):
            raise ValueError("candidate time plus confirmation window must remain finite.")
        future = []
        for later in entries[index + 1:]:
            if not _at_or_before(later.time_after_seconds, horizon):
                break
            future.append(later)
        superseding = next((later for later in future
                            if later.candidate.kind is not InvestigationCandidateKind.NONE), None)
        confirmed_at, superseded_by = None, None
        if superseding is not None:
            status = ConfirmationStatus.SUPERSEDED
            superseded_by = superseding.pair_index
        elif not _at_or_before(horizon, entries[-1].time_after_seconds):
            status = ConfirmationStatus.INSUFFICIENT_FUTURE_CONTEXT
        else:
            expected = kind is not InvestigationCandidateKind.CLOSE
            if any(later.present_after != expected for later in future):
                raise ValueError(f"Unassociated presence reversal after candidate pair {entry.pair_index}: "
                                 "full future coverage exists but no conflicting candidate supplies "
                                 "a specified status; review required.")
            status = ConfirmationStatus.CONFIRMED
            confirmed_at = horizon
            events.append(ConfirmedInvestigationEvent(kind, entry.pair_index, time, horizon))
        results.append(CandidateConfirmationResult(
            entry.pair_index, kind, time, status, confirmed_at, superseded_by))
    return events, results
