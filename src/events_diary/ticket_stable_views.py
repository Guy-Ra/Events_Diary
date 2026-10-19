"""Phase 4.3b stable ticket spans from already-computed semantic observations.

Stable views complement dense during-scroll evidence; nothing is sampled, removed,
ranked or selected here. Effective presence, no detected scroll and body strength
LOW must hold continuously for a provisional 0.5 seconds. Header/image diagnostics
do not veto stability. Every qualifying span is returned, including intermediate
reading pauses and ordinary stable regions. No OCR, identity, content deduplication,
OpenCV, new direction enum, or Phase 4.4/5 selection is implemented.

Each entry describes conditions across [time_before_seconds, time_after_seconds].
Positive-duration pairs must be chronological, contiguous and inside one supplied
Investigation. Nonuniform spacing is supported; gaps cannot establish continuous
stability and are rejected, rather than filled. A span starts at its first stable
pair's before-time and confirms at start+dwell once coverage reaches that point;
no image at that exact timestamp is required. A breaking pair ends the preceding
span at its before-time. A last stable pair reaching the Investigation end closes
the span there; otherwise an ongoing last span is right-censored with end=None.

Scroll provenance belongs to the first qualifying stable pause following detected
scroll within this Investigation. Shorter quiet gaps do not split that interaction;
opposing/unknown directions make its optional direction None. Once a qualifying
span ends, its scroll provenance is consumed; subsequent body-only interruptions
do not make later ordinary stability follow the same scroll forever. Presence
loss clears provenance. There is no arbitrary gap timeout, PAN mode or new enum.
These are annotation semantics, not a reconstruction of human interaction intent.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

from events_diary.investigation_signals import ChangeSignal, SignalStrength
from events_diary.ticket_scroll import TicketScrollState

STABLE_DWELL_SECONDS = 0.5  # Provisional Phase 4.3a calibration.
_TIME_EPSILON = 1e-12


@dataclass(frozen=True)
class TicketStableViewEntry:
    """One semantic pair, without images or newly computed measurements.

    effective_presence is True only when both endpoint frames are effectively
    present. ticket_scroll_detected is the existing pair decision; scroll_state
    is optional direction context and does not override that decision. QUIET on
    a positive pair means direction is unavailable/ambiguous. The detector validates
    these records, including existing ChangeSignal/SignalStrength types.
    """

    pair_index: int
    time_before_seconds: float
    time_after_seconds: float
    effective_presence: bool
    ticket_scroll_detected: bool
    ticket_body_change: ChangeSignal
    scroll_state: TicketScrollState | None = None


@dataclass(frozen=True)
class TicketStableView:
    """Confirmed continuous stable span and optional preceding scroll provenance.

    start is the first stable pair's before-time; confirmed is start+dwell after
    sufficient observed coverage. end is a known breaking/Investigation boundary,
    or None only for an ongoing right-censored last span. follows_scroll associates
    the first qualifying pause with prior detected scroll inside this Investigation;
    its end is the last positive pair's after-time, not the end of body settling.
    Direction is an existing active TicketScrollState, or None for missing/mixed
    direction. No frame indexes, ranking or evidence deletion are introduced.
    """

    investigation_index: int
    stable_start_time_seconds: float
    stable_confirmed_time_seconds: float
    stable_end_time_seconds: float | None
    follows_scroll: bool
    prior_scroll_end_time_seconds: float | None
    prior_scroll_direction: TicketScrollState | None


def _time(name, value, *, positive=False):
    """Validate finite nonboolean real times and positive/nonnegative bounds."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be real, not bool.")
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}.")


def _index(name, value):
    """Validate a nonboolean nonnegative integral identifier."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not bool.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate(entries, start, end):
    """Reject invalid semantic records, overlap, gaps and out-of-core observations."""
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise TypeError("entries must be a sequence of TicketStableViewEntry.")
    previous = None
    for entry in entries:
        if not isinstance(entry, TicketStableViewEntry):
            raise TypeError("entries must contain TicketStableViewEntry.")
        _index("pair_index", entry.pair_index)
        _time("time_before_seconds", entry.time_before_seconds)
        _time("time_after_seconds", entry.time_after_seconds)
        if not start <= entry.time_before_seconds < entry.time_after_seconds <= end:
            raise ValueError("positive-duration pairs must be inside Investigation bounds.")
        if not isinstance(entry.effective_presence, bool) or not isinstance(entry.ticket_scroll_detected, bool):
            raise TypeError("effective_presence and ticket_scroll_detected must be bool.")
        if (not isinstance(entry.ticket_body_change, ChangeSignal)
                or not isinstance(entry.ticket_body_change.strength, SignalStrength)):
            raise TypeError("ticket_body_change must contain an existing SignalStrength.")
        if entry.scroll_state is not None and not isinstance(entry.scroll_state, TicketScrollState):
            raise TypeError("scroll_state must be TicketScrollState or None.")
        if previous is not None:
            if entry.pair_index <= previous.pair_index:
                raise ValueError("pair indexes must strictly increase.")
            if entry.time_before_seconds != previous.time_after_seconds:
                raise ValueError("pairs must be chronological and contiguous, without gaps/overlap.")
        previous = entry


def detect_ticket_stable_views(
    *, investigation_index: int, entries: Sequence[TicketStableViewEntry],
    investigation_start_time_seconds: float, investigation_end_time_seconds: float,
    stable_dwell_seconds: float = STABLE_DWELL_SECONDS,
) -> tuple[TicketStableView, ...]:
    """Annotate all qualifying spans within one Investigation without mutating inputs.

    Stability uses ONLY effective_presence, not ticket_scroll_detected and existing
    body LOW strength. No secondary CPR/MAD gate or header/image veto is added.
    Confirmation is inclusive at dwell with 1e-12s arithmetic tolerance; a rounded
    exact boundary is clamped to the observed endpoint rather than beyond coverage.
    Dwell is seconds, independent of pair count/FPS, and may fall between samples.
    Empty input returns (). Validation occurs even for empty input. TypeError
    rejects wrong types/bool numerics; ValueError rejects invalid bounds, ordering,
    non-finite times and discontinuous coverage. No sorting/clipping is performed.
    """
    _index("investigation_index", investigation_index)
    start, end = investigation_start_time_seconds, investigation_end_time_seconds
    _time("investigation_start_time_seconds", start)
    _time("investigation_end_time_seconds", end)
    _time("stable_dwell_seconds", stable_dwell_seconds, positive=True)
    if start > end:
        raise ValueError("Investigation start must not exceed end.")
    _validate(entries, start, end)
    views = []
    stable_start = None
    scroll_end = None
    directions = set()

    def finish(observed_end, known_end):
        """Emit at most one confirmed span, consuming its prior scroll provenance."""
        nonlocal stable_start, scroll_end
        if stable_start is None:
            return
        threshold = stable_start + stable_dwell_seconds
        if threshold <= observed_end or math.isclose(threshold, observed_end, rel_tol=0, abs_tol=_TIME_EPSILON):
            direction = next(iter(directions)) if len(directions) == 1 else None
            views.append(TicketStableView(investigation_index, stable_start, min(threshold, observed_end),
                known_end, scroll_end is not None, scroll_end, direction if scroll_end is not None else None))
            scroll_end = None
            directions.clear()
        stable_start = None

    for entry in entries:
        stable = (entry.effective_presence and not entry.ticket_scroll_detected
                  and entry.ticket_body_change.strength is SignalStrength.LOW)
        if stable:
            if stable_start is None:
                stable_start = entry.time_before_seconds
            continue
        finish(entry.time_before_seconds, entry.time_before_seconds)
        if not entry.effective_presence:
            scroll_end = None
            directions.clear()
        elif entry.ticket_scroll_detected:
            scroll_end = entry.time_after_seconds
            direction = entry.scroll_state
            directions.add(direction if direction in (TicketScrollState.SCROLLING_UP,
                           TicketScrollState.SCROLLING_DOWN) else None)
    if entries:
        observed_end = entries[-1].time_after_seconds
        finish(observed_end, end if observed_end == end else None)
    return tuple(views)
