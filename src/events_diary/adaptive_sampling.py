"""Phase 4.2b time-based adaptive request planning from existing semantic signals.

No CV or decoding occurs here. Existing scroll/route decisions, meaningful ticket
changes, and valid map translation/scale directly drive sampling; no ActivityLevel
abstraction is introduced. Route evidence is labeled route_visual_change because
it also cross-triggers during pan/zoom. Generic map residual never triggers.

Provisional Phase 4.2a defaults: 0.5s pre/post padding, sparse 1 FPS, dense source
FPS, translation strictly >1px, scale delta strictly >0.01. Expand BOTH pair bounds,
clip to the InvestigationWindow, and merge overlapping or exactly touching dense
regions with deterministic reason unions. Padding supplies temporal hysteresis;
real unbridged gaps remain sparse. No additional gap tolerance forces merging.

Sparse grids are anchored at window start and excluded from dense regions. Dense
grids are anchored at each merged interval start, with interval endpoints retained.
Generate start + integer_index / FPS, avoiding repeated-addition drift. Generated
requests within 1e-9s of anchors snap to them; other near-duplicate generated values
deduplicate at that tolerance. Distinct semantic anchors are ALWAYS retained exactly,
even if closer than the tolerance. Window/core start/end are mandatory anchors.

These are timestamp requests, not source-frame indexes or selected evidence frames.
Grid phase resets and off-grid anchors can add requests beyond nominal source-density
counts; the video layer resolves actual frames later. No scroll-stop semantics,
per-request evidence metadata, OCR, VLM, identity, or Phase 4.3/4.4/5 logic exists.
Inputs are validated without sorting malformed observations or rerunning signals.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

from events_diary.investigation_signals import InvestigationSignals, SignalStrength
from events_diary.investigation_window import InvestigationWindow

DENSE_REASON_ORDER = ("ticket_scroll", "route_visual_change", "ticket_header_change",
                      "aircraft_image_change", "ticket_body_change", "map_translation", "map_scale_change")
_TIME_EPSILON = 1e-9


def _number(name, value, *, positive=False):
    """Validate finite nonboolean real parameters with nonnegative/positive bounds."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not a boolean.")
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}.")


@dataclass(frozen=True)
class SamplingSignalEntry:
    """Already-computed pair signals with index and before/after seconds, no pixels."""

    pair_index: int
    time_before_seconds: float
    time_after_seconds: float
    signals: InvestigationSignals


@dataclass(frozen=True)
class DenseSamplingInterval:
    """Clipped dense region with nonempty unique reasons in global deterministic order."""

    start_time_seconds: float
    end_time_seconds: float
    reasons: tuple[str, ...]

    def __post_init__(self):
        """Reject reversed/non-finite bounds and malformed reason tuples."""
        _number("dense start", self.start_time_seconds)
        _number("dense end", self.end_time_seconds)
        if self.start_time_seconds > self.end_time_seconds:
            raise ValueError("dense start must not exceed end.")
        if not isinstance(self.reasons, tuple):
            raise TypeError("reasons must be a tuple.")
        ordered = tuple(reason for reason in DENSE_REASON_ORDER if reason in self.reasons)
        if not self.reasons or self.reasons != ordered:
            raise ValueError("reasons must be nonempty, unique known labels in global order.")


@dataclass(frozen=True)
class AdaptiveSamplingPlan:
    """Immutable requested timestamp plan; no per-request evidence or frame metadata."""

    window: InvestigationWindow
    source_fps: float
    sparse_sampling_fps: float
    dense_sampling_fps: float
    pre_padding_seconds: float
    post_padding_seconds: float
    dense_intervals: tuple[DenseSamplingInterval, ...]
    requested_timestamps_seconds: tuple[float, ...]


def _trigger_reasons(signals, translation_threshold, scale_threshold):
    """Derive sampling reasons without interpreting residuals as operator activity."""
    reasons = []
    if signals.ticket_scroll_detected:
        reasons.append("ticket_scroll")
    if signals.route_selection_change:
        reasons.append("route_visual_change")
    for name, signal in (("ticket_header_change", signals.header_change),
                         ("aircraft_image_change", signals.aircraft_image_change),
                         ("ticket_body_change", signals.ticket_body_change)):
        if signal.strength in (SignalStrength.MEDIUM, SignalStrength.HIGH):
            reasons.append(name)
    motion = signals.map_motion
    if motion.valid:
        if (motion.dx is not None and motion.dy is not None
                and math.isfinite(motion.dx) and math.isfinite(motion.dy)
                and math.hypot(motion.dx, motion.dy) > translation_threshold):
            reasons.append("map_translation")
        # Comparing scale bounds avoids 1.01-1 rounding above the exact strict boundary.
        if (motion.scale is not None and math.isfinite(motion.scale)
                and (motion.scale > 1 + scale_threshold or motion.scale < 1 - scale_threshold)):
            reasons.append("map_scale_change")
    return tuple(reasons)


def _validate_inputs(window, entries):
    """Validate nested window bounds and chronological in-window semantic observations."""
    if not isinstance(window, InvestigationWindow):
        raise TypeError("window must be InvestigationWindow.")
    for field in ("investigation_start_time_seconds", "investigation_end_time_seconds",
                  "window_start_time_seconds", "window_end_time_seconds",
                  "context_before_seconds", "context_after_seconds"):
        _number(field, getattr(window, field))
    if not (window.window_start_time_seconds <= window.investigation_start_time_seconds
            <= window.investigation_end_time_seconds <= window.window_end_time_seconds):
        raise ValueError("window must contain an ordered Investigation core.")
    if not isinstance(entries, Sequence):
        raise TypeError("signal_entries must be a Sequence.")
    previous = None
    for entry in entries:
        if not isinstance(entry, SamplingSignalEntry) or not isinstance(entry.signals, InvestigationSignals):
            raise TypeError("entries must contain SamplingSignalEntry with InvestigationSignals.")
        if isinstance(entry.pair_index, bool) or not isinstance(entry.pair_index, Integral):
            raise TypeError("pair_index must be an integer, not a boolean.")
        if entry.pair_index < 0:
            raise ValueError("pair_index must be non-negative.")
        _number("time_before_seconds", entry.time_before_seconds)
        _number("time_after_seconds", entry.time_after_seconds)
        if not (window.window_start_time_seconds <= entry.time_before_seconds
                <= entry.time_after_seconds <= window.window_end_time_seconds):
            raise ValueError("signal pair must be ordered and entirely inside the window.")
        if previous is not None and (
            entry.pair_index <= previous.pair_index
            or entry.time_before_seconds < previous.time_before_seconds
            or entry.time_after_seconds < previous.time_after_seconds
        ):
            raise ValueError("signal entries must be chronological, with increasing pair indexes.")
        previous = entry


def _merge(intervals):
    """Merge overlapping/exactly touching dense regions and preserve ordered reason unions."""
    merged = []
    for interval in sorted(intervals, key=lambda item: (item.start_time_seconds, item.end_time_seconds)):
        if merged and interval.start_time_seconds <= merged[-1].end_time_seconds:
            previous = merged[-1]
            reasons = tuple(reason for reason in DENSE_REASON_ORDER
                            if reason in previous.reasons or reason in interval.reasons)
            merged[-1] = DenseSamplingInterval(previous.start_time_seconds,
                max(previous.end_time_seconds, interval.end_time_seconds), reasons)
        else:
            merged.append(interval)
    return tuple(merged)


def _grid(start, end, fps):
    """Generate an interval-start-anchored seconds grid without accumulation or overshoot."""
    span = (end - start) * fps
    if not math.isfinite(span):
        raise ValueError("sampling span times FPS must be finite.")
    for index in range(math.floor(span) + 1):
        value = start + index / fps
        if value <= end:
            yield value


def _timestamps(window, dense, sparse_fps, source_fps):
    """Combine sparse/dense grids and exact anchors with tolerant generated-value deduplication."""
    anchors = {window.window_start_time_seconds, window.investigation_start_time_seconds,
               window.investigation_end_time_seconds, window.window_end_time_seconds}
    generated = [time for time in _grid(window.window_start_time_seconds, window.window_end_time_seconds, sparse_fps)
                 if not any(interval.start_time_seconds <= time <= interval.end_time_seconds for interval in dense)]
    for interval in dense:
        generated.extend(_grid(interval.start_time_seconds, interval.end_time_seconds, source_fps))
        generated.append(interval.end_time_seconds)
    values = set(anchors)
    for time in generated:
        near = [anchor for anchor in anchors if abs(time - anchor) <= _TIME_EPSILON]
        values.add(min(near, key=lambda anchor: (abs(time - anchor), anchor)) if near else time)
    result = []
    for time in sorted(values):
        if time in anchors or not result or time - result[-1] > _TIME_EPSILON:
            result.append(time)
    return tuple(result)


def build_adaptive_sampling_plan(
    *, window: InvestigationWindow, signal_entries: Sequence[SamplingSignalEntry], source_fps: float,
    sparse_sampling_fps: float = 1.0, pre_padding_seconds: float = 0.5,
    post_padding_seconds: float = 0.5, map_translation_threshold_pixels: float = 1.0,
    map_scale_threshold: float = 0.01,
) -> AdaptiveSamplingPlan:
    """Build dense regions and requested times from validated already-computed signals.

    Expand each triggering pair's before/after bounds with provisional padding,
    clip to the window, merge overlap/touching, and retain ordered reasons. Dense
    FPS equals source_fps; sparse FPS must not exceed it. Semantic window/core
    anchors are exact and mandatory, including zero-duration/empty-signal cases.

    Parameters must be finite nonboolean reals: FPS >0; padding/thresholds >=0.
    TypeError rejects wrong types; ValueError rejects invalid values, out-of-window
    pairs, reversed bounds, or nonchronological observations. Inputs are unchanged.
    No decoding, source indexes, scroll stops, evidence selection, or CV occurs.
    """
    _number("source_fps", source_fps, positive=True)
    _number("sparse_sampling_fps", sparse_sampling_fps, positive=True)
    if sparse_sampling_fps > source_fps:
        raise ValueError("sparse_sampling_fps must not exceed source_fps.")
    for name, value in (("pre_padding_seconds", pre_padding_seconds),
                        ("post_padding_seconds", post_padding_seconds),
                        ("map_translation_threshold_pixels", map_translation_threshold_pixels),
                        ("map_scale_threshold", map_scale_threshold)):
        _number(name, value)
    _validate_inputs(window, signal_entries)
    intervals = []
    for entry in signal_entries:
        reasons = _trigger_reasons(entry.signals, map_translation_threshold_pixels, map_scale_threshold)
        if reasons:
            intervals.append(DenseSamplingInterval(
                max(window.window_start_time_seconds, entry.time_before_seconds - pre_padding_seconds),
                min(window.window_end_time_seconds, entry.time_after_seconds + post_padding_seconds), reasons))
    dense = _merge(intervals)
    return AdaptiveSamplingPlan(window, source_fps, sparse_sampling_fps, source_fps,
        pre_padding_seconds, post_padding_seconds, dense,
        _timestamps(window, dense, sparse_sampling_fps, source_fps))
