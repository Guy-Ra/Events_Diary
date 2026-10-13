"""Offline lifecycle tests using existing confirmed semantic event records."""

from dataclasses import FrozenInstanceError, fields, replace
from itertools import pairwise

import pytest

from events_diary.investigation_detection import (
    InvestigationCandidate,
    InvestigationCandidateKind,
)
from events_diary.investigation_lifecycle import (
    InvestigationEndReason,
    InvestigationInterval,
    InvestigationLifecycleAnomaly,
    InvestigationLifecycleState,
    InvestigationStartReason,
    build_investigation_lifecycle,
)
from events_diary.investigation_temporal import (
    CandidateTimelineEntry,
    ConfirmedInvestigationEvent,
    confirm_investigation_candidates,
)

OPEN, SWITCH, CLOSE = (InvestigationCandidateKind.OPEN, InvestigationCandidateKind.SWITCH,
                       InvestigationCandidateKind.CLOSE)
START, END = InvestigationStartReason, InvestigationEndReason


def _event(kind, time, index=0, confirmed=None):
    """Build a synthetic existing event with deliberately separate confirmation time."""
    return ConfirmedInvestigationEvent(kind, index, time, time + 0.7 if confirmed is None else confirmed)


def _build(events=(), initial=False, start=0, end=10):
    """Run lifecycle with explicit default test-video bounds and initial presence."""
    return build_investigation_lifecycle(initial_present=initial, video_start_time_seconds=start,
                                         video_end_time_seconds=end, confirmed_events=events)


def test_exact_enum_values():
    assert {e.name: e.value for e in InvestigationLifecycleState} == {
        "NO_INVESTIGATION": "no_investigation", "ACTIVE_INVESTIGATION": "active_investigation"}
    assert {e.name: e.value for e in START} == {
        "VIDEO_STARTED_ACTIVE": "video_started_active", "OPEN": "open", "SWITCH": "switch"}
    assert {e.name: e.value for e in END} == {
        "CLOSE": "close", "SWITCH": "switch", "OPEN_REPLACEMENT": "open_replacement",
        "VIDEO_END": "video_end"}


def test_frozen_records_and_exact_fields():
    interval = InvestigationInterval(0, 1, 2, START.OPEN, END.CLOSE)
    anomaly = InvestigationLifecycleAnomaly(0, CLOSE, 1,
        InvestigationLifecycleState.NO_INVESTIGATION, "reason")
    assert [f.name for f in fields(interval)] == ["investigation_index", "start_time_seconds",
        "end_time_seconds", "start_reason", "end_reason"]
    assert [f.name for f in fields(anomaly)] == ["event_index", "event_kind", "event_time_seconds",
                                               "state_before", "reason"]
    with pytest.raises(FrozenInstanceError):
        interval.end_time_seconds = 3
    with pytest.raises(FrozenInstanceError):
        anomaly.reason = "changed"


def test_initial_presence_without_events_and_nonzero_start():
    assert _build() == ([], [])
    intervals, anomalies = _build(initial=True, start=2, end=5)
    assert intervals == [InvestigationInterval(0, 2, 5, START.VIDEO_STARTED_ACTIVE, END.VIDEO_END)]
    assert anomalies == []


@pytest.mark.parametrize("kind,start_reason", [(OPEN, START.OPEN), (SWITCH, START.SWITCH)])
def test_inactive_open_or_switch_starts_interval(kind, start_reason):
    intervals, anomalies = _build([_event(kind, 3)])
    assert intervals == [InvestigationInterval(0, 3, 10, start_reason, END.VIDEO_END)]
    assert not anomalies


def test_inactive_close_records_anomaly_and_continues():
    events = [_event(CLOSE, 1, 42), _event(CLOSE, 2, 43), _event(SWITCH, 3, 44)]
    intervals, anomalies = _build(events)
    assert intervals == [InvestigationInterval(0, 3, 10, START.SWITCH, END.VIDEO_END)]
    assert anomalies == [InvestigationLifecycleAnomaly(index, CLOSE, index + 1,
        InvestigationLifecycleState.NO_INVESTIGATION,
        "confirmed CLOSE received while no Investigation was active") for index in (0, 1)]


def test_active_close_leaves_no_final_active_interval():
    intervals, anomalies = _build([_event(CLOSE, 3.8)], initial=True)
    assert intervals == [InvestigationInterval(0, 0, 3.8, START.VIDEO_STARTED_ACTIVE, END.CLOSE)]
    assert not anomalies


@pytest.mark.parametrize("kind,end_reason,start_reason", [
    (SWITCH, END.SWITCH, START.SWITCH), (OPEN, END.OPEN_REPLACEMENT, START.OPEN)])
def test_active_replacement_has_exact_shared_boundary(kind, end_reason, start_reason):
    intervals, anomalies = _build([_event(kind, 4.9)], initial=True)
    assert intervals == [InvestigationInterval(0, 0, 4.9, START.VIDEO_STARTED_ACTIVE, end_reason),
                         InvestigationInterval(1, 4.9, 10, start_reason, END.VIDEO_END)]
    assert intervals[0].end_time_seconds == intervals[1].start_time_seconds == 4.9
    assert not anomalies


def test_open_then_switch_uses_candidate_time_not_confirmation_and_preserves_inputs():
    events = [_event(OPEN, 1.6, 15, 2.3), _event(SWITCH, 4.9, 48, 5.6)]
    snapshot = [replace(event) for event in events]
    intervals, anomalies = _build(events, end=8.1)
    assert intervals == [InvestigationInterval(0, 1.6, 4.9, START.OPEN, END.SWITCH),
                         InvestigationInterval(1, 4.9, 8.1, START.SWITCH, END.VIDEO_END)]
    assert not anomalies
    assert events == snapshot


def test_startup_close_switch_leaves_gap_and_no_anomaly():
    intervals, anomalies = _build([_event(CLOSE, 3.8), _event(SWITCH, 7.6, 1)], initial=True, end=12)
    assert intervals == [InvestigationInterval(0, 0, 3.8, START.VIDEO_STARTED_ACTIVE, END.CLOSE),
                         InvestigationInterval(1, 7.6, 12, START.SWITCH, END.VIDEO_END)]
    assert not anomalies


def test_superseded_accidental_open_never_reaches_lifecycle():
    times = [3.8, 4.5, 7.1, 7.6, 8.3, 11.9]
    kinds = [CLOSE, InvestigationCandidateKind.NONE, OPEN, SWITCH,
             InvestigationCandidateKind.NONE, InvestigationCandidateKind.NONE]
    before, presence, entries = 0, True, []
    for index, (time, kind) in enumerate(zip(times, kinds)):
        after = presence if kind is InvestigationCandidateKind.NONE else kind is not CLOSE
        entries.append(CandidateTimelineEntry(index, before, time, presence, after,
                                             InvestigationCandidate(kind, "synthetic")))
        before, presence = time, after
    confirmed, results = confirm_investigation_candidates(entries)
    assert len(results) == 3 and len(confirmed) == 2
    intervals, anomalies = _build(confirmed, initial=True, end=12)
    assert [i.start_time_seconds for i in intervals] == [0, 7.6]
    assert len(intervals) == 2 and not anomalies


@pytest.mark.parametrize("events,anomaly_count", [
    ([OPEN, CLOSE], 0), ([CLOSE, OPEN], 1), ([OPEN, SWITCH], 0)])
def test_equal_event_times_preserve_input_order_and_allow_zero_duration(events, anomaly_count):
    intervals, anomalies = _build([_event(kind, 2, index) for index, kind in enumerate(events)])
    assert len(anomalies) == anomaly_count
    if events == [OPEN, CLOSE]:
        assert intervals == [InvestigationInterval(0, 2, 2, START.OPEN, END.CLOSE)]
    elif events == [CLOSE, OPEN]:
        assert intervals == [InvestigationInterval(0, 2, 10, START.OPEN, END.VIDEO_END)]
    else:
        assert intervals == [InvestigationInterval(0, 2, 2, START.OPEN, END.SWITCH),
                             InvestigationInterval(1, 2, 10, START.SWITCH, END.VIDEO_END)]


def test_event_bounds_inclusive_and_zero_length_video():
    intervals, _ = _build([_event(OPEN, 2), _event(CLOSE, 2, 1)], start=2, end=2)
    assert intervals == [InvestigationInterval(0, 2, 2, START.OPEN, END.CLOSE)]
    intervals, _ = _build(initial=True, start=2, end=2)
    assert intervals == [InvestigationInterval(0, 2, 2, START.VIDEO_STARTED_ACTIVE, END.VIDEO_END)]


def test_chronological_interval_indexes_and_invariants():
    intervals, _ = _build([_event(OPEN, 1), _event(SWITCH, 2, 1), _event(OPEN, 3, 2),
                            _event(CLOSE, 4, 3), _event(OPEN, 6, 4)])
    assert [i.investigation_index for i in intervals] == list(range(4))
    assert all(i.start_time_seconds <= i.end_time_seconds for i in intervals)
    assert all(a.end_time_seconds <= b.start_time_seconds for a, b in pairwise(intervals))


def test_confirmation_metadata_does_not_set_boundaries():
    intervals, _ = _build([_event(OPEN, 1, confirmed=100)])
    assert intervals == [InvestigationInterval(0, 1, 10, START.OPEN, END.VIDEO_END)]


def test_malformed_none_and_wrong_event_kind_rejected():
    event = _event(OPEN, 1)
    # Simulate corrupted/deserialized frozen input bypassing its own constructor guard.
    object.__setattr__(event, "kind", InvestigationCandidateKind.NONE)
    with pytest.raises(ValueError, match="NONE"):
        _build([event])
    object.__setattr__(event, "kind", "open")
    with pytest.raises(TypeError, match="kind"):
        _build([event])


def test_nonchronological_input_rejected_without_sorting():
    events = [_event(OPEN, 3), _event(CLOSE, 2, 1)]
    snapshot = list(events)
    with pytest.raises(ValueError, match="chronological"):
        _build(events)
    assert events == snapshot


@pytest.mark.parametrize("time,error", [(-1, ValueError), (11, ValueError),
    (float("nan"), ValueError), (float("inf"), ValueError), (True, TypeError), ("1", TypeError)])
def test_invalid_event_time(time, error):
    with pytest.raises(error):
        _build([_event(OPEN, time, confirmed=2)])


@pytest.mark.parametrize("argument", ["video_start_time_seconds", "video_end_time_seconds"])
@pytest.mark.parametrize("value,error", [(True, TypeError), (None, TypeError), ("1", TypeError),
    (1j, TypeError), (float("nan"), ValueError), (float("inf"), ValueError),
    (-float("inf"), ValueError)])
def test_invalid_video_timestamp_types_and_finiteness(argument, value, error):
    kwargs = {"initial_present": False, "video_start_time_seconds": 0,
              "video_end_time_seconds": 10, "confirmed_events": []}
    kwargs[argument] = value
    with pytest.raises(error):
        build_investigation_lifecycle(**kwargs)


def test_negative_start_and_end_before_start():
    with pytest.raises(ValueError, match="non-negative"):
        _build(start=-1)
    with pytest.raises(ValueError, match=">="):
        _build(start=5, end=4)


@pytest.mark.parametrize("initial", [0, 1, None, "yes"])
def test_initial_presence_requires_boolean(initial):
    with pytest.raises(TypeError, match="boolean"):
        _build(initial=initial)


def test_wrong_event_sequence_and_records_rejected():
    for events in (None, iter([]), [None], [InvestigationCandidate(OPEN, "unconfirmed")]):
        with pytest.raises(TypeError):
            _build(events)
