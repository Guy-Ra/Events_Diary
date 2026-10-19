"""Adaptive request planning tests using synthetic semantic records, no videos."""

from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest

from events_diary.adaptive_sampling import (
    DENSE_REASON_ORDER,
    AdaptiveSamplingPlan,
    DenseSamplingInterval,
    SamplingSignalEntry,
    build_adaptive_sampling_plan,
)
from events_diary.investigation_signals import (
    ChangeSignal,
    InvestigationSignals,
    SignalStrength,
)
from events_diary.investigation_window import InvestigationWindow
from events_diary.map_motion import MapMotionEstimate
from events_diary.route_selection import RouteSelectionChangeEvidence
from events_diary.ticket_scroll import TicketScrollEvidence


def _signals(**changes):
    """Build existing immutable signals, with HIGH residual deliberately ignored."""
    quiet = ChangeSignal(SignalStrength.LOW, 0, 0)
    return replace(InvestigationSignals(quiet, quiet, quiet,
        ChangeSignal(SignalStrength.HIGH, 100, 1), RouteSelectionChangeEvidence(False, "unavailable"),
        False, MapMotionEstimate(valid=True, dx=0, dy=0, scale=1),
        TicketScrollEvidence(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1), False), **changes)


def _entry(before=2, after=2.1, index=0, **changes):
    """Associate synthetic pair times with existing InvestigationSignals."""
    return SamplingSignalEntry(index, before, after, _signals(**changes))


def _plan(entries=(), window=None, **changes):
    """Build a default plan with deliberately off-grid core anchors."""
    params = {"window": window or InvestigationWindow(0, 1.33, 4.27, 0, 5, 0.5, 0.5),
              "signal_entries": entries, "source_fps": 10}
    params.update(changes)
    return build_adaptive_sampling_plan(**params)


def test_exact_models_reason_labels_frozen_and_tuple_contents():
    assert DENSE_REASON_ORDER == ("ticket_scroll", "route_visual_change", "ticket_header_change",
        "aircraft_image_change", "ticket_body_change", "map_translation", "map_scale_change")
    entry = _entry(ticket_scroll_detected=True)
    plan = _plan([entry])
    assert isinstance(plan, AdaptiveSamplingPlan)
    assert [f.name for f in fields(entry)] == ["pair_index", "time_before_seconds", "time_after_seconds", "signals"]
    assert [f.name for f in fields(plan.dense_intervals[0])] == ["start_time_seconds", "end_time_seconds", "reasons"]
    assert [f.name for f in fields(plan)] == ["window", "source_fps", "sparse_sampling_fps", "dense_sampling_fps",
        "pre_padding_seconds", "post_padding_seconds", "dense_intervals", "requested_timestamps_seconds"]
    assert isinstance(plan.dense_intervals, tuple)
    assert isinstance(plan.requested_timestamps_seconds, tuple)
    for record, field in ((entry, "pair_index"), (plan, "source_fps"), (plan.dense_intervals[0], "reasons")):
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, None)


def test_empty_signal_plan_sparse_and_exact_semantic_anchors():
    plan = _plan()
    assert plan.dense_intervals == ()
    assert plan.requested_timestamps_seconds == (0, 1, 1.33, 2, 3, 4, 4.27, 5)
    assert plan.sparse_sampling_fps == 1
    assert plan.dense_sampling_fps == plan.source_fps == 10


@pytest.mark.parametrize("changes,reason", [
    ({"ticket_scroll_detected": True}, "ticket_scroll"),
    ({"route_selection_change": True}, "route_visual_change"),
    *[({field: ChangeSignal(strength, 20, 0.8)}, reason)
      for field, reason in (("header_change", "ticket_header_change"),
                            ("aircraft_image_change", "aircraft_image_change"),
                            ("ticket_body_change", "ticket_body_change"))
      for strength in (SignalStrength.MEDIUM, SignalStrength.HIGH)],
])
def test_existing_positive_signals_trigger_with_pair_padding(changes, reason):
    plan = _plan([_entry(**changes)])
    assert plan.dense_intervals == (DenseSamplingInterval(1.5, 2.6, (reason,)),)


@pytest.mark.parametrize("field", ["header_change", "aircraft_image_change", "ticket_body_change"])
@pytest.mark.parametrize("strength", [SignalStrength.LOW, SignalStrength.NOT_AVAILABLE])
def test_weak_unavailable_ticket_and_high_map_residual_do_not_trigger(field, strength):
    assert _plan([_entry(**{field: ChangeSignal(strength, None, None)})]).dense_intervals == ()


@pytest.mark.parametrize("motion,expected", [
    (MapMotionEstimate(valid=True, dx=1.01, dy=0, scale=1), ("map_translation",)),
    (MapMotionEstimate(valid=True, dx=0.8, dy=0.8, scale=1), ("map_translation",)),
    (MapMotionEstimate(valid=True, dx=1, dy=0, scale=1), ()),
    (MapMotionEstimate(valid=False, dx=100, dy=100, scale=2), ()),
    (MapMotionEstimate(valid=True, dx=0, dy=0, scale=1.02), ("map_scale_change",)),
    (MapMotionEstimate(valid=True, dx=0, dy=0, scale=0.98), ("map_scale_change",)),
    (MapMotionEstimate(valid=True, dx=0, dy=0, scale=1.01), ()),
    (MapMotionEstimate(valid=True, dx=0, dy=0, scale=0.99), ()),
    (MapMotionEstimate(valid=True, dx=None, dy=0, scale=None), ()),
    (MapMotionEstimate(valid=True, dx=float("nan"), dy=0, scale=float("inf")), ()),
    (MapMotionEstimate(valid=True, dx=10, dy=0, scale=1.1), ("map_translation", "map_scale_change")),
])
def test_map_rules_strict_boundaries_validity_and_unavailable_values(motion, expected):
    plan = _plan([_entry(map_motion=motion)])
    assert (plan.dense_intervals[0].reasons if plan.dense_intervals else ()) == expected


def test_all_reasons_preserved_ordered_and_unique_after_merge():
    meaningful = ChangeSignal(SignalStrength.MEDIUM, 20, 0.8)
    entry = _entry(ticket_scroll_detected=True, route_selection_change=True,
                   header_change=meaningful, aircraft_image_change=meaningful,
                   ticket_body_change=meaningful, map_motion=MapMotionEstimate(valid=True, dx=2, dy=0, scale=1.1))
    plan = _plan([entry, replace(entry, pair_index=1, time_before_seconds=2.1, time_after_seconds=2.2)])
    assert len(plan.dense_intervals) == 1
    assert plan.dense_intervals[0].reasons == DENSE_REASON_ORDER


def test_both_pair_bounds_asymmetric_padding_and_window_clipping():
    window = InvestigationWindow(0, 1, 4, 0.5, 4.5, 0.5, 0.5)
    plan = _plan([_entry(0.5, 0.7, ticket_scroll_detected=True),
                  _entry(4.1, 4.5, 1, route_selection_change=True)], window,
                 pre_padding_seconds=0.2, post_padding_seconds=0.3)
    assert len(plan.dense_intervals) == 2
    assert plan.dense_intervals[0] == DenseSamplingInterval(0.5, 1, ("ticket_scroll",))
    assert plan.dense_intervals[1].start_time_seconds == pytest.approx(3.9)
    assert plan.dense_intervals[1].end_time_seconds == 4.5
    assert plan.dense_intervals[1].reasons == ("route_visual_change",)


@pytest.mark.parametrize("second_before,second_after,end", [(1.4, 1.5, 2), (2.5, 2.6, 3.1)])
def test_overlap_and_exact_touch_merge_and_union_reasons(second_before, second_after, end):
    plan = _plan([_entry(1, 1.5, ticket_scroll_detected=True),
                  _entry(second_before, second_after, 1, route_selection_change=True)])
    assert plan.dense_intervals == (DenseSamplingInterval(0.5, end, ("ticket_scroll", "route_visual_change")),)


def test_unbridged_quiet_gap_remains_sparse_with_zero_padding():
    plan = _plan([_entry(1, 1.1, ticket_scroll_detected=True),
                  _entry(1.7, 1.8, 1, route_selection_change=True)],
                 pre_padding_seconds=0, post_padding_seconds=0)
    assert len(plan.dense_intervals) == 2
    assert plan.dense_intervals[1].start_time_seconds - plan.dense_intervals[0].end_time_seconds == pytest.approx(0.6)


def test_sparse_grid_point_inside_dense_region_is_not_added_redundantly():
    plan = _plan([_entry(1.15, 1.45, ticket_scroll_detected=True)])
    assert 1.0 not in plan.requested_timestamps_seconds  # Dense phase is .65, .75, ...
    assert 1.33 in plan.requested_timestamps_seconds  # Mandatory core anchor remains.


@pytest.mark.parametrize("fps,step,count", [(10, 0.1, 3), (25, 0.04, 6)])
def test_dense_grid_uses_actual_source_rate(fps, step, count):
    window = InvestigationWindow(0, 1, 2, 0, 3, 0.5, 0.5)
    plan = _plan([_entry(1, 1.2, ticket_scroll_detected=True)], window, source_fps=fps,
                 pre_padding_seconds=0, post_padding_seconds=0)
    dense_times = [time for time in plan.requested_timestamps_seconds if 1 <= time <= 1.2]
    assert dense_times == pytest.approx([1 + index * step for index in range(count)])
    assert plan.dense_sampling_fps == fps


def test_exact_close_anchors_preserved_and_duplicate_anchors_removed():
    window = InvestigationWindow(0, 0.0000000005, 2, 0, 2, 0.5, 0.5)
    plan = _plan(window=window)
    assert window.investigation_start_time_seconds in plan.requested_timestamps_seconds
    assert 0 in plan.requested_timestamps_seconds
    assert plan.requested_timestamps_seconds.count(2) == 1


def test_sorted_unique_inside_bounds_anchors_and_input_immutability():
    entries = [_entry(ticket_scroll_detected=True)]
    window = InvestigationWindow(0, 1.33, 2.37, 0.15, 3.73, 0.5, 0.5)
    snapshot = (replace(window), [replace(entry) for entry in entries])
    plan = _plan(entries, window, source_fps=25)
    times = plan.requested_timestamps_seconds
    assert times == tuple(sorted(set(times)))
    assert all(window.window_start_time_seconds <= time <= window.window_end_time_seconds for time in times)
    assert {0.15, 1.33, 2.37, 3.73}.issubset(times)
    assert (window, entries) == snapshot
    assert plan.window is window


def test_zero_duration_window_with_empty_or_triggering_signals():
    window = InvestigationWindow(0, 2, 2, 2, 2, 0.5, 0.5)
    assert _plan(window=window).requested_timestamps_seconds == (2,)
    plan = _plan([_entry(2, 2, ticket_scroll_detected=True)], window)
    assert plan.requested_timestamps_seconds == (2,)


@pytest.mark.parametrize("name", ["source_fps", "sparse_sampling_fps", "pre_padding_seconds",
    "post_padding_seconds", "map_translation_threshold_pixels", "map_scale_threshold"])
@pytest.mark.parametrize("value,error", [(True, TypeError), (np.bool_(False), TypeError),
    ("1", TypeError), (1j, TypeError), (float("nan"), ValueError), (float("inf"), ValueError), (-1, ValueError)])
def test_invalid_numeric_parameters(name, value, error):
    with pytest.raises(error):
        _plan(**{name: value})


@pytest.mark.parametrize("overrides", [{"source_fps": 0}, {"sparse_sampling_fps": 0},
                                       {"source_fps": 0.5}, {"sparse_sampling_fps": 11}])
def test_zero_rates_and_sparse_exceeding_source_rejected(overrides):
    with pytest.raises(ValueError):
        _plan(**overrides)


@pytest.mark.parametrize("changes,error", [
    ({"pair_index": -1}, ValueError), ({"pair_index": True}, TypeError), ({"pair_index": 0.5}, TypeError),
    ({"time_before_seconds": -1}, ValueError), ({"time_after_seconds": 6}, ValueError),
    ({"time_before_seconds": 3}, ValueError), ({"time_after_seconds": float("nan")}, ValueError),
    ({"time_before_seconds": True}, TypeError), ({"time_after_seconds": "3"}, TypeError),
    ({"signals": None}, TypeError),
])
def test_invalid_semantic_entry(changes, error):
    with pytest.raises(error):
        _plan([replace(_entry(), **changes)])


def test_nonchronological_input_rejected_without_sorting():
    entries = [_entry(3, 3.1, 0), _entry(2, 2.1, 1)]
    with pytest.raises(ValueError, match="chronological"):
        _plan(entries)
    assert entries[0].time_before_seconds == 3
    with pytest.raises(ValueError, match="chronological"):
        _plan([_entry(1, 1.1, 1), _entry(2, 2.1, 1)])


def test_invalid_window_and_sequence():
    with pytest.raises(TypeError, match="window"):
        _plan(window="bad")
    with pytest.raises(ValueError, match="core"):
        _plan(window=InvestigationWindow(0, 0, 4, 1, 5, 0.5, 0.5))
    with pytest.raises(TypeError, match="Sequence"):
        _plan(iter([]))
    with pytest.raises(TypeError, match="SamplingSignalEntry"):
        _plan([None])


@pytest.mark.parametrize("reasons", [(), ("unknown",), ("ticket_scroll", "ticket_scroll"),
                                     ("map_translation", "ticket_scroll")])
def test_dense_reason_invariants(reasons):
    with pytest.raises(ValueError):
        DenseSamplingInterval(0, 1, reasons)
