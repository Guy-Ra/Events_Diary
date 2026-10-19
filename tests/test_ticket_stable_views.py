"""Stable span, time/provenance and input-validation tests without video/CV."""

from dataclasses import FrozenInstanceError, fields, replace

import pytest

from events_diary.investigation_signals import ChangeSignal, SignalStrength
from events_diary.ticket_scroll import TicketScrollState
from events_diary.ticket_stable_views import (
    TicketStableView,
    TicketStableViewEntry,
    detect_ticket_stable_views,
)

UP = TicketScrollState.SCROLLING_UP
DOWN = TicketScrollState.SCROLLING_DOWN
LOW = SignalStrength.LOW


def _entry(before, after, index=0, *, present=True, scroll=False, strength=LOW, direction=None):
    """Build a pair using the existing immutable body signal and direction enum."""
    return TicketStableViewEntry(index, before, after, present, scroll,
                                 ChangeSignal(strength, 0, 0), direction)


def _detect(entries=(), **overrides):
    """Invoke the public API with default core bounds; entries are not altered."""
    params = {"investigation_index": 3, "entries": entries,
              "investigation_start_time_seconds": 0, "investigation_end_time_seconds": 10}
    params.update(overrides)
    return detect_ticket_stable_views(**params)


def test_exact_frozen_models_and_no_new_enum():
    entry = _entry(0, 1)
    view = _detect([entry])[0]
    assert [f.name for f in fields(view)] == ["investigation_index", "stable_start_time_seconds",
        "stable_confirmed_time_seconds", "stable_end_time_seconds", "follows_scroll",
        "prior_scroll_end_time_seconds", "prior_scroll_direction"]
    assert isinstance(view, TicketStableView)
    for record, name in ((view, "follows_scroll"), (entry, "effective_presence")):
        with pytest.raises(FrozenInstanceError):
            setattr(record, name, False)


def test_empty():
    assert _detect() == ()


@pytest.mark.parametrize("duration,qualifies", [(0.499, False), (0.5, True), (0.7, True)])
def test_dwell_inclusive_and_ordinary_stable_provenance(duration, qualifies):
    views = _detect([_entry(0, duration)])
    assert bool(views) is qualifies
    if views:
        assert views[0] == TicketStableView(3, 0, 0.5, None, False, None, None)


@pytest.mark.parametrize("breaker", [
    {"present": False}, {"scroll": True, "direction": UP},
    {"strength": SignalStrength.MEDIUM}, {"strength": SignalStrength.HIGH},
    {"strength": SignalStrength.NOT_AVAILABLE},
])
def test_breakers_end_a_confirmed_span_at_before_time(breaker):
    entries = [_entry(0, 0.6), _entry(0.6, 0.7, 1, **breaker), _entry(0.7, 1.3, 2)]
    views = _detect(entries)
    assert len(views) == 2
    assert views[0].stable_end_time_seconds == 0.6
    assert views[1].stable_start_time_seconds == 0.7
    assert views[1].stable_confirmed_time_seconds == 1.2


def test_body_low_label_is_authoritative_no_second_metric_threshold():
    entry = replace(_entry(0, 1), ticket_body_change=ChangeSignal(LOW, None, float("nan")))
    assert len(_detect([entry])) == 1


def test_header_and_image_are_not_veto_inputs():
    # Only the existing body signal is projected into the minimal semantic input;
    # HIGH header/image diagnostics remain outside this API and cannot veto it.
    signals = {"header": ChangeSignal(SignalStrength.HIGH, 200, 1),
               "image": ChangeSignal(SignalStrength.HIGH, 200, 1),
               "body": ChangeSignal(LOW, 0, 0)}
    assert len(_detect([replace(_entry(0, 1), ticket_body_change=signals["body"])])) == 1


def test_first_stable_time_and_exact_confirmation_between_samples():
    view = _detect([_entry(2, 2.2), _entry(2.2, 2.9, 1)])[0]
    assert view.stable_start_time_seconds == 2
    assert view.stable_confirmed_time_seconds == 2.5
    assert view.stable_end_time_seconds is None


@pytest.mark.parametrize("fps", [7, 24, 30, 60])
def test_arbitrary_fps_without_fixed_count(fps):
    entries = [_entry(i/fps, (i+1)/fps, i) for i in range(fps)]
    views = _detect(entries, investigation_end_time_seconds=1)
    assert len(views) == 1
    assert views[0].stable_confirmed_time_seconds == 0.5
    assert views[0].stable_end_time_seconds == 1


def test_short_run_is_not_accumulated_across_body_activity():
    assert _detect([_entry(0, 0.3), _entry(0.3, 0.4, 1, strength=SignalStrength.MEDIUM),
                    _entry(0.4, 0.7, 2)]) == ()


def test_scroll_then_settling_then_stable_preserves_end_and_direction():
    entries = [_entry(0, 0.3, scroll=True, direction=UP),
               _entry(0.3, 0.4, 1, strength=SignalStrength.MEDIUM), _entry(0.4, 1, 2)]
    view = _detect(entries)[0]
    assert view == TicketStableView(3, 0.4, 0.9, None, True, 0.3, UP)


@pytest.mark.parametrize("direction", [None, TicketScrollState.QUIET])
def test_unknown_direction_is_none(direction):
    view = _detect([_entry(0, 0.1, scroll=True, direction=direction), _entry(0.1, 1, 1)])[0]
    assert view.follows_scroll
    assert view.prior_scroll_direction is None


def test_reversal_without_qualifying_pause_is_one_ambiguous_interaction():
    entries = [_entry(0, 0.2, scroll=True, direction=UP), _entry(0.2, 0.4, 1),
               _entry(0.4, 0.6, 2, scroll=True, direction=DOWN), _entry(0.6, 1.4, 3)]
    views = _detect(entries)
    assert len(views) == 1
    assert views[0].prior_scroll_end_time_seconds == 0.6
    assert views[0].prior_scroll_direction is None


def test_adjacent_reversal_and_missing_direction_remain_ambiguous():
    entries = [_entry(0, 0.1, scroll=True, direction=UP),
               _entry(0.1, 0.2, 1, scroll=True, direction=None), _entry(0.2, 1, 2)]
    assert _detect(entries)[0].prior_scroll_direction is None


def test_intermediate_pauses_final_stop_order_and_evidence_immutability():
    entries = [_entry(0, 0.5), _entry(0.5, 0.7, 1, scroll=True, direction=UP),
               _entry(0.7, 1.3, 2), _entry(1.3, 1.5, 3, scroll=True, direction=DOWN),
               _entry(1.5, 2.1, 4), _entry(2.1, 2.4, 5, scroll=True, direction=UP), _entry(2.4, 3, 6)]
    snapshot = tuple(replace(e) for e in entries)
    views = _detect(entries, investigation_end_time_seconds=3)
    assert len(views) == 4
    assert [v.stable_start_time_seconds for v in views] == [0, 0.7, 1.5, 2.4]
    assert [v.follows_scroll for v in views] == [False, True, True, True]
    assert [v.prior_scroll_direction for v in views] == [None, UP, DOWN, UP]
    assert [v.stable_end_time_seconds for v in views] == [0.5, 1.3, 2.1, 3]
    assert tuple(entries) == snapshot  # Includes every unchanged moving entry.


def test_confirmed_pause_consumes_provenance_for_later_body_only_stability():
    entries = [_entry(0, 0.1, scroll=True, direction=UP), _entry(0.1, 0.8, 1),
               _entry(0.8, 0.9, 2, strength=SignalStrength.MEDIUM), _entry(0.9, 1.5, 3)]
    views = _detect(entries)
    assert views[0].follows_scroll
    assert views[1].follows_scroll is False
    assert views[1].prior_scroll_end_time_seconds is None


def test_presence_loss_clears_scroll_provenance():
    entries = [_entry(0, 0.1, scroll=True, direction=UP), _entry(0.1, 0.2, 1, present=False),
               _entry(0.2, 1, 2)]
    assert _detect(entries)[0].follows_scroll is False


def test_final_span_known_investigation_end_or_censored_data():
    assert _detect([_entry(0, 1)], investigation_end_time_seconds=1)[0].stable_end_time_seconds == 1
    assert _detect([_entry(0, 1)], investigation_end_time_seconds=2)[0].stable_end_time_seconds is None


def test_one_continuous_span_never_emits_duplicates():
    assert len(_detect([_entry(i/10, (i+1)/10, i) for i in range(30)])) == 1


@pytest.mark.parametrize("name,value,error", [
    ("investigation_index", True, TypeError), ("investigation_index", 1.5, TypeError),
    ("investigation_index", -1, ValueError), ("investigation_start_time_seconds", True, TypeError),
    ("investigation_end_time_seconds", float("inf"), ValueError),
    ("investigation_start_time_seconds", float("nan"), ValueError),
    ("investigation_start_time_seconds", 11, ValueError),
    ("stable_dwell_seconds", True, TypeError), ("stable_dwell_seconds", "0.5", TypeError),
    ("stable_dwell_seconds", -1, ValueError), ("stable_dwell_seconds", 0, ValueError),
    ("stable_dwell_seconds", float("nan"), ValueError), ("stable_dwell_seconds", float("inf"), ValueError),
])
def test_public_parameters_validated_even_for_empty_input(name, value, error):
    with pytest.raises(error):
        _detect(**{name: value})


@pytest.mark.parametrize("field,value,error", [
    ("pair_index", True, TypeError), ("pair_index", -1, ValueError), ("pair_index", 0.5, TypeError),
    ("time_before_seconds", True, TypeError), ("time_before_seconds", -1, ValueError),
    ("time_after_seconds", float("nan"), ValueError), ("time_after_seconds", float("inf"), ValueError),
    ("time_after_seconds", 11, ValueError), ("time_after_seconds", 0, ValueError),
    ("effective_presence", 1, TypeError), ("ticket_scroll_detected", 0, TypeError),
    ("ticket_body_change", None, TypeError), ("scroll_state", "mixed", TypeError),
])
def test_malformed_entry_rejected(field, value, error):
    with pytest.raises(error):
        _detect([replace(_entry(0, 1), **{field: value})])


@pytest.mark.parametrize("entries", [None, "bad", [None]])
def test_wrong_sequence_types(entries):
    with pytest.raises(TypeError):
        _detect(entries)


@pytest.mark.parametrize("second", [_entry(0.1, 0.5, 1), _entry(0.3, 0.5, 1),
                                     _entry(0.2, 0.5, 0)])
def test_overlap_gap_and_duplicate_index_rejected(second):
    with pytest.raises(ValueError):
        _detect([_entry(0, 0.2), second])
