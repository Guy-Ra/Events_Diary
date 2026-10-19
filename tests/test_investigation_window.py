"""Fixed-context window tests with synthetic existing lifecycle intervals only."""

from dataclasses import FrozenInstanceError, fields, replace
from unittest.mock import Mock

import numpy as np
import pytest

from events_diary import investigation_window as window_module
from events_diary.investigation_lifecycle import (
    InvestigationEndReason,
    InvestigationInterval,
    InvestigationStartReason,
)
from events_diary.investigation_window import (
    InvestigationWindow,
    build_investigation_window,
    build_investigation_windows,
)


def _interval(start=1.6, end=4.9, index=7):
    """Return an existing immutable lifecycle interval without video dependencies."""
    return InvestigationInterval(index, start, end, InvestigationStartReason.OPEN,
                                 InvestigationEndReason.SWITCH)


def _build(interval=None, **overrides):
    """Build a window with test-video bounds, allowing parameter overrides."""
    params = {"video_start_time_seconds": 0, "video_end_time_seconds": 8.1}
    params.update(overrides)
    return build_investigation_window(_interval() if interval is None else interval, **params)


def test_normal_exact_fields_requested_context_and_frozen_record():
    interval = _interval()
    before = replace(interval)
    result = _build(interval)
    assert [field.name for field in fields(result)] == ["investigation_index",
        "investigation_start_time_seconds", "investigation_end_time_seconds",
        "window_start_time_seconds", "window_end_time_seconds",
        "context_before_seconds", "context_after_seconds"]
    assert result == InvestigationWindow(7, 1.6, 4.9, 1.6 - 0.5, 4.9 + 0.5, 0.5, 0.5)
    assert result.window_start_time_seconds == pytest.approx(1.1)
    assert result.window_end_time_seconds == pytest.approx(5.4)
    assert interval == before
    with pytest.raises(FrozenInstanceError):
        result.context_before_seconds = 0


@pytest.mark.parametrize("start,end,window_start,window_end", [
    (0.2, 3.8, 0, 4.3), (0, 3.8, 0, 4.3),
    (4.9, 7.9, 4.4, 8.1), (4.9, 8.1, 4.4, 8.1),
])
def test_clipping_at_or_near_video_bounds(start, end, window_start, window_end):
    result = _build(_interval(start, end))
    assert result.window_start_time_seconds == pytest.approx(window_start)
    assert result.window_end_time_seconds == pytest.approx(window_end)
    assert result.context_before_seconds == result.context_after_seconds == 0.5


def test_requested_context_does_not_shrink_when_clipped():
    result = _build(_interval(0.2, 7.9))
    assert result.context_before_seconds == result.context_after_seconds == 0.5
    assert result.investigation_start_time_seconds - result.window_start_time_seconds == 0.2
    assert result.window_end_time_seconds - result.investigation_end_time_seconds == pytest.approx(0.2)


def test_startup_active_and_video_end_reasons_do_not_change_construction():
    interval = InvestigationInterval(0, 0, 8.1, InvestigationStartReason.VIDEO_STARTED_ACTIVE,
                                     InvestigationEndReason.VIDEO_END)
    assert _build(interval) == InvestigationWindow(0, 0, 8.1, 0, 8.1, 0.5, 0.5)


@pytest.mark.parametrize("before,after", [(0, 0.5), (0.5, 0), (0, 0)])
def test_zero_context_values(before, after):
    result = _build(context_before_seconds=before, context_after_seconds=after)
    assert result.window_start_time_seconds == 1.6 - before
    assert result.window_end_time_seconds == 4.9 + after


def test_large_context_clips_to_video_and_preserves_requests():
    result = _build(context_before_seconds=100, context_after_seconds=100)
    assert (result.window_start_time_seconds, result.window_end_time_seconds) == (0, 8.1)
    assert result.context_before_seconds == result.context_after_seconds == 100


@pytest.mark.parametrize("context", [0, 0.5])
def test_zero_duration_core_and_video(context):
    result = _build(_interval(2, 2), context_before_seconds=context, context_after_seconds=context)
    assert result.window_start_time_seconds == 2 - context
    assert result.window_end_time_seconds == 2 + context
    result = _build(_interval(2, 2), video_start_time_seconds=2, video_end_time_seconds=2)
    assert result.window_start_time_seconds == result.window_end_time_seconds == 2


def test_nonzero_video_start_clips_to_supplied_range():
    result = _build(_interval(2.1, 3), video_start_time_seconds=2, video_end_time_seconds=4)
    assert (result.window_start_time_seconds, result.window_end_time_seconds) == (2, 3.5)


def test_batch_preserves_order_index_inputs_and_separate_overlaps():
    investigations = [_interval(4.9, 8.1, 9), _interval(1.6, 4.9, 3)]
    snapshot = [replace(interval) for interval in investigations]
    results = build_investigation_windows(investigations, video_start_time_seconds=0,
                                         video_end_time_seconds=8.1)
    assert [result.investigation_index for result in results] == [9, 3]
    assert len(results) == 2
    assert results[0].window_start_time_seconds < results[1].window_end_time_seconds
    assert investigations == snapshot


def test_batch_reuses_single_window_public_api(monkeypatch):
    investigations = (_interval(), _interval(0, 1, 0))
    builder = Mock(wraps=build_investigation_window)
    monkeypatch.setattr(window_module, "build_investigation_window", builder)
    results = build_investigation_windows(investigations, video_start_time_seconds=0,
                                         video_end_time_seconds=8.1,
                                         context_before_seconds=0.1, context_after_seconds=0.2)
    assert len(results) == builder.call_count == 2
    for interval, call in zip(investigations, builder.call_args_list):
        assert call.args == (interval,)
        assert call.kwargs == {"video_start_time_seconds": 0, "video_end_time_seconds": 8.1,
                               "context_before_seconds": 0.1, "context_after_seconds": 0.2}


def test_empty_batch_and_valid_real_numeric_values():
    assert build_investigation_windows([], video_start_time_seconds=0, video_end_time_seconds=0) == []
    result = _build(context_before_seconds=np.float32(0.5), context_after_seconds=0)
    assert result.window_start_time_seconds == pytest.approx(1.1)


@pytest.mark.parametrize("parameter", ["video_start_time_seconds", "video_end_time_seconds",
                                       "context_before_seconds", "context_after_seconds"])
@pytest.mark.parametrize("value,error", [(True, TypeError), (np.bool_(False), TypeError),
    ("0.5", TypeError), (None, TypeError), (1j, TypeError), (float("nan"), ValueError),
    (float("inf"), ValueError), (-float("inf"), ValueError)])
def test_invalid_parameter_types_and_finiteness_single_and_empty_batch(parameter, value, error):
    params = {"video_start_time_seconds": 0, "video_end_time_seconds": 8.1,
              "context_before_seconds": 0.5, "context_after_seconds": 0.5}
    params[parameter] = value
    with pytest.raises(error):
        build_investigation_window(_interval(), **params)
    with pytest.raises(error):
        build_investigation_windows([], **params)


@pytest.mark.parametrize("overrides", [
    {"video_start_time_seconds": -1}, {"video_start_time_seconds": 9},
    {"context_before_seconds": -0.1}, {"context_after_seconds": -0.1},
])
def test_invalid_ranges(overrides):
    with pytest.raises(ValueError):
        _build(**overrides)


@pytest.mark.parametrize("start,end", [(-0.1, 4.9), (1.6, 8.2), (4.9, 1.6)])
def test_core_outside_video_or_reversed_rejected(start, end):
    with pytest.raises(ValueError):
        _build(_interval(start, end))


@pytest.mark.parametrize("field", ["start_time_seconds", "end_time_seconds"])
@pytest.mark.parametrize("value,error", [(True, TypeError), ("1", TypeError), (1j, TypeError),
                                       (float("nan"), ValueError), (float("inf"), ValueError)])
def test_core_timestamp_validation(field, value, error):
    with pytest.raises(error):
        _build(replace(_interval(), **{field: value}))


def test_wrong_interval_and_sequence_types_rejected():
    with pytest.raises(TypeError, match="InvestigationInterval"):
        build_investigation_window(None, video_start_time_seconds=0, video_end_time_seconds=8.1)
    with pytest.raises(TypeError, match="Sequence"):
        build_investigation_windows(iter([]), video_start_time_seconds=0, video_end_time_seconds=8.1)
    with pytest.raises(TypeError, match="InvestigationInterval"):
        build_investigation_windows([None], video_start_time_seconds=0, video_end_time_seconds=8.1)
