"""Synthetic measurement and decision tests, independent of manual recordings."""

from dataclasses import FrozenInstanceError, replace

import cv2
import numpy as np
import pytest

from events_diary.config import ROI, UIConfig
from events_diary.ticket_scroll import (
    TicketScrollEvidence,
    TicketScrollState,
    analyze_ticket_scroll_sequence,
    estimate_ticket_scroll_evidence,
    is_vertical_scroll_evidence,
)


@pytest.fixture
def scene():
    config = UIConfig(reference_width=160, reference_height=180,
                      ticket_header=ROI(0, 0, 64, 20),
                      aircraft_image=ROI(0, 20, 64, 40),
                      ticket_scroll_body=ROI(0, 60, 64, 96),
                      map_analysis_region=ROI(64, 0, 96, 180))
    image = np.random.default_rng(24).integers(0, 256, (180, 160), dtype=np.uint8)
    return image, config


def _shift_body(image, config, dx, dy):
    """Return a frame with only its configured body translated by dx/dy pixels."""
    changed = image.copy()
    body = config.ticket_scroll_body
    body.crop(changed)[:] = cv2.warpAffine(body.crop(image),
                                         np.float32([[1, 0, dx], [0, 1, dy]]),
                                         (body.width, body.height))
    return changed


def _evidence(**values):
    """Build valid synthetic pair evidence with optional measurement overrides."""
    return replace(TicketScrollEvidence(0, 0, 0, 0, 30, 0.4, 0, 0, 0, 10, 0.9), **values)


def test_frozen_record():
    with pytest.raises(FrozenInstanceError):
        _evidence().dy = 1


@pytest.mark.parametrize("bgr", [False, True])
def test_identity_and_immutability(scene, bgr):
    image, config = scene
    if bgr:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    before = image.copy()
    result = estimate_ticket_scroll_evidence(image, image, config)
    for name in ("header", "aircraft_image", "body", "map"):
        assert getattr(result, f"{name}_mad") == getattr(result, f"{name}_cpr") == 0
    assert abs(result.dx) < 0.01
    assert abs(result.dy) < 0.01
    assert not is_vertical_scroll_evidence(result)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("dx,dy,expected", [(0, 9, True), (0, -9, True), (12, 2, False)])
def test_real_phase_translation(scene, dx, dy, expected):
    image, config = scene
    changed = _shift_body(image, config, dx, dy)
    before_a, before_b = image.copy(), changed.copy()
    result = estimate_ticket_scroll_evidence(image, changed, config)
    assert result.dx == pytest.approx(dx, abs=0.5)
    assert result.dy == pytest.approx(dy, abs=0.5)
    assert result.header_cpr == result.aircraft_image_cpr == result.map_cpr == 0
    assert is_vertical_scroll_evidence(result) is expected
    np.testing.assert_array_equal(image, before_a)
    np.testing.assert_array_equal(changed, before_b)


@pytest.mark.parametrize("override", [
    {"dy": 1.9}, {"dx": 5, "dy": 10}, {"body_cpr": 0.099},
    {"header_cpr": 0.011}, {"aircraft_image_cpr": 0.011}, {"map_cpr": 0.011},
    {"dy": float("nan")}, {"dx": float("inf")}, {"map_cpr": float("nan")},
])
def test_each_decision_rejection(override):
    assert not is_vertical_scroll_evidence(_evidence(**override))


def test_inclusive_boundaries_and_response_is_not_a_gate():
    evidence = _evidence(body_cpr=0.1, dx=2, dy=-6, header_cpr=0.01,
                         aircraft_image_cpr=0.01, map_cpr=0.01, motion_response=0)
    assert is_vertical_scroll_evidence(evidence)
    assert is_vertical_scroll_evidence(_evidence(dy=2))
    assert not is_vertical_scroll_evidence(replace(evidence, dy=-5.999))
    assert is_vertical_scroll_evidence(replace(evidence, motion_response=float("nan")))


@pytest.mark.parametrize("name", ["min_body_cpr", "min_vertical_shift_px",
                                  "min_vertical_dominance_ratio", "max_surrounding_cpr"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, np.bool_(False), "x"])
def test_invalid_thresholds(name, value):
    error = TypeError if isinstance(value, (bool, np.bool_, str)) else ValueError
    with pytest.raises(error):
        is_vertical_scroll_evidence(_evidence(), **{name: value})


@pytest.mark.parametrize("arguments", [{"min_body_cpr": 1.1}, {"max_surrounding_cpr": 1.1},
                                       {"min_vertical_dominance_ratio": 0}])
def test_invalid_threshold_ranges(arguments):
    with pytest.raises(ValueError):
        is_vertical_scroll_evidence(_evidence(), **arguments)


def test_allowed_zero_thresholds_and_custom_defaults():
    assert is_vertical_scroll_evidence(_evidence(dy=0, body_cpr=0),
                                       min_body_cpr=0, min_vertical_shift_px=0,
                                       max_surrounding_cpr=0)
    assert not is_vertical_scroll_evidence(_evidence(), min_body_cpr=0.5)


@pytest.mark.parametrize("bad,error", [
    (None, TypeError), (np.zeros((180, 160), np.float32), TypeError),
    (np.zeros((0, 160), np.uint8), ValueError),
    (np.zeros((180, 160, 4), np.uint8), ValueError),
    (np.zeros((181, 160), np.uint8), ValueError),
    (np.zeros((180,), np.uint8), ValueError),
])
def test_invalid_frame_inputs(scene, bad, error):
    image, config = scene
    with pytest.raises(error):
        estimate_ticket_scroll_evidence(image, bad, config)


def test_matching_noncanonical_frames_rejected(scene):
    _, config = scene
    bad = np.zeros((181, 160), np.uint8)
    with pytest.raises(ValueError, match="canonical"):
        estimate_ticket_scroll_evidence(bad, bad, config)


def test_out_of_bounds_roi_reuses_crop_validation(scene):
    image, config = scene
    config = replace(config, ticket_scroll_body=ROI(150, 100, 64, 96))
    with pytest.raises(ValueError, match="boundaries"):
        estimate_ticket_scroll_evidence(image, image, config)


def test_small_body_and_bad_config(scene):
    image, config = scene
    with pytest.raises(ValueError, match="2x2"):
        estimate_ticket_scroll_evidence(image, image,
                                       replace(config, ticket_scroll_body=ROI(0, 60, 1, 96)))
    with pytest.raises(TypeError, match="UIConfig"):
        estimate_ticket_scroll_evidence(image, image, None)
    with pytest.raises(TypeError, match="TicketScrollEvidence"):
        is_vertical_scroll_evidence(None)


@pytest.mark.parametrize("motions,expected", [
    ([], []), ([0], ["quiet"]), ([0, 0, 0], ["quiet"] * 3),
    ([-10], ["quiet"]), ([0, -10, 0], ["quiet"] * 3),
    ([-10, -12], ["scrolling_up"] * 2),
    ([10, 12], ["scrolling_down"] * 2),
    ([0, -18, -24, 0, -11, 0, 0],
     ["quiet", "scrolling_up", "scrolling_up", "scrolling_up", "scrolling_up", "quiet", "quiet"]),
    ([-10, -10, 0, 0, -10], ["scrolling_up", "scrolling_up", "quiet", "quiet", "quiet"]),
    ([-10, 10], ["quiet", "quiet"]),
    ([-10, -10, -10, 10, 10, 10], ["scrolling_up"] * 3 + ["scrolling_down"] * 3),
    ([-10, -10, 0, 0], ["scrolling_up"] * 2 + ["quiet"] * 2),
    ([0, 0, 10, 10], ["quiet"] * 2 + ["scrolling_down"] * 2),
])
def test_temporal_patterns(motions, expected):
    sequence = [_evidence(dy=dy, body_cpr=0 if dy == 0 else 0.4) for dy in motions]
    before = sequence.copy()
    states = analyze_ticket_scroll_sequence(sequence)
    assert [state.value for state in states] == expected
    assert len(states) == len(sequence)
    assert sequence == before


def test_temporal_gap_limit_and_negative_pairs_do_not_vote():
    quiet_with_opposite_motion = _evidence(dy=100, body_cpr=0)
    sequence = [_evidence(dy=-10), _evidence(dy=-10), quiet_with_opposite_motion,
                _evidence(dy=-10), _evidence(dy=-10)]
    assert analyze_ticket_scroll_sequence(sequence) == [TicketScrollState.SCROLLING_UP] * 5
    assert analyze_ticket_scroll_sequence(sequence, max_gap_pairs=0)[2] == TicketScrollState.QUIET
    assert analyze_ticket_scroll_sequence([_evidence()], window_size=1,
                                          min_positive_pairs=1) == [TicketScrollState.SCROLLING_DOWN]


def test_temporal_long_gap_cannot_borrow_support():
    sequence = [_evidence(), _evidence(body_cpr=0), _evidence(body_cpr=0), _evidence()]
    assert analyze_ticket_scroll_sequence(sequence) == [TicketScrollState.QUIET] * 4
    assert analyze_ticket_scroll_sequence(sequence, max_gap_pairs=2) == [
        TicketScrollState.QUIET, TicketScrollState.SCROLLING_DOWN,
        TicketScrollState.SCROLLING_DOWN, TicketScrollState.QUIET,
    ]


@pytest.mark.parametrize("name", ["window_size", "min_positive_pairs", "max_gap_pairs"])
@pytest.mark.parametrize("value", [True, 1.5, "2", None])
def test_temporal_parameter_types(name, value):
    with pytest.raises(TypeError, match="integer"):
        analyze_ticket_scroll_sequence([], **{name: value})


@pytest.mark.parametrize("arguments", [{"window_size": 0}, {"window_size": -1},
                                       {"min_positive_pairs": 0}, {"min_positive_pairs": -1},
                                       {"min_positive_pairs": 6}, {"max_gap_pairs": -1}])
def test_temporal_parameter_ranges(arguments):
    with pytest.raises(ValueError):
        analyze_ticket_scroll_sequence([], **arguments)
