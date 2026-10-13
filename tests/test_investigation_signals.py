"""Generic change interpretation and aggregation tests without CV recomputation."""

from dataclasses import FrozenInstanceError, fields, replace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from events_diary import investigation_signals as signals_module
from events_diary.config import ROI, UIConfig
from events_diary.investigation_signals import (
    ChangeSignal,
    InvestigationSignals,
    SignalStrength,
    interpret_change_signal,
)
from events_diary.map_motion import MapMotionEstimate
from events_diary.route_selection import RouteSelectionChangeEvidence
from events_diary.ticket_scroll import TicketScrollEvidence


def test_exact_strength_enum():
    assert {member.name: member.value for member in SignalStrength} == {
        "NOT_AVAILABLE": "not_available", "LOW": "low", "MEDIUM": "medium", "HIGH": "high",
    }


@pytest.mark.parametrize("mad,cpr", [(None, 0), (0, None), (None, None),
                                   (float("nan"), 0.3), (0, float("nan")),
                                   (float("inf"), 0.3), (0, float("inf")),
                                   (-float("inf"), 0), (0, -float("inf"))])
def test_missing_and_nonfinite_measurements_preserved(mad, cpr):
    result = interpret_change_signal(mad, cpr, medium_cpr=0.1, high_cpr=0.5)
    assert result.strength is SignalStrength.NOT_AVAILABLE
    assert result.mad is mad
    assert result.cpr is cpr


@pytest.mark.parametrize("cpr,strength", [(0, SignalStrength.LOW), (0.099, SignalStrength.LOW),
                                       (0.1, SignalStrength.MEDIUM), (0.3, SignalStrength.MEDIUM),
                                       (0.499, SignalStrength.MEDIUM), (0.5, SignalStrength.HIGH),
                                       (1, SignalStrength.HIGH)])
@pytest.mark.parametrize("mad", [0, 23.75, 255])
def test_cpr_boundaries_and_mad_does_not_vote(cpr, strength, mad):
    result = interpret_change_signal(mad, cpr, medium_cpr=0.1, high_cpr=0.5)
    assert result.strength is strength
    assert result.mad is mad
    assert result.cpr is cpr


def test_equal_thresholds_and_extreme_valid_boundaries():
    assert interpret_change_signal(0, 0.5, medium_cpr=0.5, high_cpr=0.5).strength is SignalStrength.HIGH
    assert interpret_change_signal(0, 0, medium_cpr=0, high_cpr=0).strength is SignalStrength.HIGH
    assert interpret_change_signal(0, 0, medium_cpr=1, high_cpr=1).strength is SignalStrength.LOW


@pytest.mark.parametrize("name", ["medium_cpr", "high_cpr"])
@pytest.mark.parametrize("value,error", [(-0.1, ValueError), (1.1, ValueError),
                                        (float("nan"), ValueError), (float("inf"), ValueError),
                                        (-float("inf"), ValueError), (True, TypeError),
                                        (False, TypeError), (np.bool_(True), TypeError),
                                        ("0.1", TypeError), (None, TypeError),
                                        (0.1 + 0j, TypeError)])
def test_invalid_thresholds_even_with_missing_measurements(name, value, error):
    thresholds = {"medium_cpr": 0.1, "high_cpr": 0.5}
    thresholds[name] = value
    with pytest.raises(error):
        interpret_change_signal(None, None, **thresholds)


def test_reversed_thresholds():
    with pytest.raises(ValueError, match="must not exceed"):
        interpret_change_signal(0, 0, medium_cpr=0.8, high_cpr=0.2)


def test_real_numeric_thresholds():
    assert interpret_change_signal(0, 0.5, medium_cpr=np.float32(0.25),
                                   high_cpr=1).strength is SignalStrength.MEDIUM


def test_frozen_change_signal_and_exact_fields():
    signal = ChangeSignal(SignalStrength.LOW, 0, 0)
    assert [field.name for field in fields(signal)] == ["strength", "mad", "cpr"]
    with pytest.raises(FrozenInstanceError):
        signal.cpr = 1


def test_aggregate_existing_records_immutable_and_no_input_changes():
    change = interpret_change_signal(0, 0, medium_cpr=0.1, high_cpr=0.5)
    unavailable = interpret_change_signal(None, None, medium_cpr=0.1, high_cpr=0.5)
    route = RouteSelectionChangeEvidence(False, "No map motion")
    motion = MapMotionEstimate(valid=False, reason="No descriptors")
    scroll = TicketScrollEvidence(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    before = (change, unavailable, route, motion, scroll)
    signals = InvestigationSignals(change, change, change, unavailable, route, False,
                                   motion, scroll, False)
    assert [field.name for field in fields(signals)] == [
        "header_change", "aircraft_image_change", "ticket_body_change", "map_residual_change",
        "route_selection_evidence", "route_selection_change", "map_motion",
        "ticket_scroll_evidence", "ticket_scroll_detected",
    ]
    assert signals.route_selection_evidence is route
    assert signals.map_motion is motion
    assert signals.ticket_scroll_evidence is scroll
    assert signals.header_change is change
    assert signals.map_residual_change is unavailable
    assert before == (change, unavailable, route, motion, scroll)
    with pytest.raises(FrozenInstanceError):
        signals.route_selection_change = True


@pytest.fixture
def pair_scene(monkeypatch):
    """Isolate specialized evidence while using real ticket ROI change metrics."""
    config = UIConfig(reference_width=40, reference_height=40,
                      ticket_header=ROI(0, 0, 10, 10), aircraft_image=ROI(10, 0, 10, 10),
                      ticket_scroll_body=ROI(0, 10, 10, 10),
                      map_analysis_region=ROI(20, 0, 20, 40), map_ignore_masks=())
    frame = np.zeros((40, 40, 3), np.uint8)
    records = {
        "estimate_route_selection_change": RouteSelectionChangeEvidence(True, None, 0, 0, 0, 0, 0),
        "estimate_map_motion": MapMotionEstimate(valid=True, aligned_mad=0, aligned_cpr=0),
        "estimate_ticket_scroll_evidence": TicketScrollEvidence(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    }
    mocks = {}
    for name, record in records.items():
        mocks[name] = Mock(return_value=record)
        monkeypatch.setattr(signals_module, name, mocks[name])
    return frame, config, mocks


def test_analyze_identical_frames_preserves_records_and_inputs(pair_scene):
    frame, config, mocks = pair_scene
    other = frame.copy()
    before = frame.copy()
    result = signals_module.analyze_investigation_pair(frame, other, config)
    assert isinstance(result, InvestigationSignals)
    for signal in (result.header_change, result.aircraft_image_change, result.ticket_body_change):
        assert signal == ChangeSignal(SignalStrength.LOW, 0, 0)
    assert result.route_selection_change is False
    assert result.ticket_scroll_detected is False
    assert result.route_selection_evidence is mocks["estimate_route_selection_change"].return_value
    assert result.ticket_scroll_evidence is mocks["estimate_ticket_scroll_evidence"].return_value
    assert result.map_motion is mocks["estimate_map_motion"].return_value
    np.testing.assert_array_equal(frame, before)
    np.testing.assert_array_equal(other, before)
    args, kwargs = mocks["estimate_map_motion"].call_args
    assert args[0].shape == args[1].shape == (40, 20, 3)
    assert kwargs == {"ui_config": config}


@pytest.mark.parametrize("roi,field,high", [
    ("ticket_header", "header_change", 40),
    ("aircraft_image", "aircraft_image_change", 60),
    ("ticket_scroll_body", "ticket_body_change", 60),
])
@pytest.mark.parametrize("boundary", ["below_medium", "medium", "below_high", "high"])
def test_ticket_roi_thresholds(pair_scene, roi, field, high, boundary):
    frame, config, _ = pair_scene
    other = frame.copy()
    count, expected = {
        "below_medium": (1, SignalStrength.LOW), "medium": (2, SignalStrength.MEDIUM),
        "below_high": (high - 1, SignalStrength.MEDIUM), "high": (high, SignalStrength.HIGH),
    }[boundary]
    crop = getattr(config, roi).crop(other)
    for index in range(count):
        crop[index // 10, index % 10] = 255
    before = other.copy()
    result = signals_module.analyze_investigation_pair(frame, other, config)
    signal = getattr(result, field)
    assert signal.strength is expected
    assert signal.cpr == count / 100
    assert signal.mad == pytest.approx(255 * count / 100)
    np.testing.assert_array_equal(other, before)


@pytest.mark.parametrize("valid,mad,cpr,expected", [
    (True, 3, 0.01, SignalStrength.MEDIUM), (True, 3, 0.02, SignalStrength.HIGH),
    (False, 3, 0.8, SignalStrength.NOT_AVAILABLE),
    (True, None, None, SignalStrength.NOT_AVAILABLE),
    (True, 3, None, SignalStrength.NOT_AVAILABLE),
    (True, float("nan"), 0.8, SignalStrength.NOT_AVAILABLE),
    (True, 3, float("inf"), SignalStrength.NOT_AVAILABLE),
])
def test_aligned_map_evidence_no_raw_fallback(pair_scene, valid, mad, cpr, expected):
    frame, config, mocks = pair_scene
    motion = MapMotionEstimate(valid=valid, aligned_mad=mad, aligned_cpr=cpr,
                               raw_mad=255, raw_cpr=1)
    mocks["estimate_map_motion"].return_value = motion
    result = signals_module.analyze_investigation_pair(frame, frame, config)
    assert result.map_motion is motion
    assert result.map_residual_change.strength is expected
    assert result.map_residual_change.mad is (mad if valid else None)
    assert result.map_residual_change.cpr is (cpr if valid else None)


@pytest.mark.parametrize("decision", [False, True])
def test_specialized_classifiers_receive_original_evidence(pair_scene, monkeypatch, decision):
    frame, config, mocks = pair_scene
    route_classifier, scroll_classifier = Mock(return_value=decision), Mock(return_value=decision)
    monkeypatch.setattr(signals_module, "is_route_selection_change", route_classifier)
    monkeypatch.setattr(signals_module, "is_vertical_scroll_evidence", scroll_classifier)
    result = signals_module.analyze_investigation_pair(frame, frame, config)
    route_classifier.assert_called_once_with(mocks["estimate_route_selection_change"].return_value)
    scroll_classifier.assert_called_once_with(mocks["estimate_ticket_scroll_evidence"].return_value)
    assert result.route_selection_change is decision
    assert result.ticket_scroll_detected is decision


def test_provisional_constants():
    for prefix, expected in (("HEADER", (0.02, 0.4)), ("AIRCRAFT_IMAGE", (0.02, 0.6)),
                             ("TICKET_BODY", (0.02, 0.6)), ("MAP_RESIDUAL", (0.01, 0.02))):
        assert (getattr(signals_module, prefix + "_MEDIUM_CPR"),
                getattr(signals_module, prefix + "_HIGH_CPR")) == expected


@pytest.mark.parametrize("frame,error", [(None, TypeError),
    (np.zeros((40, 40, 3), np.float32), TypeError),
    (np.zeros((40, 40), np.uint8), ValueError),
    (np.zeros((40, 40, 3), np.uint8), ValueError)])
def test_pair_input_validation_propagates(frame, error):
    with pytest.raises(error):
        signals_module.analyze_investigation_pair(frame, frame, UIConfig())


def test_subsystem_error_propagates(pair_scene):
    frame, config, mocks = pair_scene
    mocks["estimate_map_motion"].side_effect = RuntimeError("unexpected failure")
    with pytest.raises(RuntimeError, match="unexpected failure"):
        signals_module.analyze_investigation_pair(frame, frame, config)


def test_presence_evidence_exact_fields_and_frozen():
    evidence = signals_module.InvestigationPresenceEvidence(0.9, 0.1, 0.8)
    assert [f.name for f in fields(evidence)] == ["bottom_actions_dark_fraction",
                                               "ticket_body_edge_fraction",
                                               "ticket_body_light_fraction"]
    with pytest.raises(FrozenInstanceError):
        evidence.ticket_body_edge_fraction = 0


@pytest.mark.parametrize("bgr", [False, True])
def test_presence_synthetic_measurement_and_pixel_boundaries(bgr):
    config = UIConfig(reference_width=40, reference_height=40,
                      ticket_scroll_body=ROI(0, 0, 40, 32),
                      ticket_bottom_actions=ROI(0, 32, 40, 8))
    gray = np.zeros((40, 40), np.uint8)
    # Five-pixel stripes give deterministic active-style edges and half light area.
    gray[:32] = np.tile(np.repeat([191, 192], 5), (32, 4))
    # Separate high-contrast stripes exercise Canny without a weak threshold edge.
    gray[:16] = np.tile(np.repeat([0, 255], 5), (16, 4))
    gray[32:] = np.tile(np.repeat([63, 64], 20), (8, 1))
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if bgr else gray.copy()
    before = frame.copy()
    result = signals_module.estimate_investigation_presence(frame, config)
    assert result.bottom_actions_dark_fraction == 0.5
    assert result.ticket_body_light_fraction == 0.5
    body = gray[:32]
    assert result.ticket_body_edge_fraction == np.count_nonzero(cv2.Canny(body, 50, 150)) / body.size
    assert result.ticket_body_edge_fraction >= 0.05
    assert signals_module.is_investigation_present(result)
    np.testing.assert_array_equal(frame, before)


def test_presence_bgr_conversion_uses_grayscale_not_channel_average():
    config = UIConfig(reference_width=10, reference_height=10,
                      ticket_scroll_body=ROI(0, 0, 10, 5),
                      ticket_bottom_actions=ROI(0, 5, 10, 5))
    frame = np.zeros((10, 10, 3), np.uint8)
    frame[:5] = (255, 255, 0)  # Gray 179, not light.
    frame[5:] = (255, 0, 0)  # Gray 29, dark despite channel mean 85.
    result = signals_module.estimate_investigation_presence(frame, config)
    assert result.bottom_actions_dark_fraction == 1
    assert result.ticket_body_light_fraction == 0
    assert result.ticket_body_edge_fraction == 0


@pytest.mark.parametrize("dark,edge,light,expected", [
    (0.9, 0.1, 0.8, True), (0.9, 0.01, 0.8, False), (0.1, 0.1, 0.8, False),
    (0.5, 0.05, 0, True), (0.5, 0.05, 1, True),
    (0.499, 0.1, 1, False), (0.9, 0.049, 1, False),
])
def test_presence_classifier_inclusive_and_light_does_not_vote(dark, edge, light, expected):
    evidence = signals_module.InvestigationPresenceEvidence(dark, edge, light)
    before = replace(evidence)
    assert signals_module.is_investigation_present(evidence) is expected
    assert evidence == before


@pytest.mark.parametrize("field", ["bottom_actions_dark_fraction", "ticket_body_edge_fraction",
                                   "ticket_body_light_fraction"])
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -float("inf"),
                                 -0.1, 1.1, True, "0.8"])
def test_presence_malformed_evidence_returns_false(field, bad):
    evidence = signals_module.InvestigationPresenceEvidence(0.9, 0.1, 0.8)
    assert not signals_module.is_investigation_present(replace(evidence, **{field: bad}))


@pytest.mark.parametrize("name", ["min_bottom_dark_fraction", "min_body_edge_fraction"])
@pytest.mark.parametrize("value,error", [(-0.1, ValueError), (1.1, ValueError),
    (float("nan"), ValueError), (float("inf"), ValueError), (True, TypeError),
    (np.bool_(False), TypeError), ("0.1", TypeError), (None, TypeError), (1j, TypeError)])
def test_presence_invalid_thresholds(name, value, error):
    evidence = signals_module.InvestigationPresenceEvidence(None, None, None)
    with pytest.raises(error):
        signals_module.is_investigation_present(evidence, **{name: value})


def test_presence_threshold_overrides_and_wrong_evidence_type():
    evidence = signals_module.InvestigationPresenceEvidence(0, 0, 0)
    assert signals_module.is_investigation_present(
        evidence, min_bottom_dark_fraction=0, min_body_edge_fraction=0)
    assert signals_module.is_investigation_present(
        signals_module.InvestigationPresenceEvidence(1, 1, 0),
        min_bottom_dark_fraction=np.float32(1), min_body_edge_fraction=1)
    with pytest.raises(TypeError, match="InvestigationPresenceEvidence"):
        signals_module.is_investigation_present(None)


@pytest.mark.parametrize("frame,error", [(None, TypeError),
    (np.zeros((1080, 1920, 3), np.float32), TypeError),
    (np.zeros((0, 1920, 3), np.uint8), ValueError),
    (np.zeros((1080, 1920, 4), np.uint8), ValueError),
    (np.zeros((40, 40, 3), np.uint8), ValueError)])
def test_presence_invalid_frames(frame, error):
    with pytest.raises(error):
        signals_module.estimate_investigation_presence(frame, UIConfig())


def test_presence_invalid_config_and_roi():
    frame = np.zeros((40, 40, 3), np.uint8)
    with pytest.raises(TypeError, match="UIConfig"):
        signals_module.estimate_investigation_presence(frame, None)
    config = UIConfig(reference_width=40, reference_height=40)
    with pytest.raises(ValueError, match="boundaries"):
        signals_module.estimate_investigation_presence(frame, config)
