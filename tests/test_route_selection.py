"""Synthetic route-color evidence tests without manual video dependencies."""

from dataclasses import FrozenInstanceError, replace

import cv2
import numpy as np
import pytest

from events_diary import route_selection
from events_diary.config import ROI, UIConfig
from events_diary.map_motion import MapMotionEstimate


@pytest.fixture
def scene(monkeypatch):
    config = UIConfig(reference_width=120, reference_height=100,
                      map_analysis_region=ROI(20, 10, 80, 80), map_ignore_masks=())
    image = np.zeros((100, 120, 3), np.uint8)
    monkeypatch.setattr(route_selection, "estimate_map_motion", lambda *args, **kwargs:
                        MapMotionEstimate(valid=True, dx=0, dy=0, scale=1, rotation_degrees=0))
    return image, config


def _paint(image, config, x, y, width=10, height=10):
    """Paint a synthetic route-color rectangle in map-local coordinates."""
    config.map_analysis_region.crop(image)[y:y+height, x:x+width] = (255, 128, 0)


def test_identity_and_frozen_record(scene):
    image, config = scene
    _paint(image, config, 5, 5)
    result = route_selection.estimate_route_selection_change(image, image, config)
    assert result.valid
    assert result.route_fraction_a == result.route_fraction_b == 100 / 6400
    assert result.added_fraction == result.removed_fraction == result.changed_fraction == 0
    with pytest.raises(FrozenInstanceError):
        result.valid = False


@pytest.mark.parametrize("operation", ["added", "removed", "replaced"])
def test_mask_changes_and_equal_area_movement(scene, operation):
    image, config = scene
    a, b = image.copy(), image.copy()
    if operation in ("removed", "replaced"):
        _paint(a, config, 5, 5)
    if operation in ("added", "replaced"):
        _paint(b, config, 40, 40)
    before_a, before_b = a.copy(), b.copy()
    result = route_selection.estimate_route_selection_change(a, b, config)
    assert result.valid
    assert result.added_fraction == (100 / 6400 if operation != "removed" else 0)
    assert result.removed_fraction == (100 / 6400 if operation != "added" else 0)
    assert result.changed_fraction == pytest.approx(result.added_fraction + result.removed_fraction)
    if operation == "replaced":
        assert result.route_fraction_a == result.route_fraction_b
        assert result.changed_fraction > 0
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)


@pytest.mark.parametrize("hsv,expected", [
    ((60, 255, 255), True), ((131, 255, 255), True), ((95, 200, 200), True),
    ((59, 255, 255), False), ((132, 255, 255), False), ((95, 100, 255), False),
    ((95, 255, 150), False), ((0, 255, 255), False), ((30, 255, 255), False),
    ((0, 0, 255), False),
])
def test_route_hsv_membership_and_nonroute_colors(hsv, expected):
    bgr = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)
    assert bool(route_selection._build_route_mask(bgr)[0, 0]) is expected


def test_invalid_map_motion_and_no_overlap(scene, monkeypatch):
    image, config = scene
    for motion in (MapMotionEstimate(valid=False, reason="Too few inliers"),
                   MapMotionEstimate(valid=True, dx=1000, dy=0, scale=1, rotation_degrees=0)):
        monkeypatch.setattr(route_selection, "estimate_map_motion",
                            lambda *args, motion=motion, **kwargs: motion)
        result = route_selection.estimate_route_selection_change(image, image, config)
        assert not result.valid
        assert result.reason
        assert all(getattr(result, field) is None for field in (
            "route_fraction_a", "route_fraction_b", "added_fraction", "removed_fraction", "changed_fraction"))


def test_overlap_excludes_warp_borders_and_uses_overlap_denominator(scene, monkeypatch):
    image, config = scene
    a = image.copy()
    b = image.copy()
    config.map_analysis_region.crop(a)[:] = (255, 128, 0)
    config.map_analysis_region.crop(b)[:] = (255, 128, 0)
    # Borders are black after shifting B; including them would spuriously remove route.
    monkeypatch.setattr(route_selection, "estimate_map_motion", lambda *args, **kwargs:
                        MapMotionEstimate(valid=True, dx=10.5, dy=-7.25, scale=1,
                                          rotation_degrees=2))
    result = route_selection.estimate_route_selection_change(a, b, config)
    assert result.valid
    assert result.route_fraction_a == result.route_fraction_b == 1
    assert result.changed_fraction == result.added_fraction == result.removed_fraction == 0


def test_b_to_a_direction_and_diagnostic_rotation_omitted(scene, monkeypatch):
    image, config = scene
    a = image.copy()
    _paint(a, config, 20, 20)
    b = image.copy()
    _paint(b, config, 25, 27)
    monkeypatch.setattr(route_selection, "estimate_map_motion", lambda *args, **kwargs:
                        MapMotionEstimate(valid=True, dx=-5, dy=-7, scale=1,
                                          rotation_degrees=2))
    result = route_selection.estimate_route_selection_change(a, b, config)
    assert result.valid
    assert result.changed_fraction == 0
    assert result.route_fraction_a == result.route_fraction_b == 100 / (75 * 73)


def test_real_sift_translation_and_scale_preserve_route_pattern():
    config = UIConfig(reference_width=360, reference_height=320,
                      map_analysis_region=ROI(0, 0, 360, 320), map_ignore_masks=())
    rng = np.random.default_rng(38)
    gray = rng.integers(0, 256, (320, 360), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    a = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    _paint(a, config, 120, 100, 8, 100)
    b = cv2.warpAffine(a, np.float64([[1.05, 0, 10], [0, 1.05, -8]]), (360, 320))
    result = route_selection.estimate_route_selection_change(a, b, config)
    assert result.valid, result.reason
    assert result.changed_fraction < 0.002


@pytest.mark.parametrize("bad,error", [
    (None, TypeError), (np.zeros((100, 120, 3), np.float32), TypeError),
    (np.zeros((0, 120, 3), np.uint8), ValueError),
    (np.zeros((100, 120), np.uint8), ValueError),
    (np.zeros((100, 120, 4), np.uint8), ValueError),
    (np.zeros((101, 120, 3), np.uint8), ValueError),
])
def test_invalid_frames(scene, bad, error):
    image, config = scene
    with pytest.raises(error):
        route_selection.estimate_route_selection_change(image, bad, config)


def test_noncanonical_frames_invalid_config_and_roi(scene):
    image, config = scene
    bad = np.zeros((101, 120, 3), np.uint8)
    with pytest.raises(ValueError, match="canonical"):
        route_selection.estimate_route_selection_change(bad, bad, config)
    with pytest.raises(TypeError, match="UIConfig"):
        route_selection.estimate_route_selection_change(image, image, None)
    with pytest.raises(ValueError, match="boundaries"):
        route_selection.estimate_route_selection_change(image, image,
            replace(config, map_analysis_region=ROI(20, 10, 200, 200)))


def _replacement(**changes):
    """Return synthetic immutable replacement evidence with optional overrides."""
    return replace(route_selection.RouteSelectionChangeEvidence(
        True, None, 0.001, 0.001, 0.0002, 0.0002, 0.0004), **changes)


def test_bidirectional_replacement_and_immutability():
    evidence = _replacement()
    before = replace(evidence)
    assert route_selection.is_route_selection_change(evidence)
    assert evidence == before
    with pytest.raises(FrozenInstanceError):
        evidence.changed_fraction = 1


@pytest.mark.parametrize("changes", [
    {"removed_fraction": 0, "added_fraction": 0.001, "changed_fraction": 0.001},
    {"added_fraction": 0, "removed_fraction": 0.001, "changed_fraction": 0.001},
    {"changed_fraction": 0.000299}, {"added_fraction": 0.000099},
    {"removed_fraction": 0.000099}, {"valid": False, "reason": "invalid"},
])
def test_pair_classifier_rejections(changes):
    assert not route_selection.is_route_selection_change(_replacement(**changes))


def test_inclusive_classification_boundaries():
    evidence = _replacement(changed_fraction=0.0003, added_fraction=0.0001,
                            removed_fraction=0.0001)
    assert route_selection.is_route_selection_change(evidence)
    assert route_selection.is_route_selection_change(
        _replacement(changed_fraction=0, added_fraction=0, removed_fraction=0),
        min_changed_fraction=0, min_added_fraction=0, min_removed_fraction=0)


@pytest.mark.parametrize("field", ["changed_fraction", "added_fraction", "removed_fraction"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -float("inf")])
def test_unavailable_required_fractions(field, value):
    assert not route_selection.is_route_selection_change(_replacement(**{field: value}))


@pytest.mark.parametrize("name", ["min_changed_fraction", "min_added_fraction", "min_removed_fraction"])
@pytest.mark.parametrize("value,error", [(-0.1, ValueError), (1.1, ValueError),
    (float("nan"), ValueError), (float("inf"), ValueError), (True, TypeError),
    (np.bool_(False), TypeError), ("0.1", TypeError), (None, TypeError), (1j, TypeError)])
def test_classifier_threshold_validation(name, value, error):
    with pytest.raises(error):
        route_selection.is_route_selection_change(_replacement(valid=False), **{name: value})


def test_classifier_evidence_type_and_unavailable_record():
    with pytest.raises(TypeError, match="RouteSelectionChangeEvidence"):
        route_selection.is_route_selection_change(None)
    assert not route_selection.is_route_selection_change(
        route_selection.RouteSelectionChangeEvidence(False, "unavailable"))
