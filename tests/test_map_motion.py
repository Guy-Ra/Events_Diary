"""Synthetic geometry and controlled failure tests for map motion compensation."""

from dataclasses import FrozenInstanceError

import cv2
import numpy as np
import pytest

from events_diary import map_motion
from events_diary.change_metrics import compute_changed_pixel_ratio, compute_mad
from events_diary.config import ROI, UIConfig


@pytest.fixture
def texture():
    rng = np.random.default_rng(23)
    image = np.zeros((360, 420), dtype=np.uint8)
    for _ in range(180):
        x, y = rng.integers([12, 12], [408, 348])
        cv2.circle(image, (int(x), int(y)), int(rng.integers(2, 9)),
                   int(rng.integers(50, 255)), -1)
    cv2.putText(image, "MAP 234", (60, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 210, 2)
    return image


@pytest.mark.parametrize("scale,dx,dy", [(1, 0, 0), (1, 22, -17),
                                          (1.2, 0, 0), (0.85, 24, 31)])
def test_real_sift_geometry_and_residuals(texture, scale, dx, dy):
    # Generate B from A; the estimator must recover the inverse B-to-A transform.
    matrix = np.float64([[scale, 0, dx], [0, scale, dy]])
    image_b = cv2.warpAffine(texture, matrix, (420, 360))
    before_a, before_b = texture.copy(), image_b.copy()
    result = map_motion.estimate_map_motion(texture, image_b)
    assert result.valid, result.reason
    assert result.scale == pytest.approx(1 / scale, abs=0.008)
    assert result.dx == pytest.approx(-dx / scale, abs=0.7)
    assert result.dy == pytest.approx(-dy / scale, abs=0.7)
    assert abs(result.rotation_degrees) < 0.5
    if (scale, dx, dy) == (1, 0, 0):
        assert result.raw_mad == result.aligned_mad == 0
        assert result.raw_cpr == result.aligned_cpr == 0
    else:
        assert result.aligned_mad < result.raw_mad * 0.35
        assert result.aligned_cpr < result.raw_cpr * 0.4
    np.testing.assert_array_equal(texture, before_a)
    np.testing.assert_array_equal(image_b, before_b)
    with pytest.raises(FrozenInstanceError):
        result.valid = False


def test_no_features_and_raw_metrics():
    image_a = np.zeros((80, 90), dtype=np.uint8)
    image_b = np.full_like(image_a, 30)
    result = map_motion.estimate_map_motion(image_a, image_b)
    assert not result.valid
    assert "descriptors" in result.reason
    assert result.keypoints_a == result.keypoints_b == 0
    assert result.raw_mad == 30
    assert result.raw_cpr == 1
    assert result.aligned_mad is result.aligned_cpr is None


def _mock_features(monkeypatch, count=20, accepted=True, singleton=False):
    class Sift:
        def detectAndCompute(self, image, mask):
            return ([cv2.KeyPoint(float(i + 10), float(i + 10), 3)
                     for i in range(count)], np.ones((count, 128), np.float32))

    class Matcher:
        def knnMatch(self, source, destination, k):
            assert k == 2
            return [([cv2.DMatch(i, i, 1)] if singleton else
                     [cv2.DMatch(i, i, 1 if accepted else 9), cv2.DMatch(i, i, 10)])
                    for i in range(count)]

    monkeypatch.setattr(map_motion.cv2, "SIFT_create", Sift)
    monkeypatch.setattr(map_motion.cv2, "BFMatcher", lambda norm: Matcher())


@pytest.mark.parametrize("count,accepted,singleton", [(5, True, False),
                                                        (20, False, False),
                                                        (20, True, True)])
def test_insufficient_features_or_good_matches(monkeypatch, texture, count,
                                               accepted, singleton):
    _mock_features(monkeypatch, count, accepted, singleton)
    result = map_motion.estimate_map_motion(texture, texture)
    assert not result.valid
    assert "Too few good matches" in result.reason
    assert result.good_matches == (count if accepted and not singleton else 0)


@pytest.mark.parametrize("matrix,inliers,reason", [
    (None, 20, "estimation failed"),
    (np.float64([[1, 0, 0], [0, 1, 0]]), 7, "Too few RANSAC inliers"),
    (np.float64([[1, 0, 0], [0, 1, 0]]), 9, "Inlier ratio"),
    (np.float64([[0.996, -0.087, 0], [0.087, 0.996, 0]]), 20, "Rotation"),
    (np.float64([[2.1, 0, 0], [0, 2.1, 0]]), 20, "Scale"),
    (np.float64([[0.49, 0, 0], [0, 0.49, 0]]), 20, "Scale"),
    (np.float64([[np.nan, 0, 0], [0, 1, 0]]), 20, "non-finite"),
    (np.float64([[1, 0, np.inf], [0, 1, 0]]), 20, "non-finite"),
    (np.float64([[1, 0, 1000], [0, 1, 0]]), 20, "No fully supported overlap"),
])
def test_rejected_transforms(monkeypatch, texture, matrix, inliers, reason):
    _mock_features(monkeypatch)
    mask = np.uint8([1] * inliers + [0] * (20 - inliers)).reshape(-1, 1)
    monkeypatch.setattr(map_motion.cv2, "estimateAffinePartial2D",
                        lambda *args, **kwargs: (matrix, mask))
    image_b = np.full_like(texture, 45)
    result = map_motion.estimate_map_motion(texture, image_b)
    assert not result.valid
    assert reason in result.reason
    assert result.raw_mad == compute_mad(texture, image_b)
    assert result.raw_cpr == compute_changed_pixel_ratio(texture, image_b)
    assert result.aligned_mad is result.aligned_cpr is None


def test_accepted_rotation_is_diagnostic_only_and_border_is_excluded(monkeypatch):
    _mock_features(monkeypatch)
    angle = np.deg2rad(2)
    matrix = np.float64([[np.cos(angle), -np.sin(angle), 10.5],
                         [np.sin(angle), np.cos(angle), -7.25]])
    monkeypatch.setattr(map_motion.cv2, "estimateAffinePartial2D",
                        lambda *args, **kwargs: (matrix, np.ones((20, 1), np.uint8)))
    image = np.full((80, 90), 120, np.uint8)
    actual_warp = cv2.warpAffine
    warps = []

    def capture(source, warp, *args, **kwargs):
        warps.append(warp.copy())
        return actual_warp(source, warp, *args, **kwargs)

    monkeypatch.setattr(map_motion.cv2, "warpAffine", capture)
    result = map_motion.estimate_map_motion(image, image)
    assert result.valid
    assert result.rotation_degrees == pytest.approx(2)
    assert result.aligned_mad == result.aligned_cpr == 0
    assert len(warps) == 2
    for warp in warps:
        np.testing.assert_allclose(warp, [[1, 0, 10.5], [0, 1, -7.25]])


def test_bgr_and_threshold_override(texture):
    bgr = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    result = map_motion.estimate_map_motion(bgr, bgr, pixel_threshold=255)
    assert result.valid
    assert result.raw_cpr == result.aligned_cpr == 0


@pytest.mark.parametrize("image_b,error", [
    (np.zeros((10, 10), np.uint8), ValueError),
    (np.zeros((360, 420), np.float32), TypeError),
    (None, TypeError),
])
def test_input_contract(texture, image_b, error):
    with pytest.raises(error):
        map_motion.estimate_map_motion(texture, image_b)


@pytest.mark.parametrize("threshold,error", [(True, TypeError), (-1, ValueError)])
def test_invalid_threshold(texture, threshold, error):
    with pytest.raises(error):
        map_motion.estimate_map_motion(texture, texture, pixel_threshold=threshold)


def test_canonical_ui_masks_are_inside_map():
    config = UIConfig()
    region = config.map_analysis_region
    assert len(config.map_ignore_masks) == 4
    for roi in config.map_ignore_masks:
        assert region.x <= roi.x < roi.right <= region.right
        assert region.y <= roi.y < roi.bottom <= region.bottom


def test_canonical_mask_translation_and_binary_values():
    config = UIConfig()
    region = config.map_analysis_region
    mask = map_motion._build_sift_mask(config, (region.height, region.width))
    assert mask.shape == (919, 1568)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0, 255}
    # Known canonical origins subtract (352, 121), with exclusive bounds.
    expected = np.full((919, 1568), 255, np.uint8)
    expected[0:60, 0:1568] = 0
    expected[163:381, 1504:1564] = 0
    expected[837:907, 394:822] = 0
    expected[903:919, 1128:1568] = 0
    np.testing.assert_array_equal(mask, expected)
    assert mask[500, 500] == 255
    assert mask[162, 1504] == mask[381, 1504] == 255


@pytest.mark.parametrize("shape", [(918, 1568), (919, 1567)])
def test_mask_rejects_noncanonical_crop_dimensions(shape):
    with pytest.raises(ValueError, match="exact canonical"):
        map_motion._build_sift_mask(UIConfig(), shape)


def test_mask_rejects_outside_exclusions():
    config = UIConfig(map_ignore_masks=(ROI(351, 121, 10, 10),))
    with pytest.raises(ValueError, match="inside"):
        map_motion._build_sift_mask(config, (919, 1568))


def test_real_sift_excludes_feature_centers(monkeypatch, texture):
    config = UIConfig(map_analysis_region=ROI(10, 20, 420, 360),
                      map_ignore_masks=(ROI(10, 20, 210, 360),))
    original_sift = cv2.SIFT_create()
    unmasked, _ = original_sift.detectAndCompute(texture, None)
    assert any(point.pt[0] < 210 for point in unmasked)
    detections = []

    class Capture:
        def detectAndCompute(self, image, mask):
            assert mask.shape == texture.shape
            assert np.all(mask[:, :210] == 0)
            assert np.all(mask[:, 210:] == 255)
            points, descriptors = original_sift.detectAndCompute(image, mask)
            detections.append(points)
            return points, descriptors

    monkeypatch.setattr(map_motion.cv2, "SIFT_create", Capture)
    result = map_motion.estimate_map_motion(texture, texture, ui_config=config)
    assert result.valid
    assert len(detections) == 2
    for points in detections:
        assert points
        for point in points:
            x, y = round(point.pt[0]), round(point.pt[1])
            assert 210 <= x < 420 and 0 <= y < 360


def test_detection_exclusions_do_not_mask_any_metric_pixels(monkeypatch, texture):
    _mock_features(monkeypatch)
    monkeypatch.setattr(map_motion.cv2, "estimateAffinePartial2D",
                        lambda *args, **kwargs: (
                            np.float64([[1, 0, 0], [0, 1, 0]]),
                            np.ones((20, 1), np.uint8),
                        ))
    config = UIConfig(map_analysis_region=ROI(10, 20, 420, 360),
                      map_ignore_masks=(ROI(10, 20, 210, 360),))
    image_a = np.zeros_like(texture)
    image_b = image_a.copy()
    image_b[:, :210] = 100
    before_a, before_b = image_a.copy(), image_b.copy()
    result = map_motion.estimate_map_motion(image_a, image_b, ui_config=config)
    assert result.valid
    assert result.raw_mad == result.aligned_mad == 50
    assert result.raw_cpr == result.aligned_cpr == 0.5
    np.testing.assert_array_equal(image_a, before_a)
    np.testing.assert_array_equal(image_b, before_b)


@pytest.mark.parametrize("scale,dx,dy", [(1, 22, -17), (1.2, 0, 0), (0.85, 24, 31)])
def test_masked_synthetic_translation_and_scale(texture, scale, dx, dy):
    config = UIConfig(map_analysis_region=ROI(10, 20, 420, 360),
                      map_ignore_masks=(ROI(10, 20, 420, 25),))
    image_b = cv2.warpAffine(texture, np.float64([[scale, 0, dx], [0, scale, dy]]),
                             (420, 360))
    result = map_motion.estimate_map_motion(texture, image_b, ui_config=config)
    assert result.valid, result.reason
    assert result.scale == pytest.approx(1 / scale, abs=0.008)
    assert result.dx == pytest.approx(-dx / scale, abs=0.7)
    assert result.dy == pytest.approx(-dy / scale, abs=0.7)
