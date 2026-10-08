"""Focused tests for semantic-agnostic grayscale change metrics."""

import numpy as np
import pytest

from events_diary.change_metrics import compute_changed_pixel_ratio, compute_mad


@pytest.mark.parametrize("shape", [(3, 4), (3, 4, 3)])
def test_identical_images_return_zero(shape):
    image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    result = compute_mad(image, image.copy())
    assert result == 0.0
    assert isinstance(result, float)


def test_constant_grayscale_difference():
    image_a = np.full((3, 4), 20, dtype=np.uint8)
    image_b = np.full((3, 4), 55, dtype=np.uint8)
    assert compute_mad(image_a, image_b) == 35.0


def test_mixed_grayscale_differences():
    image_a = np.array([[0, 100], [200, 50]], dtype=np.uint8)
    image_b = np.array([[10, 80], [255, 50]], dtype=np.uint8)
    assert compute_mad(image_a, image_b) == 21.25
    assert compute_mad(image_b, image_a) == 21.25


def test_bgr_difference_is_computed_after_grayscale_conversion():
    # OpenCV maps pure blue, green, and red to 29, 150, and 76 respectively.
    image_a = np.zeros((1, 3, 3), dtype=np.uint8)
    image_b = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    assert compute_mad(image_a, image_b) == 85.0


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 3)])
def test_uint8_difference_does_not_wrap_or_normalize(shape):
    dark = np.zeros(shape, dtype=np.uint8)
    bright = np.full(shape, 255, dtype=np.uint8)
    assert compute_mad(dark, bright) == 255.0
    assert compute_mad(bright, dark) == 255.0


@pytest.mark.parametrize("shape", [(3, 2), (2, 3, 3)])
def test_different_shapes_are_rejected(shape):
    with pytest.raises(ValueError, match="identical shapes"):
        compute_mad(np.zeros((2, 3), dtype=np.uint8), np.zeros(shape, dtype=np.uint8))


@pytest.mark.parametrize("input_index", [0, 1])
@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 3, 3)])
def test_empty_images_are_rejected(input_index, shape):
    images = [np.zeros((2, 3), dtype=np.uint8) for _ in range(2)]
    images[input_index] = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="must not be empty"):
        compute_mad(*images)


@pytest.mark.parametrize("input_index", [0, 1])
@pytest.mark.parametrize("shape", [(), (4,), (1, 2, 3, 4)])
def test_unsupported_dimensionality_is_rejected(input_index, shape):
    images = [np.zeros((2, 3), dtype=np.uint8) for _ in range(2)]
    images[input_index] = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="2D grayscale or 3D BGR"):
        compute_mad(*images)


@pytest.mark.parametrize("input_index", [0, 1])
@pytest.mark.parametrize("channels", [1, 2, 4])
def test_invalid_channel_counts_are_rejected(input_index, channels):
    images = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(2)]
    images[input_index] = np.zeros((2, 3, channels), dtype=np.uint8)
    with pytest.raises(ValueError, match="exactly 3 BGR channels"):
        compute_mad(*images)


@pytest.mark.parametrize("input_index", [0, 1])
@pytest.mark.parametrize("value", [None, [[0, 1]], "image"])
def test_nonarrays_are_rejected(input_index, value):
    images = [np.zeros((2, 3), dtype=np.uint8) for _ in range(2)]
    images[input_index] = value
    with pytest.raises(TypeError, match="must be a NumPy array"):
        compute_mad(*images)


@pytest.mark.parametrize("input_index", [0, 1])
@pytest.mark.parametrize("dtype", [np.float32, np.int16])
@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 3)])
def test_unsupported_dtypes_are_rejected(input_index, dtype, shape):
    images = [np.zeros(shape, dtype=np.uint8) for _ in range(2)]
    images[input_index] = np.zeros(shape, dtype=dtype)
    with pytest.raises(TypeError, match="must have dtype np.uint8"):
        compute_mad(*images)


@pytest.mark.parametrize("shape", [(3, 4), (3, 4, 3)])
def test_inputs_are_not_modified(shape):
    image_a = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    image_b = np.full(shape, 200, dtype=np.uint8)
    original_a, original_b = image_a.copy(), image_b.copy()
    compute_mad(image_a, image_b)
    assert np.array_equal(image_a, original_a)
    assert np.array_equal(image_b, original_b)


@pytest.mark.parametrize("shape", [(3, 4), (3, 4, 3)])
def test_changed_ratio_identical_images(shape):
    image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    result = compute_changed_pixel_ratio(image, image.copy(), 10)
    assert result == 0.0
    assert isinstance(result, float)


def test_changed_ratio_all_pixels_above_threshold():
    image_a = np.full((2, 3), 20, dtype=np.uint8)
    image_b = np.full((2, 3), 31, dtype=np.uint8)
    assert compute_changed_pixel_ratio(image_a, image_b, 10) == 1.0


def test_changed_ratio_known_subset_and_strict_threshold():
    image_a = np.zeros((2, 2), dtype=np.uint8)
    image_b = np.array([[0, 9], [10, 11]], dtype=np.uint8)
    assert compute_changed_pixel_ratio(image_a, image_b, 10) == 0.25
    assert compute_changed_pixel_ratio(image_b, image_a, 10) == 0.25


@pytest.mark.parametrize("difference,expected", [(10, 0.0), (11, 1.0)])
def test_changed_ratio_threshold_boundary(difference, expected):
    image_a = np.zeros((1, 1), dtype=np.uint8)
    image_b = np.full((1, 1), difference, dtype=np.uint8)
    assert compute_changed_pixel_ratio(image_a, image_b, 10) == expected


def test_changed_ratio_zero_threshold_counts_nonzero_differences():
    image_a = np.zeros((2, 2), dtype=np.uint8)
    image_b = np.array([[0, 1], [10, 255]], dtype=np.uint8)
    assert compute_changed_pixel_ratio(image_a, image_b, 0) == 0.75


@pytest.mark.parametrize("threshold,expected", [(254, 1.0), (255, 0.0)])
@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 3)])
def test_changed_ratio_uint8_safety_and_max_threshold(threshold, expected, shape):
    dark = np.zeros(shape, dtype=np.uint8)
    bright = np.full(shape, 255, dtype=np.uint8)
    assert compute_changed_pixel_ratio(dark, bright, threshold) == expected
    assert compute_changed_pixel_ratio(bright, dark, threshold) == expected


def test_changed_ratio_bgr_uses_grayscale_pixel_count():
    # Grayscale intensities are 29, 150, 76; only green exceeds 100.
    image_a = np.zeros((1, 3, 3), dtype=np.uint8)
    image_b = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    assert compute_changed_pixel_ratio(image_a, image_b, 100) == 1 / 3


@pytest.mark.parametrize("threshold", [-1, 256])
def test_changed_ratio_rejects_out_of_range_threshold(threshold):
    image = np.zeros((2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="inclusive range 0 to 255"):
        compute_changed_pixel_ratio(image, image, threshold)


@pytest.mark.parametrize("threshold", [10.0, True, False, None, "10"])
def test_changed_ratio_rejects_noninteger_threshold(threshold):
    image = np.zeros((2, 3), dtype=np.uint8)
    with pytest.raises(TypeError, match="must be an integer"):
        compute_changed_pixel_ratio(image, image, threshold)


@pytest.mark.parametrize("input_index", [0, 1])
@pytest.mark.parametrize(
    "invalid,exception,message",
    [
        (None, TypeError, "NumPy array"),
        (np.zeros((2, 3), dtype=np.float32), TypeError, "dtype np.uint8"),
        (np.zeros((2, 3), dtype=np.int16), TypeError, "dtype np.uint8"),
        (np.zeros((0, 3), dtype=np.uint8), ValueError, "must not be empty"),
        (np.zeros((), dtype=np.uint8), ValueError, "2D grayscale or 3D BGR"),
        (np.zeros((3,), dtype=np.uint8), ValueError, "2D grayscale or 3D BGR"),
        (np.zeros((1, 2, 3, 4), dtype=np.uint8), ValueError, "2D grayscale or 3D BGR"),
        (np.zeros((2, 3, 1), dtype=np.uint8), ValueError, "exactly 3 BGR channels"),
        (np.zeros((2, 3, 4), dtype=np.uint8), ValueError, "exactly 3 BGR channels"),
        (np.zeros((3, 2), dtype=np.uint8), ValueError, "identical shapes"),
    ],
)
def test_changed_ratio_preserves_image_validation(input_index, invalid, exception, message):
    images = [np.zeros((2, 3), dtype=np.uint8) for _ in range(2)]
    images[input_index] = invalid
    with pytest.raises(exception, match=message):
        compute_changed_pixel_ratio(*images, pixel_threshold=10)


@pytest.mark.parametrize("shape", [(3, 4), (3, 4, 3)])
def test_changed_ratio_does_not_modify_inputs(shape):
    image_a = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    image_b = np.full(shape, 200, dtype=np.uint8)
    original_a, original_b = image_a.copy(), image_b.copy()
    compute_changed_pixel_ratio(image_a, image_b, 10)
    assert np.array_equal(image_a, original_a)
    assert np.array_equal(image_b, original_b)
