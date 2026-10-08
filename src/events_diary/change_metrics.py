"""Low-level visual change metrics between image regions.

This module compares image arrays without interpreting their contents. Metrics
are intentionally semantic-agnostic: they know nothing about UI regions,
aircraft, layouts, or Investigation Events. Higher-level interpretation belongs
to later pipeline stages.

Mean Absolute Difference (MAD) measures the average magnitude of grayscale
pixel change, expressed in intensity units. Changed Pixel Ratio measures the
fraction of pixels whose difference exceeds a caller-supplied threshold.
Threshold selection and interpretation remain external to this module.

Preprocessing is intentionally minimal: BGR images are converted to grayscale,
and grayscale images are used directly. There is no resizing, normalization,
filtering, masking, or adaptive thresholding.
"""

import cv2
import numpy as np


def compute_mad(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Return the mean absolute grayscale pixel difference between two images.

    Args:
        image_a:
            A nonempty NumPy array with dtype np.uint8, either a 2D grayscale
            image or a 3D BGR image with exactly three channels.
        image_b:
            An image with the same requirements and shape as image_a.

    Preprocessing:
        BGR inputs are converted using OpenCV's COLOR_BGR2GRAY conversion.
        Grayscale inputs are used directly. No implicit dtype conversion,
        resizing, normalization, or other preprocessing is performed.

    Returns:
        A Python float in the range 0.0 to 255.0, computed as the arithmetic
        mean of absolute grayscale pixel differences. Identical images return
        0.0. The result is not normalized to the range 0 to 1.

    Raises:
        TypeError: If either input is not a NumPy array or has a dtype other
            than np.uint8.
        ValueError: If either image is empty, the shapes differ, dimensionality
            is unsupported, or a 3D image does not have exactly three channels.

    Notes:
        cv2.absdiff avoids unsigned-integer subtraction wraparound. The inputs
        are not modified. BGR conversion uses grayscale intensity, so color
        changes with equal grayscale intensities can yield zero MAD.
    """
    return float(np.mean(_grayscale_difference(image_a, image_b)))


def compute_changed_pixel_ratio(
    image_a: np.ndarray,
    image_b: np.ndarray,
    pixel_threshold: int = 10,
) -> float:
    """Return the fraction of grayscale pixels changing above a supplied threshold.

    Args:
        image_a:
            A nonempty NumPy array with dtype np.uint8, either a 2D grayscale
            image or a 3D BGR image with exactly three channels.
        image_b:
            An image with the same requirements and shape as image_a.
        pixel_threshold:
            An integer from 0 to 255 inclusive; booleans are rejected. A pixel
            counts as changed only when its absolute grayscale difference is
            strictly greater than this value, not equal to it. Threshold
            selection is intentionally external; there is no default.

    Preprocessing:
        BGR images are converted using OpenCV's COLOR_BGR2GRAY conversion.
        Grayscale images are used directly. Differences use cv2.absdiff to
        avoid unsigned-integer wraparound. No dtype conversion, resizing,
        filtering, masking, or normalization of differences is performed.

    Returns:
        A Python float in the range 0.0 to 1.0: changed pixel count divided by
        total grayscale pixel count. Threshold 0 counts any nonzero difference;
        threshold 255 always returns 0.0 for valid uint8 images.

    Raises:
        TypeError: If pixel_threshold is not an integer or is a boolean, or
            either image is not a NumPy array with dtype np.uint8.
        ValueError: If pixel_threshold is outside 0 to 255, either image is
            empty, shapes differ, dimensions are unsupported, or a 3D image
            does not have exactly three channels.

    Notes:
        Inputs are not modified. This metric measures change coverage rather
        than average magnitude and supplies no semantic interpretation. Color
        changes with equal grayscale intensities may be missed; noise or image
        motion may count as change. Threshold calibration belongs to later
        experimentation, not this function.
    """
    if not isinstance(pixel_threshold, int) or isinstance(pixel_threshold, bool):
        raise TypeError("pixel_threshold must be an integer, not a boolean.")
    if not 0 <= pixel_threshold <= 255:
        raise ValueError("pixel_threshold must be in the inclusive range 0 to 255.")

    difference = _grayscale_difference(image_a, image_b)
    return float(np.count_nonzero(difference > pixel_threshold) / difference.size)


def _grayscale_difference(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    """Validate a uint8 image pair and return safe absolute grayscale differences."""
    for name, image in (("image_a", image_a), ("image_b", image_b)):
        if not isinstance(image, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array.")
        if image.dtype != np.uint8:
            raise TypeError(f"{name} must have dtype np.uint8; got {image.dtype}.")
        if image.size == 0:
            raise ValueError(f"{name} must not be empty.")
        if image.ndim not in (2, 3):
            raise ValueError(f"{name} must be a 2D grayscale or 3D BGR image.")
        if image.ndim == 3 and image.shape[2] != 3:
            raise ValueError(f"{name} must have exactly 3 BGR channels.")

    if image_a.shape != image_b.shape:
        raise ValueError("Images must have identical shapes.")

    if image_a.ndim == 3:
        image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
        image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    return cv2.absdiff(image_a, image_b)
