from dataclasses import FrozenInstanceError
from itertools import pairwise

import numpy as np
import pytest

from events_diary.config import ROI, UIConfig
from events_diary.layout import (
    LayoutKind,
    LayoutTransform,
    MappedROI,
    ROIVisibility,
    TransformedBounds,
    classify_layout,
    map_roi_to_frame,
    map_roi_with_transform,
    scale_roi,
)


def test_scale_roi():
    roi = ROI(x=300, y=150, width=600, height=300)

    scaled = scale_roi(
        roi=roi,
        scale_x=0.5,
        scale_y=0.5,
    )

    assert scaled == ROI(
        x=150,
        y=75,
        width=300,
        height=150,
    )


def test_map_roi_to_frame_same_resolution():
    config = UIConfig()

    mapped = map_roi_to_frame(
        roi=config.ticket_zone,
        config=config,
        frame_width=1920,
        frame_height=1080,
    )

    assert mapped == config.ticket_zone


def test_map_roi_to_frame_720p():
    config = UIConfig()

    mapped = map_roi_to_frame(
        roi=config.ticket_zone,
        config=config,
        frame_width=1280,
        frame_height=720,
    )

    assert mapped == ROI(
        x=11,
        y=81,
        width=224,
        height=591,
    )
    
def test_classify_exact_reference_layout():
    config = UIConfig()

    result = classify_layout(
        config=config,
        frame_width=1920,
        frame_height=1080,
    )

    assert result == LayoutKind.EXACT_REFERENCE

def test_classify_proportional_scaled_layout():
    config = UIConfig()

    result = classify_layout(
        config=config,
        frame_width=1280,
        frame_height=720,
    )

    assert result == LayoutKind.PROPORTIONAL_SCALE

def test_classify_partial_or_cropped_layout():
    config = UIConfig()

    result = classify_layout(
        config=config,
        frame_width=1906,
        frame_height=984,
    )

    assert result == LayoutKind.PARTIAL_OR_CROPPED

def test_map_roi_to_frame_rejects_partial_or_cropped_layout():
    config = UIConfig()

    with pytest.raises(ValueError):
        map_roi_to_frame(
            roi=config.ticket_zone,
            config=config,
            frame_width=1906,
            frame_height=984,
        )


@pytest.mark.parametrize("dimension", ["frame_width", "frame_height", "reference_width", "reference_height"])
@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize("mapping", [False, True])
def test_layout_functions_reject_nonpositive_dimensions(dimension, value, mapping):
    dimensions = {"frame_width": 1920, "frame_height": 1080}
    reference = {}
    if dimension.startswith("reference"):
        reference[dimension] = value
    else:
        dimensions[dimension] = value
    config = UIConfig(**reference)
    with pytest.raises(ValueError, match="dimensions must be positive"):
        if mapping:
            map_roi_to_frame(config.ticket_zone, config, **dimensions)
        else:
            classify_layout(config, **dimensions)


def test_classify_rejects_zero_frame_and_reference_dimensions():
    with pytest.raises(ValueError, match="dimensions must be positive"):
        classify_layout(UIConfig(reference_width=0, reference_height=0), 0, 0)


def test_layout_transform_identity_and_immutability():
    transform = LayoutTransform()
    assert transform == LayoutTransform(1.0, 1.0, 0.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        transform.offset_x = 10.0


@pytest.mark.parametrize("field", ["scale_x", "scale_y"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_layout_transform_rejects_invalid_scales(field, value):
    with pytest.raises(ValueError, match="finite and positive"):
        LayoutTransform(**{field: value})


@pytest.mark.parametrize("field", ["offset_x", "offset_y"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_layout_transform_rejects_nonfinite_offsets(field, value):
    with pytest.raises(ValueError, match="must be finite"):
        LayoutTransform(**{field: value})


@pytest.mark.parametrize(
    "transform,expected",
    [
        (LayoutTransform(), ROI(10, 20, 30, 40)),
        (LayoutTransform(0.5, 0.5), ROI(5, 10, 15, 20)),
        (LayoutTransform(2.0, 0.5), ROI(20, 10, 60, 20)),
        (LayoutTransform(offset_x=5, offset_y=7), ROI(15, 27, 30, 40)),
        (LayoutTransform(offset_x=-5, offset_y=-7), ROI(5, 13, 30, 40)),
        (LayoutTransform(2.0, 0.5, -5, 7), ROI(15, 17, 60, 20)),
    ],
)
def test_explicit_transform_maps_full_regions(transform, expected):
    result = map_roi_with_transform(ROI(10, 20, 30, 40), transform, 100, 100)
    assert result.visibility == ROIVisibility.FULL
    assert result.visible_roi == expected
    assert result.transformed_bounds == TransformedBounds(
        expected.x, expected.y, expected.right, expected.bottom,
    )


def test_full_mapping_at_exact_frame_boundaries():
    result = map_roi_with_transform(ROI(0, 0, 100, 100), LayoutTransform(), 100, 100)
    assert result == MappedROI(
        TransformedBounds(0, 0, 100, 100), ROIVisibility.FULL, ROI(0, 0, 100, 100),
    )


@pytest.mark.parametrize(
    "offset_x,offset_y,bounds,visible",
    [
        (-15, 0, TransformedBounds(-5, 10, 15, 30), ROI(0, 10, 15, 20)),
        (80, 0, TransformedBounds(90, 10, 110, 30), ROI(90, 10, 10, 20)),
        (0, -15, TransformedBounds(10, -5, 30, 15), ROI(10, 0, 20, 15)),
        (0, 80, TransformedBounds(10, 90, 30, 110), ROI(10, 90, 20, 10)),
        (-15, -15, TransformedBounds(-5, -5, 15, 15), ROI(0, 0, 15, 15)),
    ],
)
def test_partial_mapping_preserves_bounds_and_visible_crop(offset_x, offset_y, bounds, visible):
    result = map_roi_with_transform(
        ROI(10, 10, 20, 20), LayoutTransform(offset_x=offset_x, offset_y=offset_y), 100, 100,
    )
    assert result == MappedROI(bounds, ROIVisibility.PARTIAL, visible)
    frame = np.arange(10000).reshape(100, 100)
    assert np.array_equal(
        result.visible_roi.crop(frame), frame[visible.y:visible.bottom, visible.x:visible.right],
    )
    assert result.visibility == ROIVisibility.PARTIAL


@pytest.mark.parametrize(
    "offset_x,offset_y",
    [(-31, 0), (91, 0), (0, -31), (0, 91),
     (-30, 0), (90, 0), (0, -30), (0, 90)],
)
def test_outside_and_edge_contact_have_no_visible_roi(offset_x, offset_y):
    result = map_roi_with_transform(
        ROI(10, 10, 20, 20), LayoutTransform(offset_x=offset_x, offset_y=offset_y), 100, 100,
    )
    assert result.visibility == ROIVisibility.OUTSIDE
    assert result.visible_roi is None
    assert result.transformed_bounds == TransformedBounds(
        10 + offset_x, 10 + offset_y, 30 + offset_x, 30 + offset_y,
    )


def test_shared_canonical_edges_remain_aligned():
    config = UIConfig()
    regions = [config.ticket_header, config.aircraft_image,
               config.ticket_scroll_body, config.ticket_bottom_actions]
    mapped = [map_roi_to_frame(roi, config, 1280, 720) for roi in regions]
    assert all(first.bottom == second.y for first, second in pairwise(mapped))
    transform = LayoutTransform(2 / 3, 2 / 3, 0.25, 0.25)
    explicit = [map_roi_with_transform(roi, transform, 1280, 720).visible_roi for roi in regions]
    assert all(first.bottom == second.y for first, second in pairwise(explicit))


@pytest.mark.parametrize("offset", [-0.1, 0.1])
def test_fractional_overflow_remains_partial_after_rounding(offset):
    result = map_roi_with_transform(
        ROI(0, 0, 100, 100), LayoutTransform(offset_x=offset), 100, 100,
    )
    assert result.visibility == ROIVisibility.PARTIAL
    assert result.visible_roi == ROI(0, 0, 100, 100)
    assert result.transformed_bounds.left == offset


def test_edge_rounding_uses_ties_to_even_and_derives_sizes():
    result = map_roi_with_transform(ROI(1, 1, 4, 4), LayoutTransform(0.5, 0.5), 10, 10)
    assert result.transformed_bounds == TransformedBounds(0.5, 0.5, 2.5, 2.5)
    assert result.visible_roi == ROI(0, 0, 2, 2)


@pytest.mark.parametrize(
    "roi,transform",
    [(ROI(0, 0, 1, 1), LayoutTransform(0.1, 0.1)),
     (ROI(0, 0, 10, 10), LayoutTransform(offset_x=99.9))],
)
def test_visible_subpixel_geometry_raises_explicit_error(roi, transform):
    with pytest.raises(ValueError, match="zero pixel area after rounding"):
        map_roi_with_transform(roi, transform, 100, 100)


def test_scale_roi_rejects_pixel_collapse():
    with pytest.raises(ValueError, match="zero pixel area after rounding"):
        scale_roi(ROI(0, 0, 1, 1), 0.1, 0.1)


@pytest.mark.parametrize("field", ["frame_width", "frame_height"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_explicit_mapping_rejects_invalid_dimensions(field, value):
    dimensions = {"frame_width": 100, "frame_height": 100, field: value}
    with pytest.raises(ValueError, match="dimensions must be positive and finite"):
        map_roi_with_transform(ROI(0, 0, 10, 10), LayoutTransform(), **dimensions)


@pytest.mark.parametrize("field", ["frame_width", "frame_height", "reference_width", "reference_height"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_existing_layout_functions_reject_nonfinite_dimensions(field, value):
    dimensions = {"frame_width": 1920, "frame_height": 1080}
    reference = {}
    if field.startswith("reference"):
        reference[field] = value
    else:
        dimensions[field] = value
    config = UIConfig(**reference)
    with pytest.raises(ValueError, match="dimensions must be positive and finite"):
        classify_layout(config, **dimensions)
    with pytest.raises(ValueError, match="dimensions must be positive and finite"):
        map_roi_to_frame(config.ticket_zone, config, **dimensions)


def test_explicit_identity_supports_partial_dimensions_without_guessing_offset():
    config = UIConfig()
    result = map_roi_with_transform(config.map_analysis_region, LayoutTransform(), 1906, 984)
    assert result.visibility == ROIVisibility.PARTIAL
    assert result.transformed_bounds == TransformedBounds(352, 121, 1920, 1040)
    assert result.visible_roi == ROI(352, 121, 1554, 863)


def test_mapping_rejects_arithmetic_overflow():
    with pytest.raises(ValueError, match="Transformed bounds must be finite"):
        map_roi_with_transform(ROI(0, 0, 10, 10), LayoutTransform(1e308, 1.0), 100, 100)

