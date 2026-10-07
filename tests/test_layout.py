import pytest

from events_diary.config import ROI, UIConfig
from events_diary.layout import (
    LayoutKind,
    classify_layout,
    map_roi_to_frame,
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

