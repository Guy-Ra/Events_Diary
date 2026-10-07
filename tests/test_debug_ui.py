import cv2
import numpy as np

from events_diary.config import ROI
from events_diary.debug.ui import draw_rois


def test_draw_rois_preserves_frame_shape():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)

    regions = {
        "test_roi": ROI(x=20, y=30, width=100, height=50),
    }

    result = draw_rois(frame, regions)

    assert result.shape == frame.shape

def test_draw_rois_does_not_modify_original_frame():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    original = frame.copy()

    regions = {
        "test_roi": ROI(x=20, y=30, width=100, height=50),
    }

    draw_rois(frame, regions)

    assert np.array_equal(frame, original)


def test_draw_rois_renders_visible_rectangle_with_two_pixel_thickness():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = draw_rois(frame, {"": ROI(20, 30, 40, 40)})
    assert result[60, 20].tolist() == [0, 255, 0]
    assert result[60, 19].tolist() == [0, 255, 0]
    assert result[60, 18].tolist() == [0, 0, 0]


def test_draw_rois_renders_thin_antialiased_text():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    expected = frame.copy()
    cv2.putText(expected, "Test", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    result = draw_rois(frame, {"Test": ROI(20, 30, 70, 60)})
    # The interior excludes the rectangle; compare the rendered text itself.
    assert np.any(expected[34:54, 24:65])
    assert np.array_equal(result[34:54, 24:65], expected[34:54, 24:65])
