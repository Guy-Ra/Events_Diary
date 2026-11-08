"""Tests for Events Diary application configuration."""
import numpy as np
import pytest

from events_diary.config import (
    ROI,
    AppConfig,
    UIConfig,
    UnderstandingMode,
    load_runtime_config,
)


def test_default_log_level() -> None:
    """Verify that the default application log level is INFO."""

    config = AppConfig()

    assert config.log_level == "INFO"

def test_roi_stores_geometry():
    roi = ROI(x=10, y=20, width=100, height=50)

    assert roi.x == 10
    assert roi.y == 20
    assert roi.right == 110
    assert roi.bottom == 70

def test_roi_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        ROI(x=-1, y=0, width=10, height=10)

def test_roi_crop_returns_expected_shape():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    roi = ROI(x=50, y=40, width=100, height=80)

    crop = roi.crop(frame)

    assert crop.shape == (80, 100, 3)

def test_roi_crop_rejects_out_of_bounds_region():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    roi = ROI(x=80, y=80, width=30, height=30)

    with pytest.raises(ValueError):
        roi.crop(frame)

def test_ui_config_uses_expected_reference_resolution():
    config = UIConfig()

    assert config.reference_width == 1920
    assert config.reference_height == 1080

def test_ui_config_regions_fit_reference_frame():
    config = UIConfig()

    regions = (
        config.browser_chrome,
        config.app_region,
        config.ticket_zone,
        config.ticket_header,
        config.aircraft_image,
        config.ticket_scroll_body,
        config.ticket_bottom_actions,
        config.map_analysis_region,
        config.system_taskbar,
    )

    for roi in regions:
        assert roi.right <= config.reference_width
        assert roi.bottom <= config.reference_height


@pytest.mark.parametrize(
    ("raw_mode", "expected"),
    (
        ("ocr_only", UnderstandingMode.OCR_ONLY),
        ("vlm_only", UnderstandingMode.VLM_ONLY),
        ("ocr_vlm", UnderstandingMode.OCR_VLM),
    ),
)
def test_runtime_config_reads_supported_understanding_mode_from_dotenv(
    tmp_path, monkeypatch, raw_mode, expected,
):
    env_file = tmp_path / ".env"
    env_file.write_text(f"EVENTS_DIARY_UNDERSTANDING_MODE={raw_mode}\n", encoding="utf-8")
    monkeypatch.setenv("EVENTS_DIARY_UNDERSTANDING_MODE", "ignored-unless-overridden")

    config = load_runtime_config(project_root=tmp_path, env_file=env_file, override=True)

    assert config.understanding_mode is expected
    assert config.input_dir == tmp_path / "experiment" / "input"
    assert config.output_dir == tmp_path / "experiment" / "output"


def test_runtime_config_defaults_to_ocr_vlm_when_mode_is_unset(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.delenv("EVENTS_DIARY_UNDERSTANDING_MODE", raising=False)

    config = load_runtime_config(project_root=tmp_path, env_file=env_file, override=True)

    assert config.understanding_mode is UnderstandingMode.OCR_VLM


def test_runtime_config_rejects_invalid_understanding_mode(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("EVENTS_DIARY_UNDERSTANDING_MODE=foo\n", encoding="utf-8")

    with pytest.raises(ValueError, match="EVENTS_DIARY_UNDERSTANDING_MODE must be one of"):
        load_runtime_config(project_root=tmp_path, env_file=env_file, override=True)

def test_ticket_subregions_fit_inside_ticket_zone():
    config = UIConfig()

    subregions = (
        config.ticket_header,
        config.aircraft_image,
        config.ticket_scroll_body,
        config.ticket_bottom_actions,
    )

    for roi in subregions:
        assert roi.x >= config.ticket_zone.x
        assert roi.y >= config.ticket_zone.y
        assert roi.right <= config.ticket_zone.right
        assert roi.bottom <= config.ticket_zone.bottom


@pytest.mark.parametrize("field", ["x", "y", "width", "height"])
@pytest.mark.parametrize("value", [1.5, "1", None, True])
def test_roi_rejects_noninteger_geometry(field, value):
    geometry = {"x": 0, "y": 0, "width": 10, "height": 10}
    geometry[field] = value
    with pytest.raises(TypeError, match="must be an integer"):
        ROI(**geometry)


@pytest.mark.parametrize(
    "field,value",
    [("x", -1), ("y", -1), ("width", 0), ("width", -1),
     ("height", 0), ("height", -1)],
)
def test_roi_preserves_geometry_invariants(field, value):
    geometry = {"x": 0, "y": 0, "width": 10, "height": 10}
    geometry[field] = value
    with pytest.raises(ValueError):
        ROI(**geometry)
