"""Tests for Events Diary video metadata structures."""

from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from events_diary.video import (
    SampledFrame,
    VideoMetadata,
    generate_sample_timestamps,
    read_video_metadata,
    sample_video,
)


def test_video_metadata_stores_expected_values() -> None:
    """Verify that VideoMetadata stores the supplied video properties."""

    metadata = VideoMetadata(
        path=Path("example.mp4"),
        duration_seconds=10.5,
        fps=30.0,
        frame_count=315,
        width=1920,
        height=1080,
    )

    assert metadata.path == Path("example.mp4")
    assert metadata.duration_seconds == 10.5
    assert metadata.fps == 30.0
    assert metadata.frame_count == 315
    assert metadata.width == 1920
    assert metadata.height == 1080


def test_read_video_metadata_raises_for_missing_file(tmp_path) -> None:
    """Verify that a missing video file produces a clear error."""

    missing_path = tmp_path / "missing.mp4"

    with pytest.raises(FileNotFoundError):
        read_video_metadata(missing_path)


def test_generate_sample_timestamps() -> None:
    """Verify that sampling timestamps are generated at the expected interval."""

    timestamps = generate_sample_timestamps(
        duration_seconds=2.0,
        sample_fps=2.0,
    )

    assert timestamps == [0.0, 0.5, 1.0, 1.5]


def test_generate_sample_timestamps_rejects_invalid_sample_fps() -> None:
    """Verify that non-positive sampling rates are rejected."""

    with pytest.raises(ValueError):
        generate_sample_timestamps(
            duration_seconds=10.0,
            sample_fps=0.0,
        )

def test_sample_video_returns_expected_samples(tmp_path) -> None:
    """Verify that video sampling returns timestamped frames."""

    video_path = tmp_path / "test_video.avi"

    width = 64
    height = 48
    fps = 10.0
    frame_count = 20

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )

    for index in range(frame_count):
        frame = np.full(
            (height, width, 3),
            index,
            dtype=np.uint8,
        )
        writer.write(frame)

    writer.release()

    samples = sample_video(
        video_path,
        sample_fps=2.0,
    )

    assert len(samples) == 4
    assert all(isinstance(sample, SampledFrame) for sample in samples)

    assert samples[0].requested_timestamp_seconds == 0.0
    assert samples[1].requested_timestamp_seconds == 0.5
    assert samples[2].requested_timestamp_seconds == 1.0
    assert samples[3].requested_timestamp_seconds == 1.5

def test_generate_sample_timestamps_rejects_negative_duration() -> None:
    """Verify that a negative video duration is rejected."""

    with pytest.raises(ValueError):
        generate_sample_timestamps(
            duration_seconds=-1.0,
            sample_fps=2.0,
        )


def test_generate_sample_timestamps_returns_empty_list_for_zero_duration() -> None:
    """Verify that a zero-duration video produces no sampling timestamps."""

    timestamps = generate_sample_timestamps(
        duration_seconds=0.0,
        sample_fps=2.0,
    )

    assert timestamps == []


def test_sample_video_caps_sampling_rate_to_source_fps(tmp_path) -> None:
    """Verify that sampling is capped when requested FPS exceeds source FPS."""

    video_path = tmp_path / "test_video.avi"

    width = 64
    height = 48
    fps = 10.0
    frame_count = 20

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )

    for index in range(frame_count):
        frame = np.full(
            (height, width, 3),
            index,
            dtype=np.uint8,
        )
        writer.write(frame)

    writer.release()

    samples = sample_video(
        video_path,
        sample_fps=20.0,
    )

    assert len(samples) == 20

def test_sample_video_logs_warning_when_sampling_rate_is_capped(
    tmp_path,
    caplog,
) -> None:
    """Verify that a warning is logged when requested FPS exceeds source FPS."""

    video_path = tmp_path / "test_video.avi"

    width = 64
    height = 48
    fps = 10.0
    frame_count = 20

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )

    for index in range(frame_count):
        frame = np.full(
            (height, width, 3),
            index,
            dtype=np.uint8,
        )
        writer.write(frame)

    writer.release()

    with caplog.at_level("WARNING"):
        sample_video(
            video_path,
            sample_fps=20.0,
        )

    assert "exceeds source rate" in caplog.text
    assert "Using 10.000 FPS instead" in caplog.text


@pytest.mark.parametrize("duration,fps,count", [(1.0, 10.0, 10), (2.0, 10.0, 20), (1.05, 10.0, 11)])
def test_generate_timestamps_uses_integer_indices(duration, fps, count):
    timestamps = generate_sample_timestamps(duration, fps)
    assert timestamps == [index / fps for index in range(count)]
    assert all(timestamp < duration for timestamp in timestamps)


@pytest.mark.parametrize("duration", [float("inf"), float("nan"), -float("inf")])
def test_generate_sample_timestamps_rejects_nonfinite_duration(duration):
    with pytest.raises(ValueError, match="Video duration must be finite"):
        generate_sample_timestamps(duration, 2.0)


@pytest.mark.parametrize("fps", [float("inf"), float("nan"), -float("inf")])
@pytest.mark.parametrize("duration", [0.0, 1.0])
def test_generate_sample_timestamps_rejects_nonfinite_fps(fps, duration):
    with pytest.raises(ValueError, match="Sample FPS must be finite"):
        generate_sample_timestamps(duration, fps)


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_metadata_rejects_invalid_fps_and_releases_capture(monkeypatch, fps):
    capture = Mock()
    capture.get.return_value = fps
    monkeypatch.setattr(cv2, "VideoCapture", Mock(return_value=capture))
    with pytest.raises(ValueError, match="invalid FPS"):
        read_video_metadata(Path(__file__))
    capture.release.assert_called_once_with()


def test_metadata_releases_capture_after_success(monkeypatch):
    capture = Mock()
    capture.get.side_effect = [10.0, 20.0, 64.0, 48.0]
    monkeypatch.setattr(cv2, "VideoCapture", Mock(return_value=capture))
    metadata = read_video_metadata(Path(__file__))
    assert metadata.duration_seconds == 2.0
    assert (metadata.width, metadata.height) == (64, 48)
    capture.release.assert_called_once_with()


@pytest.mark.parametrize("failure", [RuntimeError("property read failed"), float("nan")])
def test_metadata_releases_capture_after_property_failure(monkeypatch, failure):
    capture = Mock()
    capture.get.side_effect = [10.0, failure]
    monkeypatch.setattr(cv2, "VideoCapture", Mock(return_value=capture))
    with pytest.raises((RuntimeError, ValueError)):
        read_video_metadata(Path(__file__))
    capture.release.assert_called_once_with()


@pytest.mark.parametrize("sampling", [False, True])
def test_video_releases_capture_after_open_failure(monkeypatch, sampling):
    capture = Mock()
    capture.isOpened.return_value = False
    monkeypatch.setattr(cv2, "VideoCapture", Mock(return_value=capture))
    if sampling:
        monkeypatch.setattr("events_diary.video.read_video_metadata", Mock(
            return_value=VideoMetadata(Path(__file__), 1.0, 10.0, 10, 64, 48)))
    with pytest.raises(ValueError, match="Unable to open"):
        if sampling:
            sample_video(Path(__file__), 2.0)
        else:
            read_video_metadata(Path(__file__))
    capture.release.assert_called_once_with()


def test_sampling_releases_capture_after_read_exception(monkeypatch):
    capture = Mock()
    capture.read.side_effect = RuntimeError("read failed")
    monkeypatch.setattr(cv2, "VideoCapture", Mock(return_value=capture))
    monkeypatch.setattr("events_diary.video.read_video_metadata", Mock(
        return_value=VideoMetadata(Path(__file__), 1.0, 10.0, 10, 64, 48)))
    with pytest.raises(RuntimeError, match="read failed"):
        sample_video(Path(__file__), 2.0)
    capture.release.assert_called_once_with()
