"""Video metadata and basic video access utilities.

This module contains lightweight structures and functions for reading
basic information from video files.

At this stage, the module is intentionally limited to metadata
inspection. Frame sampling will be added later in Phase 1.
"""
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class VideoMetadata:
    """Basic metadata describing a source video.

    Attributes:
        path:
            Path to the source video file.

        duration_seconds:
            Total video duration in seconds.

        fps:
            Frames per second reported by the video container.

        frame_count:
            Total number of frames reported by the video container.

        width:
            Video frame width in pixels.

        height:
            Video frame height in pixels.

    The object is immutable because it represents properties of the
    source video that should not change during processing.
    """

    path: Path
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True)
class SampledFrame:
    """A single video frame sampled at a requested point in time.

    Attributes:
        requested_timestamp_seconds:
            Timestamp that the sampler asked to retrieve.

        actual_timestamp_seconds:
            Timestamp reported for the frame that was actually returned.

        frame_index:
            Index of the returned frame within the source video.

        image:
            Video frame stored as a NumPy array in OpenCV's native
            BGR image format.

    The object is immutable so that timing metadata associated with a
    sampled frame remains stable throughout the processing pipeline.
    """

    requested_timestamp_seconds: float
    actual_timestamp_seconds: float
    frame_index: int
    image: np.ndarray

def read_video_metadata(path: Path) -> VideoMetadata:
    """Read basic metadata from a video file.

    Args:
        path:
            Path to the source video file.

    Returns:
        VideoMetadata:
            Structured metadata describing the video.

    Raises:
        FileNotFoundError:
            If the supplied video path does not exist.

        ValueError:
            If the video cannot be opened or reports an invalid FPS value.
    """

    if not path.exists():
        raise FileNotFoundError(f"Video file does not exist: {path}")

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    capture.release()

    if fps <= 0:
        raise ValueError(f"Video reports an invalid FPS value: {fps}")

    duration_seconds = frame_count / fps

    return VideoMetadata(
        path=path,
        duration_seconds=duration_seconds,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def generate_sample_timestamps(duration_seconds: float,sample_fps: float,) -> list[float]:

    """Generate evenly spaced sampling timestamps for a video.

    Args:
        duration_seconds:
            Total duration of the video in seconds.

        sample_fps:
            Desired number of samples per second.

    Returns:
        list[float]:
            Sampling timestamps starting at 0.0 seconds and remaining
            strictly inside the video duration.

    Raises:
        ValueError:
            If the duration is negative or if sample_fps is not positive.
    """

    if duration_seconds < 0:
        raise ValueError("Video duration cannot be negative.")

    if sample_fps <= 0:
        raise ValueError("Sample FPS must be greater than zero.")

    interval_seconds = 1.0 / sample_fps

    timestamps: list[float] = []
    timestamp = 0.0

    while timestamp < duration_seconds:
        timestamps.append(timestamp)
        timestamp += interval_seconds

    return timestamps


def sample_video(path: Path,sample_fps: float,) -> list[SampledFrame]:
    """Sample frames from a video at evenly spaced time intervals.

    Args:
        path:
            Path to the source video file.

        sample_fps:
            Desired number of sampled frames per second.

    Returns:
        list[SampledFrame]:
            Frames successfully retrieved from the video, together with
            their requested timestamps, actual timestamps, and frame indices.

    Raises:
        FileNotFoundError:
            If the supplied video path does not exist.

        ValueError:
            If the video cannot be opened, reports invalid metadata,
            or if sample_fps is not positive.

    If the requested sampling rate exceeds the source video FPS, the
    sampling rate is capped at the source FPS and a warning is logged.

    Failed frame reads are skipped and reported through logging rather
    than terminating the entire sampling operation.
    """

    if sample_fps <= 0:
        raise ValueError("Sample FPS must be greater than zero.")

    metadata = read_video_metadata(path)

    effective_sample_fps = min(sample_fps, metadata.fps)

    if sample_fps > metadata.fps:
        logger.warning(
            "Requested sampling rate %.3f FPS exceeds source rate %.3f FPS. "
            "Using %.3f FPS instead.",
            sample_fps,
            metadata.fps,
            effective_sample_fps,
        )

    timestamps = generate_sample_timestamps(
        duration_seconds=metadata.duration_seconds,
        sample_fps=effective_sample_fps,
    )

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {path}")

    sampled_frames: list[SampledFrame] = []

    try:
        for requested_timestamp in timestamps:
            capture.set(
                cv2.CAP_PROP_POS_MSEC,
                requested_timestamp * 1000.0,
            )

            success, frame = capture.read()

            if not success:
                logger.warning(
                    "Failed to read frame near %.3f seconds from %s",
                    requested_timestamp,
                    path,
                )
                continue

            actual_timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            frame_index = max(
                0,
                int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1,
            )

            sampled_frames.append(
                SampledFrame(
                    requested_timestamp_seconds=requested_timestamp,
                    actual_timestamp_seconds=actual_timestamp,
                    frame_index=frame_index,
                    image=frame,
                )
            )

    finally:
        capture.release()

    return sampled_frames