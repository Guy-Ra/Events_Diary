"""Focused synthetic tests of physical resolution, image ownership and raw diagnostics."""

from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from events_diary.config import ROI, UIConfig
from events_diary.evidence_candidate_timeline import EvidenceCandidate
from events_diary.frame_materialization import (
    MaterializedEvidenceCandidate,
    ResolvedVideoFrame,
    TicketQualityFeatures,
    VideoFrameResolver,
    materialize_evidence_candidates,
    measure_ticket_quality,
    resolve_video_frame,
)
from events_diary.video import VideoMetadata

CONFIG = UIConfig(reference_width=64, reference_height=48, ticket_scroll_body=ROI(8, 8, 16, 24))


def _candidate(time, index=0):
    """Supply existing semantic records without adding frame-materialization decisions."""
    return EvidenceCandidate(index, time, ("dense_sampling",), ("ticket_scroll",), None, False)


class FakeCapture:
    """Expose controllable decoder outcomes and observable reads/releases."""

    def __init__(self):
        self.index = 0
        self.reads = []
        self.releases = 0
        self.failures = set()
        self.opened = True
        self.shift = 0
        self.error = None
        self.seek_ok = True
        self.fps = 10

    def isOpened(self):
        return self.opened

    def set(self, prop, index):
        assert prop == cv2.CAP_PROP_POS_FRAMES
        self.index = index
        return self.seek_ok

    def read(self):
        if self.error:
            raise self.error
        self.reads.append(self.index)
        if self.index in self.failures:
            return False, None
        return True, np.full((48, 64, 3), self.index % 256, dtype=np.uint8)

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: 20 if self.fps == 10 else 200,
                cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48,
                cv2.CAP_PROP_POS_FRAMES: self.index + 1 + self.shift}[prop]

    def release(self):
        self.releases += 1


@pytest.fixture
def fake_source(tmp_path, monkeypatch):
    """Create a supplied-metadata source without requiring a real codec."""
    path = tmp_path / "source.mp4"
    path.touch()
    metadata = VideoMetadata(path, 2, 10, 20, 64, 48)
    capture = FakeCapture()
    monkeypatch.setattr("events_diary.frame_materialization.cv2.VideoCapture", lambda _: capture)
    return path, metadata, capture


@pytest.fixture
def encoded_source(tmp_path):
    """Encode a small frame-number intensity ramp for actual decoder-index checks."""
    path = tmp_path / "ramp.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    try:
        for index in range(20):
            writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
    finally:
        writer.release()
    return path


@pytest.mark.parametrize("requested,index,status", [
    (0, 0, "resolved"), (.0001, 0, "resolved"), (.049, 0, "resolved"), (.05, 1, "resolved"), (.1, 1, "resolved"),
    (.149, 1, "resolved"), (.15, 2, "resolved"), (.151, 2, "resolved"),
    (.25, 3, "resolved"), (1.9, 19, "resolved"), (1.9999, 19, "resolved"),
    (2, 19, "video_end_adjusted"),
])
def test_real_decoder_nearest_and_endpoint(encoded_source, requested, index, status):
    result = resolve_video_frame(encoded_source, requested)
    assert result.requested_timestamp_seconds == requested
    assert result.frame_index == index
    assert result.actual_timestamp_seconds == index / 10
    assert result.resolution_status == status
    assert abs(float(result.image.mean()) - index * 10) < 2
    assert not result.image.flags.writeable


def test_decimal_half_tie_avoids_product_rounding(fake_source):
    path, metadata, capture = fake_source
    metadata = replace(metadata, fps=100, frame_count=200)
    capture.fps = 100
    assert 1.005 * 100 < 100.5
    result = resolve_video_frame(path, 1.005, metadata=metadata)
    assert result.frame_index == 101
    assert capture.reads == [101]


def test_models_frozen_and_materialized_fields_are_lightweight(fake_source):
    path, metadata, _ = fake_source
    resolved = resolve_video_frame(path, .1, metadata=metadata)
    materialized = MaterializedEvidenceCandidate(_candidate(2), "video_end_adjusted", 19, 1.9, 0., 42., 0.)
    for model, attr in ((resolved, "frame_index"), (materialized, "frame_index"),
                        (TicketQualityFeatures(1., 2., 3.), "ticket_brightness_mean")):
        with pytest.raises(FrozenInstanceError):
            setattr(model, attr, 0)
    assert [f.name for f in fields(materialized)] == [
        "evidence_candidate", "resolution_status", "frame_index", "actual_timestamp_seconds",
        "ticket_sharpness_laplacian_variance", "ticket_brightness_mean", "ticket_contrast_std"]
    assert materialized.requested_timestamp_seconds == 2
    assert materialized.timestamp_error_seconds == pytest.approx(.1)
    assert all(not isinstance(getattr(materialized, f.name), np.ndarray) for f in fields(materialized))


@pytest.mark.parametrize("time,error", [(-.001, ValueError), (2.0000000001, ValueError),
    (float("nan"), ValueError), (float("inf"), ValueError), (-float("inf"), ValueError),
    (True, TypeError), (False, TypeError), (".1", TypeError), (None, TypeError)])
def test_invalid_timestamp_rejected_and_resources_released(fake_source, time, error):
    path, metadata, capture = fake_source
    with pytest.raises(error):
        resolve_video_frame(path, time, metadata=metadata)
    assert capture.reads == []
    assert capture.releases == 1


@pytest.mark.parametrize("field,value,error", [
    ("fps", 0, ValueError), ("fps", float("nan"), ValueError), ("fps", float("inf"), ValueError),
    ("fps", True, TypeError), ("fps", "10", TypeError),
    ("frame_count", 0, ValueError), ("frame_count", -1, ValueError), ("frame_count", True, TypeError),
    ("frame_count", 20., TypeError), ("width", 0, ValueError), ("height", -1, ValueError),
    ("width", True, TypeError), ("height", 48., TypeError),
    ("duration_seconds", 3, ValueError), ("duration_seconds", float("inf"), ValueError),
    ("duration_seconds", True, TypeError), ("path", Path("other.mp4"), ValueError),
])
def test_invalid_metadata_rejected_before_owned_decoder(fake_source, field, value, error):
    path, metadata, capture = fake_source
    with pytest.raises(error):
        VideoFrameResolver(path, metadata=replace(metadata, **{field: value}))
    assert capture.reads == [] and capture.releases == 0


@pytest.mark.parametrize("bound,error", [(0, ValueError), (-1, ValueError),
    (True, TypeError), (1.5, TypeError), (None, TypeError)])
def test_invalid_cache_bound(fake_source, bound, error):
    path, metadata, _ = fake_source
    with pytest.raises(error):
        VideoFrameResolver(path, metadata=metadata, max_frames=bound)
    with pytest.raises(error):
        materialize_evidence_candidates(video_path=path, candidates=[], ui_config=CONFIG, max_frames=bound)


def test_empty_candidates_dont_open_video(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("No decoder needed for empty input")
    monkeypatch.setattr("events_diary.frame_materialization.cv2.VideoCapture", fail)
    assert materialize_evidence_candidates(video_path=Path("unused.mp4"), candidates=[], ui_config=CONFIG) == ()


def test_batch_preserves_order_identity_duplicates_and_inputs(fake_source):
    path, metadata, capture = fake_source
    candidates = [_candidate(.3), _candidate(.1, 1), _candidate(.3, 2), _candidate(.31, 3)]
    original = tuple(candidates)
    with VideoFrameResolver(path, metadata=metadata) as resolver:
        result = materialize_evidence_candidates(video_path=path, candidates=candidates,
                                                  ui_config=CONFIG, resolver=resolver)
        assert [r.frame_index for r in result] == [3, 1, 3, 3]
        assert all(r.evidence_candidate is c for r, c in zip(result, candidates))
        assert len(result) == len(candidates)
        assert resolver.cache_hits == 2 and resolver.cache_misses == 2
        assert capture.releases == 0
    assert tuple(candidates) == original
    assert capture.reads == [3, 1]
    assert capture.releases == 1


def test_unresolved_not_dropped_or_given_fake_quality(fake_source):
    path, metadata, capture = fake_source
    capture.failures = {1}
    result = materialize_evidence_candidates(video_path=path, candidates=[_candidate(.1), _candidate(.2)],
                                             ui_config=CONFIG, metadata=metadata)
    failed = result[0]
    assert len(result) == 2 and failed.resolution_status == "unresolved"
    assert failed.frame_index is None and failed.actual_timestamp_seconds is None
    assert failed.ticket_sharpness_laplacian_variance is None
    assert failed.ticket_brightness_mean is None and failed.ticket_contrast_std is None
    assert failed.timestamp_error_seconds is None
    assert result[1].resolution_status == "resolved"
    assert capture.releases == 1


def test_endpoint_searches_last_decodable_only_for_nominal_end(fake_source):
    path, metadata, capture = fake_source
    capture.failures = {18, 19}
    with VideoFrameResolver(path, metadata=metadata) as resolver:
        assert resolver.resolve(1.99).resolution_status == "unresolved"
        frame = resolver.resolve(2)
        assert frame.frame_index == 17 and frame.actual_timestamp_seconds == 1.7
        assert frame.requested_timestamp_seconds == 2 and frame.resolution_status == "video_end_adjusted"
    assert capture.reads == [19, 19, 18, 17]


def test_no_decodable_endpoint_is_explicit_unresolved(fake_source):
    path, metadata, capture = fake_source
    capture.failures = set(range(20))
    result = resolve_video_frame(path, 2, metadata=metadata)
    assert result == ResolvedVideoFrame(2, "unresolved", None, None, None)
    assert capture.releases == 1


def test_lru_is_bounded_and_hits_refresh_recency(fake_source):
    path, metadata, capture = fake_source
    with VideoFrameResolver(path, metadata=metadata, max_frames=2) as resolver:
        first = resolver.resolve(0)
        resolver.resolve(.1)
        assert resolver.resolve(0).image is first.image
        resolver.resolve(.2)  # Evicts frame1, not the recently accessed frame0.
        assert resolver.cache_size == 2
        resolver.resolve(.1)
        assert resolver.cache_size == 2
        assert resolver.decode_attempts == 4 and resolver.cache_hits == 1
    assert capture.reads == [0, 1, 2, 1]
    assert resolver.cache_size == 0
    assert first.image[0, 0, 0] == 0  # Externally retained handle survives cache eviction/close.
    with pytest.raises(ValueError):
        first.image[0, 0, 0] = 255
    copy = first.image.copy()
    copy[0, 0, 0] = 255
    assert first.image[0, 0, 0] == 0


def test_decoder_position_disagreement_is_not_silent_fallback(fake_source):
    path, metadata, capture = fake_source
    capture.shift = -1
    with pytest.raises(RuntimeError, match="disagrees"):
        resolve_video_frame(path, .1, metadata=metadata)
    assert capture.releases == 1


def test_failed_seek_is_unresolved(fake_source):
    path, metadata, capture = fake_source
    capture.seek_ok = False
    assert resolve_video_frame(path, .1, metadata=metadata).resolution_status == "unresolved"
    assert capture.releases == 1


def test_opencv_decode_failure_is_unresolved(fake_source):
    path, metadata, capture = fake_source
    capture.error = cv2.error("failed codec read")
    result = materialize_evidence_candidates(video_path=path, candidates=[_candidate(.1)],
                                              ui_config=CONFIG, metadata=metadata)
    assert len(result) == 1 and result[0].resolution_status == "unresolved"
    assert capture.releases == 1


@pytest.mark.parametrize("opening", [True, False])
def test_resources_released_on_open_or_read_exception(fake_source, opening):
    path, metadata, capture = fake_source
    if opening:
        capture.opened = False
    else:
        capture.error = RuntimeError("decoder error")
    with pytest.raises((RuntimeError, ValueError)):
        resolve_video_frame(path, 0, metadata=metadata)
    assert capture.releases == 1


def test_batch_validates_all_timestamps_before_decoding(fake_source):
    path, metadata, capture = fake_source
    with pytest.raises(ValueError):
        materialize_evidence_candidates(video_path=path, candidates=[_candidate(0), _candidate(3)],
                                         ui_config=CONFIG, metadata=metadata)
    assert capture.reads == [] and capture.releases == 1


def test_shared_resolver_must_match_source_and_remain_active(fake_source):
    path, metadata, capture = fake_source
    with VideoFrameResolver(path, metadata=metadata) as resolver, pytest.raises(ValueError):
        materialize_evidence_candidates(video_path=Path("other.mp4"), candidates=[],
                                         ui_config=CONFIG, resolver=resolver)
    with pytest.raises(RuntimeError):
        resolver.resolve(0)
    assert capture.releases == 1


def test_quality_checkerboard_has_known_raw_values():
    gray = (np.indices((48, 64)).sum(axis=0) % 2 * 255).astype(np.uint8)
    image = np.repeat(gray[:, :, None], 3, axis=2)
    before = image.copy()
    quality = measure_ticket_quality(image, CONFIG)
    assert quality.ticket_sharpness_laplacian_variance == 1020.0**2
    assert quality.ticket_brightness_mean == 127.5
    assert quality.ticket_contrast_std == 127.5
    assert all(isinstance(getattr(quality, f.name), float) for f in fields(quality))
    assert np.array_equal(image, before)


def test_constant_pixels_remain_raw_not_normalized():
    quality = measure_ticket_quality(np.full((48, 64, 3), 200, dtype=np.uint8), CONFIG)
    assert quality == TicketQualityFeatures(0., 200., 0.)


def test_quality_proportional_layout_mapping_uses_body_only():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Independently known mapped canonical body bounds, rounded shared edges.
    image[257:635, 11:235] = 200
    assert measure_ticket_quality(image, UIConfig()) == TicketQualityFeatures(0., 200., 0.)


@pytest.mark.parametrize("image,error", [
    (None, TypeError), (np.zeros((48, 64, 3), dtype=float), TypeError),
    (np.zeros((0, 64, 3), dtype=np.uint8), ValueError),
    (np.zeros((48, 64), dtype=np.uint8), ValueError),
    (np.zeros((48, 64, 4), dtype=np.uint8), ValueError),
])
def test_quality_rejects_invalid_images(image, error):
    with pytest.raises(error):
        measure_ticket_quality(image, CONFIG)


@pytest.mark.parametrize("config,shape,error", [
    (UIConfig(), (100, 100, 3), ValueError),
    (replace(CONFIG, ticket_scroll_body=ROI(64, 0, 10, 10)), (48, 64, 3), ValueError),
    (replace(CONFIG, ticket_scroll_body=None), (48, 64, 3), TypeError),
    (UIConfig(ticket_scroll_body=ROI(1, 1, 1, 1)), (9, 16, 3), ValueError),
])
def test_quality_rejects_unsupported_or_empty_mapped_regions(config, shape, error):
    with pytest.raises(error):
        measure_ticket_quality(np.zeros(shape, dtype=np.uint8), config)


def test_batch_roi_error_releases_decoder(fake_source):
    path, metadata, capture = fake_source
    with pytest.raises(ValueError):
        materialize_evidence_candidates(video_path=path, candidates=[_candidate(0)],
                                         ui_config=UIConfig(), metadata=metadata)
    assert capture.releases == 1


def test_supplied_metadata_must_match_actual_decoder(fake_source):
    path, metadata, capture = fake_source
    with pytest.raises(ValueError, match="Decoder metadata"):
        resolve_video_frame(path, 0, metadata=replace(metadata, fps=20, duration_seconds=1))
    assert capture.reads == [] and capture.releases == 1


def test_missing_source_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_video_frame(tmp_path / "missing.mp4", 0)


@pytest.mark.parametrize("candidates,error", [([None], TypeError), (iter([]), TypeError),
    ([_candidate(True)], TypeError), ([_candidate(float("nan"))], ValueError)])
def test_invalid_candidate_inputs(fake_source, candidates, error):
    path, metadata, capture = fake_source
    with pytest.raises(error):
        materialize_evidence_candidates(video_path=path, candidates=candidates, ui_config=CONFIG, metadata=metadata)
    assert capture.reads == [] and capture.releases == 0
