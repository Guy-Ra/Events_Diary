"""Synthetic geometry, guard, ownership and bounded pair-wrapper regressions."""

from dataclasses import FrozenInstanceError, asdict, fields, replace

import cv2
import numpy as np
import pytest

from events_diary.change_metrics import compute_changed_pixel_ratio, compute_mad
from events_diary.config import ROI, UIConfig
from events_diary.evidence_candidate_timeline import EvidenceCandidate
from events_diary.frame_materialization import (
    VideoFrameResolver,
    materialize_evidence_candidates,
)
from events_diary.ticket_view_similarity import (
    TicketViewComparison,
    _body,
    compare_ticket_body_crops,
    compare_ticket_view_pairs,
)


@pytest.fixture
def texture():
    """Provide reproducible nonperiodic content for real phase estimation."""
    return np.random.default_rng(17).integers(0, 256, (128, 96), dtype=np.uint8)


def phase(monkeypatch, dx=0.0, dy=0.0, response=0.9):
    """Inject a measured fit to isolate guard and support geometry from estimation."""
    monkeypatch.setattr(cv2, "phaseCorrelate", lambda *args: ((dx, dy), response))


def assert_rejected(result, reason):
    """Require retained raw measurements and absent, never fabricated residuals."""
    assert result.alignment_rejection_reason == reason
    assert not result.alignment_valid
    assert result.overlap_fraction is result.aligned_mad is result.aligned_cpr is None
    assert np.isfinite(result.raw_mad) and np.isfinite(result.raw_cpr)


def test_identical_brightness_and_threshold_semantics(texture):
    result = compare_ticket_body_crops(texture, texture)
    assert result.raw_mad == result.raw_cpr == 0
    a = np.zeros((32, 32), np.uint8)
    b = np.full_like(a, 10)
    result = compare_ticket_body_crops(a, b, attempt_alignment=False)
    assert result.raw_mad == 10 and result.raw_cpr == 0
    assert compare_ticket_body_crops(a, b, changed_pixel_threshold=9).raw_cpr == 1
    assert_rejected(result, "not_requested")


@pytest.mark.parametrize("dy", [-8, 8])
def test_real_vertical_estimation(texture, dy):
    b = cv2.warpAffine(texture, np.float32([[1, 0, 0], [0, 1, dy]]), (96, 128))
    result = compare_ticket_body_crops(texture, b)
    assert result.raw_mad > 60
    assert result.dy_pixels == pytest.approx(dy, abs=0.15)
    assert abs(result.dx_pixels) < 0.15
    assert result.alignment_valid and result.aligned_mad < 12


def test_real_horizontal_diagnostic_is_not_applied(texture):
    b = np.roll(texture, 1, axis=1)
    result = compare_ticket_body_crops(texture, b)
    assert result.dx_pixels == pytest.approx(1, abs=0.15)
    assert result.alignment_valid and result.aligned_mad > 60


@pytest.mark.parametrize("dy", [-8, 0, 8])
def test_exact_rows_exclude_padding(texture, monkeypatch, dy):
    phase(monkeypatch, dy=dy)
    b = cv2.warpAffine(
        texture, np.float32([[1, 0, 0], [0, 1, dy]]), (96, 128), borderValue=231
    )
    result = compare_ticket_body_crops(texture, b)
    assert result.aligned_mad == result.aligned_cpr == 0
    assert result.overlap_fraction == (128 - abs(dy)) / 128


@pytest.mark.parametrize("dy", [-2.5, 2.5])
def test_fractional_full_support_oracle(texture, monkeypatch, dy):
    phase(monkeypatch, dy=dy)
    b = np.flipud(texture).copy()
    matrix = np.float64([[1, 0, 0], [0, 1, -dy]])
    warped = cv2.warpAffine(b, matrix, (96, 128), flags=cv2.INTER_LINEAR)
    support = cv2.warpAffine(
        np.ones(texture.shape, np.float32), matrix, (96, 128), flags=cv2.INTER_LINEAR
    )
    mask = support == 1
    result = compare_ticket_body_crops(texture, b)
    assert result.overlap_fraction == 125 / 128
    assert result.aligned_mad == compute_mad(texture[mask, None], warped[mask, None])
    assert result.aligned_cpr == compute_changed_pixel_ratio(
        texture[mask, None], warped[mask, None]
    )


@pytest.mark.parametrize(
    "dx,dy,response,kwargs,reason",
    [
        (0, 0, 0.59, {}, "low_response"),
        (2.01, 0, 0.9, {}, "excessive_dx"),
        (0, 39, 0.9, {}, "excessive_dy"),
        (0, 16, 0.9, {"min_overlap_fraction": 0.9}, "low_overlap"),
        (0, 128, 0.9, {"max_vertical_shift_fraction": 1}, "no_overlap"),
        (float("nan"), 3, 0.9, {}, "non_finite_motion"),
        (0, float("inf"), 0.9, {}, "non_finite_motion"),
        (0, 3, float("nan"), {}, "non_finite_motion"),
        (0, 3, float("inf"), {}, "non_finite_motion"),
    ],
)
def test_rejections_keep_raw_and_finite_diagnostics(
    texture, monkeypatch, dx, dy, response, kwargs, reason
):
    phase(monkeypatch, dx, dy, response)
    result = compare_ticket_body_crops(texture, np.flipud(texture), **kwargs)
    assert_rejected(result, reason)
    assert result.dx_pixels == (dx if np.isfinite(dx) else None)
    assert result.dy_pixels == (dy if np.isfinite(dy) else None)
    assert result.alignment_response == (response if np.isfinite(response) else None)


def test_inclusive_guard_boundaries_and_unclipped_response(monkeypatch):
    a = np.zeros((100, 50), np.uint8)
    phase(monkeypatch, dx=2, dy=30, response=0.6)
    result = compare_ticket_body_crops(a, a)
    assert result.alignment_valid and result.overlap_fraction == 0.7
    phase(monkeypatch, response=1.0001)
    assert compare_ticket_body_crops(a, a).alignment_response == 1.0001


def test_phase_failure_and_tiny_crops(texture, monkeypatch):
    def fail(*args):
        """Simulate OpenCV estimator failure without invalidating raw pixels."""
        raise cv2.error("phase failed")

    monkeypatch.setattr(cv2, "phaseCorrelate", fail)
    assert_rejected(
        compare_ticket_body_crops(texture, texture), "phase_correlation_failed"
    )
    assert_rejected(
        compare_ticket_body_crops(texture[:1], texture[:1]), "crop_too_small"
    )


def test_warp_failure_retains_measured_fit(texture, monkeypatch):
    phase(monkeypatch, dy=0.5)

    def fail(*args, **kwargs):
        """Simulate a failed supplemental warp after successful phase estimation."""
        raise cv2.error("warp failed")

    monkeypatch.setattr(cv2, "warpAffine", fail)
    result = compare_ticket_body_crops(texture, texture)
    assert_rejected(result, "warp_failed")
    assert result.dy_pixels == 0.5 and result.alignment_response == 0.9


@pytest.mark.parametrize(
    "name",
    [
        "min_alignment_response",
        "max_abs_dx_pixels",
        "max_vertical_shift_fraction",
        "min_overlap_fraction",
    ],
)
@pytest.mark.parametrize(
    "value,error",
    [
        (True, TypeError),
        ("1", TypeError),
        (None, TypeError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (-0.1, ValueError),
    ],
)
def test_invalid_guards(texture, name, value, error):
    with pytest.raises(error):
        compare_ticket_body_crops(texture, texture, **{name: value})


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"min_overlap_fraction": 0}, ValueError),
        ({"min_overlap_fraction": 1.01}, ValueError),
        ({"max_vertical_shift_fraction": 1.01}, ValueError),
        ({"attempt_alignment": 1}, TypeError),
        ({"changed_pixel_threshold": True}, TypeError),
        ({"changed_pixel_threshold": 256}, ValueError),
    ],
)
def test_other_invalid_parameters(texture, kwargs, error):
    with pytest.raises(error):
        compare_ticket_body_crops(texture, texture, **kwargs)


@pytest.mark.parametrize(
    "b",
    [
        np.empty((0, 2), np.uint8),
        np.zeros((5, 5), np.uint8),
        np.zeros((128, 96), np.float32),
        np.zeros((128, 96, 4), np.uint8),
    ],
)
def test_invalid_crops(texture, b):
    with pytest.raises((TypeError, ValueError)):
        compare_ticket_body_crops(texture, b)


def test_immutable_deterministic_records_and_inputs(texture):
    a = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    before = a.copy()
    a.setflags(write=False)
    measured = compare_ticket_body_crops(a, a)
    assert measured == compare_ticket_body_crops(a, a)
    assert np.array_equal(a, before)
    result = TicketViewComparison(3, 5, **asdict(measured))
    with pytest.raises(FrozenInstanceError):
        result.raw_mad = 4
    assert {f.name for f in fields(result)} == {
        "candidate_a_index",
        "candidate_b_index",
        "raw_mad",
        "raw_cpr",
        "dx_pixels",
        "dy_pixels",
        "alignment_response",
        "overlap_fraction",
        "aligned_mad",
        "aligned_cpr",
        "alignment_rejection_reason",
    }
    assert all(not isinstance(value, np.ndarray) for value in asdict(result).values())


@pytest.mark.parametrize("width,height,box", [
    (1920, 1080, (16, 386, 336, 566)),
    (1280, 720, (11, 257, 224, 378)),
])
def test_canonical_and_proportional_body_mapping(width, height, box):
    x, y, w, h = box
    frame = np.zeros((height, width, 3), np.uint8)
    frame[y:y+h, x:x+w] = 111
    crop = _body(frame, UIConfig())
    assert crop.shape == (h, w, 3)
    assert np.all(crop == 111)


def test_unsupported_layout_rejected():
    with pytest.raises(ValueError):
        _body(np.zeros((400, 400, 3), np.uint8), UIConfig())


@pytest.fixture(params=[1, 2])
def source(tmp_path, request):
    """Encode map-only changes outside a ticket ROI at reference/proportional sizes."""
    scale = request.param
    path = tmp_path / "roi.avi"
    config = UIConfig(
        reference_width=64, reference_height=48, ticket_scroll_body=ROI(8, 8, 16, 24)
    )
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64 * scale, 48 * scale)
    )
    assert writer.isOpened()
    try:
        for value in (0, 120, 240):
            frame = np.full((48 * scale, 64 * scale, 3), value, np.uint8)
            frame[8 * scale : 32 * scale, 8 * scale : 24 * scale] = 64
            writer.write(frame)
    finally:
        writer.release()
    candidates = tuple(
        EvidenceCandidate(0, t, ("dense_sampling",), (), None, False)
        for t in (0.0, 0.1, 0.2)
    )
    return path, config, candidates


def test_batch_roi_order_duplicates_cache_and_resource_ownership(source):
    path, config, candidates = source
    with VideoFrameResolver(path, max_frames=3) as resolver:
        materialized = materialize_evidence_candidates(
            video_path=path, candidates=candidates, ui_config=config, resolver=resolver
        )
        snapshot = tuple(asdict(c) for c in materialized)
        attempts = resolver.decode_attempts
        pairs = [(1, 0), (0, 2), (1, 0)]
        results = compare_ticket_view_pairs(
            video_path=path,
            materialized_candidates=materialized,
            pairs=pairs,
            ui_config=config,
            resolver=resolver,
            attempt_alignment=False,
        )
        assert [(r.candidate_a_index, r.candidate_b_index) for r in results] == pairs
        assert results[0] == results[2] and len(results) == 3
        assert all(r.raw_mad < 2 for r in results)
        assert resolver.decode_attempts == attempts and resolver.cache_hits >= 6
        assert tuple(asdict(c) for c in materialized) == snapshot
        assert resolver.resolve(0).image is not None
        changed_quality = tuple(
            replace(c, ticket_brightness_mean=-999) for c in materialized
        )
        assert results == compare_ticket_view_pairs(
            video_path=path,
            materialized_candidates=changed_quality,
            pairs=pairs,
            ui_config=config,
            resolver=resolver,
            attempt_alignment=False,
        )
    assert resolver.cache_size == 0
    with pytest.raises(RuntimeError):
        resolver.resolve(0)


def test_owned_resolver_released_on_success_and_error(source, monkeypatch):
    path, config, candidates = source
    materialized = materialize_evidence_candidates(
        video_path=path, candidates=candidates, ui_config=config
    )
    closed = []
    original = VideoFrameResolver.close

    def close(self):
        """Observe production cleanup while preserving its actual behavior."""
        original(self)
        closed.append(self)

    monkeypatch.setattr(VideoFrameResolver, "close", close)
    compare_ticket_view_pairs(
        video_path=path,
        materialized_candidates=materialized,
        pairs=[(0, 1)],
        ui_config=config,
    )
    assert len(closed) == 1 and closed[-1].cache_size == 0
    wrong = (replace(materialized[0], frame_index=2), *materialized[1:])
    with pytest.raises(ValueError, match="disagrees"):
        compare_ticket_view_pairs(
            video_path=path,
            materialized_candidates=wrong,
            pairs=[(0, 1)],
            ui_config=config,
        )
    assert len(closed) == 2 and closed[-1].cache_size == 0


@pytest.mark.parametrize(
    "pair,error",
    [
        ((-1, 0), ValueError),
        ((0, 3), ValueError),
        ((True, 0), TypeError),
        ((0.5, 0), TypeError),
        ((0,), TypeError),
        ("01", TypeError),
    ],
)
def test_batch_index_validation(source, pair, error):
    path, config, candidates = source
    materialized = materialize_evidence_candidates(
        video_path=path, candidates=candidates, ui_config=config
    )
    with pytest.raises(error):
        compare_ticket_view_pairs(
            video_path=path,
            materialized_candidates=materialized,
            pairs=[pair],
            ui_config=config,
        )


def test_unresolved_and_empty_pairs(source):
    path, config, candidates = source
    materialized = materialize_evidence_candidates(
        video_path=path, candidates=candidates, ui_config=config
    )
    assert (
        compare_ticket_view_pairs(
            video_path=path,
            materialized_candidates=materialized,
            pairs=[],
            ui_config=config,
        )
        == ()
    )
    unresolved = (
        replace(materialized[0], resolution_status="decode_failed", frame_index=None),
    )
    with pytest.raises(ValueError, match="resolved"):
        compare_ticket_view_pairs(
            video_path=path,
            materialized_candidates=unresolved,
            pairs=[(0, 0)],
            ui_config=config,
        )
