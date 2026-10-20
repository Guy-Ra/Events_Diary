"""Isolated raw OCR, physical provenance, role and geometry tests; no OCR downloads."""

import builtins
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict, fields, replace
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from events_diary import ticket_ocr as ocr
from events_diary.config import ROI, UIConfig
from events_diary.evidence_candidate_timeline import EvidenceCandidate
from events_diary.frame_materialization import (
    MaterializedEvidenceCandidate,
    ResolvedVideoFrame,
    VideoFrameResolver,
)
from events_diary.representative_selection import RepresentativeSelection
from events_diary.video import VideoMetadata

CONFIG = UIConfig(reference_width=64, reference_height=48,
    ticket_zone=ROI(8, 4, 16, 40), ticket_bottom_actions=ROI(8, 36, 16, 8))
QUAD = [[2., 4.], [18., 2.], [20., 14.], [4., 16.]]


@pytest.fixture
def native(monkeypatch):
    """Replace only optional OCR construction, retaining all production processing."""
    class Native:
        instances: ClassVar[list] = []
        output: ClassVar[list | None] = [(QUAD, "  AGE (Mar 2010): O0 I1\n", .987), (QUAD, "16 years", .95)]

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.images = []
            self.instances.append(self)

        def __call__(self, image):
            self.images.append(image.copy())
            return self.output, [0.1, 0.2, 0.3]

    monkeypatch.setattr(ocr, "_load_rapidocr", lambda: (Native, "RapidOCR/test CPU"))
    return Native


@pytest.fixture
def image():
    """Use unequal color channels and spatial texture to expose altered processing."""
    return np.random.default_rng(41).integers(0, 256, (48, 64, 3), dtype=np.uint8)


def batch(image, *, roles=(("stable_ticket",),), times=None):
    """Build aligned real upstream records with an isolated physical-frame loader."""
    times = [i/10 for i in range(len(roles))] if times is None else times
    candidates = tuple(EvidenceCandidate(3, t, ("dense_sampling",), (), None, False) for t in times)
    materialized = tuple(MaterializedEvidenceCandidate(c, "resolved", round(c.timestamp_seconds*10),
        round(c.timestamp_seconds*10)/10, 100., 100., 100.) for c in candidates)
    resolver = MagicMock(spec=VideoFrameResolver)
    resolver.video_path = Path("source.mp4").resolve()
    resolver.resolve.side_effect = lambda t: ResolvedVideoFrame(t, "resolved", round(t*10), round(t*10)/10, image)
    return {"video_path": resolver.video_path, "investigation_index": 3,
        "selections": tuple(RepresentativeSelection(i, r) for i, r in enumerate(roles)),
        "candidates": candidates, "materialized_candidates": materialized, "ui_config": CONFIG, "resolver": resolver}


def test_models_are_frozen_and_deeply_immutable(native, image):
    result = ocr.run_ocr_for_representatives(**batch(image))[0]
    for model, name in [(result, "candidate_index"), (result.detections[0], "text")]:
        with pytest.raises(FrozenInstanceError):
            setattr(model, name, "changed")
    with pytest.raises(TypeError):
        result.detections[0].source_polygon[0][0] = 4
    assert isinstance(result.detections, tuple)
    polygon = [p.copy() for p in QUAD]
    detection = ocr.OCRDetection("raw", None, polygon)
    polygon[0][0] = 100
    assert detection.source_polygon[0][0] == 2


def test_exact_color_crop_cubic2x_without_other_filters(native, image, monkeypatch):
    expected = cv2.resize(image[4:36, 8:24], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    before = image.copy()
    for name in ("cvtColor", "createCLAHE", "threshold", "GaussianBlur", "addWeighted"):
        monkeypatch.setattr(ocr.cv2, name, MagicMock(side_effect=AssertionError("Uncalibrated preprocessing")))
    result = ocr.run_ticket_ocr(image, ui_config=CONFIG)
    np.testing.assert_array_equal(native.instances[0].images[0], expected)
    np.testing.assert_array_equal(image, before)
    assert result.crop_roi == ROI(8, 4, 16, 32)
    assert result.upscale_factor == 2
    assert result.preprocessing_name == "full_ticket_color_cubic_2x"


def test_full_reference_crop_includes_upper_and_body_excludes_external_ui(native):
    image = np.zeros((1080, 1920, 3), np.uint8)
    image[121:197, 16:352] = [11, 22, 33]
    image[197:386, 16:352] = [44, 55, 66]
    image[386:952, 16:352] = [77, 88, 99]
    image[952:1008, 16:352] = 255
    result = ocr.run_ticket_ocr(image, ui_config=UIConfig())
    assert result.crop_roi == ROI(16, 121, 336, 831)
    processed = native.instances[0].images[0]
    assert processed.shape == (1662, 672, 3)
    np.testing.assert_array_equal(processed[20, 20], [11, 22, 33])
    np.testing.assert_array_equal(processed[200, 20], [44, 55, 66])
    np.testing.assert_array_equal(processed[-20, 20], [77, 88, 99])


@pytest.mark.parametrize("scale", [1, 2])
def test_polygon_maps_fractional_rotated_vertices_to_source(native, scale):
    native.output = [(QUAD, "TRACK", .9)]
    result = ocr.run_ticket_ocr(np.zeros((48*scale, 64*scale, 3), np.uint8), ui_config=CONFIG)
    expected = tuple((x/2+8*scale, y/2+4*scale) for x, y in QUAD)
    assert result.detections[0].source_polygon == expected
    assert result.crop_roi == ROI(8*scale, 4*scale, 16*scale, 32*scale)


def test_raw_text_confidence_multiple_detections_and_native_order(native, image):
    native.output = [(QUAD, "  AGE (Mar 2010): O0 I1\n", .987), (QUAD, "16 years", None)]
    result = ocr.run_ticket_ocr(image, ui_config=CONFIG)
    assert [(d.text, d.confidence) for d in result.detections] == [
        ("  AGE (Mar 2010): O0 I1\n", .987), ("16 years", None)]
    assert [f.name for f in fields(ocr.OCRDetection)] == ["text", "confidence", "source_polygon"]
    assert not any("field" in f.name or "label" in f.name for f in fields(result))


@pytest.mark.parametrize("output", [None, []])
def test_no_text_is_valid_empty_execution(native, image, output):
    native.output = output
    assert ocr.run_ticket_ocr(image, ui_config=CONFIG).detections == ()


def test_cpu_engine_settings_and_one_initialization_per_batch(native, image):
    data = batch(image, roles=(("stable_ticket",), ("post_scroll_stable",)))
    result = ocr.run_ocr_for_representatives(**data)
    assert len(native.instances) == 1 and len(native.instances[0].images) == 2
    assert native.instances[0].kwargs == {"intra_op_num_threads": 2, "inter_op_num_threads": 1,
        "det_use_cuda": False, "cls_use_cuda": False, "rec_use_cuda": False,
        "det_use_dml": False, "cls_use_dml": False, "rec_use_dml": False}
    assert [r.candidate_index for r in result] == [0, 1]
    assert [(r.investigation_index, r.frame_index, r.actual_timestamp_seconds, r.resolution_status)
        for r in result] == [(3, 0, 0., "resolved"), (3, 1, .1, "resolved")]


@pytest.mark.parametrize("role", ["map_activity", "route_activity", "scroll_provenance", "investigation_boundary"])
def test_context_roles_alone_skip_all_ocr_and_frame_loading(native, image, role):
    data = batch(image, roles=((role,),))
    assert ocr.run_ocr_for_representatives(**data) == ()
    assert not native.instances
    data["resolver"].resolve.assert_not_called()


def test_multi_role_and_duplicate_candidate_execute_once(native, image):
    data = batch(image, roles=(("map_activity",), ("stable_ticket", "map_activity", "post_scroll_stable")))
    data["selections"] += (data["selections"][1],)
    result = ocr.run_ocr_for_representatives(**data)
    assert len(result) == 1 and result[0].candidate_index == 1
    assert len(native.instances[0].images) == 1
    data["resolver"].resolve.assert_called_once_with(.1)


def test_distinct_candidates_sharing_frame_keep_distinct_provenance(native, image):
    data = batch(image, roles=(("stable_ticket",), ("post_scroll_stable",)), times=[.1, .11])
    result = ocr.run_ocr_for_representatives(**data)
    assert [r.candidate_index for r in result] == [0, 1]
    assert [r.frame_index for r in result] == [1, 1]


def test_reuses_supplied_engine_and_inputs_are_unchanged(native, image):
    data = batch(image)
    snapshots = (tuple(asdict(c) for c in data["candidates"]), tuple(asdict(m) for m in data["materialized_candidates"]), data["selections"])
    engine = ocr.RapidOCREngine()
    first = ocr.run_ocr_for_representatives(**data, engine=engine)
    assert first == ocr.run_ocr_for_representatives(**data, engine=engine)
    assert len(native.instances) == 1
    assert snapshots == (tuple(asdict(c) for c in data["candidates"]), tuple(asdict(m) for m in data["materialized_candidates"]), data["selections"])
    assert all(not isinstance(value, np.ndarray) for value in asdict(first[0]).values())


@pytest.mark.parametrize("index,error", [(-1, ValueError), (2, ValueError), (True, TypeError), (1.5, TypeError)])
def test_bad_selected_index_fails_before_loading(native, image, index, error):
    data = batch(image)
    data["selections"] = (RepresentativeSelection(index, ("stable_ticket",)),)
    with pytest.raises(error, match="candidate_index"):
        ocr.run_ocr_for_representatives(**data)
    assert not native.instances
    data["resolver"].resolve.assert_not_called()


@pytest.mark.parametrize("change,match", [
    ({"resolution_status": "unresolved", "frame_index": None, "actual_timestamp_seconds": None}, "unresolved"),
    ({"evidence_candidate": EvidenceCandidate(7, 0., (), (), None, False)}, "correspondence"),
    ({"frame_index": -1}, "frame_index"),
    ({"actual_timestamp_seconds": float("nan")}, "finite")])
def test_bad_materialized_selection_rejected(native, image, change, match):
    data = batch(image)
    data["materialized_candidates"] = (replace(data["materialized_candidates"][0], **change),)
    with pytest.raises((ValueError, TypeError), match=match):
        ocr.run_ocr_for_representatives(**data)
    assert not native.instances


def test_wrong_investigation_lengths_and_unsorted_selection(native, image):
    data = batch(image, roles=(("stable_ticket",), ("post_scroll_stable",)))
    with pytest.raises(ValueError, match="ownership"):
        ocr.run_ocr_for_representatives(**{**data, "investigation_index": 4})
    with pytest.raises(ValueError, match="lengths"):
        ocr.run_ocr_for_representatives(**{**data, "materialized_candidates": ()})
    with pytest.raises(ValueError, match="chronological"):
        ocr.run_ocr_for_representatives(**{**data, "selections": data["selections"][::-1]})


@pytest.mark.parametrize("change", [{"image": None}, {"frame_index": 5}, {"actual_timestamp_seconds": .2}, {"resolution_status": "video_end_adjusted"}])
def test_resolved_frame_mismatch_is_explicit(native, image, change):
    data = batch(image)
    data["resolver"].resolve.side_effect = None
    data["resolver"].resolve.return_value = replace(ResolvedVideoFrame(0., "resolved", 0, 0., image), **change)
    with pytest.raises(RuntimeError, match="disagrees"):
        ocr.run_ocr_for_representatives(**data)
    assert not native.instances[0].images


def test_caller_resolver_path_must_match(native, image):
    data = batch(image)
    data["video_path"] = Path("different.mp4")
    with pytest.raises(ValueError, match="resolver video"):
        ocr.run_ocr_for_representatives(**data)


def test_actual_resolver_cache_reused_without_extra_decode(native, image, tmp_path, monkeypatch):
    path = tmp_path/"source.mp4"
    path.touch()
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.set.return_value = True
    capture.read.return_value = (True, image)
    capture.get.side_effect = lambda prop: {cv2.CAP_PROP_FPS: 10., cv2.CAP_PROP_FRAME_COUNT: 10,
        cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48, cv2.CAP_PROP_POS_FRAMES: 1}[prop]
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    data = batch(image)
    with VideoFrameResolver(path, metadata=VideoMetadata(path, 1., 10., 10, 64, 48)) as resolver:
        resolver.resolve(0.)
        data.update(video_path=path, resolver=resolver)
        ocr.run_ocr_for_representatives(**data)
        assert resolver.decode_attempts == 1 and resolver.cache_hits == 1
    capture.release.assert_called_once()


def test_owned_resolver_is_closed_and_only_selected_requests_load(native, image, monkeypatch):
    data = batch(image, roles=(("map_activity",), ("stable_ticket",)))
    supplied = data.pop("resolver")
    factory = MagicMock()
    factory.return_value.__enter__.return_value = supplied
    monkeypatch.setattr(ocr, "VideoFrameResolver", factory)
    assert ocr.run_ocr_for_representatives(**data)[0].candidate_index == 1
    factory.assert_called_once_with(data["video_path"])
    factory.return_value.__exit__.assert_called_once()
    supplied.resolve.assert_called_once_with(.1)


def test_optional_import_boundary_and_actionable_missing_dependency(image, monkeypatch):
    subprocess.run([sys.executable, "-c", "import sys; import events_diary.ticket_ocr; assert 'rapidocr_onnxruntime' not in sys.modules; assert 'onnxruntime' not in sys.modules"], check=True)
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "rapidocr_onnxruntime":
            raise ModuleNotFoundError("isolated unavailable OCR runtime")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(RuntimeError, match="uv sync --group ocr-calibration"):
        ocr.run_ticket_ocr(image, ui_config=CONFIG)


def test_native_failure_propagates_without_fallback(native, image, monkeypatch):
    def fail(self, image):
        raise RuntimeError("native inference failure")
    monkeypatch.setattr(native, "__call__", fail)
    with pytest.raises(RuntimeError, match="native inference failure"):
        ocr.run_ticket_ocr(image, ui_config=CONFIG)
    assert len(native.instances) == 1


@pytest.mark.parametrize("bad,error", [(None, TypeError), (np.zeros((48,64,3), float), TypeError),
    (np.zeros((48,64), np.uint8), ValueError), (np.zeros((0,64,3), np.uint8), ValueError),
    (np.zeros((48,64,4), np.uint8), ValueError), (np.zeros((49,64,3), np.uint8), ValueError)])
def test_bad_images_fail_before_engine_creation(native, bad, error):
    with pytest.raises(error):
        ocr.run_ticket_ocr(bad, ui_config=CONFIG)
    assert not native.instances


def test_empty_batch_does_not_load_engine_or_open_video(native):
    assert ocr.run_ocr_for_representatives(video_path=Path("absent.mp4"), investigation_index=0,
        selections=(), candidates=(), materialized_candidates=(), ui_config=CONFIG) == ()
    assert not native.instances


def test_incompatible_crop_geometry_rejected(native, image):
    with pytest.raises(ValueError, match="full-width"):
        ocr.run_ticket_ocr(image, ui_config=replace(CONFIG, ticket_bottom_actions=ROI(9,36,15,8)))


@pytest.mark.parametrize("polygon,confidence", [([[0,0]]*3,.9), ([[0,float('inf')]]*4,.9), (QUAD,float('nan')), (QUAD,1.1)])
def test_malformed_native_detection_is_not_repaired(polygon, confidence):
    with pytest.raises(ValueError):
        ocr.OCRDetection("raw", confidence, polygon)
