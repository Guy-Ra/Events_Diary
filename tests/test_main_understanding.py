"""Phase 9.3 mode dispatch tests using doubles at existing runtime boundaries."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from events_diary.config import UnderstandingMode
from events_diary.main import (
    VideoPipelineResult,
    _understand_one,
    _UnderstandingStageError,
    run_understanding,
)
from events_diary.video import VideoMetadata


def _prepared():
    """Build a small duck-typed prepared Investigation for orchestration tests."""
    interval = SimpleNamespace(start_time_seconds=1.0, end_time_seconds=2.0)
    return SimpleNamespace(
        investigation_id=0, interval=interval, confirmed_events=(), window=object(),
        sampling_plan=object(), stable_views=(), evidence_candidates=(),
        materialized_candidates=(), representative_selection=SimpleNamespace(selections=()),
    )


def _metadata(tmp_path):
    """Return source identity without requiring a video decoder in mode tests."""
    return SimpleNamespace(path=tmp_path / "source.mp4")


@pytest.mark.parametrize(
    ("mode", "expected_ocr", "expected_pass1", "expected_fusion", "expected_pass2"),
    (
        (UnderstandingMode.OCR_ONLY, 1, 0, 0, 0),
        (UnderstandingMode.VLM_ONLY, 0, 1, 0, 0),
        (UnderstandingMode.OCR_VLM, 1, 1, 1, 1),
    ),
)
def test_understanding_modes_call_only_their_approved_branches(
    mode, expected_ocr, expected_pass1, expected_fusion, expected_pass2,
    tmp_path, monkeypatch,
):
    import events_diary.fusion as fusion_module
    import events_diary.investigation_record as record_module
    import events_diary.main as main_module
    import events_diary.vlm_review as review_module
    from events_diary.fusion import FusedFieldObservation

    counts = {"ocr": 0, "pass1": 0, "fusion": 0, "pass2": 0, "ticket_crops": 0}
    pass1_images = []

    class Runtime:
        """Count requests while standing in for the production runtime protocol."""

        def run(self, request):
            counts["pass2"] += 1
            return object()

    source_item = SimpleNamespace(
        field_name="REGISTRATION", canonical_label="REGISTRATION", status="extracted",
        provenance=SimpleNamespace(actual_timestamp_seconds=1.5, candidate_index=0),
    )
    pass1 = SimpleNamespace(observations=(source_item,))
    review = SimpleNamespace(
        raw_result=SimpleNamespace(execution_status="success", error=None),
        diagnostics=(), reviewed_fields={name: object() for name in review_module.REVIEW_FIELD_NAMES},
        route_reviews={name: object() for name in review_module.REVIEW_ROUTE_FIELDS},
    )

    def run_ocr(*args, **kwargs):
        counts["ocr"] += 1
        return (), (source_item,)

    def run_pass1(*args, **kwargs):
        counts["pass1"] += 1
        pass1_images.append(kwargs.get("ticket_images", ()))
        return object(), pass1

    def make_ticket_images(*args):
        counts["ticket_crops"] += 1
        return ("selected ticket crop",)

    def compare(*args):
        counts["fusion"] += 1
        return FusedFieldObservation(args[0], "unavailable")

    def temporal(item):
        return item

    monkeypatch.setattr(main_module, "_run_ocr", run_ocr)
    monkeypatch.setattr(main_module, "_make_continuous_clip", lambda *args: None)
    monkeypatch.setattr(main_module, "_run_pass1", run_pass1)
    monkeypatch.setattr(main_module, "_vlm_ticket_images", make_ticket_images)
    monkeypatch.setattr(main_module, "_ticket_detail_images", lambda *args: (object(),))
    monkeypatch.setattr(fusion_module, "compare_field_observations", compare)
    monkeypatch.setattr(fusion_module, "apply_temporal_awareness", temporal)
    monkeypatch.setattr(fusion_module, "resolve_single_source_investigation", lambda **kwargs: object())
    monkeypatch.setattr(fusion_module, "resolve_investigation", lambda *args: object())
    monkeypatch.setattr(review_module, "build_vlm_review_request", lambda *args, **kwargs: object())
    monkeypatch.setattr(review_module, "parse_vlm_review_result", lambda *args: review)
    monkeypatch.setattr(record_module, "assemble_investigation_record", lambda **kwargs: kwargs)

    result = _understand_one(
        _prepared(), _metadata(tmp_path), mode, ocr_engine=object(), vlm_runtime=Runtime(),
    )

    assert result["vlm_pass1"] is (pass1 if expected_pass1 else None)
    assert (result["fusion_result"] is not None) is bool(expected_fusion)
    assert (result["vlm_pass2"] is not None) is bool(expected_pass2)
    assert counts == {
        "ocr": expected_ocr, "pass1": expected_pass1,
        "fusion": expected_fusion, "pass2": expected_pass2,
        "ticket_crops": int(mode is UnderstandingMode.VLM_ONLY),
    }
    if mode is UnderstandingMode.OCR_ONLY:
        assert result["vlm_pass1"] is result["fusion_result"] is result["vlm_pass2"] is None
    if mode is UnderstandingMode.VLM_ONLY:
        assert result["ocr_results"] == result["ocr_field_occurrences"] == ()
        assert result["fusion_result"] is result["vlm_pass2"] is None
        assert pass1_images == [("selected ticket crop",)]
    elif expected_pass1:
        assert pass1_images == [()]


def test_vlm_ticket_images_use_selected_views_and_clean_temporary_crops(tmp_path, monkeypatch):
    """Build separate crops from existing stable representatives inside temp storage."""
    import tempfile

    import numpy as np

    import events_diary.frame_materialization as frame_module
    import events_diary.main as main_module
    from events_diary.config import UIConfig
    from events_diary.representative_selection import RepresentativeSelection

    class FakeResolver:
        """Return one deterministic full-size frame for each chosen physical time."""

        def __init__(self, *_args, **_kwargs):
            self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def resolve(self, timestamp):
            index = int(timestamp * 10)
            return SimpleNamespace(frame_index=index, image=self.frame)

    monkeypatch.setattr(frame_module, "VideoFrameResolver", FakeResolver)
    prepared = SimpleNamespace(
        investigation_id=1,
        representative_selection=SimpleNamespace(selections=(
            RepresentativeSelection(0, ("stable_ticket",)),
            RepresentativeSelection(1, ("post_scroll_stable",)),
        )),
        materialized_candidates=(
            SimpleNamespace(resolution_status="resolved", frame_index=10, actual_timestamp_seconds=1.0),
            SimpleNamespace(resolution_status="resolved", frame_index=20, actual_timestamp_seconds=2.0),
        ),
        evidence_candidates=(SimpleNamespace(), SimpleNamespace()),
    )
    metadata = SimpleNamespace(path=tmp_path / "source.mp4")
    with tempfile.TemporaryDirectory(dir=tmp_path) as temporary:
        images = main_module._vlm_ticket_images(
            prepared, metadata, UIConfig(), Path(temporary),
        )
        assert len(images) == 2
        assert [image.role for image in images] == [
            "upper_ticket_header_and_route_times", "post_scroll_ticket_details",
        ]
        assert all(image.image_path.is_file() for image in images)
        assert all(image.image_path.stat().st_size > 0 for image in images)
        paths = tuple(image.image_path for image in images)
    assert all(not path.exists() for path in paths)


def test_vlm_ticket_images_allow_no_optional_representatives(tmp_path, monkeypatch):
    """Permit a valid video-only Pass 1 when no selected ticket crop is available."""
    import events_diary.main as main_module

    prepared = SimpleNamespace(
        investigation_id=0,
        representative_selection=SimpleNamespace(selections=()),
        materialized_candidates=(), evidence_candidates=(),
    )
    monkeypatch.setattr(
        "events_diary.frame_materialization.VideoFrameResolver",
        lambda *_args, **_kwargs: pytest.fail("resolver is unnecessary without image candidates"),
    )

    assert main_module._vlm_ticket_images(prepared, object(), object(), tmp_path) == ()


def test_vlm_ticket_images_skip_unresolved_optional_representatives(tmp_path):
    """Skip unavailable selected pixels without blocking the continuous-video request."""
    import events_diary.main as main_module
    from events_diary.representative_selection import RepresentativeSelection

    prepared = SimpleNamespace(
        investigation_id=0,
        representative_selection=SimpleNamespace(selections=(
            RepresentativeSelection(0, ("stable_ticket",)),
        )),
        materialized_candidates=(SimpleNamespace(
            resolution_status="unresolved", frame_index=None, actual_timestamp_seconds=None,
        ),),
        evidence_candidates=(SimpleNamespace(),),
    )
    assert main_module._vlm_ticket_images(prepared, object(), object(), tmp_path) == ()


def test_pass1_prompt_explains_video_and_ticket_image_roles_only_when_images_exist(tmp_path, monkeypatch):
    """Add only the approved media-source guidance and keep the Pass 1 contract."""
    import events_diary.main as main_module
    import events_diary.vlm_parsing as parsing_module
    from events_diary.vlm import RawVLMResult, VLMImageInput
    from events_diary.vlm_parsing import VLMParseDiagnostics
    from events_diary.vlm_prompts import FR24_VLM_V1, build_vlm_prompt

    interval = SimpleNamespace(start_time_seconds=1.0, end_time_seconds=2.0)
    investigation = SimpleNamespace(investigation_id=0, interval=interval)
    image = VLMImageInput(tmp_path / "ticket.png", 1.5, "upper_ticket", "identity and times")
    captured = []

    class Runtime:
        """Capture the Pass 1 request and return a valid-shaped raw response."""

        def run(self, request):
            captured.append(request)
            return RawVLMResult(
                "success", 0, tmp_path / "source.mp4", request.video_clip_path,
                1.0, 2.0, "model", "revision", FR24_VLM_V1, raw_text="{}",
            )

    monkeypatch.setattr(parsing_module, "parse_vlm_result", lambda request, raw: SimpleNamespace(
        raw_result=raw, parse_diagnostics=VLMParseDiagnostics("valid_json"),
        visual_route=object(), observations=tuple(range(36)),
    ))
    main_module._run_pass1(
        Runtime(), SimpleNamespace(path=tmp_path / "source.mp4"), investigation,
        tmp_path / "clip.mp4", ticket_images=(image,),
    )
    request = captured[0]
    assert request.ticket_images == (image,)
    assert request.prompt.startswith(build_vlm_prompt(FR24_VLM_V1))
    assert "continuous video for scrolling" in request.prompt
    assert "clearest supplied ticket crop first for small ticket text" in request.prompt
    assert "never infer registration or aircraft variant/code" in request.prompt
    assert "use N/A only when that exact field visibly says N/A" in request.prompt
    assert "Keep ticket TRACK separate from map direction" in request.prompt
    assert "OCR" not in request.prompt


def test_ocr_failure_is_reported_without_switching_modes(tmp_path, monkeypatch):
    import events_diary.main as main_module

    def fail_ocr(*args, **kwargs):
        raise RuntimeError("OCR unavailable")

    monkeypatch.setattr(main_module, "_run_ocr", fail_ocr)

    with pytest.raises(_UnderstandingStageError, match="OCR unavailable") as error:
        _understand_one(
            _prepared(), _metadata(tmp_path), UnderstandingMode.OCR_ONLY,
            ocr_engine=object(), vlm_runtime=object(),
        )

    assert error.value.stage == "ocr"


def test_pass2_provider_failure_reports_exact_stage(tmp_path, monkeypatch):
    import events_diary.fusion as fusion_module
    import events_diary.main as main_module
    import events_diary.vlm_review as review_module
    from events_diary.fusion import FusedFieldObservation

    source_item = SimpleNamespace(
        field_name="REGISTRATION", canonical_label="REGISTRATION", status="extracted",
        raw_value_text="N123AB",
        provenance=SimpleNamespace(actual_timestamp_seconds=1.5, candidate_index=0),
    )
    pass1 = SimpleNamespace(observations=(source_item,))

    class FailingRuntime:
        """Raise a provider-like failure only when Pass 2 sends its request."""

        def run(self, request):
            raise RuntimeError("Gemini API returned HTTP 503.")

    monkeypatch.setattr(main_module, "_run_ocr", lambda *args, **kwargs: ((), (source_item,)))
    monkeypatch.setattr(main_module, "_make_continuous_clip", lambda *args: None)
    monkeypatch.setattr(main_module, "_run_pass1", lambda *args, **kwargs: (object(), pass1))
    monkeypatch.setattr(main_module, "_ticket_detail_images", lambda *args: (object(),))
    monkeypatch.setattr(fusion_module, "compare_field_observations", lambda *args: (
        FusedFieldObservation("REGISTRATION", "unavailable")
    ))
    monkeypatch.setattr(fusion_module, "apply_temporal_awareness", lambda item: item)
    monkeypatch.setattr(review_module, "build_vlm_review_request", lambda *args, **kwargs: object())

    with pytest.raises(_UnderstandingStageError, match="HTTP 503") as error:
        _understand_one(
            _prepared(), _metadata(tmp_path), UnderstandingMode.OCR_VLM,
            ocr_engine=object(), vlm_runtime=FailingRuntime(),
        )

    assert error.value.stage == "vlm_pass2"


def test_zero_investigations_returns_empty_video_result_without_loading_branches(
    tmp_path, monkeypatch,
):
    metadata = VideoMetadata(tmp_path / "empty.mp4", 1.0, 10.0, 10, 1920, 1080)
    prepared = VideoPipelineResult(metadata, ())

    result = run_understanding(prepared, UnderstandingMode.OCR_VLM, vlm_settings=object())

    assert result.source_video == metadata.path
    assert result.investigations == result.failures == ()
