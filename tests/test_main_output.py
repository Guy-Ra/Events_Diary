"""Phase 9.4 tests for canonical rendering and final diary file output."""

import importlib.util
import json
from pathlib import Path

import pytest

from events_diary.canonical_events import CanonicalSession, CanonicalSourceVideo
from events_diary.config import UnderstandingMode, VLMAPISettings
from events_diary.investigation_record import (
    InvestigationFailure,
    VideoInvestigationResult,
)
from events_diary.main import (
    DETAILED_OUTPUT_FILENAME,
    OPERATIONAL_OUTPUT_FILENAME,
    VLM_RESULTS_FILENAME,
    IncompleteVideoRunError,
    PipelineOutput,
    VideoPipelineResult,
    finalize_pipeline_result,
    run_understanding,
)
from events_diary.video import VideoMetadata


def _empty_video_result(path: Path) -> VideoInvestigationResult:
    """Build a valid zero-Investigation result for output-boundary tests."""
    metadata = VideoMetadata(path, 0.0, 10.0, 0, 1920, 1080)
    return VideoInvestigationResult(path, metadata, ())


def _saved_record(index: int = 0):
    """Load one typed successful Phase 9 fixture for partial-output tests."""
    adapter_path = Path(__file__).parents[1] / "scripts/validation/validate_phase7_4_saved_artifacts.py"
    spec = importlib.util.spec_from_file_location("phase7_partial_adapter", adapter_path)
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    payload = adapter._load_json(adapter.ARTIFACT_ROOT / "session_result.json")
    item = payload["completed_records"][index]
    record_dir = adapter.ARTIFACT_ROOT / f"investigation_{item['investigation_id']:03}"
    record = adapter._typed_record(
        adapter._load_json(record_dir / "investigation_record.json"),
        adapter._load_json(record_dir / "final_result.json"),
        adapter._load_json(record_dir / "vlm_pass2_result.json"),
    )
    if not hasattr(record, "vlm_pass1"):
        from types import SimpleNamespace

        object.__setattr__(record, "vlm_pass1", SimpleNamespace(
            raw_result=SimpleNamespace(
                model_name="fixture-model", model_revision="fixture-revision",
                prompt_version="fr24_vlm_v1",
            ),
            observations=(SimpleNamespace(
                field_name="REGISTRATION", raw_value="4X-ABC", status="observed",
                evidence_time_s=record.start_time_s + 0.5,
            ),),
            visual_route=None,
        ))
    video_data = payload["video_metadata"]
    metadata = VideoMetadata(
        Path(video_data["path"]), video_data["duration_seconds"], video_data["fps"],
        video_data["frame_count"], video_data["width"], video_data["height"],
    )
    return record, metadata


@pytest.mark.parametrize(
    ("mode", "label"),
    (
        (UnderstandingMode.OCR_ONLY, "OCR only"),
        (UnderstandingMode.VLM_ONLY, "VLM only"),
        (UnderstandingMode.OCR_VLM, "OCR + VLM"),
    ),
)
def test_finalizer_adds_same_mode_header_and_preserves_renderer_bodies(
    mode, label, tmp_path, monkeypatch,
):
    """Pass one session to both renderers and persist exactly their UTF-8 text."""
    import events_diary.canonical_events as canonical_module
    import events_diary.detailed_diary as detailed_module
    import events_diary.operational_diary as operational_module

    source = tmp_path / "flight.mp4"
    result = _empty_video_result(source)
    output_dir = tmp_path / "new-output"
    output_dir.mkdir()
    operational_path = output_dir / OPERATIONAL_OUTPUT_FILENAME
    detailed_path = output_dir / DETAILED_OUTPUT_FILENAME
    operational_path.write_text("stale operational", encoding="utf-8")
    detailed_path.write_text("stale detailed", encoding="utf-8")
    session = CanonicalSession(CanonicalSourceVideo(source.name, 0.0), ())
    calls = []

    def build(actual_result):
        calls.append(("canonical", actual_result))
        return session

    def render_operational(actual_session):
        calls.append(("operational", actual_session))
        return "Operational – café ✈\n"

    def render_detailed(actual_session):
        calls.append(("detailed", actual_session))
        return "Detailed – résumé 🛫\n"

    monkeypatch.setattr(canonical_module, "build_canonical_session", build)
    monkeypatch.setattr(operational_module, "render_operational_diary", render_operational)
    monkeypatch.setattr(detailed_module, "render_detailed_diary", render_detailed)
    before = result

    output = finalize_pipeline_result(result, output_dir, mode)

    expected_output = PipelineOutput(
        operational_path, detailed_path, "completed",
        output_dir / VLM_RESULTS_FILENAME if mode is UnderstandingMode.VLM_ONLY else None,
    )
    assert output == expected_output
    assert calls == [
        ("canonical", result), ("operational", session), ("detailed", session),
    ]
    header = f"Understanding mode: {label}"
    if mode is UnderstandingMode.VLM_ONLY:
        header = f"Run status: completed\n{header}"
    assert operational_path.read_text(encoding="utf-8") == (
        f"{header}\n\nOperational – café ✈\n"
    )
    assert detailed_path.read_text(encoding="utf-8") == (
        f"{header}\n\nDetailed – résumé 🛫\n"
    )
    assert result is before
    assert not hasattr(session, "understanding_mode")
    assert all(not hasattr(event, "understanding_mode") for event in session.events)
    expected_files = {OPERATIONAL_OUTPUT_FILENAME, DETAILED_OUTPUT_FILENAME}
    if mode is UnderstandingMode.VLM_ONLY:
        expected_files.add(VLM_RESULTS_FILENAME)
        assert output.run_status == "completed"
        assert output.vlm_results_path == output_dir / VLM_RESULTS_FILENAME
    assert {path.name for path in output_dir.iterdir()} == expected_files


def test_finalizer_creates_output_directory_and_empty_diaries(tmp_path):
    """Preserve existing renderer semantics for a successful empty session."""
    output_dir = tmp_path / "missing" / "output"
    result = _empty_video_result(tmp_path / "empty.mp4")

    output = finalize_pipeline_result(result, output_dir, UnderstandingMode.OCR_ONLY)

    assert output.operational_diary_path.read_text(encoding="utf-8") == (
        "Understanding mode: OCR only"
    )
    assert output.detailed_diary_path.read_text(encoding="utf-8") == (
        "Understanding mode: OCR only"
    )
    assert sorted(path.name for path in output_dir.iterdir()) == sorted((
        OPERATIONAL_OUTPUT_FILENAME, DETAILED_OUTPUT_FILENAME,
    ))


@pytest.mark.parametrize("mode", (UnderstandingMode.OCR_ONLY, UnderstandingMode.OCR_VLM))
def test_failed_investigations_are_all_reported_before_canonicalization(
    mode, tmp_path, monkeypatch, caplog,
):
    """Report safe per-Investigation causes and never build or write a partial session."""
    import events_diary.canonical_events as canonical_module

    source = tmp_path / "failed.mp4"
    metadata = VideoMetadata(source, 1.0, 10.0, 10, 1920, 1080)
    failures = (
        InvestigationFailure(0, source, 0.0, 0.4, "ocr", "reader failed", error_type="RuntimeError"),
        InvestigationFailure(
            1, source, 0.4, 1.0, "vlm_pass2",
            '{"api_key":"secret","error":"rejected"}', error_type="HTTPError",
        ),
    )
    result = VideoInvestigationResult(source, metadata, (), failures)
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        canonical_module, "build_canonical_session",
        lambda _: pytest.fail("canonical conversion must not run for failed Investigations"),
    )

    with pytest.raises(IncompleteVideoRunError, match=r"IDs: 0, 1") as raised:
        finalize_pipeline_result(result, output_dir, mode)

    assert "Investigation 0 [0.000, 0.400)" in caplog.text
    assert "stage=ocr | error_type=RuntimeError | reason=reader failed" in caplog.text
    assert "Investigation 1 [0.400, 1.000)" in caplog.text
    assert "stage=vlm_pass2 | error_type=HTTPError" in caplog.text
    assert "secret" not in caplog.text
    assert "secret" not in str(raised.value)
    assert not output_dir.exists()


@pytest.mark.parametrize(("success_index", "failure_first"), ((0, False), (1, True)))
def test_vlm_only_partial_diary_preserves_success_and_failure_order(
    tmp_path, success_index, failure_first,
):
    """Render both outcome types in source-time order without dropping success data."""
    record, metadata = _saved_record(success_index)
    failure_id = 1 - success_index
    if failure_first:
        failure_start = 0.0
        failure_end = 0.0
    else:
        failure_start = record.end_time_s + 1.0
        failure_end = failure_start + 1.0
    failure = InvestigationFailure(
        failure_id, metadata.path, failure_start, failure_end, "vlm_pass1",
        "Gemini API returned HTTP 429 — Resource Exhausted.", error_type="RuntimeError",
        provider_diagnostics_json='{"attempt_count":3,"classification":"unknown"}',
    )
    result = VideoInvestigationResult(
        metadata.path, metadata, (record,), (failure,),
    )

    output = finalize_pipeline_result(result, tmp_path, UnderstandingMode.VLM_ONLY)

    detailed = output.detailed_diary_path.read_text(encoding="utf-8")
    assert output.run_status == "partial"
    assert "Run status: partial" in detailed
    assert "VLM processing failed" in detailed
    assert "HTTP 429" in detailed
    failed_position = detailed.index("VLM processing failed")
    flight_position = detailed.index("Flight:")
    if failure_first:
        assert failed_position < flight_position
    else:
        assert failed_position > flight_position
    artifact = json.loads(output.vlm_results_path.read_text(encoding="utf-8"))
    assert artifact["run_status"] == "partial"
    assert artifact["successful_investigations"][0]["investigation_id"] == record.investigation_id
    assert artifact["successful_investigations"][0]["fields"]
    assert artifact["failed_investigations"][0]["provider_diagnostics"]["attempt_count"] == 3


def test_vlm_only_all_failures_still_write_failure_diaries(tmp_path):
    """Write an explicit failed run artifact even when no Investigation succeeds."""
    source = tmp_path / "failed.mp4"
    metadata = VideoMetadata(source, 8.0, 25.0, 200, 1920, 1080)
    failures = (
        InvestigationFailure(0, source, 1.0, 2.0, "vlm_pass1", "Gemini API returned HTTP 429 — Resource Exhausted."),
        InvestigationFailure(1, source, 4.0, 5.0, "vlm_pass1", "Gemini API returned HTTP 503 — UNAVAILABLE."),
    )
    result = VideoInvestigationResult(source, metadata, (), failures)

    output = finalize_pipeline_result(result, tmp_path / "out", UnderstandingMode.VLM_ONLY)

    assert output.run_status == "failed"
    detailed = output.detailed_diary_path.read_text(encoding="utf-8")
    assert "Run status: failed" in detailed
    assert detailed.count("VLM processing failed") == 2
    assert detailed.index("HTTP 429") < detailed.index("HTTP 503")
    artifact = json.loads(output.vlm_results_path.read_text(encoding="utf-8"))
    assert artifact["successful_investigations"] == []
    assert [item["investigation_id"] for item in artifact["failed_investigations"]] == [0, 1]


@pytest.mark.parametrize("success_index", (0, 1))
def test_vlm_only_understanding_continues_after_failure_in_either_position(
    success_index, monkeypatch,
):
    """Keep a successful record whether it precedes or follows a failed call."""
    from types import SimpleNamespace

    import events_diary.main as main_module

    record, metadata = _saved_record(success_index)
    prepared = VideoPipelineResult(
        metadata,
        tuple(SimpleNamespace(
            investigation_id=index,
            interval=SimpleNamespace(start_time_seconds=float(index), end_time_seconds=float(index + 1)),
        ) for index in range(2)),
    )
    calls = []

    def understand(item, *args, **kwargs):
        calls.append(item.investigation_id)
        if item.investigation_id == success_index:
            return record
        raise RuntimeError("Gemini API returned HTTP 429 — Resource Exhausted.")

    monkeypatch.setattr(main_module, "_understand_one", understand)
    settings = VLMAPISettings(
        "gemini", "", "https://example.invalid", "v1beta", "test-model", 10,
    )

    result = run_understanding(prepared, UnderstandingMode.VLM_ONLY, vlm_settings=settings)

    assert calls == [0, 1]
    assert result.investigations == (record,)
    assert len(result.failures) == 1
    assert result.failures[0].investigation_id == 1 - success_index
    assert "HTTP 429" in result.failures[0].reason


@pytest.mark.parametrize("renderer", ["operational", "detailed"])
def test_render_failure_does_not_create_final_output(tmp_path, monkeypatch, renderer):
    """Render both strings before creating or writing final output files."""
    import events_diary.canonical_events as canonical_module
    import events_diary.detailed_diary as detailed_module
    import events_diary.operational_diary as operational_module

    result = _empty_video_result(tmp_path / "source.mp4")
    session = CanonicalSession(CanonicalSourceVideo("source.mp4", 0.0), ())
    monkeypatch.setattr(canonical_module, "build_canonical_session", lambda _: session)
    monkeypatch.setattr(operational_module, "render_operational_diary", lambda _: (
        (_ for _ in ()).throw(RuntimeError("operational render failed"))
        if renderer == "operational" else "operational"
    ))
    monkeypatch.setattr(detailed_module, "render_detailed_diary", lambda _: (
        (_ for _ in ()).throw(RuntimeError("detailed render failed"))
        if renderer == "detailed" else "detailed"
    ))
    output_dir = tmp_path / "output"

    with pytest.raises(RuntimeError, match="render failed"):
        finalize_pipeline_result(result, output_dir, UnderstandingMode.OCR_ONLY)

    assert not output_dir.exists()


def test_saved_combined_result_renders_to_accepted_phase8_diaries(tmp_path):
    """Replay the saved combined-mode records through the Phase 9.4 back half."""
    adapter_path = Path(__file__).parents[1] / "scripts/validation/validate_phase7_4_saved_artifacts.py"
    spec = importlib.util.spec_from_file_location("phase7_saved_adapter", adapter_path)
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    payload = adapter._load_json(adapter.ARTIFACT_ROOT / "session_result.json")
    video_data = payload["video_metadata"]
    metadata = VideoMetadata(
        Path(video_data["path"]), video_data["duration_seconds"], video_data["fps"],
        video_data["frame_count"], video_data["width"], video_data["height"],
    )
    records = []
    for item in payload["completed_records"]:
        record_dir = adapter.ARTIFACT_ROOT / f"investigation_{item['investigation_id']:03}"
        records.append(adapter._typed_record(
            adapter._load_json(record_dir / "investigation_record.json"),
            adapter._load_json(record_dir / "final_result.json"),
            adapter._load_json(record_dir / "vlm_pass2_result.json"),
        ))
    result = VideoInvestigationResult(metadata.path, metadata, tuple(records))
    snapshots = tuple(record.__dict__.copy() for record in result.investigations)

    output = finalize_pipeline_result(result, tmp_path, UnderstandingMode.OCR_VLM)

    accepted_operational = Path("data/debug/events_diary/phase_8_2/operational_diary.txt")
    accepted_detailed = Path("data/debug/events_diary/phase_8_3/detailed_diary.txt")
    assert output.operational_diary_path.read_text(encoding="utf-8").splitlines()[2:] == (
        accepted_operational.read_text(encoding="utf-8").splitlines()
    )
    assert output.detailed_diary_path.read_text(encoding="utf-8").splitlines()[2:] == (
        accepted_detailed.read_text(encoding="utf-8").splitlines()
    )
    assert output.operational_diary_path.read_text(encoding="utf-8").splitlines()[0] == (
        "Understanding mode: OCR + VLM"
    )
    assert output.detailed_diary_path.read_text(encoding="utf-8").splitlines()[0] == (
        "Understanding mode: OCR + VLM"
    )
    assert tuple(record.__dict__.copy() for record in result.investigations) == snapshots
    assert {path.name for path in tmp_path.iterdir()} == {
        OPERATIONAL_OUTPUT_FILENAME, DETAILED_OUTPUT_FILENAME,
    }
    assert not any(path.suffix == ".json" for path in tmp_path.iterdir())

