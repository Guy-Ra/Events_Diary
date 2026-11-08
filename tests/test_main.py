"""Focused tests for Phase 9.1 configuration and orchestration contracts."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from events_diary.config import (
    RuntimeConfig,
    UnderstandingMode,
    VLMAPISettings,
    load_runtime_config,
)
from events_diary.main import (
    DETAILED_OUTPUT_FILENAME,
    MODE_READINESS,
    OPERATIONAL_OUTPUT_FILENAME,
    InputVideoSelectionError,
    InvestigationDetectionResult,
    UnderstandingModeReadiness,
    build_pipeline_run_plan,
    main,
    prepare_investigation_evidence,
    prepare_video_pipeline,
    run_pipeline,
    select_input_video,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ("ocr_only", UnderstandingMode.OCR_ONLY),
        ("vlm_only", UnderstandingMode.VLM_ONLY),
        ("ocr_vlm", UnderstandingMode.OCR_VLM),
    ),
)
def test_runtime_config_loads_each_understanding_mode(raw, expected, tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("EVENTS_DIARY_UNDERSTANDING_MODE", raw)

    config = load_runtime_config(project_root=tmp_path, env_file=env_file)

    assert config.understanding_mode is expected
    assert config.input_dir == tmp_path / "experiment" / "input"
    assert config.output_dir == tmp_path / "experiment" / "output"
    assert (config.vlm_api_settings is not None) is expected.uses_vlm


def test_runtime_config_rejects_invalid_understanding_mode(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("EVENTS_DIARY_UNDERSTANDING_MODE", "ocr_and_maybe_vlm")

    with pytest.raises(ValueError, match="EVENTS_DIARY_UNDERSTANDING_MODE"):
        load_runtime_config(project_root=tmp_path, env_file=env_file)


def test_understanding_modes_have_distinct_values():
    assert [mode.value for mode in UnderstandingMode] == ["ocr_only", "vlm_only", "ocr_vlm"]
    assert len(set(UnderstandingMode)) == 3


def test_runtime_config_rejects_untyped_mode(tmp_path):
    with pytest.raises(TypeError, match="UnderstandingMode"):
        RuntimeConfig("ocr_only", tmp_path, tmp_path)  # type: ignore[arg-type]


def test_zero_supported_input_videos_fails_clearly(tmp_path):
    (tmp_path / ".gitkeep").touch()
    (tmp_path / "notes.txt").touch()

    with pytest.raises(InputVideoSelectionError, match=r"exactly one \.mp4"):
        select_input_video(tmp_path)


def test_multiple_supported_input_videos_fails_instead_of_choosing(tmp_path):
    (tmp_path / "a.mp4").touch()
    (tmp_path / "b.MP4").touch()

    with pytest.raises(InputVideoSelectionError, match="found 2"):
        select_input_video(tmp_path)


def test_exactly_one_supported_video_is_selected_and_other_files_ignored(tmp_path):
    video = tmp_path / "capture.Mp4"
    video.touch()
    (tmp_path / ".gitkeep").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "thumbnail.png").touch()
    (tmp_path / "nested").mkdir()

    assert select_input_video(tmp_path) == video.resolve()


def test_default_workspace_paths_are_under_experiment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("EVENTS_DIARY_UNDERSTANDING_MODE", "ocr_only")

    config = load_runtime_config(project_root=tmp_path, env_file=env_file)

    assert config.input_dir == tmp_path / "experiment" / "input"
    assert config.output_dir == tmp_path / "experiment" / "output"


def test_run_plan_has_fixed_output_filenames_and_no_filesystem_effects(tmp_path):
    input_dir = tmp_path / "experiment" / "input"
    input_dir.mkdir(parents=True)
    video = input_dir / "one.mp4"
    video.touch()
    output_dir = tmp_path / "experiment" / "output"
    config = RuntimeConfig(UnderstandingMode.OCR_ONLY, input_dir, output_dir)

    plan = build_pipeline_run_plan(config)

    assert plan.video_path == video.resolve()
    assert plan.operational_output_path == output_dir / OPERATIONAL_OUTPUT_FILENAME
    assert plan.detailed_output_path == output_dir / DETAILED_OUTPUT_FILENAME
    assert OPERATIONAL_OUTPUT_FILENAME == "operational_diary.txt"
    assert DETAILED_OUTPUT_FILENAME == "detailed_diary.txt"
    assert not output_dir.exists()


@pytest.mark.parametrize("mode", tuple(UnderstandingMode))
def test_all_modes_share_phase_9_2_preparation(mode, tmp_path, monkeypatch):
    import events_diary.main as main_module

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "one.mp4").touch()
    settings = (
        VLMAPISettings("gemini", "", "https://example.invalid", "v1beta", "model", 180)
        if mode.uses_vlm
        else None
    )
    config = RuntimeConfig(mode, input_dir, tmp_path / "output", settings)
    expected = object()
    output = object()
    calls = []
    monkeypatch.setattr(main_module, "prepare_video_pipeline", lambda received: calls.append(received) or expected)
    successful_result = SimpleNamespace(failures=(), investigations=())
    monkeypatch.setattr(main_module, "run_understanding", lambda *args, **kwargs: successful_result)
    monkeypatch.setattr(main_module, "finalize_pipeline_result",
                        lambda result, directory, received_mode: (
                            calls.append((result, directory, received_mode)) or output
                        ))

    assert run_pipeline(config) is output
    assert calls == [config, (successful_result, config.output_dir, mode)]


def test_prepare_pipeline_resolves_input_then_calls_video_and_investigation_stages(tmp_path, monkeypatch):
    import events_diary.main as main_module
    import events_diary.video as video_module

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    video = input_dir / "chosen.mp4"
    video.touch()
    config = RuntimeConfig(UnderstandingMode.OCR_ONLY, input_dir, tmp_path / "output")
    metadata = object()
    detection = object()
    prepared = (object(), object())
    order = []

    monkeypatch.setattr(video_module, "read_video_metadata", lambda path: order.append(("metadata", path)) or metadata)
    monkeypatch.setattr(main_module, "detect_investigations", lambda received: order.append(("detection", received)) or detection)
    monkeypatch.setattr(main_module, "prepare_investigation_evidence",
                        lambda received, evidence: order.append(("evidence", received, evidence)) or prepared)

    result = prepare_video_pipeline(config)

    assert result.metadata is metadata
    assert result.investigations == prepared
    assert order == [("metadata", video.resolve()), ("detection", metadata), ("evidence", metadata, detection)]
    assert not config.output_dir.exists()


def test_empty_detection_result_produces_valid_zero_investigation_result(tmp_path):
    from events_diary.video import VideoMetadata

    metadata = VideoMetadata(tmp_path / "empty.mp4", 3.0, 10.0, 30, 1920, 1080)
    empty = InvestigationDetectionResult((), (), (), (), (), ())

    assert prepare_investigation_evidence(metadata, empty) == ()


def test_evidence_preparation_preserves_interval_order_and_closes_each_resolver(
    tmp_path, monkeypatch,
):
    import events_diary.frame_materialization as materialization_module
    import events_diary.representative_selection as selection_module
    from events_diary.investigation_lifecycle import (
        InvestigationEndReason,
        InvestigationInterval,
        InvestigationStartReason,
    )
    from events_diary.representative_selection import InvestigationSelection
    from events_diary.video import VideoMetadata

    class Resolver:
        """Track the orchestration-owned lifetime of each decoder resolver."""

        def __init__(self, *args, **kwargs):
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.closed = True

    resolvers = []

    def resolver_factory(*args, **kwargs):
        resolver = Resolver(*args, **kwargs)
        resolvers.append(resolver)
        return resolver

    monkeypatch.setattr(materialization_module, "VideoFrameResolver", resolver_factory)
    monkeypatch.setattr(materialization_module, "materialize_evidence_candidates",
                        lambda **kwargs: ())
    monkeypatch.setattr(
        selection_module,
        "select_investigation_representatives",
        lambda **kwargs: InvestigationSelection(kwargs["window"].investigation_index, ()),
    )
    intervals = (
        InvestigationInterval(0, 0.5, 1.0, InvestigationStartReason.OPEN,
                              InvestigationEndReason.CLOSE),
        InvestigationInterval(1, 2.0, 2.5, InvestigationStartReason.OPEN,
                              InvestigationEndReason.VIDEO_END),
    )
    metadata = VideoMetadata(tmp_path / "video.mp4", 3.0, 10.0, 30, 1920, 1080)
    detection = InvestigationDetectionResult((), (), (), (), (), intervals)

    prepared = prepare_investigation_evidence(metadata, detection)

    assert tuple(item.investigation_id for item in prepared) == (0, 1)
    assert tuple(item.interval for item in prepared) == intervals
    assert tuple(item.window.investigation_index for item in prepared) == (0, 1)
    assert len(resolvers) == 2
    assert all(resolver.closed for resolver in resolvers)


def test_detection_helper_calls_existing_stages_in_order(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import events_diary.investigation_detection as detection_module
    import events_diary.investigation_lifecycle as lifecycle_module
    import events_diary.investigation_signals as signal_module
    import events_diary.investigation_temporal as temporal_module
    import events_diary.ticket_scroll as scroll_module
    import events_diary.video as video_module
    from events_diary.investigation_detection import (
        InvestigationCandidate,
        InvestigationCandidateKind,
    )
    from events_diary.main import detect_investigations
    from events_diary.video import SampledFrame, VideoMetadata

    video_path = tmp_path / "source.mp4"
    metadata = VideoMetadata(video_path, 0.3, 10.0, 3, 1920, 1080)
    samples = [
        SampledFrame(index / 10, index / 10, index, np.zeros((1080, 1920, 3), dtype=np.uint8))
        for index in range(3)
    ]
    order = []
    evidence = object()
    signal = SimpleNamespace(ticket_scroll_evidence=evidence)
    candidate = InvestigationCandidate(InvestigationCandidateKind.NONE, "test pair")

    monkeypatch.setattr(video_module, "sample_video", lambda path, fps: order.append("sample") or samples)
    monkeypatch.setattr(signal_module, "estimate_investigation_presence",
                        lambda frame, config: order.append("presence") or object())
    monkeypatch.setattr(signal_module, "analyze_investigation_pair",
                        lambda first, second, config: order.append("pair") or signal)
    monkeypatch.setattr(signal_module, "is_investigation_present",
                        lambda item: order.append("classify presence") or False)
    monkeypatch.setattr(temporal_module, "bridge_transient_absence",
                        lambda timeline: order.append("bridge") or (False, False, False))
    monkeypatch.setattr(detection_module, "detect_investigation_candidate",
                        lambda *args, **kwargs: order.append("candidate") or candidate)
    monkeypatch.setattr(scroll_module, "analyze_ticket_scroll_sequence",
                        lambda items: order.append("scroll sequence") or ("quiet", "quiet"))
    monkeypatch.setattr(temporal_module, "confirm_investigation_candidates",
                        lambda entries: order.append("confirm") or ((), ()))
    monkeypatch.setattr(lifecycle_module, "build_investigation_lifecycle",
                        lambda **kwargs: order.append("lifecycle") or ((), ()))

    result = detect_investigations(metadata)

    assert len(result.presences) == 3
    assert len(result.signals) == len(result.timeline) == 2
    assert result.intervals == ()
    assert order == [
        "sample", "presence", "presence", "presence", "pair", "pair",
        "classify presence", "classify presence", "classify presence", "bridge",
        "candidate", "candidate", "scroll sequence", "confirm", "lifecycle",
    ]


def test_all_understanding_modes_are_supported_after_phase_9_3():
    assert MODE_READINESS[UnderstandingMode.OCR_ONLY] is UnderstandingModeReadiness.SUPPORTED_NOW
    assert MODE_READINESS[UnderstandingMode.VLM_ONLY] is UnderstandingModeReadiness.SUPPORTED_NOW
    assert MODE_READINESS[UnderstandingMode.OCR_VLM] is UnderstandingModeReadiness.SUPPORTED_NOW


def test_vlm_configuration_remains_centralized_and_secret_is_hidden(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("EVENTS_DIARY_UNDERSTANDING_MODE", "vlm_only")
    monkeypatch.setenv("EVENTS_DIARY_VLM_PROVIDER", "gemini")
    monkeypatch.setenv("EVENTS_DIARY_VLM_API_KEY", "test-secret-value")
    monkeypatch.setenv("EVENTS_DIARY_VLM_API_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("EVENTS_DIARY_VLM_API_VERSION", "v1beta")
    monkeypatch.setenv("EVENTS_DIARY_VLM_MODEL", "test-model")
    monkeypatch.setenv("EVENTS_DIARY_VLM_REQUEST_TIMEOUT_S", "222")
    monkeypatch.setenv("EVENTS_DIARY_VLM_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("EVENTS_DIARY_VLM_RETRY_BASE_DELAY_S", "7")
    monkeypatch.setenv("EVENTS_DIARY_VLM_RETRY_MAX_DELAY_S", "40")
    monkeypatch.setenv("EVENTS_DIARY_VLM_RETRY_JITTER_S", "0.5")

    settings = load_runtime_config(project_root=tmp_path, env_file=env_file).vlm_api_settings

    assert settings is not None
    assert settings.provider == "gemini"
    assert settings.api_key == "test-secret-value"
    assert settings.base_url == "https://example.invalid"
    assert settings.api_version == "v1beta"
    assert settings.model_name == "test-model"
    assert settings.request_timeout_s == 222
    assert settings.max_attempts == 2
    assert settings.retry_base_delay_s == 7
    assert settings.retry_max_delay_s == 40
    assert settings.retry_jitter_s == 0.5
    assert "test-secret-value" not in repr(settings)


def test_importing_main_and_package_entrypoint_has_no_processing_side_effects():
    code = """
import sys
import events_diary.main
import events_diary.__main__
forbidden = {
    'events_diary.video', 'events_diary.investigation_signals',
    'events_diary.investigation_temporal', 'events_diary.frame_materialization',
    'events_diary.ticket_ocr', 'events_diary.vlm', 'events_diary.gemini_api',
    'events_diary.fusion', 'events_diary.investigation_record',
}
assert forbidden.isdisjoint(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

def test_main_returns_final_output_result(tmp_path, monkeypatch) -> None:
    import events_diary.main as main_module

    input_dir = tmp_path / "experiment" / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "sample.mp4").touch()
    config = RuntimeConfig(
        UnderstandingMode.OCR_ONLY,
        input_dir,
        tmp_path / "experiment" / "output",
    )
    from types import SimpleNamespace

    expected = SimpleNamespace(failures=(), investigations=())
    output = object()
    monkeypatch.setattr(main_module, "prepare_video_pipeline", lambda received: object())
    monkeypatch.setattr(main_module, "run_understanding", lambda *args, **kwargs: expected)
    monkeypatch.setattr(
        main_module, "finalize_pipeline_result",
        lambda result, directory, mode: output,
    )

    assert main(config) is output


def test_main_uses_dotenv_loaded_mode_without_code_changes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import events_diary.main as main_module

    config = RuntimeConfig(
        UnderstandingMode.VLM_ONLY,
        tmp_path / "input",
        tmp_path / "output",
        VLMAPISettings("gemini", "", "https://example.invalid", "v1beta", "model", 180),
    )
    expected = object()
    received = []
    monkeypatch.setattr(main_module, "configure_logging", lambda _: None)
    monkeypatch.setattr(
        main_module, "get_runtime_profile",
        lambda: SimpleNamespace(operating_system="test", python_version="3", machine="test"),
    )
    monkeypatch.setattr(main_module, "load_runtime_config", lambda: config)
    monkeypatch.setattr(main_module, "run_pipeline", lambda value: received.append(value) or expected)

    assert main() is expected
    assert received == [config]
    assert received[0].understanding_mode is UnderstandingMode.VLM_ONLY
