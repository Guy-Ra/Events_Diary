"""Focused tests for model-independent VLM data and runtime contracts."""

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from events_diary.vlm import (
    RawVLMResult,
    VLMFieldObservation,
    VLMInvestigationResult,
    VLMRequest,
    VLMRuntime,
)

SOURCE_VIDEO = Path("data/videos/session.mp4")
CLIP = Path("data/debug/vlm/investigation_2.mp4")
MODEL = "gemini-flash-latest"
REVISION = "pinned-revision"
PROMPT_VERSION = "fr24_vlm_v1"


def request(**overrides):
    """Create a valid continuous Investigation request for contract tests."""
    values = {
        "investigation_index": 2,
        "source_video": SOURCE_VIDEO,
        "video_clip_path": CLIP,
        "source_start_s": 12.0,
        "source_end_s": 20.4,
        "prompt": "Read this complete aircraft Investigation.",
    }
    values.update(overrides)
    return VLMRequest(**values)


def raw_result(**overrides):
    """Create a valid raw runtime result with reproducible source metadata."""
    values = {
        "execution_status": "success",
        "investigation_index": 2,
        "source_video": SOURCE_VIDEO,
        "video_clip_path": CLIP,
        "source_start_s": 12.0,
        "source_end_s": 20.4,
        "model_name": MODEL,
        "model_revision": REVISION,
        "prompt_version": PROMPT_VERSION,
        "raw_text": '{"airline_name":"Israir Airlines"}',
        "inference_time_s": 4.2,
        "ttft_s": 1.1,
        "generated_tokens": 18,
    }
    values.update(overrides)
    return RawVLMResult(**values)


def observation(**overrides):
    """Create a valid observed canonical field with matching provenance."""
    values = {
        "field_name": "airline_name",
        "raw_value": "Israir Airlines",
        "status": "observed",
        "investigation_index": 2,
        "source_video": SOURCE_VIDEO,
        "source_start_s": 12.0,
        "source_end_s": 20.4,
        "model_name": MODEL,
        "model_revision": REVISION,
        "prompt_version": PROMPT_VERSION,
    }
    values.update(overrides)
    return VLMFieldObservation(**values)


def test_request_preserves_continuous_video_and_prompt_identity():
    result = request()
    assert result.investigation_index == 2
    assert result.source_video == SOURCE_VIDEO
    assert result.video_clip_path == CLIP
    assert (result.source_start_s, result.source_end_s) == (12.0, 20.4)
    assert result.prompt_version == "fr24_vlm_v1"
    assert not hasattr(result, "frames")
    with pytest.raises(FrozenInstanceError):
        result.prompt = "changed"


def test_successful_raw_result_preserves_diagnostics_and_provenance():
    result = raw_result()
    assert result.execution_status == "success"
    assert result.raw_text == '{"airline_name":"Israir Airlines"}'
    assert result.inference_time_s == 4.2
    assert result.ttft_s == 1.1
    assert result.generated_tokens == 18
    assert result.model_revision == REVISION
    assert result.prompt_version == PROMPT_VERSION


@pytest.mark.parametrize("status", ["timeout", "runtime_error", "input_error", "invalid_response"])
def test_failed_raw_result_keeps_failure_as_data(status):
    result = raw_result(execution_status=status, raw_text=None, error="diagnostic")
    assert result.execution_status == status
    assert result.raw_text is None
    assert result.error == "diagnostic"


def test_field_observation_supports_observed_unavailable_and_optional_time():
    observed = observation(evidence_time_s=16.2)
    unavailable = observation(field_name="serial_number", raw_value=None, status="unavailable")
    assert observed.raw_value == "Israir Airlines"
    assert observed.evidence_time_s == 16.2
    assert unavailable.status == "unavailable"
    assert unavailable.raw_value is None
    assert observation().evidence_time_s is None


def test_field_status_does_not_infer_absence_and_ambiguous_can_retain_raw_text():
    result = observation(field_name="SERIAL_NUMBER", raw_value="1100 or 738209", status="ambiguous")
    assert result.status == "ambiguous"
    assert result.raw_value == "1100 or 738209"


def test_canonical_names_remain_open_without_ocr_model_coupling():
    names = ("airline_name", "primary_flight_id", "origin_airport_code", "SERIAL_NUMBER")
    fields = tuple(observation(field_name=name) for name in names)
    assert tuple(item.field_name for item in fields) == names


def test_successful_investigation_result_exposes_nested_provenance():
    raw = raw_result()
    result = VLMInvestigationResult(raw, [observation()])
    assert result.execution_status == "success"
    assert result.investigation_index == 2
    assert result.model_name == MODEL
    assert result.model_revision == REVISION
    assert result.prompt_version == PROMPT_VERSION
    assert (result.source_start_s, result.source_end_s) == (12.0, 20.4)
    assert result.inference_time_s == 4.2
    assert result.error is None
    assert isinstance(result.observations, tuple)


@pytest.mark.parametrize("status", ["timeout", "runtime_error", "input_error", "invalid_response"])
def test_failed_investigation_result_is_valid_with_empty_observations(status):
    failed = raw_result(execution_status=status, raw_text=None, inference_time_s=None,
        ttft_s=None, generated_tokens=None, error="runtime unavailable")
    result = VLMInvestigationResult(failed)
    assert result.execution_status == status
    assert result.observations == ()
    assert result.error == "runtime unavailable"


def test_failed_result_rejects_observations_and_mismatched_provenance():
    failed = raw_result(execution_status="timeout", raw_text=None, error="timeout")
    with pytest.raises(ValueError, match="empty observations"):
        VLMInvestigationResult(failed, (observation(),))
    with pytest.raises(ValueError, match="provenance"):
        VLMInvestigationResult(raw_result(), (observation(investigation_index=9),))


def test_runtime_protocol_has_only_the_minimal_run_contract():
    class Runtime:
        def run(self, item: VLMRequest) -> RawVLMResult:
            return raw_result()

    runtime = Runtime()
    assert callable(VLMRuntime.run)
    assert callable(runtime.run)
    assert runtime.run(request()).execution_status == "success"


def test_vlm_module_does_not_import_ocr_models_or_functions():
    module_path = Path(__import__("events_diary.vlm", fromlist=["__file__"]).__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
    assert not any(name == "events_diary.ocr" or name.startswith("events_diary.ocr_")
        for name in imported_modules)
