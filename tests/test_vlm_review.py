"""Contract, prompt, parser, and mocked Gemini tests for full-video Pass 2."""

import base64
import json
from pathlib import Path

import pytest

from events_diary import gemini_api
from events_diary.config import ROI, VLMAPISettings
from events_diary.fusion import (
    FusedFieldObservation,
    FusedInvestigationResult,
    resolve_investigation,
)
from events_diary.gemini_api import GeminiAPIRuntime
from events_diary.ocr_fields import extract_ocr_fields
from events_diary.ticket_ocr import OCRDetection, OCRTicketResult
from events_diary.vlm import (
    RawVLMResult,
    VLMFieldObservation,
    VLMInvestigationResult,
    VLMRequest,
    VLMVisualRouteClaim,
    VLMVisualRouteObservation,
)
from events_diary.vlm_prompts import build_vlm_prompt
from events_diary.vlm_review import (
    EVIDENCE_QUALITIES,
    FR24_VLM_REVIEW_V1,
    HIGH_PRIORITY_STATUSES,
    REVIEW_FIELD_NAMES,
    REVIEW_STATUSES,
    VLMReviewObservation,
    build_vlm_review_context,
    build_vlm_review_prompt,
    build_vlm_review_request,
    parse_vlm_review_result,
)

SOURCE = Path("data/videos/session.mp4")
CLIP = Path("data/debug/vlm/investigation_2.mp4")
MODEL = "gemini-flash-latest"
REVISION = "provider-version-123"


def base_request(**overrides):
    """Create a Pass 1 request identifying a full continuous Investigation clip."""
    values = {
        "investigation_index": 2,
        "source_video": SOURCE,
        "video_clip_path": CLIP,
        "source_start_s": 12.0,
        "source_end_s": 20.4,
        "prompt": build_vlm_prompt(),
    }
    values.update(overrides)
    return VLMRequest(**values)


def ocr_field(value="4X-BAL", *, label="REGISTRATION", timestamp=16.2):
    """Build one real OCR extraction record without an OCR runtime call."""
    roi = ROI(16, 121, 336, 831)
    detections = (
        OCRDetection(label, .99, ((26, 141), (126, 141), (126, 153), (26, 153))),
        OCRDetection(value, .98, ((26, 161), (86, 161), (86, 177), (26, 177))),
    )
    ticket = OCRTicketResult(
        "RapidOCR/test", "full_ticket_color_cubic_2x", roi, 2, detections,
        2, 7, 40, timestamp, "resolved",
    )
    return next(
        field for field in extract_ocr_fields(ticket)
        if field.canonical_label == label
    )


def vlm_field(name="REGISTRATION", value="4X-BAL", *, evidence_time_s=None):
    """Create one source-specific Pass 1 field observation."""
    return VLMFieldObservation(
        field_name=name,
        raw_value=value,
        status="observed" if value is not None else "unavailable",
        investigation_index=2,
        source_video=SOURCE,
        source_start_s=12.0,
        source_end_s=20.4,
        model_name=MODEL,
        model_revision=REVISION,
        prompt_version="fr24_vlm_v1",
        evidence_time_s=evidence_time_s,
    )


def fusion_result():
    """Create accepted Pass 1 and cross-check evidence for review tests."""
    ocr_registration = ocr_field(" 4X-BAL ")
    pass1_registration = vlm_field()
    pass1_track = vlm_field("TRACK", "281ֲ°")
    raw_pass1 = RawVLMResult(
        execution_status="success",
        investigation_index=2,
        source_video=SOURCE,
        video_clip_path=CLIP,
        source_start_s=12.0,
        source_end_s=20.4,
        model_name=MODEL,
        model_revision=REVISION,
        prompt_version="fr24_vlm_v1",
        raw_text='{"registration":"4X-BAL"}',
    )
    pass1_route = VLMVisualRouteObservation(
        VLMVisualRouteClaim("ESE", "observed"),
        VLMVisualRouteClaim("Generally east-southeast", "inferred"),
        VLMVisualRouteClaim("Trail extends west of aircraft", "observed"),
        VLMVisualRouteClaim(None, "not_visible"),
        2, SOURCE, 12.0, 20.4, MODEL, REVISION, "fr24_vlm_v1",
    )
    pass1_result = VLMInvestigationResult(
        raw_pass1, (pass1_registration, pass1_track), visual_route=pass1_route
    )
    fields = {
        "REGISTRATION": FusedFieldObservation(
            "REGISTRATION", "agreement", ocr_registration, pass1_registration,
            "4X-BAL", "4X-BAL", "4X-BAL", ("values match after normalization",),
        ),
        "TRACK": FusedFieldObservation(
            "TRACK", "temporal_mismatch", ocr_field("278ֲ°", label="TRACK"), pass1_track,
            "278", "281", None,
            ("dynamic field values differ at non-comparable source times",),
        ),
        "SQUAWK": FusedFieldObservation("SQUAWK", "conflict"),
    }
    result = FusedInvestigationResult(2, SOURCE, 12.0, 20.4, fields)
    return result, pass1_result, ocr_registration


def review_request(fused=None):
    """Build one full-video review request from matching Pass 1 provenance."""
    generated_fused, pass1, _ = fusion_result()
    return build_vlm_review_request(
        base_request(), fused or generated_fused, pass1_result=pass1
    )


def raw_review_result(request, text, **overrides):
    """Create a Pass 2 raw runtime result with matching clip provenance."""
    values = {
        "execution_status": "success",
        "investigation_index": request.investigation_index,
        "source_video": request.source_video,
        "video_clip_path": request.video_clip_path,
        "source_start_s": request.source_start_s,
        "source_end_s": request.source_end_s,
        "model_name": MODEL,
        "model_revision": REVISION,
        "prompt_version": request.prompt_version,
        "raw_text": text,
    }
    values.update(overrides)
    return RawVLMResult(**values)


def review_payload(overrides=None):
    """Return a complete schema-shaped review response with nullable values."""
    fields = {}
    for name in REVIEW_FIELD_NAMES:
        fields[name] = {
            "reviewed_value": None,
            "review_status": "unresolved",
            "evidence_quality": "weak",
            "evidence_time_s": None,
            "notes": None,
        }
    fields.update(overrides or {})
    route = {
        name: {
            "reviewed_value": None,
            "review_status": "not_visible",
            "evidence_quality": "not_visible",
            "observation_status": "not_visible",
        }
        for name in (
            "visual_flight_direction_sector", "visual_flight_direction_text",
            "visual_route_description", "visual_departure_context",
        )
    }
    return {"fields": fields, "visual_route": route}


def test_review_context_contains_every_canonical_field_and_priorities():
    fused, _, _ = fusion_result()
    context = build_vlm_review_context(fused)
    assert set(context.fields) == set(REVIEW_FIELD_NAMES)
    assert context.fields["REGISTRATION"].priority == "normal"
    assert context.fields["TRACK"].priority == "high"
    assert context.fields["SQUAWK"].priority == "high"
    assert context.fields["TRACK"].fusion_status == "temporal_mismatch"
    context = build_vlm_review_context(fused, pass1_result=fusion_result()[1])
    assert context.visual_route.direction_sector.value == "ESE"
    assert context.direction_consistency.status == "inconsistent"
    assert context.direction_consistency.derived_direction.source_observation is not None
    assert HIGH_PRIORITY_STATUSES == {
        "conflict", "temporal_mismatch", "ocr_only", "vlm_only", "unavailable"
    }


@pytest.mark.parametrize(
    "status",
    ["conflict", "temporal_mismatch", "ocr_only", "vlm_only", "unavailable"],
)
def test_each_nonagreement_status_is_high_priority(status):
    from events_diary.vlm_review import VLMReviewFieldContext

    field = VLMReviewFieldContext(
        field_name="TRACK", fusion_status=status,
        ocr_raw_value=None, ocr_available=False,
        vlm_pass1_raw_value=None, vlm_pass1_available=False,
        ocr_normalized_value=None, vlm_normalized_value=None,
    )
    assert field.priority == "high"


def test_agreement_is_present_at_normal_priority_with_source_values():
    context = build_vlm_review_context(fusion_result()[0])
    registration = context.fields["REGISTRATION"]
    assert registration.priority == "normal"
    assert registration.ocr_raw_value == " 4X-BAL "
    assert registration.vlm_pass1_raw_value == "4X-BAL"
    assert registration.ocr_available is registration.vlm_pass1_available is True
    assert registration.ocr_normalized_value == registration.vlm_normalized_value == "4X-BAL"


def test_temporal_review_prompt_uses_ocr_time_and_does_not_invent_missing_time():
    request = review_request()
    assert "inspect around original-video time 16.2s" in request.prompt
    assert "fr24_vlm_review_v1" == request.prompt_version
    assert "Inspect every field" in request.prompt
    from events_diary.vlm_review import VLMReviewContext, VLMReviewFieldContext

    fields = dict(request.context.fields)
    fields["TRACK"] = VLMReviewFieldContext(
        field_name="TRACK", fusion_status="temporal_mismatch",
        ocr_raw_value="278ֲ°", ocr_available=True,
        vlm_pass1_raw_value="281ֲ°", vlm_pass1_available=True,
        ocr_normalized_value="278", vlm_normalized_value="281",
        ocr_timestamp_s=None,
    )
    no_time = VLMReviewContext(2, SOURCE, 12.0, 20.4, fields)
    prompt = build_vlm_review_prompt(no_time)
    assert "No field-specific OCR timestamp was supplied." in prompt
    assert "around original-video time 0s" not in prompt
    assert '"ocr_timestamp_s":null' in prompt


def test_prompt_is_full_review_and_does_not_trust_either_source():
    prompt = review_request().prompt
    assert "complete continuous Investigation video again" in prompt
    assert "do not skip agreement fields" in prompt
    assert "Do not assume OCR or VLM Pass 1 is correct" in prompt
    assert "Treat both values only as hypotheses" in prompt
    assert "Do not vote" in prompt
    assert "blindly repeat prior values" in prompt


@pytest.mark.parametrize("status", REVIEW_STATUSES)
def test_review_observation_accepts_each_review_status(status):
    observation = VLMReviewObservation("REGISTRATION", None, status, "not_visible")
    assert observation.review_status == status
    assert observation.reviewed_value is None


@pytest.mark.parametrize("quality", EVIDENCE_QUALITIES)
def test_review_observation_accepts_each_evidence_quality(quality):
    observation = VLMReviewObservation("TRACK", "281ֲ°", "confirmed", quality)
    assert observation.evidence_quality == quality


def test_review_observation_rejects_unknown_literals_and_preserves_text():
    with pytest.raises(ValueError, match="review_status"):
        VLMReviewObservation("TRACK", None, "maybe", "weak")
    with pytest.raises(ValueError, match="evidence_quality"):
        VLMReviewObservation("TRACK", None, "unresolved", "guess")
    raw = " 281ן¿½ "
    observation = VLMReviewObservation("TRACK", raw, "corrected", "partial")
    assert observation.reviewed_value == raw


@pytest.mark.parametrize("status", REVIEW_STATUSES)
def test_parser_accepts_review_statuses(status):
    request = review_request()
    payload = review_payload({
        "TRACK": {
            "reviewed_value": None,
            "review_status": status,
            "evidence_quality": "partial",
            "evidence_time_s": 16.2,
            "notes": "visible near the OCR sample",
        }
    })
    result = parse_vlm_review_result(request, raw_review_result(request, json.dumps(payload)))
    assert result.reviewed_fields["TRACK"].review_status == status
    assert result.reviewed_fields["TRACK"].reviewed_value is None
    assert result.reviewed_fields["TRACK"].evidence_time_s == 16.2


def test_parser_preserves_source_faithful_values_and_result_provenance():
    request = review_request()
    raw_value = " 281ן¿½ "
    payload = review_payload({
        "TRACK": {
            "reviewed_value": raw_value,
            "review_status": "corrected",
            "evidence_quality": "clear",
            "evidence_time_s": 16.2,
            "notes": "As displayed.",
        }
    })
    raw = raw_review_result(request, json.dumps(payload, ensure_ascii=False))
    result = parse_vlm_review_result(request, raw)
    assert result.reviewed_fields["TRACK"].reviewed_value == raw_value
    assert result.raw_result is raw
    assert result.investigation_id == 2
    assert result.source_video == SOURCE
    assert (result.source_start_s, result.source_end_s) == (12.0, 20.4)
    assert result.model_name == MODEL
    assert result.model_revision == REVISION
    assert result.prompt_version == FR24_VLM_REVIEW_V1
    assert set(result.route_reviews) == {
        "visual_flight_direction_sector", "visual_flight_direction_text",
        "visual_route_description", "visual_departure_context",
    }


def test_pass2_review_context_keeps_route_and_track_cross_check_separate_from_ticket():
    request = review_request()
    assert request.context.visual_route.direction_sector.value == "ESE"
    assert request.context.direction_consistency.status == "inconsistent"
    assert request.context.direction_consistency.derived_direction.text == "West"
    context_json = json.loads(request.prompt.split("Review context JSON:\n", 1)[1].split("\n\nReview the map", 1)[0])
    assert context_json["fields"]["origin_airport_code"]["ocr_raw_value"] is None
    assert context_json["visual_route"]["visual_departure_context"]["status"] == "not_visible"
    assert context_json["track_direction_cross_check"]["status"] == "inconsistent"


def test_pass2_parser_preserves_route_claim_statuses():
    request = review_request()
    payload = review_payload()
    payload["visual_route"]["visual_flight_direction_sector"] = {
        "reviewed_value": "ESE", "review_status": "confirmed",
        "evidence_quality": "clear", "observation_status": "observed",
    }
    result = parse_vlm_review_result(request, raw_review_result(request, json.dumps(payload)))
    item = result.route_reviews["visual_flight_direction_sector"]
    assert item.reviewed_value == "ESE"
    assert item.observation_status == "observed"
    final = resolve_investigation(fusion_result()[0], result)
    assert final.direction_consistency.status == "inconsistent"
    assert final.direction_consistency.derived_direction.source_observation.raw_value_text == (
        "278" + chr(0x05B2) + chr(176)
    )


def test_parser_rejects_malformed_json_with_diagnostic():
    request = review_request()
    result = parse_vlm_review_result(request, raw_review_result(request, "{not-json"))
    assert result.reviewed_fields == {}
    assert result.diagnostics == ("Pass 2 response is not valid JSON.",)


def test_parser_reports_missing_review_field():
    request = review_request()
    payload = review_payload()
    del payload["fields"]["TRACK"]
    result = parse_vlm_review_result(request, raw_review_result(request, json.dumps(payload)))
    assert "TRACK" not in result.reviewed_fields
    assert "Missing review field: TRACK." in result.diagnostics


def test_pass1_and_ocr_observations_remain_unchanged_after_review_request_build():
    fused, pass1, ocr = fusion_result()
    original_pass1 = tuple((item.field_name, item.raw_value) for item in pass1.observations)
    original_ocr = (ocr.raw_label_text, ocr.raw_value_text, ocr.status)
    request = build_vlm_review_request(base_request(), fused, pass1_result=pass1)
    assert request.context.fields["REGISTRATION"].ocr_raw_value == " 4X-BAL "
    assert tuple((item.field_name, item.raw_value) for item in pass1.observations) == original_pass1
    assert (ocr.raw_label_text, ocr.raw_value_text, ocr.status) == original_ocr


def test_gemini_review_request_sends_full_video_and_json_mode_without_schema(tmp_path, monkeypatch):
    clip = tmp_path / "complete_investigation.mp4"
    video_bytes = b"complete continuous Investigation MP4"
    clip.write_bytes(video_bytes)
    base = base_request(video_clip_path=clip)
    fused, pass1, _ = fusion_result()
    request = build_vlm_review_request(base, fused, pass1_result=pass1)
    captured = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({
                "candidates": [{"content": {"parts": [{"text": "{}"}]}}],
            }).encode()

    def fake_urlopen(http_request, *, timeout):
        captured.append(json.loads(http_request.data))
        return FakeResponse()

    monkeypatch.setattr(gemini_api, "urlopen", fake_urlopen)
    settings = VLMAPISettings(
        provider="gemini", api_key="fake-key",
        base_url="https://generativelanguage.googleapis.com", api_version="v1beta",
        model_name=MODEL, request_timeout_s=10.0,
    )
    result = GeminiAPIRuntime(settings).run_with_diagnostics(request)
    body = captured[0]
    parts = body["contents"][0]["parts"]
    media = [part for part in parts if "inlineData" in part]
    text_parts = [part["text"] for part in parts if "text" in part]
    assert len(media) == 1
    assert media[0]["inlineData"]["mimeType"] == "video/mp4"
    assert base64.b64decode(media[0]["inlineData"]["data"]) == video_bytes
    assert text_parts == [request.prompt]
    assert body["generationConfig"] == {"responseMimeType": "application/json"}
    assert "responseSchema" not in body["generationConfig"]
    assert "maxOutputTokens" not in body["generationConfig"]
    assert "temperature" not in body["generationConfig"]
    assert result.raw_result.prompt_version == FR24_VLM_REVIEW_V1


def test_gemini_adapter_imports_no_ocr_runtime_modules():
    source = Path(gemini_api.__file__).read_text(encoding="utf-8")
    assert "events_diary.ticket_ocr" not in source
    assert "rapidocr" not in source.lower()
