"""Deterministic OCR TRACK-to-compass conversion and provenance tests."""

from dataclasses import replace

import pytest

from events_diary.config import ROI
from events_diary.ocr_fields import extract_ocr_fields
from events_diary.ticket_ocr import OCRDetection, OCRTicketResult
from events_diary.track_direction import (
    compare_track_direction,
    compass_direction,
    derive_track_direction,
)


def _track_observation(value: str):
    """Build a real extracted TRACK field from local synthetic OCR detections."""
    roi = ROI(0, 0, 400, 800)
    detections = (
        OCRDetection("TRACK", .99, ((10, 10), (80, 10), (80, 24), (10, 24))),
        OCRDetection(value, .98, ((10, 34), (80, 34), (80, 50), (10, 50))),
    )
    result = OCRTicketResult(
        "RapidOCR/test", "full_ticket_color_cubic_2x", roi, 2, detections,
        0, 0, 1, 1.0, "resolved",
    )
    return next(item for item in extract_ocr_fields(result) if item.canonical_label == "TRACK")


@pytest.mark.parametrize(("degrees", "sector", "name"), [
    (0, "N", "North"), (1, "N", "North"), (11.24, "N", "North"),
    (11.25, "NNE", "North-Northeast"), (22.5, "NNE", "North-Northeast"),
    (45, "NE", "Northeast"), (90, "E", "East"), (103, "ESE", "East-Southeast"),
    (180, "S", "South"), (270, "W", "West"), (337.5, "NNW", "North-Northwest"),
    (348.74, "NNW", "North-Northwest"), (348.75, "N", "North"),
    (359, "N", "North"), (360, "N", "North"),
])
def test_nearest_centered_sixteen_point_compass(degrees, sector, name):
    assert compass_direction(degrees) == (sector, name)


@pytest.mark.parametrize("raw", ["103°", "103", " 103 deg "])
def test_derived_direction_preserves_original_track_and_ocr_provenance(raw):
    observation = _track_observation(raw)
    derived = derive_track_direction(observation)
    assert derived is not None
    assert derived.source_observation is observation
    assert derived.source_observation.raw_value_text == raw
    assert (derived.sector, derived.text) == ("ESE", "East-Southeast")
    assert derived.provenance == "deterministic_derived_from_ocr_track"


@pytest.mark.parametrize("raw", ["", "east", "361°", "-1°", "nan", "inf"])
def test_missing_malformed_and_out_of_range_track_values_are_unavailable(raw):
    observation = _track_observation(raw)
    assert derive_track_direction(observation) is None
    assert derive_track_direction(replace(observation, status="no_value_selected")) is None


def test_direction_cross_check_preserves_both_values_and_never_selects_a_winner():
    derived = derive_track_direction(_track_observation("103°"))
    same = compare_track_direction(derived, "ESE")
    mismatch = compare_track_direction(derived, "W")
    unavailable = compare_track_direction(derived, None)
    assert same.status == "consistent"
    inferred = compare_track_direction(
        derived, "ESE", visual_direction_status="inferred"
    )
    assert inferred.status == "consistent"
    assert inferred.visual_direction_status == "inferred"
    assert mismatch.status == "inconsistent"
    assert mismatch.derived_direction is derived
    assert mismatch.visual_direction_sector == "W"
    assert unavailable.status == "unavailable"
    assert compare_track_direction(None, "ESE").status == "unavailable"
