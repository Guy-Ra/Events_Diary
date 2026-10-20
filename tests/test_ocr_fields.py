"""Canonical recognition, conservative extraction and raw-provenance regression tests."""

from dataclasses import FrozenInstanceError, asdict, fields, replace

import pytest

from events_diary import ocr_fields as api
from events_diary.config import ROI
from events_diary.ocr_spatial import build_spatial_observations, build_spatial_relations
from events_diary.ticket_ocr import OCRDetection, OCRTicketResult


def result(*items,inv=2,candidate=7):
    """Build real typed OCR records without invoking an OCR runtime or reading images."""
    roi = ROI(16,121,336,831)
    detections = tuple(OCRDetection(text,confidence,((roi.x+x,roi.y+y),(roi.x+x+w,roi.y+y),
        (roi.x+x+w,roi.y+y+h),(roi.x+x,roi.y+y+h))) for text,(x,y,w,h),confidence in items)
    return OCRTicketResult("RapidOCR/test","full_ticket_color_cubic_2x",roi,2,detections,inv,candidate,40,4.,"resolved")


def recognize(text,**options):
    """Recognize one raw text observation with known nonzero polygon geometry."""
    return api.recognize_canonical_labels(build_spatial_observations(result((text,(10,20,100,12),.9))),**options)


@pytest.mark.parametrize("raw,canonical", [
    (" SCHEDULED ","SCHEDULED"),("actual","ACTUAL"),("ESTIMATED","ESTIMATED"),
    ("REGISTRATION","REGISTRATION"),("COUNTRY OFREG.","COUNTRY_OF_REGISTRATION"),
    ("SERIAL NUMBER (MSN)","SERIAL_NUMBER"),("AGE (Mar 2010)","AGE"),
    ("AIRCRAFTCATEGORY","AIRCRAFT_CATEGORY"),("AIRCRAFT TYPE (A332)","AIRCRAFT_TYPE"),
    ("BAROMETRICALT.","BAROMETRIC_ALTITUDE"),("VERTICALSPEED","VERTICAL_SPEED"),
    ("GPS ALTITUDE","GPS_ALTITUDE"),("TRACK","TRACK"),("GROUND SPEED","GROUND_SPEED"),
    ("TRUEAIRSPEED","TRUE_AIRSPEED"),("INDICATED AIRSPEED","INDICATED_AIRSPEED"),
    ("MACH","MACH"),("WIND","WIND"),("TEMPERATURE","TEMPERATURE"),("FIR/UIR","FIR_UIR"),
    ("ICAO24-BITADDRESS","ICAO_ADDRESS"),("SQUAWK","SQUAWK"),("LATITUDE","LATITUDE"),("LONGITUDE","LONGITUDE")])
def test_all_calibrated_identities_preserve_raw_text(raw,canonical):
    matches = recognize(raw)
    assert len(matches)==1 and matches[0].canonical_label==canonical
    assert matches[0].raw_text==raw
    assert not matches[0].compound


@pytest.mark.parametrize("text,canonical", [("AGE","AGE"),("AGE(Feb1999)","AGE"),("age (Jul 2024)","AGE"),
    ("SERIALNUMBER","SERIAL_NUMBER"),("AIRCRAFT TYPE (B738)","AIRCRAFT_TYPE"),("AIRCRAFT TYPE (A21N)","AIRCRAFT_TYPE")])
def test_stable_qualifiers_do_not_become_canonical_identities(text,canonical):
    assert recognize(text)[0].canonical_label==canonical


@pytest.mark.parametrize("text", ["16 years","1100","281°","9 kts","N/A","Israir Airlines","ISR972", "TLV",
    "NEW YORK","arbitrary text","AGELESS","MACHINERY","TRACKING","GROUND SPEEDING","SCHEDULED123",
    "AGE (unknown)","SERIAL NUMBER (XYZ)","", "   "])
def test_values_names_identifiers_and_unsupported_qualifiers_remain_unknown(text):
    assert recognize(text)==()


def test_compound_matches_share_whole_raw_geometry_with_distinct_character_spans():
    raw = result(("SERIALNUMBER(MSN)AGE(Mar2010)",(10,20,280,12),.99),
        ("1100",(10,36,55,16),.93),("16 years",(180,36,60,16),.94))
    obs = build_spatial_observations(raw)
    matches = api.recognize_canonical_labels(obs)
    assert [m.canonical_label for m in matches]==["SERIAL_NUMBER","AGE"]
    assert all(m.compound and m.raw_detection_index==0 for m in matches)
    assert matches[0].observation is matches[1].observation is obs[0]
    assert matches[0].source_polygon is matches[1].source_polygon is raw.detections[0].source_polygon
    assert matches[0].matched_text_span!=matches[1].matched_text_span
    assert raw.detections[0].text[slice(*matches[1].matched_text_span)]=="AGE(Mar2010)"
    extracted = api.extract_ocr_fields(raw)
    assert len(extracted)==2
    assert all(f.status=="ambiguous" and f.compound_label and f.raw_value_text is None for f in extracted)
    assert all(f.reason=="compound_shared_geometry" and f.relationship_score is None for f in extracted)
    assert "source_bounds" not in {f.name for f in fields(matches[0])}


def test_compact_and_stable_success_bypass_fuzzy(monkeypatch):
    def forbidden(_):
        raise AssertionError("Fuzzy ran despite successful calibrated grammar")
    monkeypatch.setattr(api,"_fuzzy_candidates",forbidden)
    assert recognize("BAROMETRICALT.")[0].match_method=="compact"
    assert recognize("AGE(May2020)")[0].match_method=="stable_token"
    assert all(m.match_method=="compound_stable" for m in recognize("SERIALNUMBER(MSN)AGE(Mar2010)"))


def test_bounded_fuzzy_keeps_raw_text_and_respects_inclusive_similarity():
    match = recognize("1CAO 24-BIT ADDRESS",fuzzy_threshold=.9375)[0]
    assert match.canonical_label=="ICAO_ADDRESS" and match.match_method=="bounded_fuzzy"
    assert match.match_score==.9375 and match.raw_text=="1CAO 24-BIT ADDRESS"
    assert match.matched_text_span is None
    assert recognize("1CAO 24-BIT ADDRESS",fuzzy_threshold=.94)==()


def test_fuzzy_margin_and_ties_abstain(monkeypatch):
    monkeypatch.setattr(api,"_fuzzy_candidates",lambda _:[(.95,"TRACK"),(.90,"MACH")])
    assert recognize("unknown near label")==()
    monkeypatch.setattr(api,"_fuzzy_candidates",lambda _:[(1.,"TRACK"),(.5,"MACH")])
    assert recognize("unknown near label",fuzzy_margin=.5)[0].canonical_label=="TRACK"
    monkeypatch.setattr(api,"_fuzzy_candidates",lambda _:[(.95,"TRACK"),(.95,"MACH")])
    assert recognize("unknown near label",fuzzy_margin=0)==()


def test_clean_stack_selection_opposite_column_competitor_and_confidence_separation():
    raw = result(("BAROMETRICALT.",(10,20,100,12),.81),("0ft",(10,36,50,16),.92),
        ("opposite",(200,36,50,16),.999))
    field = api.extract_ocr_fields(raw)[0]
    assert field.canonical_label=="BAROMETRIC_ALTITUDE" and field.status=="extracted"
    assert field.raw_value_text=="0ft" and field.value_detection_index==1
    assert field.relationship_type=="stack" and field.relationship_score>.5
    assert (field.ocr_label_confidence,field.ocr_value_confidence,field.label_match_score)==(.81,.92,1.)
    assert field.selected_candidate.observation.detection is raw.detections[1]
    assert field.selected_candidate.relation.source_index==0 and field.selected_candidate.relation.target_index==1
    obs = build_spatial_observations(raw)
    matches = api.recognize_canonical_labels(obs)
    ranked = api.rank_ocr_field_candidates(matches[0],obs,build_spatial_relations(obs),label_matches=matches)
    assert [c.observation.detection_index for c in ranked]==[1,2]


def test_repeated_inline_labels_select_independent_times_without_semantic_roles():
    raw = result(("SCHEDULED",(0,20,75,20),.8),("2:30 pM",(90,20,65,20),.9),
        ("SCHEDULED",(165,20,75,20),.82),("8:45 AM",(253,20,65,20),.91))
    fields_out = api.extract_ocr_fields(raw)
    assert [f.canonical_label for f in fields_out]==["SCHEDULED","SCHEDULED"]
    assert [f.label_detection_index for f in fields_out]==[0,2]
    assert [f.raw_value_text for f in fields_out]==["2:30 pM","8:45 AM"]
    assert all(f.relationship_type=="inline" and f.status=="extracted" for f in fields_out)
    assert all(f.selected_candidate.relation.delta_center_y==0 for f in fields_out)
    assert not any("arrival" in f.canonical_label or "departure" in f.canonical_label for f in fields_out)


def test_ambiguous_near_candidates_and_equal_ties_are_not_forced():
    raw = result(("TRACK",(10,20,100,12),.9),("one",(10,36,50,16),.9),("two",(11,36,50,16),.9))
    field = api.extract_ocr_fields(raw)[0]
    assert field.status=="ambiguous" and field.reason=="candidate_margin"
    assert field.raw_value_text is None and field.value_detection_index is None
    equal = replace(raw,detections=(raw.detections[0],raw.detections[1],replace(raw.detections[1],text="other")))
    assert api.extract_ocr_fields(equal,min_candidate_margin=0)[0].ambiguous


def test_distinct_labels_with_identical_native_polygons_keep_shared_geometry_ambiguity():
    raw = result(("TRACK",(10,20,100,12),.9),("GROUND SPEED",(10,20,100,12),.9),
        ("arbitrary value",(10,36,50,16),.9))
    fields_out = api.extract_ocr_fields(raw)
    assert all(f.ambiguous and f.shared_geometry and not f.compound_label for f in fields_out)
    assert all(f.reason=="shared_label_geometry" and f.raw_value_text is None for f in fields_out)


@pytest.mark.parametrize("text", ["COUNTRY OF REG.","FIR/UIR"])
def test_non_text_special_fields_never_force_nearby_value(text):
    field = api.extract_ocr_fields(result((text,(10,20,100,12),.9),("nearby unrelated",(10,36,50,16),.99)))[0]
    assert field.status=="special_non_text" and field.raw_value_text is None
    assert field.raw_label_text==text and field.value_detection_index is None


def test_obvious_label_candidate_is_kept_for_diagnostics_but_not_selected_as_value():
    raw = result(("TRACK",(10,20,100,12),.9),("GROUND SPEED",(10,34,100,12),.9),("281°",(10,48,50,16),.9))
    obs = build_spatial_observations(raw)
    matches = api.recognize_canonical_labels(obs)
    ranked = api.rank_ocr_field_candidates(matches[0],obs,build_spatial_relations(obs),label_matches=matches)
    assert ranked[0].is_label_candidate and ranked[0].observation.text=="GROUND SPEED"
    field = api.extract_ocr_fields(raw)[0]
    assert field.raw_value_text=="281°" and field.value_detection_index==2


def test_weak_or_absent_candidates_abstain_without_parsing_value_formats():
    alone = api.extract_ocr_fields(result(("TRACK",(10,20,100,12),.9)))[0]
    assert alone.status=="no_value_selected" and alone.reason=="no_eligible_candidate"
    wrong_direction = api.extract_ocr_fields(result(("TRACK",(10,200,100,12),.9),("whatever",(10,20,50,16),.9)))[0]
    assert wrong_direction.status=="no_value_selected" and wrong_direction.reason=="weak_spatial_candidate"
    arbitrary = api.extract_ocr_fields(result(("TRACK",(10,20,100,12),.9),("not a degree value",(10,36,50,16),.9)))[0]
    assert arbitrary.raw_value_text=="not a degree value"


def test_raw_models_frozen_provenance_and_input_immutability_determinism():
    raw = result(("REGISTRATION",(10,20,100,12),.7),("4X-BAL",(10,36,50,16),.8))
    snapshot = asdict(raw)
    field = api.extract_ocr_fields(raw)[0]
    for model,name in [(field,"status"),(field.label_match,"canonical_label"),(field.provenance,"frame_index"),(field.selected_candidate,"is_label_candidate")]:
        with pytest.raises(FrozenInstanceError):
            setattr(model,name,"changed")
    assert asdict(raw)==snapshot and field==api.extract_ocr_fields(raw)[0]
    assert (field.provenance.investigation_index,field.provenance.candidate_index,field.provenance.frame_index,
        field.provenance.actual_timestamp_seconds)==(2,7,40,4.)
    assert field.provenance.resolution_status=="resolved"
    assert field.label_match.source_polygon is raw.detections[0].source_polygon
    assert field.selected_candidate.observation.source_polygon is raw.detections[1].source_polygon
    assert not {f.name for f in fields(field)} & {"global_confidence","origin","destination","departure","arrival"}


def test_empty_unknown_only_and_batch_order_keep_representatives_separate():
    first = result(("TRACK",(10,20,100,12),.9),("281°",(10,36,50,16),.9),inv=3,candidate=9)
    second = result(("TRACK",(10,20,100,12),.9),("295°",(10,36,50,16),.9),inv=4,candidate=1)
    batch = api.extract_ocr_fields_batch((first,result(),second))
    assert len(batch)==3 and batch[1]==()
    assert [batch[0][0].raw_value_text,batch[2][0].raw_value_text]==["281°","295°"]
    assert batch[0][0].provenance.investigation_index==3 and batch[2][0].provenance.investigation_index==4
    assert api.extract_ocr_fields(result(("TLV",(10,20,100,12),.9),("NEW YORK",(10,36,50,16),.9)))==()
    assert api.extract_ocr_fields_batch(())==()


@pytest.mark.parametrize("option", ["fuzzy_threshold","fuzzy_margin","min_relationship_score","min_candidate_margin"])
@pytest.mark.parametrize("value,error", [(True,TypeError),(".9",TypeError),(None,TypeError),(-.1,ValueError),(1.1,ValueError),(float('nan'),ValueError),(float('inf'),ValueError)])
def test_invalid_policy_inputs_are_not_silently_repaired(option,value,error):
    with pytest.raises(error,match=option):
        api.extract_ocr_fields(result(),**{option:value})


def test_bad_parent_provenance_rejected_without_model_execution():
    with pytest.raises(TypeError):
        api.extract_ocr_fields(None)
    with pytest.raises(ValueError,match="provenance"):
        api.extract_ocr_fields(replace(result(),resolution_status="unresolved"))
    with pytest.raises(TypeError):
        api.extract_ocr_fields_batch("not a batch")
