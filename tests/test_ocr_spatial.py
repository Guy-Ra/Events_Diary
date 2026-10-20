"""Synthetic geometry tests for raw spatial observations and ambiguous candidates."""

import math
from dataclasses import FrozenInstanceError, asdict, fields, replace

import pytest

from events_diary.config import ROI
from events_diary.ocr_spatial import (
    SpatialBounds,
    SpatialOCRObservation,
    build_spatial_observations,
    build_spatial_relations,
    interval_overlap,
    signed_interval_gap,
)
from events_diary.ticket_ocr import OCRDetection, OCRTicketResult

ROI_DEFAULT = ROI(100,50,200,400)
SCORES = ("same_column_score","same_row_score","vertical_stack_score","inline_score","local_neighbor_score")


def result(*boxes,roi=ROI_DEFAULT,texts=None):
    """Construct actual production OCR records from simple ticket-local rectangles."""
    texts = texts or [f"raw{i}" for i in range(len(boxes))]
    detections = tuple(OCRDetection(texts[i],.987,((x+roi.x,y+roi.y),
        (x+w+roi.x,y+roi.y),(x+w+roi.x,y+h+roi.y),(x+roi.x,y+h+roi.y)))
        for i,(x,y,w,h) in enumerate(boxes))
    return OCRTicketResult("RapidOCR/test","full_ticket_color_cubic_2x",roi,2,detections,0,1,4,.4,"resolved")


def relations(*boxes,**kwargs):
    """Call both production APIs without invoking OCR or calibration logic."""
    return build_spatial_relations(build_spatial_observations(result(*boxes,**kwargs)))


def test_models_frozen_and_raw_identity_text_confidence_preserved():
    raw = result((10,20,40,10),(10,35,40,10),texts=["  AGE O0:I1\n", "16 years"])
    observations = build_spatial_observations(raw)
    relation = build_spatial_relations(observations)[0]
    for model,name in [(observations[0],"detection_index"),(relation,"source_index"),(observations[0].source_bounds,"left")]:
        with pytest.raises(FrozenInstanceError):
            setattr(model,name,4)
    assert observations[0].detection is raw.detections[0]
    assert observations[0].source_polygon is raw.detections[0].source_polygon
    assert observations[0].text == "  AGE O0:I1\n"
    assert observations[0].ocr_confidence == .987
    assert isinstance(observations,tuple)


def test_rotated_polygon_aabb_centers_and_normalization_do_not_replace_vertices():
    polygon = ((110.,70.),(151.,65.),(154.,83.),(113.,88.))
    raw = replace(result((0,0,1,1)),detections=(OCRDetection(" raw ",None,polygon),))
    before = asdict(raw)
    o = build_spatial_observations(raw)[0]
    assert o.source_polygon==polygon
    assert o.source_bounds==SpatialBounds(110,65,154,88)
    assert (o.center_x,o.center_y,o.width,o.height)==(132,76.5,44,23)
    assert (o.normalized_left,o.normalized_top,o.normalized_right,o.normalized_bottom)==(.05,.0375,.27,.095)
    assert o.normalized_center_x==pytest.approx(.16)
    assert o.normalized_center_y==pytest.approx(.06625)
    assert o.normalized_width==pytest.approx(.22)
    assert o.normalized_height==pytest.approx(.0575)
    assert o.ocr_confidence is None
    assert asdict(raw)==before


def test_actual_crop_metadata_drives_proportional_normalization_and_scores():
    a = result((10,20,40,10),(12,35,35,12))
    b = result((5,10,20,5),(6,17.5,17.5,6),roi=ROI(50,25,100,200))
    oa,ob = build_spatial_observations(a),build_spatial_observations(b)
    assert oa[0].normalized_bounds==ob[0].normalized_bounds
    ra,rb = build_spatial_relations(oa)[0],build_spatial_relations(ob)[0]
    for feature in SCORES:
        assert getattr(ra,feature)==pytest.approx(getattr(rb,feature))
    assert rb.delta_center_y==ra.delta_center_y/2
    assert rb.normalized_delta_center_y==ra.normalized_delta_center_y


def test_real_crop_edge_overhang_is_not_clipped():
    o = build_spatial_observations(result((-2,-3,204,406)))[0]
    assert o.normalized_left<0 and o.normalized_top<0
    assert o.normalized_right>1 and o.normalized_bottom>1


@pytest.mark.parametrize("a,b,expected", [((0,10),(0,10),(10,1,1)),
    ((0,10),(10,20),(0,0,0)),((0,10),(12,20),(0,0,0)),
    ((0,10),(5,15),(5,.5,1/3)),((0,10),(2,6),(4,1,.4))])
def test_interval_overlap_exact_partial_containment_touch_and_disjoint(a,b,expected):
    assert interval_overlap(*a,*b)==pytest.approx(expected)
    assert interval_overlap(*b,*a)==pytest.approx(expected)


@pytest.mark.parametrize("a,b,gap", [((0,10),(15,25),5),((15,25),(0,10),5),
    ((0,10),(10,20),0),((0,10),(5,15),-5),((0,10),(2,6),-4)])
def test_signed_gap_symmetric_separation_contact_overlap_and_containment(a,b,gap):
    assert signed_interval_gap(*a,*b)==gap
    assert signed_interval_gap(*b,*a)==gap


def test_relation_overlaps_gaps_deltas_and_direction_reverse_correctly():
    a,b = relations((10,20,40,10),(30,25,40,20))
    assert (a.horizontal_overlap_pixels,a.horizontal_overlap_min_ratio,a.horizontal_overlap_union_ratio)==pytest.approx((20,.5,1/3))
    assert (a.vertical_overlap_pixels,a.vertical_overlap_min_ratio,a.vertical_overlap_union_ratio)==pytest.approx((5,.5,.2))
    assert a.signed_horizontal_gap_pixels==-20 and a.signed_vertical_gap_pixels==-5
    assert a.normalized_signed_horizontal_gap==-.1
    assert a.normalized_signed_vertical_gap==-.0125
    assert (a.delta_center_x,a.delta_center_y)==(20,10)
    assert a.target_is_below and a.target_is_right and not a.target_is_above and not a.target_is_left
    assert b.target_is_above and b.target_is_left
    assert b.delta_center_x==-a.delta_center_x and b.delta_center_y==-a.delta_center_y
    assert b.signed_horizontal_gap_pixels==a.signed_horizontal_gap_pixels
    assert b.signed_vertical_gap_pixels==a.signed_vertical_gap_pixels


def test_zero_overlap_and_positive_vertical_horizontal_gap():
    r = relations((0,0,10,10),(20,25,10,10))[0]
    assert r.horizontal_overlap_pixels==0 and r.vertical_overlap_pixels==0
    assert r.signed_horizontal_gap_pixels==10 and r.signed_vertical_gap_pixels==15


def test_equal_centers_neither_direction_is_asserted():
    r = relations((0,0,10,10),(0,0,10,10))[0]
    assert not any((r.target_is_above,r.target_is_below,r.target_is_left,r.target_is_right))
    assert r.same_column_score==r.same_row_score==r.local_neighbor_score==1
    assert r.vertical_stack_score==r.inline_score==0


def test_vertical_stack_prefers_aligned_near_below_and_suppresses_above_opposite():
    rs = relations((10,20,40,10),(10,32,40,10),(130,32,40,10),(10,2,40,10))
    by_target = {r.target_index:r for r in rs if r.source_index==0}
    aligned,opposite,above = (by_target[i] for i in (1,2,3))
    assert aligned.same_column_score>.95 and aligned.vertical_stack_score>.8
    assert opposite.same_column_score<.2 and opposite.vertical_stack_score<aligned.vertical_stack_score
    assert above.vertical_stack_score==0
    assert aligned.inline_score<.2


def test_inline_and_same_row_are_independent_of_stack_and_decay_smoothly():
    rs = relations((0,20,20,10),(21,20,20,10),(21,24,20,10),(21,70,20,10))
    near,shifted,far = [next(r for r in rs if (r.source_index,r.target_index)==(0,i)) for i in (1,2,3)]
    assert near.same_row_score==1 and near.inline_score>.9
    assert near.vertical_stack_score==0
    assert near.same_row_score>shifted.same_row_score>far.same_row_score
    assert near.inline_score>shifted.inline_score>far.inline_score


def test_local_neighbor_decreases_with_distance_without_selecting_a_winner():
    rs = relations((0,0,10,10),(20,0,10,10),(100,200,10,10))
    scores = {r.target_index:r.local_neighbor_score for r in rs if r.source_index==0}
    assert scores[1]>scores[2]>0
    assert len(rs)==6


def test_scores_are_text_and_confidence_independent():
    raw = result((0,0,20,10),(1,13,20,10),texts=["AGE","16 years"])
    altered = replace(raw,detections=tuple(replace(d,text="unrelated! O0",confidence=None) for d in raw.detections))
    assert build_spatial_relations(build_spatial_observations(raw))==build_spatial_relations(build_spatial_observations(altered))


def test_all_pairs_preserve_indexes_multiple_candidates_and_determinism():
    raw = result((0,0,20,10),(0,12,20,10),(1,13,20,10))
    snapshot = asdict(raw)
    obs = build_spatial_observations(raw)
    rs = build_spatial_relations(obs)
    assert [(r.source_index,r.target_index) for r in rs]==[(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)]
    assert all(r.vertical_stack_score>.8 for r in rs if r.source_index==0)
    assert build_spatial_relations(obs)==rs
    assert asdict(raw)==snapshot
    shuffled = build_spatial_relations((obs[2],obs[0],obs[1]))
    assert [(r.source_index,r.target_index) for r in shuffled][:2]==[(2,0),(2,1)]


def test_no_semantic_fields_group_ids_columns_or_reading_order():
    obs = build_spatial_observations(result((0,0,20,10),(1,13,20,10)))
    for model in (obs[0],build_spatial_relations(obs)[0]):
        names = {f.name for f in fields(model)}
        assert names.isdisjoint({"field_id","section_id","same_field","label_value_pair","canonical_label","canonical_column","reading_order"})


@pytest.mark.parametrize("n", [0,1,40])
def test_empty_single_and_realistic_pair_counts_without_ocr(n):
    raw = result(*[(i%2*100,i//2*15,40,10) for i in range(n)])
    obs = build_spatial_observations(raw)
    rs = build_spatial_relations(obs)
    assert len(obs)==n and len(rs)==n*(n-1)
    for r in rs:
        assert all(0<=getattr(r,name)<=1 and math.isfinite(getattr(r,name)) for name in SCORES)
        assert all(math.isfinite(v) for v in asdict(r).values())


@pytest.mark.parametrize("polygon", [((0,0),(1,0),(2,0),(3,0)),
    ((0,0),(0,1),(0,2),(0,3)),((0,0),(1,1),(2,2),(3,3)),((0,0),(2,2),(0,2),(2,0))])
def test_zero_area_polygons_fail_clearly(polygon):
    raw = replace(result((0,0,1,1)),detections=(OCRDetection("raw",.9,polygon),))
    with pytest.raises(ValueError,match="area"):
        build_spatial_observations(raw)


@pytest.mark.parametrize("interval", [(0,0,1,2),(1,0,1,2),(0,float('inf'),1,2)])
def test_invalid_interval_geometry_rejected(interval):
    with pytest.raises(ValueError):
        interval_overlap(*interval)


def test_bad_api_inputs_and_mixed_crops_or_duplicate_indexes_rejected():
    with pytest.raises(TypeError):
        build_spatial_observations(None)
    with pytest.raises(TypeError):
        build_spatial_relations("not observations")
    obs = build_spatial_observations(result((0,0,20,10),(1,13,20,10)))
    with pytest.raises(ValueError,match="unique"):
        build_spatial_relations((obs[0],obs[0]))
    with pytest.raises(ValueError,match="same ticket"):
        build_spatial_relations((obs[0],replace(obs[1],ticket_roi=ROI(0,0,20,40))))
    with pytest.raises(TypeError):
        SpatialOCRObservation(True,obs[0].detection,ROI_DEFAULT)
    with pytest.raises(ValueError):
        SpatialOCRObservation(-1,obs[0].detection,ROI_DEFAULT)
