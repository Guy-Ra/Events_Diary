"""Phase 6.2b geometry-only observations and directed OCR candidate relations.

Raw OCRDetection polygons/text/confidence remain unchanged. An observation holds
that immutable detection and its production crop ROI; its derived AABB is only a
convenience, never replacement geometry. Normalize x=(source_x-crop.x)/crop.width
and y=(source_y-crop.y)/crop.height. Coordinates outside [0,1] are retained, not
clipped. No image arrays, semantic columns, fields or section identities exist.

Signed interval gap = max(start_a,start_b)-min(end_a,end_b): positive means
separation in EITHER direction, zero means edge contact, negative means overlap
(including containment). Direction is independently determined by target minus
source CENTER, with equality neither above/below nor left/right. Thus reversing
a relation negates center deltas/direction but preserves signed interval gaps.
This differs intentionally from a directed target.top-source.bottom edge offset.

All five scores are bounded geometric compatibility features, not probabilities,
OCR/field/semantic/fusion confidence or final pairing decisions. Fixed smooth
scales express broad ticket fractions and box sizes; they are not fitted to the
seven Phase6.2a crops. In particular, 0.02H is only a smooth row decay scale, not
a same-row cutoff. Inline and downward-stack scores are separate, while generic
neighbor proximity is a separate fallback feature, never combined as a decision.

Every ordered pair of distinct observations is retained, including weak, tied
and competing candidates. No grouping, threshold acceptance, reading order,
canonical-label recognition, label/value interpretation or text normalization
occurs. Phase6.3 may later interpret this geometry. Existing OCR runtime/model
execution and Phases1-5 remain independent of this module.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real

from events_diary.config import ROI
from events_diary.ticket_ocr import OCRDetection, RawTicketOCRResult


@dataclass(frozen=True)
class SpatialBounds:
    """Continuous source-pixel or normalized AABB edges; width/centers are derived."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self):
        """Reject nonfinite, collapsed or overflowing geometry instead of returning NaN."""
        edges = (self.left,self.top,self.right,self.bottom)
        if any(isinstance(v,bool) or not isinstance(v,Real) for v in edges):
            raise TypeError("Bounds must be real pixel coordinates.")
        if not all(math.isfinite(v) for v in (*edges,self.width,self.height,self.center_x,self.center_y)):
            raise ValueError("Bounds and derived dimensions must be finite.")
        if self.width<=0 or self.height<=0:
            raise ValueError("Spatial bounds must have positive area.")

    @property
    def width(self) -> float:
        """Return continuous horizontal extent."""
        return self.right-self.left

    @property
    def height(self) -> float:
        """Return continuous vertical extent."""
        return self.bottom-self.top

    @property
    def center_x(self) -> float:
        """Return the AABB center, not the polygon area centroid."""
        return self.left+self.width/2

    @property
    def center_y(self) -> float:
        """Return the vertical AABB center."""
        return self.top+self.height/2


def _polygon_bounds(detection):
    """Validate nonzero polygon area and derive bounds without changing vertices."""
    polygon = detection.source_polygon
    x0,y0 = polygon[0]
    # Translate before shoelace products to avoid cancellation at large origins.
    translated = [(x-x0,y-y0) for x,y in polygon]
    try:
        twice_area = math.fsum(translated[i][0]*translated[(i+1)%4][1]
            -translated[(i+1)%4][0]*translated[i][1] for i in range(4))
    except (OverflowError,ValueError) as error:
        raise ValueError("Polygon area must be finite and nonzero.") from error
    if not math.isfinite(twice_area) or twice_area==0:
        raise ValueError("Polygon must have finite nonzero area.")
    return SpatialBounds(min(x for x,_ in polygon),min(y for _,y in polygon),
        max(x for x,_ in polygon),max(y for _,y in polygon))


@dataclass(frozen=True)
class SpatialOCRObservation:
    """One indexed raw detection and its derived geometry, with no field semantics.

    Keep one reference to OCRDetection rather than duplicate text/polygon/score.
    Store only its derived bounds; dimensions, centers and normalization are
    properties. ticket_roi is the actual source-frame crop from the OCR result.
    """

    detection_index: int
    detection: OCRDetection
    ticket_roi: ROI
    source_bounds: SpatialBounds = field(init=False)

    def __post_init__(self):
        """Require valid immutable input metadata and nondegenerate geometry."""
        if isinstance(self.detection_index,bool) or not isinstance(self.detection_index,Integral):
            raise TypeError("detection_index must be an integer, not bool.")
        if self.detection_index<0:
            raise ValueError("detection_index must be nonnegative.")
        if not isinstance(self.detection,OCRDetection) or not isinstance(self.ticket_roi,ROI):
            raise TypeError("Observation requires OCRDetection and production crop ROI.")
        object.__setattr__(self,"source_bounds",_polygon_bounds(self.detection))
        # Validate normalized geometry too; allow real detector edge overhang.
        _ = self.normalized_bounds

    @property
    def text(self) -> str:
        """Expose exact engine text without cleanup or semantic interpretation."""
        return self.detection.text

    @property
    def ocr_confidence(self) -> float | None:
        """Expose the unchanged OCR-engine score, never geometric/field confidence."""
        return self.detection.confidence

    @property
    def source_polygon(self):
        """Expose the original immutable quadrilateral in original source pixels."""
        return self.detection.source_polygon

    @property
    def normalized_bounds(self) -> SpatialBounds:
        """Derive unclipped ticket-relative bounds using the actual crop origin/size."""
        b,r = self.source_bounds,self.ticket_roi
        return SpatialBounds((b.left-r.x)/r.width,(b.top-r.y)/r.height,
            (b.right-r.x)/r.width,(b.bottom-r.y)/r.height)

    @property
    def center_x(self):
        """Return source-frame AABB center x."""
        return self.source_bounds.center_x

    @property
    def center_y(self):
        """Return source-frame AABB center y."""
        return self.source_bounds.center_y

    @property
    def width(self):
        """Return source-frame AABB width."""
        return self.source_bounds.width

    @property
    def height(self):
        """Return source-frame AABB height."""
        return self.source_bounds.height

    @property
    def normalized_left(self):
        """Return the unclipped left ticket fraction."""
        return self.normalized_bounds.left

    @property
    def normalized_top(self):
        """Return the unclipped top ticket fraction."""
        return self.normalized_bounds.top

    @property
    def normalized_right(self):
        """Return the unclipped right ticket fraction."""
        return self.normalized_bounds.right

    @property
    def normalized_bottom(self):
        """Return the unclipped bottom ticket fraction."""
        return self.normalized_bounds.bottom

    @property
    def normalized_center_x(self):
        """Return the AABB center relative to ticket width, without column assignment."""
        return self.normalized_bounds.center_x

    @property
    def normalized_center_y(self):
        """Return the AABB center relative to ticket height."""
        return self.normalized_bounds.center_y

    @property
    def normalized_width(self):
        """Return AABB width divided by ticket crop width."""
        return self.normalized_bounds.width

    @property
    def normalized_height(self):
        """Return AABB height divided by ticket crop height."""
        return self.normalized_bounds.height


@dataclass(frozen=True)
class OCRSpatialRelation:
    """Directed geometry and five independent compatibility scores, with no grouping.

    Indices are original OCR detection indexes, never sorted/renumbered. Deltas
    use target minus source; direction is based on centers. Signed gaps are the
    symmetric interval separation/overlap defined in the module documentation.
    All ratios/scores are dimensionless; all *_pixels fields use source pixels.
    """

    source_index: int
    target_index: int
    delta_center_x: float
    delta_center_y: float
    normalized_delta_center_x: float
    normalized_delta_center_y: float
    horizontal_overlap_pixels: float
    horizontal_overlap_min_ratio: float
    horizontal_overlap_union_ratio: float
    vertical_overlap_pixels: float
    vertical_overlap_min_ratio: float
    vertical_overlap_union_ratio: float
    signed_horizontal_gap_pixels: float
    signed_vertical_gap_pixels: float
    normalized_signed_horizontal_gap: float
    normalized_signed_vertical_gap: float
    target_is_above: bool
    target_is_below: bool
    target_is_left: bool
    target_is_right: bool
    same_column_score: float
    same_row_score: float
    vertical_stack_score: float
    inline_score: float
    local_neighbor_score: float


def interval_overlap(start_a: float,end_a: float,start_b: float,end_b: float) -> tuple[float,float,float]:
    """Return intersection length, intersection/min-length and intersection/union.

    Continuous intervals must have finite positive lengths. Union length is
    length_a+length_b-intersection. Touching/disjoint intervals return three zeros.
    Invalid zero-length/infinite intervals raise ValueError instead of NaN.
    """
    values = (start_a,end_a,start_b,end_b)
    if any(isinstance(v,bool) or not isinstance(v,Real) for v in values):
        raise TypeError("Interval edges must be real, not bool.")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Interval edges must be finite.")
    wa,wb = end_a-start_a,end_b-start_b
    if min(wa,wb)<=0 or not all(math.isfinite(v) for v in (wa,wb,wa+wb)):
        raise ValueError("Intervals must have finite positive lengths and union.")
    intersection = max(0.,min(end_a,end_b)-max(start_a,start_b))
    return intersection,intersection/min(wa,wb),intersection/(wa+wb-intersection)


def signed_interval_gap(start_a: float,end_a: float,start_b: float,end_b: float) -> float:
    """Return positive separation, zero touch or negative overlap in either direction.

    For containment, negative magnitude is the smaller interval's length.
    Direction comes from center deltas, not this symmetric measurement.
    """
    interval_overlap(start_a,end_a,start_b,end_b)
    return max(start_a,start_b)-min(end_a,end_b)


def build_spatial_observations(ocr_result: RawTicketOCRResult) -> tuple[SpatialOCRObservation,...]:
    """Derive one observation per detection in exact original engine/index order.

    Accept the core RawTicketOCRResult or its provenance-bearing OCRTicketResult
    subclass. Caller retains the parent result/video provenance; it is not copied
    into every observation. No OCR execution or optional model dependency is used.
    """
    if not isinstance(ocr_result,RawTicketOCRResult):
        raise TypeError("ocr_result must be RawTicketOCRResult or OCRTicketResult.")
    return tuple(SpatialOCRObservation(i,d,ocr_result.crop_roi) for i,d in enumerate(ocr_result.detections))


def _relation(source,target):
    """Compute one directed pair using broad fixed smooth scales, never raw text."""
    a,b,r = source.source_bounds,target.source_bounds,source.ticket_roi
    dx,dy = b.center_x-a.center_x,b.center_y-a.center_y
    nx,ny = dx/r.width,dy/r.height
    hx,hmin,hunion = interval_overlap(a.left,a.right,b.left,b.right)
    vy,vmin,vunion = interval_overlap(a.top,a.bottom,b.top,b.bottom)
    gx,gy = max(a.left,b.left)-min(a.right,b.right),max(a.top,b.top)-min(a.bottom,b.bottom)
    mean_w,mean_h = a.width/2+b.width/2,a.height/2+b.height/2
    if not all(math.isfinite(v) for v in (dx,dy,nx,ny,gx,gy,mean_w,mean_h)):
        raise ValueError("Pair geometry must remain finite in the source/ticket space.")
    column = .5*hmin+.5*math.exp(-abs(nx)/.25)
    row_scale = max(.02,mean_h/r.height)
    row = .5*vmin+.5*math.exp(-abs(ny)/row_scale)
    # Smooth direction/overlap attenuation: no exact edge-alignment gate.
    down = max(0.,min(1.,dy/mean_h))
    right = max(0.,min(1.,dx/mean_w))
    stack = column*down*math.exp(-max(0.,gy/r.height)/.05)
    inline = row*right*math.exp(-max(0.,gx/r.width)/.1)
    neighbor = 1/(1+math.hypot(nx,ny))
    return OCRSpatialRelation(source.detection_index,target.detection_index,dx,dy,nx,ny,
        hx,hmin,hunion,vy,vmin,vunion,gx,gy,gx/r.width,gy/r.height,
        dy<0,dy>0,dx<0,dx>0,column,row,stack,inline,neighbor)


def build_spatial_relations(observations: Sequence[SpatialOCRObservation]) -> tuple[OCRSpatialRelation,...]:
    """Retain all N*(N-1) directed non-self pairs in caller observation order.

    Inputs must describe one ticket/crop. Unique original indexes are preserved
    even if caller order differs. No sort, threshold, top-k truncation, semantic
    pairing, grouping or confidence-based filtering occurs. Empty/single inputs
    return (). Observations and raw OCR objects are never mutated.
    """
    if not isinstance(observations,Sequence) or isinstance(observations,(str,bytes)):
        raise TypeError("observations must be a sequence.")
    items = tuple(observations)
    if any(not isinstance(o,SpatialOCRObservation) for o in items):
        raise TypeError("observations must contain SpatialOCRObservation records.")
    if len({o.detection_index for o in items})!=len(items):
        raise ValueError("Original detection indexes must be unique.")
    if items and any(o.ticket_roi!=items[0].ticket_roi for o in items):
        raise ValueError("All observations must use the same ticket crop coordinate space.")
    return tuple(_relation(a,b) for a in items for b in items if a.detection_index!=b.detection_index)
