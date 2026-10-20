"""FR24 MVP canonical-label recognition and per-representative OCR fields.

Phase6.3a supports compact, case-insensitive label matching and label-specific
stable qualifiers: AGE month/year context, optional SERIAL NUMBER MSN, and an
AIRCRAFT TYPE parenthetical tag. A complete compound string yields multiple
semantic identities sharing ONE original observation/polygon, with character
spans only. No per-label pixel geometry is invented. Raw label/value text is
never normalized, corrected, parsed into numbers, or overwritten.

Bounded fuzzy fallback runs only after compact/stable matching fails: normalized
Levenshtein similarity>=.90 and best-minus-second margin>=.10 by default. It
compares stable labels/fixed MSN qualification, not memorized dates/type values.
Weak/tied matches remain unknown. No global O/0/I/1 repair, arbitrary parenthesis
stripping, multi-system profiles or calibration-module dependency is present.

Time labels use unchanged Phase6.2 inline scores; technical labels (including
full-width aircraft rows) use stack scores. COUNTRY_OF_REGISTRATION/FIR_UIR are
calibrated non-text cases, not inferred flag/lock readers. Obvious exact/stable
label detections remain in diagnostics but are ineligible as textual values.
No expected value formats, airport roles or departure/arrival semantics are used.

Compound geometry always abstains. Other values require provisional configurable
spatial guards: score>=.25 and margin>=.10; exact/near numerical ties abstain even
with a zero margin setting. These are working extraction policies, not calibrated
probabilities or changes to Phase6.2 formulas. All alternatives can be regenerated
with rank_ocr_field_candidates. No score is combined with OCR confidence.

Frozen records retain source observation references and scalar physical provenance,
not image arrays or duplicated polygons. Each occurrence/representative remains
separate. Status is local extraction state (extracted, no_value_selected, ambiguous,
special_non_text), not a cross-source evidence-status system. No VLM, fusion,
multi-view merging, final Investigation record or Phase6.4 behavior is included.
"""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

from events_diary.ocr_spatial import (
    OCRSpatialRelation,
    SpatialOCRObservation,
    build_spatial_observations,
    build_spatial_relations,
)
from events_diary.ticket_ocr import OCRTicketResult

# Immutable FR24-only vocabulary and observed regimes; never a mutable singleton.
_LABELS = (
    ("SCHEDULED","SCHEDULED","inline"),("ACTUAL","ACTUAL","inline"),("ESTIMATED","ESTIMATED","inline"),
    ("REGISTRATION","REGISTRATION","stack"),("COUNTRY_OF_REGISTRATION","COUNTRYOFREG","special_non_text"),
    ("SERIAL_NUMBER","SERIALNUMBER","stack"),("AGE","AGE","stack"),
    ("AIRCRAFT_CATEGORY","AIRCRAFTCATEGORY","stack"),("AIRCRAFT_TYPE","AIRCRAFTTYPE","stack"),
    ("BAROMETRIC_ALTITUDE","BAROMETRICALT","stack"),("VERTICAL_SPEED","VERTICALSPEED","stack"),
    ("GPS_ALTITUDE","GPSALTITUDE","stack"),("TRACK","TRACK","stack"),("GROUND_SPEED","GROUNDSPEED","stack"),
    ("TRUE_AIRSPEED","TRUEAIRSPEED","stack"),("INDICATED_AIRSPEED","INDICATEDAIRSPEED","stack"),
    ("MACH","MACH","stack"),("WIND","WIND","stack"),("TEMPERATURE","TEMPERATURE","stack"),
    ("FIR_UIR","FIRUIR","special_non_text"),("ICAO_ADDRESS","ICAO24BITADDRESS","stack"),
    ("SQUAWK","SQUAWK","stack"),("LATITUDE","LATITUDE","stack"),("LONGITUDE","LONGITUDE","stack"),
)
CANONICAL_LABELS = tuple(label for label,_,_ in _LABELS)
_MONTH_CONTEXT = re.compile(r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{4}")


def _bounded(name,value):
    """Validate finite real nonboolean score policy inputs in [0,1]."""
    if isinstance(value,bool) or not isinstance(value,Real):
        raise TypeError(f"{name} must be real, not bool.")
    if not math.isfinite(value) or not 0<=value<=1:
        raise ValueError(f"{name} must be finite and within [0,1].")


def _observations(observations):
    """Validate one indexed crop's observation sequence without reordering or copying pixels."""
    if not isinstance(observations,Sequence) or isinstance(observations,(str,bytes)):
        raise TypeError("observations must be a sequence.")
    items = tuple(observations)
    if any(not isinstance(o,SpatialOCRObservation) for o in items):
        raise TypeError("Expected SpatialOCRObservation records.")
    if len({o.detection_index for o in items})!=len(items):
        raise ValueError("Detection indexes must be unique.")
    if items and any(o.ticket_roi!=items[0].ticket_roi for o in items):
        raise ValueError("Observations must share the same ticket crop.")
    return items


@dataclass(frozen=True)
class CanonicalLabelMatch:
    """One identity with raw observation provenance and optional literal character span.

    Compound matches retain the identical observation reference; the text span
    is never a sub-polygon. match_score is text compatibility, not OCR confidence.
    """

    canonical_label: str
    observation: SpatialOCRObservation
    match_method: str
    match_score: float
    compound: bool
    matched_text_span: tuple[int,int] | None

    @property
    def raw_detection_index(self):
        """Return the unchanged original OCR detection index."""
        return self.observation.detection_index

    @property
    def raw_text(self):
        """Expose exact whole raw text, including other labels in a compound box."""
        return self.observation.text

    @property
    def source_polygon(self):
        """Expose original shared source-frame geometry, never a fabricated label box."""
        return self.observation.source_polygon


@dataclass(frozen=True)
class OCRFieldProvenance:
    """Scalar physical provenance shared by one representative's extracted fields.

    Surrounding operation supplies video identity; detection observations retain
    the actual crop metadata, polygons and OCR confidence without duplicate storage.
    """

    investigation_index: int
    candidate_index: int
    frame_index: int
    actual_timestamp_seconds: float
    engine_name: str
    preprocessing_name: str
    resolution_status: str


@dataclass(frozen=True)
class OCRValueCandidate:
    """Regenerable ranked value alternative with unchanged raw spatial evidence.

    is_label_candidate marks obvious compact/stable labels ineligible for default
    selection, while preserving them in diagnostics. No value-format check occurs.
    """

    observation: SpatialOCRObservation
    relation: OCRSpatialRelation
    relationship_type: str
    is_label_candidate: bool

    @property
    def relationship_score(self):
        """Return the unchanged Phase6.2 score for this calibrated regime."""
        return self.relation.inline_score if self.relationship_type=="inline" else self.relation.vertical_stack_score


@dataclass(frozen=True)
class OCRFieldObservation:
    """One canonical occurrence, optional selected raw value and explicit local state.

    No alternative list, raw polygon/text or confidence is stored redundantly;
    references expose these exactly. Unselected states retain no selected value or
    relationship. Reasons distinguish compound/margin ambiguity from weak/no value.
    """

    label_match: CanonicalLabelMatch
    provenance: OCRFieldProvenance
    status: str
    selected_candidate: OCRValueCandidate | None
    reason: str | None

    @property
    def canonical_label(self):
        """Return stable canonical identity, independent of visible qualifiers."""
        return self.label_match.canonical_label

    @property
    def raw_label_text(self):
        """Return the exact whole raw label detection text."""
        return self.label_match.raw_text

    @property
    def raw_value_text(self):
        """Return exact selected raw text, or None for explicit abstention/special cases."""
        return self.selected_candidate.observation.text if self.selected_candidate else None

    @property
    def label_detection_index(self):
        """Return the unchanged source detection index, shared for compounds."""
        return self.label_match.raw_detection_index

    @property
    def value_detection_index(self):
        """Return the selected detection index or None; never a guessed value identity."""
        return self.selected_candidate.observation.detection_index if self.selected_candidate else None

    @property
    def label_match_method(self):
        """Expose recognition method separately from spatial/OCR confidence."""
        return self.label_match.match_method

    @property
    def label_match_score(self):
        """Expose text compatibility without interpreting it as final field confidence."""
        return self.label_match.match_score

    @property
    def relationship_type(self):
        """Return stack/inline only for a selected association."""
        return self.selected_candidate.relationship_type if self.selected_candidate else None

    @property
    def relationship_score(self):
        """Expose the unchanged spatial score independently of recognition/OCR scores."""
        return self.selected_candidate.relationship_score if self.selected_candidate else None

    @property
    def ocr_label_confidence(self):
        """Return original OCR-engine label confidence, including None."""
        return self.label_match.observation.ocr_confidence

    @property
    def ocr_value_confidence(self):
        """Return original selected OCR-engine value confidence, including None."""
        return self.selected_candidate.observation.ocr_confidence if self.selected_candidate else None

    @property
    def compound_label(self):
        """Report that semantic labels share an unsplit native detection polygon."""
        return self.label_match.compound

    @property
    def ambiguous(self):
        """Identify local unresolved association, not a cross-source CONFLICT status."""
        return self.status=="ambiguous"

    @property
    def shared_geometry(self):
        """Expose compound or identical-polygon label geometry without inventing a split."""
        return self.compound_label or self.reason=="shared_label_geometry"


def _compact(text):
    """Create an uppercase alphanumeric matching view plus exact raw character offsets."""
    chars,positions = [],[]
    for i,char in enumerate(text):
        if char.isalnum():
            upper = char.upper()
            chars.extend(upper)
            positions.extend([i]*len(upper))
    return "".join(chars),positions


def _stable_spans(text):
    """Parse complete calibrated label chains; reject arbitrary text around a label."""
    compact,positions = _compact(text)
    offset,hits = 0,[]
    while offset<len(compact):
        possible = []
        for canonical,stable,_ in _LABELS:
            if not compact.startswith(stable,offset):
                continue
            end = offset+len(stable)
            if canonical=="AGE":
                context = _MONTH_CONTEXT.match(compact,end)
                end = context.end() if context else end
            elif canonical=="SERIAL_NUMBER" and compact.startswith("MSN",end):
                end+=3
            elif canonical=="AIRCRAFT_TYPE":
                tag = re.match(r"\s*\(\s*([A-Za-z0-9]{2,6})\s*\)",text[positions[end-1]+1:])
                if tag:
                    end+=len(tag[1])
            possible.append((end,canonical))
        if not possible:
            return ()
        end,canonical = max(possible)
        hits.append((canonical,(positions[offset],positions[end] if end<len(positions) else len(text))))
        offset = end
    return tuple(hits)


def _distance(a,b):
    """Compute bounded-memory Levenshtein distance without OCR text repair."""
    row = list(range(len(b)+1))
    for i,left in enumerate(a,1):
        following = [i]
        for j,right in enumerate(b,1):
            following.append(min(following[-1]+1,row[j]+1,row[j-1]+(left!=right)))
        row = following
    return row[-1]


def _fuzzy_candidates(compact):
    """Compare stable spellings/fixed MSN only; do not memorize dates or aircraft tags."""
    scores = []
    for canonical,stable,_ in _LABELS:
        targets = (stable,stable+"MSN") if canonical=="SERIAL_NUMBER" else (stable,)
        score = max(1-_distance(compact,target)/max(len(compact),len(target),1) for target in targets)
        scores.append((score,canonical))
    return sorted(scores,reverse=True)


def recognize_canonical_labels(observations: Sequence[SpatialOCRObservation], *,
        fuzzy_threshold: float=.90,fuzzy_margin: float=.10) -> tuple[CanonicalLabelMatch,...]:
    """Recognize calibrated identities in original occurrence/span order, or none.

    Compact/stable success bypasses fuzzy matching. Fuzzy requires both inclusive
    similarity/margin bounds and a unique best identity; otherwise return unknown.
    Whole raw text/polygons are preserved even when literal character spans differ.
    """
    items = _observations(observations)
    _bounded("fuzzy_threshold",fuzzy_threshold)
    _bounded("fuzzy_margin",fuzzy_margin)
    matches = []
    for observation in items:
        compact,_ = _compact(observation.text)
        spans = _stable_spans(observation.text)
        if spans:
            compound = len(spans)>1
            literal = any(stable==compact for _,stable,_ in _LABELS)
            method = "compound_stable" if compound else "compact" if literal else "stable_token"
            matches.extend(CanonicalLabelMatch(c,observation,method,1.,compound,span) for c,span in spans)
            continue
        if not compact:
            continue
        scores = _fuzzy_candidates(compact)
        best,second = scores[0][0],scores[1][0]
        if best>=fuzzy_threshold and best>second and best-second>=fuzzy_margin:
            matches.append(CanonicalLabelMatch(scores[0][1],observation,"bounded_fuzzy",best,False,None))
    return tuple(matches)


def rank_ocr_field_candidates(label_match: CanonicalLabelMatch,
        observations: Sequence[SpatialOCRObservation], relations: Sequence[OCRSpatialRelation], *,
        label_matches: Sequence[CanonicalLabelMatch]) -> tuple[OCRValueCandidate,...]:
    """Regenerate all spatial alternatives; mark obvious label targets, never invent boxes.

    Scores/order use only the calibrated regime and unchanged Phase6.2 relations.
    Exact/stable canonical labels are flagged as ineligible values; uncertain fuzzy
    targets are retained to avoid over-filtering potential legitimate raw values.
    Special/non-text labels return no textual alternatives. No value format is used.
    """
    if not isinstance(label_match,CanonicalLabelMatch):
        raise TypeError("label_match must be CanonicalLabelMatch.")
    items = _observations(observations)
    by_index = {o.detection_index:o for o in items}
    if not isinstance(relations,Sequence) or not isinstance(label_matches,Sequence):
        raise TypeError("relations and label_matches must be sequences.")
    if any(not isinstance(m,CanonicalLabelMatch) for m in label_matches):
        raise TypeError("label_matches must contain CanonicalLabelMatch records.")
    if any(by_index.get(m.raw_detection_index)!=m.observation for m in label_matches):
        raise ValueError("Label matches must belong to this representative.")
    if by_index.get(label_match.raw_detection_index)!=label_match.observation:
        raise ValueError("Label observation must belong to this representative.")
    regime = next((r for c,_,r in _LABELS if c==label_match.canonical_label),None)
    if regime is None:
        raise ValueError("Unknown canonical label identity.")
    if regime=="special_non_text":
        return ()
    obvious = {m.raw_detection_index for m in label_matches if m.match_method!="bounded_fuzzy"}
    candidates = []
    for relation in relations:
        if not isinstance(relation,OCRSpatialRelation):
            raise TypeError("relations must contain OCRSpatialRelation records.")
        if relation.source_index!=label_match.raw_detection_index:
            continue
        if relation.target_index not in by_index or relation.target_index==relation.source_index:
            raise ValueError("Relation target must be another original detection.")
        candidates.append(OCRValueCandidate(by_index[relation.target_index],relation,regime,relation.target_index in obvious))
    return tuple(sorted(candidates,key=lambda c:(-c.relationship_score,c.observation.detection_index)))


def extract_ocr_fields(ocr_result: OCRTicketResult, *, fuzzy_threshold: float=.90,
        fuzzy_margin: float=.10,min_relationship_score: float=.25,
        min_candidate_margin: float=.10) -> tuple[OCRFieldObservation,...]:
    """Extract independent labeled occurrences with explicit selection/abstention states.

    Compound matches always abstain, even if one shared box has a strong neighbor.
    Ordinary selections require a plausible non-label target and sufficient margin;
    ties abstain. Policies are configurable provisional guards, not probabilities.
    Retain all physical/source provenance and original text; never merge occurrences.
    """
    if not isinstance(ocr_result,OCRTicketResult):
        raise TypeError("ocr_result must be a provenance-bearing OCRTicketResult.")
    for name in ("investigation_index","candidate_index","frame_index"):
        value = getattr(ocr_result,name)
        if isinstance(value,bool) or not isinstance(value,Integral):
            raise TypeError(f"{name} must be an integer, not bool.")
        if value<0:
            raise ValueError(f"{name} must be nonnegative.")
    time = ocr_result.actual_timestamp_seconds
    if isinstance(time,bool) or not isinstance(time,Real):
        raise TypeError("actual_timestamp_seconds must be real, not bool.")
    if not math.isfinite(time) or time<0 or ocr_result.resolution_status not in ("resolved","video_end_adjusted"):
        raise ValueError("OCR result must carry resolved finite physical provenance.")
    _bounded("min_relationship_score",min_relationship_score)
    _bounded("min_candidate_margin",min_candidate_margin)
    observations = build_spatial_observations(ocr_result)
    matches = recognize_canonical_labels(observations,fuzzy_threshold=fuzzy_threshold,fuzzy_margin=fuzzy_margin)
    relations = build_spatial_relations(observations) if matches else ()
    provenance = OCRFieldProvenance(ocr_result.investigation_index,ocr_result.candidate_index,ocr_result.frame_index,
        time,ocr_result.engine_name,ocr_result.preprocessing_name,ocr_result.resolution_status)
    fields = []
    for match in matches:
        regime = next(r for c,_,r in _LABELS if c==match.canonical_label)
        chosen,status,reason = None,"no_value_selected","no_eligible_candidate"
        if match.compound:
            status,reason = "ambiguous","compound_shared_geometry"
        elif any(other.raw_detection_index!=match.raw_detection_index and other.source_polygon==match.source_polygon for other in matches):
            status,reason = "ambiguous","shared_label_geometry"
        elif regime=="special_non_text":
            status,reason = "special_non_text","calibrated_non_text_case"
        else:
            alternatives = rank_ocr_field_candidates(match,observations,relations,label_matches=matches)
            eligible = [c for c in alternatives if not c.is_label_candidate]
            if eligible:
                best = eligible[0]
                runner = eligible[1].relationship_score if len(eligible)>1 else 0.
                gap = best.relationship_score-runner
                if best.relationship_score<min_relationship_score:
                    reason = "weak_spatial_candidate"
                elif math.isclose(gap,0.,abs_tol=1e-12) or gap<min_candidate_margin:
                    status,reason = "ambiguous","candidate_margin"
                else:
                    chosen,status,reason = best,"extracted",None
        fields.append(OCRFieldObservation(match,provenance,status,chosen,reason))
    return tuple(fields)


def extract_ocr_fields_batch(ocr_results: Sequence[OCRTicketResult], **options) -> tuple[tuple[OCRFieldObservation,...],...]:
    """Preserve caller Investigation/representative/occurrence order without merging views."""
    if not isinstance(ocr_results,Sequence) or isinstance(ocr_results,(str,bytes)):
        raise TypeError("ocr_results must be a sequence.")
    return tuple(extract_ocr_fields(result,**options) for result in ocr_results)
