# Events Diary — Product Requirements Document (PRD)

**Version 1.0 — October 2026**

> A local visual-intelligence system that converts Flightradar24 screen recordings into a structured, evidence-backed aircraft event journal.

## 1. Executive Summary

Events Diary is a local visual-intelligence system designed to analyze Flightradar24 screen recordings and convert operator activity into a structured chronological aircraft event journal. The product is intended to understand what the operator investigated, extract the information visible in the ticket and map, preserve uncertainty and provenance, and generate machine-readable and human-readable outputs. Each Investigation remains independent; revisit matching is a future capability.

The system combines deterministic computer vision, temporal reasoning, OCR,
visible route context and an independent full-Investigation VLM branch. The MVP VLM
path sends continuous video directly to the Google Gemini API. Separate evidence
provenance supports later cross-checking without making either branch depend on
the other.

Phase 7 structures final Phase 6 results without adding inference. The canonical
structured session record is the source of truth. Operational and detailed
reports are deterministic renderings of that record.

## 2. Problem Statement

A Flightradar24 recording contains a temporal sequence of operator actions: aircraft selections, direct switches, ticket loading, scrolling, map pan/zoom, route changes, accidental clicks, and inactive periods.

Key challenges:
- Processing every frame equally creates redundancy.
- OCR/VLM on every frame is expensive and noisy.
- A single end-to-end VLM response is difficult to audit field-by-field.
- Investigation boundaries and temporal context must remain available to video analysis.
- Each Investigation remains an independent chronological event; revisit
  matching is a future capability, not an MVP requirement.

## 3. Product Vision and Goals

**Vision:** Given a Flightradar24 screen recording, automatically produce a reliable chronological record of operator investigations and associated aircraft information while distinguishing observed, enriched, inferred, ambiguous, and conflicting information.

Core goals:
- Detect Investigation Events and their time boundaries.
- Extract complete ticket information across scroll states.
- Analyze visible map/route context and derive compass direction from OCR TRACK.
- Use OCR, CV and VLM as complementary evidence sources.
- Keep runtime configuration explicit without changing evidence semantics.
- Preserve each Investigation as an independent event; no revisit inference is
  required for the MVP.
- Produce a canonical session record and derived reports.
- Preserve provenance, confidence and uncertainty.

## 4. Users and Operational Context

Primary user: an analyst, engineer or operational reviewer who needs to reconstruct what was investigated during a Flightradar24 session without manually reviewing the entire recording.

Initial assumptions:
- Short recordings (~30–40 s), extensible later.
- North-up map; no rotation expected initially.
- Pan/zoom may occur.
- One selected aircraft at a time.
- Scrollable left-side ticket.
- Local/internal-network deployment.

## 5. Inputs, Outputs and Core Concepts

**Input:** Flightradar24 screen recording.

**Outputs:**
- `session.json`
- `operational_log.txt`
- `detailed_report.txt`
- optional debug artifacts

**Investigation Event:** a period in which the operator focuses on one selected aircraft.

**Canonical Event:** exactly one final Phase 6 `InvestigationRecord` represented
in the Phase 7 Events Diary data layer.

**Canonical Session:** the ordered collection of canonical events and
session-level data. Its JSON representation is the Events Diary source of truth.

## 6. MVP Scope and Non-Goals

### In scope
- Video decode/timestamping.
- Semantic UI analysis.
- Investigation detection and lifecycle.
- Adaptive evidence extraction.
- Role-based frame selection.
- OCR ticket extraction and consensus.
- Deterministic TRACK-to-compass derivation and visual route understanding.
- One full-Investigation VLM branch that analyzes ticket and map together.
- Evidence fusion.
- Canonical event/session field selection and output contracts.
- Deterministic operational and detailed diary rendering from canonical data.

### Out of scope initially
- Live website interaction.
- Arbitrary aviation UI support.
- Unconstrained VLM generation as source of truth.
- Unsupported route/airport inference.
- Audio fusion in the initial MVP.
- Production accuracy claims before larger ground-truth validation.

## 7. End-to-End Architecture

```text
Flightradar24 Video
        │
        ▼
Video Decode + Timestamping
        │
        ▼
Operator Activity Understanding
(CV + temporal reasoning)
        │
        ▼
Investigation Intervals
        │
        ▼
Adaptive Evidence Extraction
        │
        ▼
Role-Based Evidence Selection
        │
   ┌────┴───────────────┐
   ▼                    ▼
Ticket OCR                 Full-video VLM (ticket + map / route)
   |                                      |
   +---------- Ticket fusion + route-direction cross-check
           │
           ▼
 InvestigationRecord 0...N
           │
           ▼
 CanonicalEvent records
           │
           ▼
 CanonicalSession (source of truth)
           │
    ┌──────┴────────┐
    ▼               ▼
Operational Log  Detailed Report
```

## 8. Functional Requirements

- FR-01 Video ingestion
- FR-02 Investigation detection
- FR-03 Investigation transitions
- FR-04 False interaction filtering
- FR-05 Adaptive sampling
- FR-06 Ticket scroll coverage
- FR-07 Role-based evidence
- FR-08 OCR extraction
- FR-09 Multi-frame consensus
- FR-10 Route analysis
- FR-11 VLM assistance
- FR-12 Independent full-Investigation Gemini VLM branch
- FR-13 Evidence fusion
- FR-14 Canonical event/session representation
- FR-15 Deterministic diary/report rendering
- FR-16 Runtime configuration without schema changes

One Phase 6 `InvestigationRecord` maps to one `CanonicalEvent`; events are
ordered by Investigation start time and are never merged or grouped. Phase 7
does not match aircraft/flights or perform semantic inference. Revisit analysis
is deferred to Phase 10.

Phase 7 is **COMPLETE**: **7.1 Define Events Diary field set is COMPLETE**;
**7.2 `CanonicalEvent` and `CanonicalSession` contracts is COMPLETE**;
**7.3 Missing / unresolved data representation is COMPLETE**;
**7.4 Session aggregation and real-artifact validation are COMPLETE**.
**7.5 Renderer-ready output contract is COMPLETE**; **7.6 final canonical-layer
validation is COMPLETE**. **Phase 8 — Deterministic Events Diary Renderer — is
NEXT** and will produce actual Operational and Detailed Diary outputs from
`CanonicalSession`.

The approved Phase 7.1 field families are event metadata (`event_id`,
`investigation_id`, `start_time_s`, `end_time_s`); flight IDs and airline;
origin/destination airport codes and scheduled/actual/estimated times; aircraft
registration/code/type; and TRACK, derived TRACK direction, visual flight
direction, visual route description and visual departure context. Secondary
ticket fields and detailed technical/OCR/VLM diagnostics remain in Phase 6
evidence layers rather than the canonical diary.

Canonical values remain directly readable. Missingness and uncertainty are
summarized separately under `data_quality` (for example,
`unresolved_fields` and `not_available_fields`), without requiring nested
value/status/provenance wrappers for every field.

The operational diary is concise and emphasizes flight IDs, airline, route,
times, aircraft code/type and registration, with direction where useful. The
detailed diary/report carries the same core event fields and may add TRACK and
human-readable direction, visual route context, data-quality summaries and
useful event context. Neither renderer exposes unnecessary Phase 6 internals.
Both are deterministic renderings of `CanonicalSession` JSON, which remains the
source of truth.

Canonical JSON stores null for unresolved, not-available and absent values.
`data_quality.unresolved_fields` means relevant information could not be
determined reliably; `data_quality.not_available_fields` means the displayed or
source data did not provide it and does not imply classified/restricted data.
A Phase 6 field absent entirely remains unclassified. Future renderers may show
“Unresolved” / “Could not determine” or “N/A” / “Not available”; these labels
are never stored as canonical values. Canonical quality paths are unique,
disjoint, lexicographically ordered, and consistent with null values.

Phase 7.4 consumes the existing Phase 6 `VideoInvestigationResult` and creates
one `CanonicalSession` per source video and one
`CanonicalEvent` per Investigation. Events sort chronologically by event time
and Investigation ID; sequential `event_id` values are assigned after sorting,
while original Investigation IDs remain intact. Authoritative video metadata
provides the source filename and duration. A video with no Investigations may
produce an empty session; a video result with failed Investigations is rejected
instead of silently dropping them. Aggregation performs no inference or revisit matching,
and data-quality fields remain local to each event. Canonical
`movement.direction` is derived only when final Phase 6 TRACK is resolved;
unresolved TRACK leaves both canonical TRACK and deterministic direction null
and unresolved. An intermediate OCR-derived direction never overrides final
Phase 6 resolution. The separate `movement.visual_flight_direction` comes from
final visual review and may remain available when TRACK is unresolved.

Future renderers consume `CanonicalSession` only. The Operational profile uses
event time, flight identity, route and schedule/time fields, aircraft identity,
and deterministic movement direction. The Detailed profile preserves that
field order, then adds Investigation identity and bounds, TRACK, visual movement
context, and data-quality lists. Profile paths are canonical fields only;
OCR/VLM internals, fusion candidates, provider metadata and Phase 6 evidence
graphs are excluded. `field_state` keeps `absent` distinct from
`not_available`; presentation labels remain renderer decisions. Sessions
provide stable chronological event ordering and source metadata, so renderers
do not sort or inspect upstream evidence.

## 9. OCR, CV and VLM Responsibilities

**CV:** deterministic temporal/UI understanding, pan/zoom/scroll and visual timing.

**OCR:** visible ticket text; preserve raw observations and combine across frames.

**VLM:** independent full-Investigation interpretation through the Google Gemini API. It receives the continuous Investigation video and the versioned `fr24_vlm_v1` prompt, not OCR values. Its structured ticket and map/route responses remain source-specific evidence for later fusion.

Phase 6 Pass 1 analyzes both ticket content and the visible map / selected-aircraft
route in its existing full-video call. Visual direction sector/text, route
description and departure context are separate VLM observations with observed,
inferred or not_visible status. Map-derived departure context never fills or
overwrites ticket origin or destination. Pass 2 v3 reviews the full Investigation
video together with Phase 5-selected ticket crops and OCR, VLM Pass 1, and fusion
context in one Gemini request. The video supports temporal, map/route, scrolling,
movement and dynamic-field interpretation; crops support precise small-text
ticket reading. No image-only request or Pass 3 is added.

OCR TRACK remains the original degree observation. Events Diary derives a
16-point compass direction deterministically and marks it as derived from OCR.
The derived value and VLM visual direction are compared without overwriting either
source. A missing side yields an unavailable comparison.

**Fusion layer:** reconcile evidence with provenance and conflict handling.

The MVP has one VLM mode: send the complete continuous Investigation MP4 directly
to Gemini, request structured JSON, and parse the result without semantic repair.
There is no application-side VLM frame sampling. OCR and VLM remain independent
branches; Phase 6.5 implements their fusion and cross-check. The VLM is evidence,
not unrestricted authority.

## 10. Runtime and Hardware Adaptation

The final validated MVP model was `gemini-3.6-flash`, used throughout the
successful complete validation dataset. Request timeout was 900 seconds, with
at most three attempts and 503/timeout backoff of 30 seconds then 90 seconds.
`gemini-3.8-flash` was an optional fallback only for confirmed 3.6 Free Tier daily
quota exhaustion and was not used in that dataset. Hardware-specific model
execution profiles, alternate runtimes and optimization are deferred to post-MVP
Phase 10. The canonical output schema and core event semantics remain stable.

## 11. Evidence, Provenance and Confidence

Statuses:
- OBSERVED
- ENRICHED
- INFERRED
- EXPLICIT_NA
- UNREADABLE
- NOT_PRESENT
- AMBIGUOUS
- CONFLICT

Confidence: HIGH / MEDIUM / LOW.

Core rules:
- observed > unsupported inferred;
- repeated independent evidence > equivalent single evidence;
- multi-frame OCR consensus > single visual guess;
- deterministic geometry > VLM direction guess when reliable;
- strong contradictions remain visible.

## 12. Non-Functional Requirements

- Local/internal-network capable.
- Modular and testable.
- Auditable provenance.
- Efficient evidence reduction before expensive processing.
- Explicit Gemini API failure and unavailable-evidence handling.
- Deterministic processing where appropriate.
- Configuration-driven tuning.
- Explicit unknown/ambiguous/conflict states.
- Stable output contract.

## 13. Success Criteria and Validation

Success should be measured across:
- Investigation boundary accuracy.
- Evidence coverage and redundancy reduction.
- Field-level extraction accuracy.
- Route/direction correctness.
- Fusion quality and conflict handling.
- Correct canonical representation and deterministic rendering of each
  Investigation as an independent event.
- Runtime portability.
- End-to-end journal agreement with human review.

A larger labeled ground-truth dataset is required before production-readiness claims.

## 14. Risks and Mitigations

- UI/layout variation → configurable/mapped ROIs.
- VLM hallucination → evidence-first guardrails and provenance.
- OCR instability → preserve uncertainty and source-specific observations.
- Gemini API unavailability → retain explicit VLM branch failure status.
- Route ambiguity → explicit AMBIGUOUS states.
- Limited validation set → expand labeled regression corpus.
- Correlated VLM re-checks → retain independent OCR/CV evidence.

## 15. Product Roadmap

- Milestone A — Visual Foundation (Phases 0–2)
- Milestone B — Operator Activity Understanding (Phases 3–5)
- Milestone C — Complete Investigation Understanding (Phase 6)
- Milestone D — Canonical Events Diary Foundation and outputs (Phase 7 onward)

Phase 6 — Complete Investigation Understanding — is COMPLETE for MVP purposes.
It includes OCR extraction and spatial structure, structured ticket extraction,
deterministic TRACK direction, Pass 1 ticket and map/route understanding,
provenance-preserving fusion, temporal awareness, direction consistency, Pass 2
v3 crop-assisted review, conservative resolution, multi-Investigation assembly
and end-to-end validation. Phase 6.6 passed record isolation: multiple real
Investigations completed the pipeline with crops scoped correctly and no evidence
leakage. Phase 6.7 validated `ticket_scroll_test.mp4` and
`investigation_switch_test.mp4`; Pass 2 v3 materially improved small-text reading,
and the targeted ticket-scroll regression achieved 11/11 manually reviewed
targets. This is representative validation, not a single overall accuracy score.

An accepted MVP limitation is that occasional low-priority small-text errors may
remain after crop-assisted review. One AGE value was visibly 6 years while Pass 2
returned 4 years; this is deferred to Phase 10. The former standalone route /
direction analysis remains absorbed into Phase 6. Roadmap status: Phase 6
COMPLETE, Phase 7 COMPLETE (7.1–7.6 COMPLETE), Phase 8 NEXT
(Deterministic Events Diary Renderer), Phase 9 planned,
and Phase 10 System Optimization & Improvement post-MVP.

## 16. Open Product Decisions

- Gemini API runtime and latency targets for post-MVP evaluation.
- OCR engine and calibration.
- Final accuracy thresholds.
- Debug evidence retention policy.
- Future audio-fusion requirements.
- Production session duration/throughput targets.

## Appendix A. Current Project Status

Phases 0–6 are complete. Phase 7.1 field selection, Phase 7.2 canonical
contracts, Phase 7.3 missing-data semantics, Phase 7.4 session aggregation
with saved-artifact validation, Phase 7.5 renderer-ready contract, and Phase
7.6 final canonical-layer validation are complete. Phase 7 is complete; Phase 8, the deterministic Operational and
Detailed Diary renderer, is next. Phase 10 is
reserved for post-MVP optimization,
including residual VLM small-text errors, low-priority field accuracy and AGE
review, model comparison/upgrades, media-resolution and crop-selection
improvements, prompt, latency/runtime, provider/quota optimization, and future
future capability work: detecting repeated Investigations of the same aircraft,
matching the same aircraft or flight across Investigations, creating revisit
links and entity groups, and (if implemented) defining `MATCH`, `AMBIGUOUS` and
`NO_MATCH` semantics. This capability is not required to create the Events
Diary; each Investigation remains an independent event.
