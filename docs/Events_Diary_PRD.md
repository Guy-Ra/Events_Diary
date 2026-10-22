# Events Diary — Product Requirements Document (PRD)

**Version 1.0 — October 2026**

> A local visual-intelligence system that converts Flightradar24 screen recordings into a structured, evidence-backed aircraft event journal.

## 1. Executive Summary

Events Diary is a local visual-intelligence system designed to analyze Flightradar24 screen recordings and convert operator activity into a structured chronological aircraft event journal. The product is intended to understand what the operator investigated, extract the information visible in the ticket and map, preserve uncertainty and provenance, associate repeated investigations with the same aircraft where justified, and generate both machine-readable and human-readable outputs.

The system combines deterministic computer vision, temporal reasoning, OCR,
visible route context and an independent full-Investigation VLM branch. The MVP VLM
path sends continuous video directly to the Google Gemini API. Separate evidence
provenance supports later cross-checking without making either branch depend on
the other.

The canonical structured session record is the source of truth. Operational and detailed reports are derived from that record rather than generated independently.

## 2. Problem Statement

A Flightradar24 recording contains a temporal sequence of operator actions: aircraft selections, direct switches, ticket loading, scrolling, map pan/zoom, route changes, accidental clicks, and inactive periods.

Key challenges:
- Processing every frame equally creates redundancy.
- OCR/VLM on every frame is expensive and noisy.
- A single end-to-end VLM response is difficult to audit field-by-field.
- Investigation boundaries and temporal context must remain available to video analysis.
- The same aircraft may be investigated multiple times.

## 3. Product Vision and Goals

**Vision:** Given a Flightradar24 screen recording, automatically produce a reliable chronological record of operator investigations and associated aircraft information while distinguishing observed, enriched, inferred, ambiguous, and conflicting information.

Core goals:
- Detect Investigation Events and their time boundaries.
- Extract complete ticket information across scroll states.
- Analyze visible map/route context and derive compass direction from OCR TRACK.
- Use OCR, CV and VLM as complementary evidence sources.
- Keep runtime configuration explicit without changing evidence semantics.
- Link repeated aircraft investigations only when justified.
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

**Tracked Aircraft Entity:** an aircraft-level entity that may link multiple Investigation Events when evidence supports the match.

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
- Entity matching.
- Canonical output and reports.

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
 Aircraft Entity Matching
           │
           ▼
 Canonical Session Record
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
- FR-14 Entity matching
- FR-15 Canonical output
- FR-16 Runtime configuration without schema changes

## 9. OCR, CV and VLM Responsibilities

**CV:** deterministic temporal/UI understanding, pan/zoom/scroll and visual timing.

**OCR:** visible ticket text; preserve raw observations and combine across frames.

**VLM:** independent full-Investigation interpretation through the Google Gemini API. It receives the continuous Investigation video and the versioned `fr24_vlm_v1` prompt, not OCR values. Its structured ticket and map/route responses remain source-specific evidence for later fusion.

Phase 6 Pass 1 analyzes both ticket content and the visible map / selected-aircraft
route in its existing full-video call. Visual direction sector/text, route
description and departure context are separate VLM observations with observed,
inferred or not_visible status. Map-derived departure context never fills or
overwrites ticket origin or destination. Pass 2 reviews these claims in its
existing full-video review call; no route-only call or Pass 3 is added.

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

The MVP VLM configuration selects the Google Gemini API endpoint/model and
request timeout. Hardware-specific model execution profiles, alternate runtimes
and optimization are deferred to post-MVP Phase 10. The canonical output schema
and core event semantics remain stable.

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
- Aircraft revisit matching.
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
- Milestone D — Complete MVP (Phases 7–9)

Phase 6 includes OCR/VLM ticket and map observations, deterministic TRACK
direction, ticket fusion, direction cross-checking, Pass 2 review, multi-
Investigation assembly and closure validation. The former standalone route /
direction phase is absorbed into Phase 6. The former Phase 8–11 roadmap entries
are renumbered as Phases 7–10; Phase 10 is System Optimization & Improvement and
starts only after Phases 0–9 and end-to-end MVP validation.

## 16. Open Product Decisions

- Gemini API runtime and latency targets for post-MVP evaluation.
- OCR engine and calibration.
- Final accuracy thresholds.
- Debug evidence retention policy.
- Future audio-fusion requirements.
- Production session duration/throughput targets.

## Appendix A. Current Project Status

Phases 0–5 are complete. Phase 6 implementation scope is in progress. Existing
Gemini integration and ticket fusion have been validated on real video. The
approved Pass 1 map/route extension, route review in Pass 2, TRACK direction
cross-check, Phase 6.6 provider validation and Phase 6.7 closure validation are
pending Gemini quota. Synthetic tests do not establish provider visual accuracy.
