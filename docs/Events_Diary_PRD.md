# Events Diary — Product Requirements Document (PRD)

**Version 1.0 — October 2026**

> A local visual-intelligence system that converts Flightradar24 screen recordings into a structured, evidence-backed aircraft event journal.

## 1. Executive Summary

Events Diary is a local visual-intelligence system designed to analyze Flightradar24 screen recordings and convert operator activity into a structured chronological aircraft event journal. The product is intended to understand what the operator investigated, extract the information visible in the ticket and map, preserve uncertainty and provenance, associate repeated investigations with the same aircraft where justified, and generate both machine-readable and human-readable outputs.

The system deliberately does not rely on a single model or a single inference method. It combines deterministic computer vision, temporal reasoning, OCR, route geometry, and selective VLM assistance. This approach reduces unnecessary computation, improves traceability, allows the system to operate on heterogeneous hardware, and enables independent evidence sources to cross-check each other.

The canonical structured session record is the source of truth. Operational and detailed reports are derived from that record rather than generated independently.

## 2. Problem Statement

A Flightradar24 recording contains a temporal sequence of operator actions: aircraft selections, direct switches, ticket loading, scrolling, map pan/zoom, route changes, accidental clicks, and inactive periods.

Key challenges:
- Processing every frame equally creates redundancy.
- OCR/VLM on every frame is expensive and noisy.
- A single end-to-end VLM response is difficult to audit field-by-field.
- Internal-network machines may have very different GPU/model capabilities.
- The same aircraft may be investigated multiple times.

## 3. Product Vision and Goals

**Vision:** Given a Flightradar24 screen recording, automatically produce a reliable chronological record of operator investigations and associated aircraft information while distinguishing observed, enriched, inferred, ambiguous, and conflicting information.

Core goals:
- Detect Investigation Events and their time boundaries.
- Extract complete ticket information across scroll states.
- Analyze route geometry and direction where supported.
- Use OCR, CV and VLM as complementary evidence sources.
- Adapt expensive processing to hardware/model capability.
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
- Route/direction analysis.
- Selective or parallel VLM assistance.
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
Ticket OCR         Map / Route CV
   │                    │
   └───────┬────────────┘
           ▼
 VLM Assistance / Parallel Interpretation
           │
           ▼
 Evidence Fusion + Cross-Check
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
- FR-12 Parallel VLM mode on capable hardware
- FR-13 Evidence fusion
- FR-14 Entity matching
- FR-15 Canonical output
- FR-16 Runtime adaptation without schema changes

## 9. OCR, CV and VLM Responsibilities

**CV:** temporal/UI understanding, pan/zoom/scroll, route geometry, direction.

**OCR:** visible ticket text; preserve raw observations and combine across frames.

**VLM:** OCR fallback, semantic enrichment, difficult visual interpretation, Investigation-level parallel interpretation, ambiguity resolution, final cross-check.

**Fusion layer:** reconcile evidence with provenance and conflict handling.

### VLM operating modes
- Off / minimal
- Targeted VLM
- Parallel Investigation VLM
- Parallel + final cross-check

The VLM is an evidence source, not unrestricted authority.

## 10. Runtime and Hardware Adaptation

```text
HardwareProfile + ModelProfile → ExecutionProfile
```

**HardwareProfile:** CPU, RAM, GPU availability/type, VRAM, backend support.

**ModelProfile:** model family/size, precision/quantization, memory needs, visual capabilities, context constraints.

**ExecutionProfile:** chooses VLM enablement, model size, number of calls, image resolution, batch size and evidence budget.

Invariant: the canonical output schema and core event semantics remain stable across hardware profiles.

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
- Graceful hardware degradation.
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
- OCR instability → role-based evidence + consensus + targeted VLM.
- Hardware variability → runtime execution profiles.
- Route ambiguity → explicit AMBIGUOUS states.
- Limited validation set → expand labeled regression corpus.
- Correlated VLM re-checks → retain independent OCR/CV evidence.

## 15. Product Roadmap

- Milestone A — Visual Foundation (Phases 0–2)
- Milestone B — Operator Activity Understanding (Phases 3–5)
- Milestone C — Semantic Flight Understanding (Phases 6–8)
- Milestone D — Complete MVP (Phases 9–10)

## 16. Open Product Decisions

- Hardware thresholds for execution profiles.
- Approved internal VLM models.
- Default policy for parallel VLM interpretation.
- OCR engine and calibration.
- Final accuracy thresholds.
- Debug evidence retention policy.
- Future audio-fusion requirements.
- Production session duration/throughput targets.

## Appendix A. Current Project Status

Phases 0–3 are complete. Phase 4 (Investigation Extraction & Adaptive Sampling) is next. Phases 5–10 remain planned.
