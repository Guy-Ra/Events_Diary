# Events Diary

Local project building structured aircraft event logs from Flightradar24 screen
recordings using deterministic computer vision, OCR and Gemini VLM observations.

## Project Status

**Phase 5 - Role-Based Frame & Ticket-Section Selection is COMPLETE.**

The offline system supports FR24 UI/layout mapping, visual-change and map-motion
analysis, ticket scrolling, Investigation lifecycle detection, adaptive sampling,
stable ticket views, and evidence candidate timelines. Raw and effective presence
remain distinct.

Phase 5 provides the production flow:
`EvidenceCandidate -> physical frame materialization -> ticket-view comparison
-> representative selection`. Comparisons are supporting measurements; the pure
selector does not require image access or similarity decisions. Selection retains
one primary representative per stable ticket view and conservatively keeps every
post-scroll view, with separate bounded map/route evidence and optional scroll
provenance/boundary context. Evidence volume follows role coverage rather than a
global frame quota.

Validation is representative/manual, not a labeled benchmark. Thresholds remain
provisional; stable views do not prove loading completeness. The complete-video
Gemini VLM path and Phase 6.5 OCR/VLM fusion/cross-check are implemented and
validated on real video.

**Phase 6 — Complete Investigation Understanding is COMPLETE for MVP purposes.**
The validated pipeline includes OCR extraction and spatial structure, structured
ticket extraction, deterministic TRACK-to-compass conversion, Pass 1 ticket and
map/route observations, provenance-preserving fusion and temporal cross-checks,
Pass 2 v3 crop-assisted review, conservative resolution, and isolated
multi-Investigation assembly. Phase 6.6 and 6.7 real-video validation completed.

Pass 2 v3 sends the full Investigation video, Phase 5-selected ticket crops,
OCR evidence, VLM Pass 1 evidence and fusion/context information in one Gemini
request. The video provides temporal and map/route context; crops support precise
small-text ticket reading. No image-only request or Pass 3 exists. The final
validated model was `gemini-3.6-flash` throughout the successful complete
validation dataset. That validation used a 900-second timeout and its then-current
503/timeout backoff of 30 seconds then 90 seconds. `gemini-3.8-flash` was an
optional fallback only for confirmed 3.6 Free Tier daily quota exhaustion and
was not used in that dataset.

Validation used `ticket_scroll_test.mp4` and `investigation_switch_test.mp4`.
The ticket-scroll targeted regression achieved 11/11 manually reviewed targets;
the multi-Investigation regression completed structurally with record isolation
and correctly scoped crops. Validation is representative rather than a benchmark
and does not claim a single overall accuracy score.

Known limitation: occasional low-priority small-text errors may remain after
crop-assisted review. In one Investigation AGE was visibly 6 years while Pass 2
returned 4 years. This is accepted for MVP and deferred to Phase 10.

**Roadmap:** Phases 6 through 9 are COMPLETE, including Phase 9.5 real end-to-end validation. Phase 10 System Optimization & Improvement is NEXT / NOT STARTED. The former standalone route-analysis phase remains absorbed into Phase 6.

**Phase 7 — Canonical Event Data & Events Diary Foundation is COMPLETE.**
Phase 7.1 field selection, Phase 7.2 (`CanonicalEvent` and
`CanonicalSession` contracts), and Phase 7.3 missing-data semantics are complete;
Phase 7.4 session aggregation and saved-artifact validation are complete.
Canonical direction follows final TRACK; intermediate OCR-derived direction is
not propagated when final TRACK is unresolved. Phase 7.5 renderer-ready field
profiles and final canonical-layer validation are complete. Phase 8.1 provides
the shared renderer core. Phase 8.2 renders a concise plain-text Operational
Diary from `CanonicalSession` only, with one interval-headed block per
Investigation and missing operational values omitted. Phase 8.3 shows resolved
TRACK and deterministic direction together, omits that line when TRACK is
unusable, and groups source-faithful visual-route and departure-context text
under one heading. Phase 8.4 establishes the fixed output filenames and
single-video workspace conventions; it does not add a production file writer.
Phase 8 renderers return deterministic UTF-8-safe strings:
`CanonicalSession → render_operational_diary(session) -> str` and
`render_detailed_diary(session) -> str`. They make no inference and have no
Phase 6 runtime dependency. `main.py` selects the input, runs the configured
pipeline, receives both strings, and writes them to
`experiment/output/operational_diary.txt` and
`experiment/output/detailed_diary.txt`. Empty sessions continue to render as
empty strings. The single-video MVP workspace uses `experiment/input/` and
`experiment/output/`; generated contents remain ignored and `.gitkeep`
placeholders remain trackable.
Phase 9 full-system integration and real end-to-end validation are COMPLETE. All three modes were run through the production entry point; detailed comparisons and request diagnostics are recorded in `data/validation/phase9_5/phase9_5_validation_report.md`. Phase 10 is reserved for quality improvements identified by that representative validation.

**Phase 9: Full System Integration & Main Orchestrator is COMPLETE.**
The production entry point runs the configured `ocr_only`, `vlm_only`, or
`ocr_vlm` path through canonical conversion and the two pure diary renderers.
Phase 9.5 validated all three modes on `investigation_switch_test.mp4` with
`gemini-3.6-flash`: OCR-only made zero Gemini requests, VLM-only made two Pass 1
requests, and combined mode made two Pass 1 plus two Pass 2 requests. Combined
mode remains the MVP default. It added route/visual context and retained
unresolved values conservatively; OCR-only's focused refinement now connects
existing upper-ticket extraction so it attempts meaningful flight, route, time,
aircraft and TRACK fields. It remains source-faithful and does not infer visual
map claims from ticket text. Failed Investigations are reported with safe
per-Investigation diagnostics before canonicalization and prevent all diary
writes for that video. VLM-only made confident errors on Investigation 1. Earlier VLM-only attempts
received transient HTTP 503 responses, followed by a successful full run. The
runs preserve the two fixed output files and clean temporary media. These are
representative manual results, not a benchmark; further quality work is deferred
to Phase 10. See the detailed comparison at
[`data/validation/phase9_5/phase9_5_validation_report.md`](data/validation/phase9_5/phase9_5_validation_report.md).

See the [project plan](docs/project_plan.md) and [system design](docs/system_design.md).
Run local closure checks with `uv run python scripts/validation/validate_phase3.py`,
`uv run python scripts/validation/validate_phase4.py`, and
`uv run python scripts/validation/validate_phase5.py` when the required recordings
are available under `data/videos/`. Detailed measurements stay in local reports.

## Goals

The system is intended to:

- detect aircraft investigation events in Flightradar24 screen recordings;
- extract structured information from the aircraft ticket;
- describe visible route context and movement direction;
- use OCR, computer vision, and selective VLM assistance;
- preserve each Investigation as an independent event;
- generate a canonical structured session record;
- render a concise operational log and a detailed report.

## Project Structure

```text
Events_Diary/
├── docs/
│   ├── system_design.md
│   └── project_plan.md
├── src/
│   └── events_diary/
│       └── __init__.py
├── tests/
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
```

## Development Environment

Current baseline:

- Python 3.10
- uv
- pytest
- Ruff

## Setup

Clone the repository:

```bash
git clone https://github.com/Guy-Ra/Events_Diary.git
cd Events_Diary
```

Install the project and development dependencies:

```bash
uv sync
```

Verify the package:

```bash
uv run python -c "import events_diary; print(events_diary)"
```

Run tests:

```bash
uv run pytest
```

Run Ruff:

```bash
uv run ruff check .
```

## Documentation

Detailed system design:

```text
docs/system_design.md
```

Implementation roadmap:

```text
docs/project_plan.md
```

## Development Approach

The project is developed incrementally.

Each phase follows the same process:

```text
Plan
→ Implement
→ Unit tests
→ Real-example validation
→ Integration test
→ Regression tests
→ Refactor and document
```

The goal is to keep the implementation modular, testable, and evidence-driven without introducing unnecessary abstractions.

## VLM-only reliability and partial runs

VLM-only processing continues after an individual Investigation failure while
preserving successful results. Both diaries keep failed intervals in
chronological position with an explicit `VLM processing failed` marker and a
concise reason. No values are invented for failed Investigations. Overall status
is `completed` when all succeed, `partial` when successes and failures coexist,
and `failed` when none succeeds. Successful structured fields, provenance,
selected evidence references, and sanitized failure diagnostics are retained in
`experiment/output/vlm_run_results.json`.

Gemini uses direct HTTP without SDK retries and makes at most three total
requests per Investigation, including the initial call and schema-compatibility
calls. Only HTTP 429/503 and request timeouts are retried; explicit daily quota
exhaustion stops immediately. Configurable exponential backoff with jitter
honors provider `Retry-After`/`retryDelay` metadata. A 429 without explanatory
metadata is classified as unknown. Real-provider reliability validation of
these rules remains pending; the MVP is not fully reliability-validated until
that validation succeeds.
