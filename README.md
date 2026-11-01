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
validation dataset. Timeout was 900 seconds with up to three attempts and
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

**Roadmap:** Phase 6 COMPLETE; Phase 7 COMPLETE (7.1–7.6 COMPLETE); Phase 8
COMPLETE (8.1–8.5 COMPLETE); full system integration is next; Phase 9 planned; Phase 10 System
Optimization & Improvement is post-MVP. The former standalone route-analysis
phase remains absorbed into Phase 6.

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
Phase 6 runtime dependency. The future orchestrator will select/load the input
video, run the full pipeline, receive both strings, and write them to
`experiment/output/operational_diary.txt` and
`experiment/output/detailed_diary.txt`. Empty sessions continue to render as
empty strings. The single-video MVP workspace uses `experiment/input/` and
`experiment/output/`; generated contents remain ignored and `.gitkeep`
placeholders remain trackable.
Historical `data/` artifacts remain until full-pipeline integration is validated
and a deliberate cleanup is performed. Phase 8.5 real-data validation is
complete. The next major task is full system integration from video input
through detection, sampling, Investigation extraction, configured understanding,
CanonicalSession, and both diaries. Integration configuration must support
`ocr_only`, `vlm_only`, and `ocr_vlm` while preserving existing VLM
provider/model/endpoint/runtime settings. This work has not started. A unified
VLM visual-route description remains future backlog.
One Phase 6
`InvestigationRecord` becomes
one chronologically ordered canonical event, with no merging or revisit
inference. `CanonicalSession` JSON will be the source of truth for future
deterministic operational and detailed diary renderers.

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
