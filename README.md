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

**Current: Phase 6 — Complete Investigation Understanding (implementation in
progress).** The approved scope includes deterministic OCR TRACK-to-compass
conversion, map/visible-route observations in the existing Pass 1 call, route
review in the existing Pass 2 call, and a provenance-preserving direction
cross-check. Real provider validation for these additions, Phase 6.6 and Phase
6.7 remains pending Gemini quota. No extra VLM call is planned.

The former standalone route/direction work is absorbed into Phase 6. The remaining
roadmap is Phase 7 (absorbed former Phase 8 boundary), Phase 8 aircraft entity
matching, Phase 9 canonical output, and Phase 10 post-MVP system optimization.

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
- identify repeated investigations of the same aircraft;
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
