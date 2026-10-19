# Events Diary

Local project building toward structured aircraft event logs from Flightradar24
screen recordings, using computer vision and planned OCR/VLM assistance.

## Project Status

**Phase 4 - Investigation Extraction & Adaptive Sampling is COMPLETE.**

The offline system supports video metadata/time sampling, FR24 UI regions and
layout transforms, visual-change analysis, map motion compensation, ticket scroll
detection, and Investigation lifecycle detection. It also builds adaptive
Investigation sampling plans, detects stable ticket views, and composes unified
evidence candidate timelines. Raw and effective presence remain distinct.

Validation is representative/manual and thresholds remain provisional. OCR, final
frame selection, structured ticket extraction, and VLM logic are not implemented.

**Next planned: Phase 5 - Role-Based Frame & Ticket-Section Selection (NOT STARTED).**

See the [project plan](docs/project_plan.md) and [system design](docs/system_design.md).
Run local closure validation with `uv run python scripts/validate_phase3.py` and
`uv run python scripts/validate_phase4.py` when the required local recordings are
available under `data/videos/`. Phase 4 covers seven representative recordings;
detailed measurements remain in local debug reports.

## Goals

The system is intended to:

- detect aircraft investigation events in Flightradar24 screen recordings;
- extract structured information from the aircraft ticket;
- analyze route geometry and movement direction;
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
