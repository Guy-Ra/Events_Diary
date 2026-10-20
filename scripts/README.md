# Standalone scripts

- `calibration/`: empirical design, threshold and parameter studies, including
  manual layout/color exploration.
- `evaluation/`: feature-level real-video/screenshot evaluation and diagnostic
  visualization of implemented behavior.
- `validation/`: regression and phase-closure checks with pass/fail invariants.

Production and reusable importable logic belongs under `src/events_diary/`;
scripts should orchestrate and review it rather than duplicate it. Existing
script-to-script preparation helpers retain their explicit imports; this
organization does not refactor those helpers into production code.

Run from the repository with `uv run python scripts/<category>/<script>.py`.
Examples:

```text
uv run python scripts/validation/validate_phase3.py
uv run python scripts/validation/validate_phase4.py
uv run python scripts/validation/validate_phase5.py
uv run python scripts/evaluation/evaluate_frame_materialization.py
uv run python scripts/evaluation/evaluate_ticket_view_similarity.py
uv run python scripts/evaluation/evaluate_representative_selection.py
uv run python scripts/calibration/evaluate_ticket_view_similarity_calibration.py
```

Output locations under `data/debug/` are unchanged. Scripts that accept explicit
relative input paths retain their existing working-directory semantics.
