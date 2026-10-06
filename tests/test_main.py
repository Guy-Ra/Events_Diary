"""Tests for the Events Diary application entry point."""

from events_diary.__main__ import main


def test_main_runs_without_error() -> None:
    """Verify that the application entry point initializes successfully."""

    main()