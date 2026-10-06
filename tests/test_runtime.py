"""Tests for Events Diary runtime environment inspection."""

from events_diary.runtime import get_runtime_profile


def test_runtime_profile_contains_basic_information() -> None:
    """Verify that the runtime profile contains essential environment data."""

    profile = get_runtime_profile()

    assert profile.python_version
    assert profile.operating_system
    assert profile.machine