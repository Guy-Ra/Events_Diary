"""Tests for Events Diary application configuration."""

from events_diary.config import AppConfig


def test_default_log_level() -> None:
    """Verify that the default application log level is INFO."""

    config = AppConfig()

    assert config.log_level == "INFO"