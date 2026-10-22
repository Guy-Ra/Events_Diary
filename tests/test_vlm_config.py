"""Focused tests for central dotenv-backed VLM API configuration."""

from events_diary.config import VLMAPISettings, load_vlm_api_settings


def test_dotenv_configuration_loads_provider_model_and_timeout(tmp_path, monkeypatch):
    """Load all user settings centrally from one isolated dotenv file."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "EVENTS_DIARY_VLM_PROVIDER=gemini\n"
        "EVENTS_DIARY_VLM_API_KEY=\n"
        "EVENTS_DIARY_VLM_API_BASE_URL=https://generativelanguage.googleapis.com\n"
        "EVENTS_DIARY_VLM_API_VERSION=v1beta\n"
        "EVENTS_DIARY_VLM_MODEL=gemini-flash-test\n"
        "EVENTS_DIARY_VLM_REQUEST_TIMEOUT_S=42\n",
        encoding="utf-8",
    )
    for name in (
        "EVENTS_DIARY_VLM_PROVIDER", "EVENTS_DIARY_VLM_API_KEY",
        "EVENTS_DIARY_VLM_API_BASE_URL", "EVENTS_DIARY_VLM_API_VERSION",
        "EVENTS_DIARY_VLM_MODEL", "EVENTS_DIARY_VLM_REQUEST_TIMEOUT_S",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = load_vlm_api_settings(env_file=env_file, override=True)
    assert settings.provider == "gemini"
    assert settings.model_name == "gemini-flash-test"
    assert settings.request_timeout_s == 42.0
    assert not settings.api_key_configured


def test_process_environment_can_switch_model_without_source_changes(tmp_path, monkeypatch):
    """Give process environment precedence over the local model default."""
    env_file = tmp_path / ".env"
    env_file.write_text("EVENTS_DIARY_VLM_MODEL=gemini-flash-latest\n", encoding="utf-8")
    monkeypatch.setenv("EVENTS_DIARY_VLM_MODEL", "gemini-next-test")
    settings = load_vlm_api_settings(env_file=env_file)
    assert settings.model_name == "gemini-next-test"


def test_endpoint_is_assembled_from_configuration():
    """Construct endpoint path from the configured base, API version and model."""
    settings = VLMAPISettings(
        provider="gemini", api_key="", base_url="https://api.example.test/",
        api_version="v1", model_name="model-from-config", request_timeout_s=5,
    )
    assert settings.endpoint_url == "https://api.example.test/v1/models/model-from-config:generateContent"


def test_api_key_is_excluded_from_repr_and_empty_key_is_reported(tmp_path, monkeypatch):
    """Keep credentials out of repr and expose only a configured boolean."""
    env_file = tmp_path / ".env"
    env_file.write_text("EVENTS_DIARY_VLM_API_KEY=\n", encoding="utf-8")
    monkeypatch.delenv("EVENTS_DIARY_VLM_API_KEY", raising=False)
    empty = load_vlm_api_settings(env_file=env_file, override=True)
    assert empty.api_key == ""
    assert empty.api_key_configured is False

    secret = "fake-test-credential-only"
    configured = VLMAPISettings(
        provider="gemini", api_key=secret, base_url="https://example.test",
        api_version="v1beta", model_name="model", request_timeout_s=1,
    )
    assert configured.api_key_configured is True
    assert secret not in repr(configured)
