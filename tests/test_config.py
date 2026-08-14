"""Tests for settings loading and the transport values a planted `.env` must not reach."""

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

import repo_map.llm_service as llm_module
from repo_map.config import ConfigurationError, Settings, load_settings

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
BUILT_IN_REFERER = "https://github.com/cyanheads/repo-map"
BUILT_IN_APP_NAME = "repo-map"

REPO_MAP_ENV_VARS = (
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL_NAME",
    "API_SEMAPHORE_LIMIT",
    "OPENROUTER_API_URL",
    "HTTP_REFERER",
    "APP_NAME",
)


@pytest.fixture
def scratch_cwd(tmp_path, monkeypatch):
    """Run from an empty working directory with every repo-map variable unset."""
    for name in REPO_MAP_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _capturing_session() -> tuple[type, dict[str, Any]]:
    """Return an ``aiohttp.ClientSession`` stand-in plus the dict it records into."""
    captured: dict[str, Any] = {}

    class Response:
        status = 200

        async def __aenter__(self) -> "Response":
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def json(self) -> dict[str, Any]:
            return {"choices": []}

    class Session:
        async def __aenter__(self) -> "Session":
            return self

        async def __aexit__(self, *args) -> None:
            return None

        def post(self, url: str, *, headers: dict[str, str], **kwargs) -> Response:
            captured["url"] = url
            captured["headers"] = headers
            return Response()

    return Session, captured


def _capture_request(monkeypatch, settings: Settings) -> dict[str, Any]:
    """Run one OpenRouter call against ``settings`` and return the intercepted request."""
    session_cls, captured = _capturing_session()
    monkeypatch.setattr(llm_module.aiohttp, "ClientSession", session_cls)
    monkeypatch.setattr(llm_module, "settings", settings)

    asyncio.run(llm_module.rate_limited_api_call([], "test-model", 0.0))

    return captured


def test_documented_settings_load_from_project_root_env(scratch_cwd) -> None:
    """`.env` in the working directory supplies key, model, and concurrency."""
    (scratch_cwd / ".env").write_text(
        "OPENROUTER_API_KEY=sk-from-env-file\n"
        "OPENROUTER_MODEL_NAME=vendor/model-from-env-file\n"
        "API_SEMAPHORE_LIMIT=7\n",
        encoding="utf-8",
    )

    settings = Settings()

    assert settings.openrouter_api_key == "sk-from-env-file"
    assert settings.openrouter_model_name == "vendor/model-from-env-file"
    assert settings.api_semaphore_limit == 7
    assert settings.has_api_key() is True


def test_shell_environment_wins_over_env_file(scratch_cwd, monkeypatch) -> None:
    """Exported variables take precedence over the same keys in `.env`."""
    (scratch_cwd / ".env").write_text(
        "OPENROUTER_API_KEY=sk-from-env-file\n"
        "OPENROUTER_MODEL_NAME=vendor/model-from-env-file\n"
        "API_SEMAPHORE_LIMIT=7\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-shell")
    monkeypatch.setenv("OPENROUTER_MODEL_NAME", "vendor/model-from-shell")
    monkeypatch.setenv("API_SEMAPHORE_LIMIT", "5")

    settings = Settings()

    assert settings.openrouter_api_key == "sk-from-shell"
    assert settings.openrouter_model_name == "vendor/model-from-shell"
    assert settings.api_semaphore_limit == 5


def test_defaults_apply_without_env_file_or_environment(scratch_cwd) -> None:
    """With nothing configured the key is absent and the documented defaults hold."""
    settings = Settings()

    assert settings.openrouter_api_key is None
    assert settings.has_api_key() is False
    assert settings.openrouter_model_name == "anthropic/claude-sonnet-4.6"
    assert settings.api_semaphore_limit == 3


def test_blank_api_key_is_not_treated_as_configured(scratch_cwd) -> None:
    """An empty `OPENROUTER_API_KEY` fails `has_api_key()` rather than reading as set."""
    (scratch_cwd / ".env").write_text("OPENROUTER_API_KEY=\n", encoding="utf-8")

    assert Settings().has_api_key() is False


def test_non_integer_concurrency_limit_is_rejected(scratch_cwd) -> None:
    """A malformed `API_SEMAPHORE_LIMIT` fails validation instead of silently defaulting."""
    (scratch_cwd / ".env").write_text("API_SEMAPHORE_LIMIT=many\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("value", ["0", "abc"])
def test_load_settings_reports_a_bad_limit_in_one_actionable_line(
    scratch_cwd, value: str
) -> None:
    """The singleton's construction names the value and the valid range."""
    (scratch_cwd / ".env").write_text(
        f"API_SEMAPHORE_LIMIT={value}\n", encoding="utf-8"
    )

    with pytest.raises(ConfigurationError) as failure:
        load_settings()

    message = str(failure.value)
    assert "API_SEMAPHORE_LIMIT" in message
    assert repr(value) in message
    assert "must be an integer of 1 or greater" in message
    assert "\n" not in message


def test_configuration_error_is_reachable_from_the_import_boundary() -> None:
    """`run_main()` guards its deferred import with `except ValueError`."""
    assert issubclass(ConfigurationError, ValueError)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_out_of_range_concurrency_limit_is_rejected(scratch_cwd, value: str) -> None:
    """A concurrency limit below one fails validation rather than reaching a semaphore."""
    (scratch_cwd / ".env").write_text(
        f"API_SEMAPHORE_LIMIT={value}\n", encoding="utf-8"
    )

    with pytest.raises(ValidationError):
        Settings()


def test_request_targets_the_built_in_openrouter_endpoint(
    scratch_cwd, monkeypatch
) -> None:
    """A clean environment produces the built-in URL and headers."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-real-key")

    captured = _capture_request(monkeypatch, Settings())

    assert captured["url"] == OPENROUTER_ENDPOINT
    assert captured["headers"] == {
        "Authorization": "Bearer sk-real-key",
        "HTTP-Referer": BUILT_IN_REFERER,
        "X-Title": BUILT_IN_APP_NAME,
    }


def test_planted_env_cannot_redirect_requests_or_rewrite_headers(
    scratch_cwd, monkeypatch
) -> None:
    """A `.env` in the analyzed repository cannot move the endpoint or forge headers."""
    (scratch_cwd / ".env").write_text(
        "OPENROUTER_API_URL=https://evil.example/collect\n"
        "HTTP_REFERER=https://evil.example\n"
        "APP_NAME=evil\n"
        "OPENROUTER_API_KEY=sk-attacker-placeholder\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-real-key")

    captured = _capture_request(monkeypatch, Settings())

    assert captured["url"] == OPENROUTER_ENDPOINT
    assert captured["headers"]["HTTP-Referer"] == BUILT_IN_REFERER
    assert captured["headers"]["X-Title"] == BUILT_IN_APP_NAME
    assert captured["headers"]["Authorization"] == "Bearer sk-real-key"
