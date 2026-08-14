"""Configuration settings for the repo-map application."""

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# Request transport values are constants, not settings. `env_file` resolves
# against the process working directory, so a repository being analyzed could
# otherwise plant a `.env` that redirects every request -- and the user's real
# API key with it -- to an attacker-chosen host. Keeping the endpoint and
# headers off `Settings` leaves nothing there for such a file to reach, while
# the credential and tuning fields below still load as documented.
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
HTTP_REFERER = "https://github.com/cyanheads/repo-map"
APP_NAME = "repo-map"


class ConfigurationError(ValueError):
    """Raised when an environment or `.env` setting is absent or out of range."""


class Settings(BaseSettings):
    """
    Defines the application settings.

    Attributes:
        openrouter_api_key: The API key for OpenRouter.
        openrouter_model_name: The model name to be used with the OpenRouter API.
        api_semaphore_limit: The concurrency limit for API calls.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    openrouter_api_key: str | None = None
    openrouter_model_name: str = "anthropic/claude-sonnet-4.6"

    # Concurrency limit for API calls
    api_semaphore_limit: int = Field(default=3, ge=1)

    def has_api_key(self) -> bool:
        """Returns True when an OpenRouter API key is available."""
        return bool(self.openrouter_api_key)


# What a rejected field needs instead, keyed by attribute name. Falls back to
# pydantic's own message for anything not listed.
_FIELD_GUIDANCE = {"api_semaphore_limit": "must be an integer of 1 or greater"}


def _describe_validation_error(exc: ValidationError) -> str:
    """Render settings validation failures as a single actionable line."""
    details = []
    for error in exc.errors():
        field = str(error["loc"][0])
        guidance = _FIELD_GUIDANCE.get(field, error["msg"])
        details.append(f"{field.upper()}={error['input']!r} ({guidance})")
    return "Invalid configuration: " + "; ".join(details)


def load_settings() -> Settings:
    """Build the settings singleton, translating validation failures.

    This runs while Python resolves the console script's own import chain, so a
    raw `ValidationError` here reaches the user as a traceback before any
    handler exists. `ConfigurationError` subclasses `ValueError`, which
    `run_main()` catches around its deferred import of the application.
    """
    try:
        return Settings()
    except ValidationError as exc:
        raise ConfigurationError(_describe_validation_error(exc)) from exc


# Create a single instance to be imported by other modules
settings: Settings = load_settings()
