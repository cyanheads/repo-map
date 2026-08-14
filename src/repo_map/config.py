"""Configuration settings for the repo-map application."""

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
    api_semaphore_limit: int = 3

    def has_api_key(self) -> bool:
        """Returns True when an OpenRouter API key is available."""
        return bool(self.openrouter_api_key)


# Create a single instance to be imported by other modules
settings: Settings = Settings()
