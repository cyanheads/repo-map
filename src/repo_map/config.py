"""Configuration settings for the repo-map application."""

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Defines the application settings.

    Attributes:
        openrouter_api_key: The API key for OpenRouter.
        openrouter_model_name: The model name to be used with the OpenRouter API.
        openrouter_api_url: The URL for the OpenRouter API.
        http_referer: The HTTP referer to be used in API requests.
        app_name: The name of the application.
        api_semaphore_limit: The concurrency limit for API calls.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    openrouter_api_key: str = "dummy"  # Default for static analysis
    openrouter_model_name: str = "google/gemini-2.5-flash-preview-09-2025"
    openrouter_api_url: str = "https://openrouter.ai/api/v1/chat/completions"

    # For OpenRouter headers
    http_referer: str = "https://github.com/cyanheads/repo-map"
    app_name: str = "repo-map"

    # Concurrency limit for API calls
    api_semaphore_limit: int = 3

    @model_validator(mode="after")
    def check_api_key(self) -> "Settings":
        """Checks that the API key is not the dummy value."""
        if self.openrouter_api_key == "dummy":
            raise ValueError(
                "OPENROUTER_API_KEY must be set in the environment or .env file."
            )
        return self


# Create a single instance to be imported by other modules
settings: Settings = Settings()
