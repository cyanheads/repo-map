"""
Main script for the repo-map tool.

Provides a CLI for generating a structured summary of a software repository and
enhancing it with AI-generated documentation hints.
"""

import asyncio
import logging
import sys

from repo_map.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def run_main() -> None:
    """Run the application and capture top-level exceptions."""
    setup_logging()

    # Deferred deliberately: importing the application constructs the settings
    # singleton and the shared rate limiter, so an invalid `API_SEMAPHORE_LIMIT`
    # fails here. At module scope that failure lands in the console script's own
    # import statement, where no handler of ours can reach it. `ValueError` is
    # what both `ConfigurationError` and the concurrency guard raise.
    try:
        from repo_map.cli_handler import RepoMapApp
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    try:
        asyncio.run(RepoMapApp().run())
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user. Exiting.")
        sys.exit(130)
    except Exception:
        logger.error("An unexpected error occurred:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_main()
