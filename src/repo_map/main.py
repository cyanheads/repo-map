"""
Main script for the repo-map tool.

Provides a CLI for generating a structured summary of a software repository and
enhancing it with AI-generated documentation hints.
"""
import asyncio
import logging
import sys

from repo_map.cli_handler import RepoMapApp

logger = logging.getLogger(__name__)


def run_main() -> None:
    """Run the application and capture top-level exceptions."""
    try:
        app = RepoMapApp()
        asyncio.run(app.run())
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user. Exiting.")
        sys.exit(130)
    except RuntimeError:
        logger.error("An unexpected runtime error occurred:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_main()
