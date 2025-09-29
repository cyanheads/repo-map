"""Logging configuration for the repo-map tool."""

import logging
from logging import LogRecord

from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Redirect logging records through tqdm to avoid breaking progress bars."""

    def emit(self, record: LogRecord) -> None:
        try:
            message = self.format(record)
            tqdm.write(message)
            self.flush()
        except (OSError, ValueError, TypeError):
            self.handleError(record)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging with tqdm compatibility."""
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[TqdmLoggingHandler()],
    )
