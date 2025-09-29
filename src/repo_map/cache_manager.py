"""Manages the cache for LLM responses."""

import logging
import os
import sqlite3

logger = logging.getLogger(__name__)


def load_cache(repo_root: str) -> sqlite3.Connection:
    """
    Loads the LLM response cache from a SQLite3 database.
    Creates the database and table if they don't exist.
    Also, adds new columns if they are missing.
    """
    cache_file_path = os.path.join(repo_root, ".repo-map-cache.db")
    conn = sqlite3.connect(cache_file_path)
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS cache (
            path TEXT PRIMARY KEY,
            hash TEXT,
            description TEXT,
            developer_consideration TEXT,
            imports TEXT,
            functions TEXT
        )
    """
    )

    # Add new columns if they don't exist
    cursor.execute("PRAGMA table_info(cache)")
    existing_columns = [info[1] for info in cursor.fetchall()]

    new_columns = {
        "developer_consideration",
    }

    for column in new_columns:
        if column not in existing_columns:
            cursor.execute(f"ALTER TABLE cache ADD COLUMN {column} TEXT")

    conn.commit()
    return conn
