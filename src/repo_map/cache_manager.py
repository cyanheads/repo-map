"""Manages the cache for LLM responses."""

import logging
import os
import sqlite3

logger = logging.getLogger(__name__)


def load_cache(repo_root: str) -> sqlite3.Connection:
    """Load or initialize the SQLite cache database for a repository."""
    cache_file_path = os.path.join(repo_root, ".repo-map-cache.db")
    conn = sqlite3.connect(cache_file_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS cache (
            path TEXT PRIMARY KEY,
            hash TEXT,
            description TEXT,
            developer_consideration TEXT,
            imports TEXT,
            functions TEXT,
            maintenance_flag TEXT,
            critical_dependencies TEXT,
            architectural_role TEXT,
            code_quality_score INTEGER,
            refactoring_suggestions TEXT,
            security_assessment TEXT
        )
        """
    )

    cursor.execute("PRAGMA table_info(cache)")
    existing_columns = {info[1] for info in cursor.fetchall()}

    new_columns = {
        "developer_consideration": "TEXT",
        "imports": "TEXT",
        "functions": "TEXT",
        "maintenance_flag": "TEXT",
        "critical_dependencies": "TEXT",
        "architectural_role": "TEXT",
        "code_quality_score": "INTEGER",
        "refactoring_suggestions": "TEXT",
        "security_assessment": "TEXT",
    }

    for column, column_type in new_columns.items():
        if column not in existing_columns:
            logger.info("Adding missing cache column '%s'", column)
            cursor.execute(f"ALTER TABLE cache ADD COLUMN {column} {column_type}")

    conn.commit()
    return conn
