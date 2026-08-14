"""Manages the cache for LLM responses."""

import logging
import os
import sqlite3

logger = logging.getLogger(__name__)

CACHE_FILE_NAME = ".repo-map-cache.db"


class CacheError(RuntimeError):
    """Raised when the repository cache cannot be opened or initialized."""


def load_cache(repo_root: str) -> sqlite3.Connection:
    """Load or initialize the SQLite cache database for a repository.

    Raises:
        CacheError: The cache file is unreachable, unwritable, or not a
            database. The message names the path and the remedy, so callers
            report it rather than inspecting the underlying `sqlite3` failure.
    """
    cache_file_path = os.path.join(repo_root, CACHE_FILE_NAME)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(cache_file_path)
        _apply_schema(connection)
    except (sqlite3.Error, OSError) as exc:
        if connection is not None:
            connection.close()
        raise CacheError(
            f"Cannot use the repo-map cache at {cache_file_path}: {exc}. "
            f"Delete {CACHE_FILE_NAME} to force a clean rebuild."
        ) from exc
    return connection


def relative_cache_key(repo_root: str, file_path: str) -> str:
    """Return a file's cache key: its repository-relative path, forward-slash separated.

    Keying on the relative path is what lets the database travel with the
    repository it describes -- through a rename, a fresh clone, a CI checkout,
    or a container mount at another prefix. Separators are normalized so a
    cache written on Windows still reads on POSIX.

    A row keyed any other way -- an absolute path written before this key
    format -- never matches a key produced here, so it reads as a miss and is
    swept by `prune_cache` at the end of the run. Nothing rewrites such a row
    in place: two absolute keys recorded under different prefixes reduce to the
    same relative key, and updating the second would violate the primary key.
    """
    return os.path.relpath(file_path, repo_root).replace(os.sep, "/")


def prune_cache(conn: sqlite3.Connection, scanned_keys: set[str]) -> int:
    """Delete rows outside `scanned_keys`, returning how many were removed.

    Orphans are collected in Python and deleted one key at a time rather than
    with a single ``NOT IN`` clause, which would bind one parameter per scanned
    file and can exceed SQLite's variable limit on a large repository.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT path FROM cache")
    orphaned = [(row[0],) for row in cursor.fetchall() if row[0] not in scanned_keys]
    if orphaned:
        cursor.executemany("DELETE FROM cache WHERE path = ?", orphaned)
        conn.commit()
    return len(orphaned)


def _apply_schema(conn: sqlite3.Connection) -> None:
    """Create the cache table and add any columns a legacy database lacks."""
    cursor = conn.cursor()

    cursor.execute("""
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
            refactoring_suggestions TEXT,
            security_assessment TEXT
        )
        """)

    cursor.execute("PRAGMA table_info(cache)")
    existing_columns = {info[1] for info in cursor.fetchall()}

    new_columns = {
        "developer_consideration": "TEXT",
        "imports": "TEXT",
        "functions": "TEXT",
        "maintenance_flag": "TEXT",
        "critical_dependencies": "TEXT",
        "architectural_role": "TEXT",
        "refactoring_suggestions": "TEXT",
        "security_assessment": "TEXT",
    }

    for column, column_type in new_columns.items():
        if column not in existing_columns:
            logger.info("Adding missing cache column '%s'", column)
            cursor.execute(f"ALTER TABLE cache ADD COLUMN {column} {column_type}")

    conn.commit()
