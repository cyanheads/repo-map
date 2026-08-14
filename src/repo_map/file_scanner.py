"""Scans the repository to identify and summarize files."""

import hashlib
import json
import logging
import os
import sqlite3
import stat
from dataclasses import dataclass
from typing import Any

import pathspec

from repo_map.code_parser import get_imports, get_module_docstring, get_structure
from repo_map.models import (
    MAX_SOURCE_BYTES,
    SUPPORTED_LANGUAGES,
    SUPPORTED_TEXT_LANGUAGES,
)

logger = logging.getLogger(__name__)

_LANGUAGE_SUFFIXES = tuple(
    sorted(
        (key for key in SUPPORTED_LANGUAGES if key.startswith(".")),
        key=lambda suffix: (-len(suffix), suffix),
    )
)

DEFAULT_IGNORE_PATTERNS = [
    ".git/",
    ".hg/",
    ".svn/",
    "CVS/",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".pytest_cache/",
    ".mypy_cache/",
    ".venv/",
    "venv/",
    "env/",
    ".env",
    "build/",
    "dist/",
    "*.egg-info/",
    "node_modules/",
    ".DS_Store",
    "*.db",
    "*.sqlite3",
    "*.log",
    ".repo-map-cache.db",
    ".repo_map_structure.json",
    "*_repo_map.md",
]


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """A complete eligible source file and the hash of its exact bytes."""

    source: str
    sha256: str


def get_ignore_spec(root_dir: str) -> pathspec.GitIgnoreSpec:
    """Create a GitIgnoreSpec combining default patterns with the root .gitignore.

    Uses ``GitIgnoreSpec`` (rather than ``PathSpec`` with the deprecated
    ``"gitwildmatch"`` alias) so matching mirrors Git's actual behavior on
    edge cases like negation inside ignored directories — important since the
    point is to skip exactly what Git would skip.
    """
    patterns = list(DEFAULT_IGNORE_PATTERNS)
    gitignore_path = os.path.join(root_dir, ".gitignore")
    if os.path.exists(gitignore_path):
        try:
            with open(gitignore_path, encoding="utf-8") as handle:
                patterns.extend(handle.read().splitlines())
        except OSError as exc:
            logger.warning("Could not read root .gitignore: %s", exc)

    filtered = [p for p in patterns if p.strip() and not p.strip().startswith("#")]
    return pathspec.GitIgnoreSpec.from_lines(filtered)


def load_source_snapshot(
    file_path: str,
    language: str | None,
    expected_hash: str | None = None,
) -> SourceSnapshot | None:
    """Load a complete eligible UTF-8 source file without following symlinks."""
    if language not in SUPPORTED_TEXT_LANGUAGES:
        return None

    try:
        file_stat = os.lstat(file_path)
        if not stat.S_ISREG(file_stat.st_mode):
            return None
        with open(file_path, "rb") as handle:
            raw_source = handle.read(MAX_SOURCE_BYTES + 1)
    except OSError as exc:
        logger.warning("Cannot read source file %s: %s", file_path, exc)
        return None

    if len(raw_source) > MAX_SOURCE_BYTES or b"\0" in raw_source:
        return None

    try:
        source = raw_source.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None

    source_hash = hashlib.sha256(raw_source).hexdigest()
    if expected_hash is not None and source_hash != expected_hash:
        logger.warning("Source file changed after scanning; skipping %s", file_path)
        return None

    return SourceSnapshot(source=source, sha256=source_hash)


def _detect_language(file_path: str) -> str | None:
    """Detect a language by exact basename, then longest matching suffix."""
    basename = os.path.basename(file_path).lower()
    if language := SUPPORTED_LANGUAGES.get(basename):
        return language

    for suffix in _LANGUAGE_SUFFIXES:
        if basename.endswith(suffix):
            return SUPPORTED_LANGUAGES[suffix]
    return None


def _process_file(
    full_path: str, level: int, cache_conn: sqlite3.Connection
) -> dict[str, Any]:
    """Return metadata for a single file, preferring cached data."""
    language = _detect_language(full_path)

    file_info: dict[str, Any] = {
        "name": os.path.basename(full_path),
        "path": full_path,
        "level": level,
        "type": "file",
        "language": language,
        "source_eligible": False,
    }

    if not language:
        return file_info

    snapshot = load_source_snapshot(full_path, language)
    if snapshot is None:
        return file_info

    file_hash = snapshot.sha256
    file_info["source_eligible"] = True
    cursor = cache_conn.cursor()
    cursor.execute(
        """
        SELECT hash,
               description,
               developer_consideration,
               imports,
               functions,
               maintenance_flag,
               critical_dependencies,
               architectural_role,
               refactoring_suggestions,
               security_assessment
        FROM cache
        WHERE path = ?
        """,
        (full_path,),
    )
    row = cursor.fetchone()

    if row and row[0] == file_hash:
        file_info.update(
            {
                "hash": file_hash,
                "description": row[1] or "",
                "developer_consideration": row[2] or "",
                "imports": json.loads(row[3]) if row[3] else [],
                "functions": json.loads(row[4]) if row[4] else [],
                "maintenance_flag": row[5] or "Unknown",
                "critical_dependencies": row[6] or "{}",
                "architectural_role": row[7] or "Unknown",
                "refactoring_suggestions": row[8] or "None",
                "security_assessment": row[9] or "None",
            }
        )
        return file_info

    try:
        classes, funcs, consts = get_structure(full_path, language)
        docstring = get_module_docstring(full_path, language)
        imports = get_imports(full_path, language)
    except UnicodeDecodeError as exc:
        logger.warning("Cannot decode source file %s as UTF-8: %s", full_path, exc)
        file_info["source_eligible"] = False
        return file_info
    file_info.update(
        {
            "classes": classes,
            "functions": funcs,
            "constants": consts,
            "imports": imports,
            "description": docstring,
            "hash": file_hash,
        }
    )
    return file_info


def summarize_repo(
    root_dir: str, cache_conn: sqlite3.Connection
) -> list[dict[str, Any]]:
    """Summarize the repository by recursively scanning directories and files."""
    summary: list[dict[str, Any]] = []
    abs_root_dir = os.path.abspath(root_dir)
    ignore_spec = get_ignore_spec(abs_root_dir)

    def _scan(current_path: str, level: int) -> None:
        try:
            with os.scandir(current_path) as directory:
                entries = [entry for entry in directory if not entry.is_symlink()]
        except OSError as exc:
            logger.warning("Cannot read directory %s: %s", current_path, exc)
            return

        entries.sort(
            key=lambda entry: (
                not entry.is_dir(follow_symlinks=False),
                entry.name,
            )
        )

        for entry in entries:
            full_path = entry.path
            relative_path = os.path.relpath(full_path, abs_root_dir)

            if ignore_spec.match_file(relative_path):
                continue

            if entry.is_dir(follow_symlinks=False):
                summary.append(
                    {
                        "name": entry.name,
                        "path": full_path,
                        "level": level,
                        "type": "directory",
                    }
                )
                _scan(full_path, level + 1)
            elif entry.is_file(follow_symlinks=False):
                summary.append(_process_file(full_path, level, cache_conn))

    _scan(abs_root_dir, 0)
    return summary
