"""Scans the repository to identify and summarize files."""

import hashlib
import json
import logging
import os
import sqlite3
from typing import Any

import pathspec

from repo_map.code_parser import get_imports, get_module_docstring, get_structure
from repo_map.models import SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

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


def get_ignore_spec(root_dir: str) -> pathspec.PathSpec:
    """Create a PathSpec combining default patterns with the root .gitignore."""
    patterns = list(DEFAULT_IGNORE_PATTERNS)
    gitignore_path = os.path.join(root_dir, ".gitignore")
    if os.path.exists(gitignore_path):
        try:
            with open(gitignore_path, encoding="utf-8") as handle:
                patterns.extend(handle.read().splitlines())
        except OSError as exc:
            logger.warning("Could not read root .gitignore: %s", exc)

    filtered = [p for p in patterns if p.strip() and not p.strip().startswith("#")]
    return pathspec.PathSpec.from_lines("gitwildmatch", filtered)


def compute_file_hash(file_path: str) -> str:
    """Compute the SHA-256 hash of a file's bytes."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except OSError as exc:
        logger.error("Error reading file %s for hashing: %s", file_path, exc)
        return ""


def _process_file(
    full_path: str, level: int, cache_conn: sqlite3.Connection
) -> dict[str, Any]:
    """Return metadata for a single file, preferring cached data."""
    _, ext = os.path.splitext(full_path)
    language = SUPPORTED_LANGUAGES.get(ext.lower())

    file_info: dict[str, Any] = {
        "name": os.path.basename(full_path),
        "path": full_path,
        "level": level,
        "type": "file",
        "language": language,
    }

    if not language:
        return file_info

    file_hash = compute_file_hash(full_path)
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
               code_quality_score,
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
                "code_quality_score": row[8] or 0,
                "refactoring_suggestions": row[9] or "None",
                "security_assessment": row[10] or "None",
            }
        )
        return file_info

    classes, funcs, consts = get_structure(full_path, language)
    docstring = get_module_docstring(full_path, language)
    imports = get_imports(full_path, language)
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
            entries = sorted(os.listdir(current_path))
        except OSError as exc:
            logger.warning("Cannot read directory %s: %s", current_path, exc)
            return

        entries.sort(key=lambda entry: not os.path.isdir(os.path.join(current_path, entry)))

        for name in entries:
            full_path = os.path.join(current_path, name)
            relative_path = os.path.relpath(full_path, abs_root_dir)

            if ignore_spec.match_file(relative_path):
                continue

            if os.path.isdir(full_path):
                summary.append(
                    {
                        "name": name,
                        "path": full_path,
                        "level": level,
                        "type": "directory",
                    }
                )
                _scan(full_path, level + 1)
            elif os.path.isfile(full_path):
                summary.append(_process_file(full_path, level, cache_conn))

    _scan(abs_root_dir, 0)
    return summary
