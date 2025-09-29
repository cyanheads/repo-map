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

# A robust list of default patterns to ignore
DEFAULT_IGNORE_PATTERNS = [
    # VCS directories
    ".git/", ".hg/", ".svn/", "CVS/",
    # Python specific
    "__pycache__/", "*.pyc", "*.pyo", "*.pyd",
    ".pytest_cache/", ".mypy_cache/",
    # Virtual environments
    ".venv/", "venv/", "env/", ".env",
    # Build artifacts
    "build/", "dist/", "*.egg-info/",
    # Node.js
    "node_modules/",
    # OS generated files
    ".DS_Store",
    # Tool-specific
    "*.db", "*.sqlite3", "*.log",
    # Repo-map specific
    ".repo-map-cache.db", ".repo_map_structure.json", "*_repo_map.md"
]


def get_ignore_spec(root_dir: str) -> pathspec.PathSpec:
    """Creates a PathSpec object from default and .gitignore patterns."""
    patterns = list(DEFAULT_IGNORE_PATTERNS)
    gitignore_path = os.path.join(root_dir, ".gitignore")
    if os.path.exists(gitignore_path):
        try:
            with open(gitignore_path, encoding="utf-8") as f:
                patterns.extend(f.read().splitlines())
        except OSError as e:
            logger.warning("Could not read root .gitignore: %s", e)

    # Filter out empty lines and comments from the final list
    final_patterns = [p for p in patterns if p.strip() and not p.strip().startswith("#")]
    return pathspec.PathSpec.from_lines("gitwildmatch", final_patterns)


def compute_file_hash(file_path: str) -> str:
    """Computes the SHA-256 hash of the given file."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except OSError as e:
        logger.error("Error reading file %s for hashing: %s", file_path, e)
        return ""


def _process_file(full_path: str, level: int, cache_conn: sqlite3.Connection) -> dict[str, Any]:
    """Processes a single file, checking cache and parsing if necessary."""
    _, ext = os.path.splitext(full_path)
    language = SUPPORTED_LANGUAGES.get(ext.lower())

    file_info = {
        "name": os.path.basename(full_path),
        "path": full_path,
        "level": level,
        "type": "file",
        "language": language,
    }

    if language:
        file_hash = compute_file_hash(full_path)
        cursor = cache_conn.cursor()
        cursor.execute(
            "SELECT hash, description, developer_consideration, imports, functions FROM cache WHERE path = ?",
            (full_path,),
        )
        row = cursor.fetchone()

        if row and row[0] == file_hash:
            file_info.update({
                "description": row[1], "developer_consideration": row[2],
                "imports": json.loads(row[3]) if row[3] else [],
                "functions": json.loads(row[4]) if row[4] else [], "hash": file_hash,
            })
        else:
            classes, funcs, consts = get_structure(full_path, language)
            docstring = get_module_docstring(full_path, language)
            imports = get_imports(full_path, language)
            file_info.update({
                "classes": classes, "functions": funcs, "constants": consts,
                "imports": imports, "description": docstring, "hash": file_hash,
            })
    return file_info


def summarize_repo(
    root_dir: str, cache_conn: sqlite3.Connection
) -> list[dict[str, Any]]:
    """Summarizes the repository by recursively scanning directories and files."""
    summary: list[dict[str, Any]] = []
    abs_root_dir = os.path.abspath(root_dir)
    ignore_spec = get_ignore_spec(abs_root_dir)

    def _scan(current_path: str, level: int):
        try:
            entries = sorted(os.listdir(current_path))
        except OSError as e:
            logger.warning("Cannot read directory %s: %s", current_path, e)
            return

        # Sort to prioritize directories
        entries.sort(key=lambda e: not os.path.isdir(os.path.join(current_path, e)))

        for name in entries:
            full_path = os.path.join(current_path, name)
            relative_path = os.path.relpath(full_path, abs_root_dir)

            if ignore_spec.match_file(relative_path):
                continue

            if os.path.isdir(full_path):
                dir_info = {"name": name, "path": full_path, "level": level, "type": "directory"}
                summary.append(dir_info)
                _scan(full_path, level + 1)
            elif os.path.isfile(full_path):
                file_info = _process_file(full_path, level, cache_conn)
                summary.append(file_info)

    _scan(abs_root_dir, 0)
    return summary
