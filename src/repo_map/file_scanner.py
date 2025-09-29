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


def parse_gitignore(root_dir: str) -> list[str]:
    """
    Parses all .gitignore files in the repository to extract ignore patterns.
    This considers .gitignore files in nested directories as well.
    """
    ignore_patterns = []
    for dirpath, _, filenames in os.walk(root_dir):
        if ".gitignore" in filenames:
            gitignore_path = os.path.join(dirpath, ".gitignore")
            try:
                with open(gitignore_path, encoding="utf-8") as f:
                    patterns = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.startswith("#")
                    ]
                    # Prepend the relative path to patterns for nested .gitignore
                    rel_path = os.path.relpath(dirpath, root_dir)
                    if rel_path != ".":
                        patterns = [
                            os.path.join(rel_path, pattern) for pattern in patterns
                        ]
                    ignore_patterns.extend(patterns)
            except OSError as e:
                logger.error(
                    "Error reading .gitignore file at %s: %s", gitignore_path, e
                )
    return ignore_patterns


def should_ignore(path: str, ignore_spec: pathspec.PathSpec) -> bool:
    """
    Determines if a given path should be ignored based on the PathSpec.

    Args:
        path (str): The file or directory path to check.
        ignore_spec (pathspec.PathSpec): The compiled PathSpec object.

    Returns:
        bool: True if the path should be ignored, False otherwise.
    """
    return ignore_spec.match_file(path)


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


def summarize_repo(
    root_dir: str, cache_conn: sqlite3.Connection
) -> list[dict[str, Any]]:
    """
    Summarizes the repository by walking through directories and files.
    Includes directories in the summary with appropriate levels.
    Utilizes cache to skip processing unchanged files.
    """
    summary = []
    ignore_patterns = parse_gitignore(root_dir)

    # Add a manual ignore list for generated files
    manual_ignore_patterns = [".repo_map_structure.json", ".repo-map-cache.db"]
    additional_patterns = ["*.pkl"] + manual_ignore_patterns
    combined_patterns = ignore_patterns + additional_patterns
    ignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", combined_patterns)

    cursor = cache_conn.cursor()

    for root, dirs, files in os.walk(root_dir):
        relative_root = os.path.relpath(root, root_dir)
        if relative_root == ".":
            relative_root = ""

        # Modify dirs in-place to skip ignored directories
        dirs[:] = [
            d
            for d in dirs
            if not should_ignore(os.path.join(relative_root, d), ignore_spec)
        ]

        # Add the current directory to the summary
        if relative_root != "":
            dir_info = {
                "name": os.path.basename(root),
                "path": root,
                "level": relative_root.count(os.sep),
                "type": "directory",
                "language": None,
            }
            summary.append(dir_info)

        for file in sorted(files):
            full_path = os.path.join(root, file)
            relative_file_path = os.path.relpath(full_path, root_dir)
            if should_ignore(relative_file_path, ignore_spec):
                continue

            _, ext = os.path.splitext(file)
            language = SUPPORTED_LANGUAGES.get(ext.lower())
            file_info = {
                "name": file,
                "path": full_path,
                "level": relative_file_path.count(os.sep),
                "type": "file",
                "language": language,
            }

            if language:
                file_hash = compute_file_hash(full_path)
                cursor.execute(
                    "SELECT hash, description, developer_consideration, imports, functions FROM cache WHERE path = ?",
                    (full_path,),
                )
                row = cursor.fetchone()

                if row and row[0] == file_hash:
                    # Use cached descriptions
                    file_info.update(
                        {
                            "description": row[1],
                            "developer_consideration": row[2],
                            "imports": json.loads(row[3]) if row[3] else [],
                            "functions": json.loads(row[4]) if row[4] else [],
                            "hash": file_hash,
                        }
                    )
                else:
                    # Need to process this file
                    (
                        classes,
                        functions_extracted,
                        constants,
                    ) = get_structure(full_path, language)
                    module_doc = get_module_docstring(full_path, language)
                    imports = get_imports(full_path, language)
                    file_info.update(
                        {
                            "classes": classes,
                            "functions": functions_extracted,
                            "constants": constants,
                            "imports": imports,
                            "description": module_doc,
                            "hash": file_hash,
                        }
                    )

            summary.append(file_info)

    return summary
