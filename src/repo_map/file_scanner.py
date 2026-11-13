"""Scans the repository to identify and summarize files."""

import codecs
import hashlib
import json
import logging
import os
import re
import sqlite3
import stat
from dataclasses import dataclass, replace
from typing import Any, Self

import pathspec
from pathspec.patterns.gitignore.spec import GitIgnoreSpecPattern

from repo_map.cache_manager import relative_cache_key
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

# A PEM BEGIN line's label. The run is possessive and the closing dashes are a
# lookahead, so each label is read once and those dashes can open the next line.
_PEM_BEGIN_LABEL = re.compile(r"-----BEGIN ([A-Z0-9 ]*+)(?=-----)")
_KEY_LABEL_SUFFIX = re.compile(r"[A-Z ]*")

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
    "*.env",
    "*.envrc",
    "*.tfvars",
    "*.tfvars.json",
    "*.tfstate",
    "*.tfstate.*",
    "*.ini",
    "*.conf",
    "*.cfg",
    "*secrets.toml",
    "credentials.toml",
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
    """A complete eligible source file, the hash of its exact bytes, and their count."""

    source: str
    sha256: str
    size_bytes: int


def _read_rules(path: str) -> list[GitIgnoreSpecPattern]:
    """Compile one rules file: a ``.gitignore`` or a ``.git/info/exclude``.

    Every rules file comes from the analyzed repository, so none can fail the
    scan. A missing file has no rules; a non-regular or unreadable one is
    skipped with a warning. Bytes that are not UTF-8 decode with
    ``surrogateescape``, as ``os.scandir`` decodes such names, so a pattern
    still matches the name it was written for. A leading byte-order mark is
    dropped, as Git drops it, and a line that does not compile is dropped with
    a warning while the others apply. Symlinks are followed: a rule can only
    exclude more.
    """
    try:
        if not stat.S_ISREG(os.stat(path).st_mode):
            logger.warning("Skipping ignore rules in %s: not a regular file", path)
            return []
        with open(path, "rb") as handle:
            raw = handle.read()
    except (FileNotFoundError, NotADirectoryError):
        return []
    except OSError as exc:
        logger.warning("Cannot read ignore rules in %s: %s", path, exc)
        return []

    text = raw.removeprefix(codecs.BOM_UTF8).decode("utf-8", errors="surrogateescape")
    rules: list[GitIgnoreSpecPattern] = []
    for number, line in enumerate(text.splitlines(), start=1):
        try:
            rule = GitIgnoreSpecPattern(line)
        except (ValueError, re.error) as exc:
            logger.warning(
                "Ignoring pattern %r on line %d of %s: %s", line, number, path, exc
            )
            continue
        if rule.include is not None:
            rules.append(rule)
    return rules


def _directory_rules(
    directory: str, gitignore: list[GitIgnoreSpecPattern] | None = None
) -> list[GitIgnoreSpecPattern]:
    """A directory's ``.git/info/exclude`` rules, then its ``.gitignore`` rules.

    The ``.gitignore`` comes last so it wins a conflict, as Git ranks every
    ``.gitignore`` above ``info/exclude``. ``info/exclude`` exists only where
    ``.git`` is a directory; a ``.git`` file (a linked worktree or submodule)
    is never followed. ``gitignore`` passes rules the caller already read.
    """
    exclude = _read_rules(os.path.join(directory, ".git", "info", "exclude"))
    if gitignore is None:
        gitignore = _read_rules(os.path.join(directory, ".gitignore"))
    return [*exclude, *gitignore]


def _work_tree_top(root_dir: str) -> str | None:
    """The nearest directory at or above ``root_dir`` that holds a ``.git`` entry."""
    directory = root_dir
    while not os.path.lexists(os.path.join(directory, ".git")):
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent
    return directory


@dataclass(frozen=True, slots=True)
class _IgnoreLevel:
    """One directory's Git rules, matched against paths relative to it."""

    base: str
    """The directory relative to the work-tree top, ending in ``/`` (``""`` for the top)."""
    spec: pathspec.GitIgnoreSpec


def _level(base: str, rules: list[GitIgnoreSpecPattern]) -> tuple[_IgnoreLevel, ...]:
    """``rules`` as the level at ``base``, or no level when there are none."""
    return (_IgnoreLevel(base, pathspec.GitIgnoreSpec(rules)),) if rules else ()


def _levels_exclude(levels: tuple[_IgnoreLevel, ...], path: str) -> bool:
    """Whether the deepest level with a pattern matching ``path`` excludes it.

    ``path`` is relative to the work-tree top and ends in ``/`` for a directory.
    """
    for level in reversed(levels):
        include = level.spec.check_file(path[len(level.base) :]).include
        if include is not None:
            return include
    return False


@dataclass(frozen=True, slots=True)
class _IgnoreRules:
    """Decides which paths under the scanned root are excluded.

    A path is excluded when either of two rule sets excludes it, so a rules
    file below the root can narrow eligibility but never widen it:

    - ``root``: ``DEFAULT_IGNORE_PATTERNS``, then the root ``.gitignore``, last
      match winning. Only here can a negation re-include a default exclusion.
    - ``levels``: Git's own precedence over ``.git/info/exclude`` and every
      ``.gitignore`` from the work-tree top (the nearest directory at or above
      the root holding a ``.git`` entry) down to the path's directory. The
      deepest level with a matching pattern decides.

    The index is never read, so a pattern applies whether or not Git tracks
    the path. Neither is the global excludes file, so the outcome depends on
    the repository alone, and nothing is executed.
    """

    root: pathspec.GitIgnoreSpec
    prefix: str
    """The scanned root relative to the work-tree top, ending in ``/`` (``""`` for the top)."""
    levels: tuple[_IgnoreLevel, ...]

    @classmethod
    def for_root(cls, root_dir: str) -> Self:
        """The rules at ``root_dir``, seeded with any enclosing work tree's."""
        gitignore = _read_rules(os.path.join(root_dir, ".gitignore"))
        defaults = [
            GitIgnoreSpecPattern(pattern) for pattern in DEFAULT_IGNORE_PATTERNS
        ]
        top = _work_tree_top(root_dir) or root_dir
        names = [] if top == root_dir else os.path.relpath(root_dir, top).split(os.sep)

        levels: tuple[_IgnoreLevel, ...] = ()
        for depth in range(len(names)):
            base = "".join(f"{name}/" for name in names[:depth])
            levels += _level(base, _directory_rules(os.path.join(top, *names[:depth])))
        prefix = "".join(f"{name}/" for name in names)
        if prefix and _levels_exclude(levels, prefix):
            logger.warning(
                "%s is ignored by the Git rules of the work tree at %s, "
                "so nothing under it is scanned or sent.",
                root_dir,
                top,
            )
        levels += _level(prefix, _directory_rules(root_dir, gitignore))
        return cls(pathspec.GitIgnoreSpec([*defaults, *gitignore]), prefix, levels)

    def excludes(self, path: str) -> bool:
        """Whether ``path``, root-relative and ending in ``/`` for a directory, is excluded."""
        return self.root.match_file(path) or _levels_exclude(
            self.levels, self.prefix + path
        )

    def descend(self, directory: str, path: str) -> Self:
        """The rules inside ``directory``, whose root-relative ``path`` ends in ``/``."""
        level = _level(self.prefix + path, _directory_rules(directory))
        return replace(self, levels=self.levels + level) if level else self


def _has_private_key_header(source: str) -> bool:
    """Whether ``source`` holds a PEM BEGIN line that labels a private key.

    Matches what ``-----BEGIN [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----`` matches, in
    linear time: that pattern backtracks quadratically through a long label
    of repeated ``PRIVATE KEY`` runs. Only a label's last run needs checking,
    since every earlier run's suffix contains it.
    """
    for match in _PEM_BEGIN_LABEL.finditer(source):
        _, key, suffix = match[1].rpartition("PRIVATE KEY")
        if key and _KEY_LABEL_SUFFIX.fullmatch(suffix):
            return True
    return False


def load_source_snapshot(
    file_path: str,
    language: str | None,
    expected_hash: str | None = None,
) -> SourceSnapshot | None:
    """Load a complete eligible UTF-8 source file without following symlinks.

    Returns ``None`` for anything that must not be uploaded: an unsupported or
    non-text language, a non-regular file, more than ``MAX_SOURCE_BYTES``, a
    NUL byte, invalid UTF-8, a PEM private-key header, or bytes that no longer
    hash to ``expected_hash``. The key check reads content, so no ignore-file
    negation re-includes such a file, and a document that only quotes a key
    header is skipped too.
    """
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

    if _has_private_key_header(source):
        logger.warning(
            "Skipping %s: its source contains a private key header", file_path
        )
        return None

    source_hash = hashlib.sha256(raw_source).hexdigest()
    if expected_hash is not None and source_hash != expected_hash:
        logger.warning("Source file changed after scanning; skipping %s", file_path)
        return None

    return SourceSnapshot(source=source, sha256=source_hash, size_bytes=len(raw_source))


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
    full_path: str, rel_path: str, level: int, cache_conn: sqlite3.Connection
) -> dict[str, Any]:
    """Return metadata for a single file, preferring cached data.

    The two path fields are not interchangeable: ``path`` is absolute and is
    what every disk read resolves (the process working directory is not
    necessarily the repository root), while ``rel_path`` is the portable cache
    key and is the only value the cache is ever queried or written with.

    This lookup matches on the hash alone and ignores the row's ``model`` and
    ``contract_version``. It decides only whether structure has to be
    re-extracted from disk; whether a stored analysis is still valid is
    ``cli_handler.plan_upload()``'s call, and any field hydrated here
    from a row that fails it is overwritten by the analysis that follows.
    """
    language = _detect_language(full_path)

    file_info: dict[str, Any] = {
        "name": os.path.basename(full_path),
        "path": full_path,
        "rel_path": rel_path,
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
    file_info["size_bytes"] = snapshot.size_bytes
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
        (rel_path,),
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
    """Summarize the repository by recursively scanning directories and files.

    Directory candidates are matched with a trailing separator so directory-only
    ignore patterns (``build/``) apply to them, pruning the walk instead of
    emitting an empty node and discarding the subtree one file at a time. A
    pruned directory is never read, its rules files included; each directory
    that is entered adds its own rules (``_IgnoreRules``) for the entries below.
    """
    summary: list[dict[str, Any]] = []
    abs_root_dir = os.path.abspath(root_dir)

    def _scan(
        current_path: str, relative_dir: str, level: int, rules: _IgnoreRules
    ) -> None:
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
            is_directory = entry.is_dir(follow_symlinks=False)
            relative_path = f"{relative_dir}{entry.name}"
            candidate = f"{relative_path}/" if is_directory else relative_path
            if rules.excludes(candidate):
                continue

            if is_directory:
                summary.append(
                    {
                        "name": entry.name,
                        "path": full_path,
                        "level": level,
                        "type": "directory",
                    }
                )
                _scan(
                    full_path, candidate, level + 1, rules.descend(full_path, candidate)
                )
            elif entry.is_file(follow_symlinks=False):
                summary.append(
                    _process_file(
                        full_path,
                        relative_cache_key(abs_root_dir, full_path),
                        level,
                        cache_conn,
                    )
                )

    _scan(abs_root_dir, "", 0, _IgnoreRules.for_root(abs_root_dir))
    return summary
