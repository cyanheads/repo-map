"""Tests for the repository-local SQLite cache."""

import os

import pytest

from repo_map.cache_manager import (
    CACHE_FILE_NAME,
    CacheError,
    load_cache,
    prune_cache,
    relative_cache_key,
)


def test_load_cache_creates_current_schema(tmp_path) -> None:
    connection = load_cache(str(tmp_path))

    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(cache)").fetchall()
    }

    connection.close()
    assert (tmp_path / ".repo-map-cache.db").is_file()
    assert columns == {
        "path",
        "hash",
        "description",
        "developer_consideration",
        "imports",
        "functions",
        "maintenance_flag",
        "critical_dependencies",
        "architectural_role",
        "refactoring_suggestions",
        "security_assessment",
    }


def test_load_cache_migrates_legacy_schema(tmp_path) -> None:
    connection = load_cache(str(tmp_path))
    connection.execute("DROP TABLE cache")
    connection.execute(
        "CREATE TABLE cache (path TEXT PRIMARY KEY, hash TEXT, description TEXT)"
    )
    connection.commit()
    connection.close()

    migrated = load_cache(str(tmp_path))
    columns = {
        row[1] for row in migrated.execute("PRAGMA table_info(cache)").fetchall()
    }
    migrated.close()

    assert "security_assessment" in columns
    assert "architectural_role" in columns


def test_unwritable_cache_directory_raises_actionable_error(tmp_path) -> None:
    """`sqlite3.connect` failures name the cache path and the remedy -- issue #21."""
    repository = tmp_path / "readonly"
    repository.mkdir()
    repository.chmod(0o555)

    try:
        with pytest.raises(CacheError) as failure:
            load_cache(str(repository))
    finally:
        repository.chmod(0o755)

    message = str(failure.value)
    assert str(repository / CACHE_FILE_NAME) in message
    assert "clean rebuild" in message


def test_relative_cache_key_is_repository_relative_and_posix_separated() -> None:
    """The key drops the prefix and never carries a platform separator -- issue #24."""
    root = os.path.join(os.sep, "checkout", "project")

    assert relative_cache_key(root, os.path.join(root, "mod.py")) == "mod.py"
    assert (
        relative_cache_key(root, os.path.join(root, "src", "pkg", "deep.py"))
        == "src/pkg/deep.py"
    )


def test_relative_cache_key_is_independent_of_the_repository_prefix() -> None:
    """The same file under two checkouts reduces to one key -- issue #24."""
    first = os.path.join(os.sep, "checkout-a", "project")
    second = os.path.join(os.sep, "elsewhere", "b", "project")

    assert relative_cache_key(
        first, os.path.join(first, "src", "mod.py")
    ) == relative_cache_key(second, os.path.join(second, "src", "mod.py"))


def test_prune_cache_removes_only_rows_absent_from_the_scan(tmp_path) -> None:
    connection = load_cache(str(tmp_path))
    for path in ("kept.py", "src/nested.py", "deleted.py", "/legacy/abs/kept.py"):
        connection.execute(
            "INSERT INTO cache (path, hash) VALUES (?, ?)", (path, "seeded")
        )
    connection.commit()

    removed = prune_cache(connection, {"kept.py", "src/nested.py"})
    remaining = {row[0] for row in connection.execute("SELECT path FROM cache")}
    connection.close()

    assert removed == 2
    assert remaining == {"kept.py", "src/nested.py"}


def test_prune_cache_on_a_fully_matched_cache_changes_nothing(tmp_path) -> None:
    connection = load_cache(str(tmp_path))
    connection.execute("INSERT INTO cache (path, hash) VALUES (?, ?)", ("mod.py", "h"))
    connection.commit()

    removed = prune_cache(connection, {"mod.py", "never-cached.py"})
    remaining = {row[0] for row in connection.execute("SELECT path FROM cache")}
    connection.close()

    assert removed == 0
    assert remaining == {"mod.py"}


def test_prune_cache_empties_a_cache_whose_files_are_all_gone(tmp_path) -> None:
    connection = load_cache(str(tmp_path))
    connection.execute("INSERT INTO cache (path, hash) VALUES (?, ?)", ("mod.py", "h"))
    connection.commit()

    removed = prune_cache(connection, set())
    remaining = connection.execute("SELECT path FROM cache").fetchall()
    connection.close()

    assert removed == 1
    assert remaining == []


def test_corrupt_cache_file_raises_actionable_error(tmp_path) -> None:
    """Schema setup on a non-database file names the path and the remedy -- issue #21."""
    (tmp_path / CACHE_FILE_NAME).write_text("not a database", encoding="utf-8")

    with pytest.raises(CacheError) as failure:
        load_cache(str(tmp_path))

    message = str(failure.value)
    assert str(tmp_path / CACHE_FILE_NAME) in message
    assert "clean rebuild" in message
