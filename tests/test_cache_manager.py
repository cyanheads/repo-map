"""Tests for the repository-local SQLite cache."""

import os
import sqlite3

import pytest

from repo_map.cache_manager import (
    CACHE_FILE_NAME,
    CacheError,
    load_cache,
    load_cache_snapshot,
    prune_cache,
    relative_cache_key,
)

#: The schema columns a fully migrated cache carries.
CURRENT_COLUMNS = {
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
    "model",
    "contract_version",
}


def _columns(connection: sqlite3.Connection) -> set[str]:
    return {row[1] for row in connection.execute("PRAGMA table_info(cache)")}


def _seed_legacy_cache(repository) -> bytes:
    """Write a cache in the oldest schema with one row, returning the file's bytes."""
    connection = sqlite3.connect(repository / CACHE_FILE_NAME)
    connection.execute(
        "CREATE TABLE cache (path TEXT PRIMARY KEY, hash TEXT, description TEXT)"
    )
    connection.execute(
        "INSERT INTO cache (path, hash, description) VALUES (?, ?, ?)",
        ("mod.py", "unchanged", "written by an older repo-map"),
    )
    connection.commit()
    connection.close()
    return (repository / CACHE_FILE_NAME).read_bytes()


def test_load_cache_creates_current_schema(tmp_path) -> None:
    connection = load_cache(str(tmp_path))

    columns = _columns(connection)

    connection.close()
    assert (tmp_path / ".repo-map-cache.db").is_file()
    assert columns == CURRENT_COLUMNS


def test_load_cache_adds_identity_columns_without_touching_existing_rows(
    tmp_path,
) -> None:
    """Rows written before the identity columns survive and read NULL -- issue #13."""
    connection = load_cache(str(tmp_path))
    connection.execute("DROP TABLE cache")
    connection.execute(
        "CREATE TABLE cache (path TEXT PRIMARY KEY, hash TEXT, description TEXT)"
    )
    connection.execute(
        "INSERT INTO cache (path, hash, description) VALUES (?, ?, ?)",
        ("mod.py", "unchanged", "written by an older repo-map"),
    )
    connection.commit()
    connection.close()

    migrated = load_cache(str(tmp_path))
    row = migrated.execute(
        "SELECT hash, description, model, contract_version FROM cache WHERE path = ?",
        ("mod.py",),
    ).fetchone()
    migrated.close()

    assert row == ("unchanged", "written by an older repo-map", None, None)


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


def test_load_cache_reports_each_column_it_adds(tmp_path, caplog) -> None:
    _seed_legacy_cache(tmp_path)

    with caplog.at_level("INFO"):
        load_cache(str(tmp_path)).close()

    assert "Adding missing cache column 'model'" in caplog.text
    assert "Adding missing cache column 'contract_version'" in caplog.text


def test_snapshot_of_an_absent_cache_is_empty_and_creates_no_file(tmp_path) -> None:
    snapshot = load_cache_snapshot(str(tmp_path))
    columns = _columns(snapshot)
    rows = snapshot.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    snapshot.close()

    assert columns == CURRENT_COLUMNS
    assert rows == 0
    assert list(tmp_path.iterdir()) == []


def test_snapshot_reads_the_cache_and_writes_nothing_back(tmp_path) -> None:
    """The copy answers like the file; changes to the copy stay in memory -- issue #34."""
    connection = load_cache(str(tmp_path))
    connection.execute(
        "INSERT INTO cache (path, hash, model) VALUES (?, ?, ?)",
        ("mod.py", "h", "vendor/model-a"),
    )
    connection.commit()
    connection.close()
    before = (tmp_path / CACHE_FILE_NAME).read_bytes()

    snapshot = load_cache_snapshot(str(tmp_path))
    row = snapshot.execute(
        "SELECT hash, model FROM cache WHERE path = ?", ("mod.py",)
    ).fetchone()
    snapshot.execute("DELETE FROM cache")
    snapshot.commit()
    snapshot.close()

    assert row == ("h", "vendor/model-a")
    assert (tmp_path / CACHE_FILE_NAME).read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == [CACHE_FILE_NAME]


def test_snapshot_migrates_a_legacy_schema_in_memory_only(tmp_path, caplog) -> None:
    """An older database reads at the current schema and is left byte-identical."""
    before = _seed_legacy_cache(tmp_path)

    with caplog.at_level("INFO"):
        snapshot = load_cache_snapshot(str(tmp_path))
    columns = _columns(snapshot)
    row = snapshot.execute(
        "SELECT hash, model, contract_version FROM cache WHERE path = ?", ("mod.py",)
    ).fetchone()
    snapshot.close()

    assert columns == CURRENT_COLUMNS
    assert row == ("unchanged", None, None)
    assert (tmp_path / CACHE_FILE_NAME).read_bytes() == before
    assert "Adding missing cache column" not in caplog.text


def test_snapshot_works_in_a_read_only_directory(tmp_path) -> None:
    repository = tmp_path / "readonly"
    repository.mkdir()
    before = _seed_legacy_cache(repository)
    repository.chmod(0o555)

    try:
        snapshot = load_cache_snapshot(str(repository))
        rows = snapshot.execute("SELECT path FROM cache").fetchall()
        snapshot.close()
    finally:
        repository.chmod(0o755)

    assert rows == [("mod.py",)]
    assert (repository / CACHE_FILE_NAME).read_bytes() == before
    assert sorted(path.name for path in repository.iterdir()) == [CACHE_FILE_NAME]


def test_snapshot_of_a_corrupt_cache_raises_the_same_actionable_error(
    tmp_path,
) -> None:
    (tmp_path / CACHE_FILE_NAME).write_text("not a database", encoding="utf-8")

    with pytest.raises(CacheError) as failure:
        load_cache_snapshot(str(tmp_path))

    message = str(failure.value)
    assert str(tmp_path / CACHE_FILE_NAME) in message
    assert "clean rebuild" in message
