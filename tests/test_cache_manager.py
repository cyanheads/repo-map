"""Tests for the repository-local SQLite cache."""

import pytest

from repo_map.cache_manager import CACHE_FILE_NAME, CacheError, load_cache


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


def test_corrupt_cache_file_raises_actionable_error(tmp_path) -> None:
    """Schema setup on a non-database file names the path and the remedy -- issue #21."""
    (tmp_path / CACHE_FILE_NAME).write_text("not a database", encoding="utf-8")

    with pytest.raises(CacheError) as failure:
        load_cache(str(tmp_path))

    message = str(failure.value)
    assert str(tmp_path / CACHE_FILE_NAME) in message
    assert "clean rebuild" in message
