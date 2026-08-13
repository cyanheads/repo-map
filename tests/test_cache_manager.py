"""Tests for the repository-local SQLite cache."""

from repo_map.cache_manager import load_cache


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
