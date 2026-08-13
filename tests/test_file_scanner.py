"""Tests for repository traversal and file metadata."""

from repo_map.cache_manager import load_cache
from repo_map.file_scanner import compute_file_hash, summarize_repo


def test_compute_file_hash_is_stable_and_content_sensitive(tmp_path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first_hash = compute_file_hash(str(source))

    assert first_hash == compute_file_hash(str(source))

    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert compute_file_hash(str(source)) != first_hash


def test_summarize_repo_orders_directories_first_and_honors_gitignore(
    tmp_path,
) -> None:
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / "zeta.py").write_text("def run():\n    pass\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("SECRET = 1\n", encoding="utf-8")
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "nested.py").write_text("VALUE = 1\n", encoding="utf-8")
    connection = load_cache(str(tmp_path))

    summary = summarize_repo(str(tmp_path), connection)
    connection.close()

    names = [item["name"] for item in summary]
    assert names[:2] == ["alpha", "nested.py"]
    assert "ignored.py" not in names
    assert "zeta.py" in names


def test_summarize_repo_reuses_matching_cache_entry(tmp_path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("def run():\n    pass\n", encoding="utf-8")
    connection = load_cache(str(tmp_path))
    first = summarize_repo(str(tmp_path), connection)
    file_data = next(item for item in first if item["name"] == "sample.py")
    connection.execute(
        """
        INSERT INTO cache (
            path, hash, description, imports, functions, maintenance_flag,
            critical_dependencies, architectural_role,
            refactoring_suggestions, security_assessment
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(source),
            file_data["hash"],
            "Cached description",
            "[]",
            "[]",
            "Stable",
            "{}",
            "Utility",
            "None",
            "None",
        ),
    )
    connection.commit()

    cached = summarize_repo(str(tmp_path), connection)
    connection.close()

    cached_file = next(item for item in cached if item["name"] == "sample.py")
    assert cached_file["description"] == "Cached description"
    assert cached_file["architectural_role"] == "Utility"
