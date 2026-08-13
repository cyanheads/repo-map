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


def test_summarize_repo_skips_file_and_directory_symlinks(tmp_path) -> None:
    repository = tmp_path / "repository"
    outside = tmp_path / "outside"
    repository.mkdir()
    outside.mkdir()
    (outside / "external.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "alpha").mkdir()
    (repository / "alpha" / "nested.py").write_text("VALUE = 2\n", encoding="utf-8")
    (repository / "zeta.py").write_text("VALUE = 3\n", encoding="utf-8")
    (repository / "linked-directory").symlink_to(outside, target_is_directory=True)
    (repository / "linked-file.py").symlink_to(outside / "external.py")
    (repository / "alpha" / "cycle").symlink_to(repository, target_is_directory=True)
    connection = load_cache(str(repository))

    summary = summarize_repo(str(repository), connection)
    connection.close()

    assert [item["name"] for item in summary] == ["alpha", "nested.py", "zeta.py"]


def test_summarize_repo_detects_exact_filenames_and_longest_suffixes(tmp_path) -> None:
    expected = {
        ".envrc": "Config",
        ".gitignore": "Git",
        "Dockerfile": "Docker",
        "container.dockerfile": "Docker",
        "main.py": "Python",
        "main.tfstate.backup": "Terraform",
    }
    for name in [*expected, ".env"]:
        (tmp_path / name).write_text("content\n", encoding="utf-8")
    connection = load_cache(str(tmp_path))

    summary = summarize_repo(str(tmp_path), connection)
    connection.close()
    detected = {item["name"]: item.get("language") for item in summary}

    assert {name: detected[name] for name in expected} == expected
    assert ".env" not in detected


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
