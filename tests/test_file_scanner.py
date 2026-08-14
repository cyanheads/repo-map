"""Tests for repository traversal and file metadata."""

import builtins
import os

from repo_map import file_scanner
from repo_map.cache_manager import load_cache
from repo_map.file_scanner import (
    MAX_SOURCE_BYTES,
    load_source_snapshot,
    summarize_repo,
)


def _snapshot_hash(path) -> str:
    snapshot = load_source_snapshot(str(path), "Python")
    assert snapshot is not None
    return snapshot.sha256


def test_source_snapshot_hash_is_stable_and_content_sensitive(tmp_path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first_hash = _snapshot_hash(source)

    assert first_hash == _snapshot_hash(source)

    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert _snapshot_hash(source) != first_hash


def test_source_snapshot_rejects_hash_that_no_longer_matches(tmp_path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    stale_hash = _snapshot_hash(source)
    source.write_text("VALUE = 2\n", encoding="utf-8")

    assert load_source_snapshot(str(source), "Python", stale_hash) is None
    assert (
        load_source_snapshot(str(source), "Python", _snapshot_hash(source)) is not None
    )


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
        ".gitignore": "Git",
        "Dockerfile": "Docker",
        "container.dockerfile": "Docker",
        "main.py": "Python",
        "main.tfstate.backup": "Terraform",
    }
    for name in [*expected, ".env", ".envrc"]:
        (tmp_path / name).write_text("content\n", encoding="utf-8")
    connection = load_cache(str(tmp_path))

    summary = summarize_repo(str(tmp_path), connection)
    connection.close()
    detected = {item["name"]: item.get("language") for item in summary}

    assert {name: detected[name] for name in expected} == expected
    assert ".env" not in detected
    assert ".envrc" not in detected


def _seed_cache_row(connection, key: str, file_hash: str, description: str) -> None:
    """Insert a fully populated cache row under an explicit key."""
    connection.execute(
        """
        INSERT INTO cache (
            path, hash, description, imports, functions, maintenance_flag,
            critical_dependencies, architectural_role,
            refactoring_suggestions, security_assessment
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            key,
            file_hash,
            description,
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


def test_summarize_repo_reuses_matching_cache_entry(tmp_path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("def run():\n    pass\n", encoding="utf-8")
    connection = load_cache(str(tmp_path))
    first = summarize_repo(str(tmp_path), connection)
    file_data = next(item for item in first if item["name"] == "sample.py")
    _seed_cache_row(connection, "sample.py", file_data["hash"], "Cached description")

    cached = summarize_repo(str(tmp_path), connection)
    connection.close()

    cached_file = next(item for item in cached if item["name"] == "sample.py")
    assert cached_file["description"] == "Cached description"
    assert cached_file["architectural_role"] == "Utility"


def test_summarize_repo_keys_files_by_posix_relative_path(tmp_path) -> None:
    """Cache keys are relative and forward-slash separated; `path` stays absolute."""
    nested = tmp_path / "src" / "pkg"
    nested.mkdir(parents=True)
    (nested / "deep.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "root.py").write_text("VALUE = 2\n", encoding="utf-8")
    connection = load_cache(str(tmp_path))

    summary = summarize_repo(str(tmp_path), connection)
    connection.close()
    files = {item["name"]: item for item in summary if item["type"] == "file"}

    assert files["deep.py"]["rel_path"] == "src/pkg/deep.py"
    assert files["root.py"]["rel_path"] == "root.py"
    assert files["deep.py"]["path"] == str(nested / "deep.py")


def test_summarize_repo_reuses_nested_cache_entries_after_a_repository_move(
    tmp_path,
) -> None:
    """A relative key survives the repository moving to another prefix -- issue #24."""
    original = tmp_path / "project"
    (original / "src" / "pkg").mkdir(parents=True)
    (original / "src" / "pkg" / "deep.py").write_text("VALUE = 1\n", encoding="utf-8")
    connection = load_cache(str(original))
    scanned = summarize_repo(str(original), connection)
    deep = next(item for item in scanned if item["name"] == "deep.py")
    _seed_cache_row(connection, "src/pkg/deep.py", deep["hash"], "Cached description")
    connection.close()

    moved = tmp_path / "project-renamed"
    original.rename(moved)
    connection = load_cache(str(moved))
    summary = summarize_repo(str(moved), connection)
    connection.close()

    cached_file = next(item for item in summary if item["name"] == "deep.py")
    assert cached_file["description"] == "Cached description"
    assert cached_file["path"] == str(moved / "src" / "pkg" / "deep.py")


def test_summarize_repo_treats_a_legacy_absolute_path_row_as_a_miss(tmp_path) -> None:
    """A pre-upgrade absolute key matches nothing and is re-extracted -- issue #24."""
    source = tmp_path / "sample.py"
    source.write_text('"""Module."""\n\nVALUE = 1\n', encoding="utf-8")
    connection = load_cache(str(tmp_path))
    scanned = summarize_repo(str(tmp_path), connection)
    file_data = next(item for item in scanned if item["name"] == "sample.py")
    _seed_cache_row(connection, str(source), file_data["hash"], "Legacy description")

    summary = summarize_repo(str(tmp_path), connection)
    connection.close()

    cached_file = next(item for item in summary if item["name"] == "sample.py")
    assert cached_file["description"] == "Module."
    assert "constants" in cached_file


def test_source_snapshot_accepts_empty_and_exact_limit_but_not_one_over(
    tmp_path,
) -> None:
    empty = tmp_path / "empty.txt"
    exact = tmp_path / "exact.txt"
    oversized = tmp_path / "oversized.txt"
    empty.write_bytes(b"")
    exact.write_bytes(b"x" * MAX_SOURCE_BYTES)
    oversized.write_bytes(b"x" * (MAX_SOURCE_BYTES + 1))

    empty_snapshot = load_source_snapshot(str(empty), "Text")
    exact_snapshot = load_source_snapshot(str(exact), "Text")

    assert empty_snapshot is not None
    assert empty_snapshot.source == ""
    assert exact_snapshot is not None
    assert len(exact_snapshot.source.encode()) == MAX_SOURCE_BYTES
    assert load_source_snapshot(str(oversized), "Text") is None


def test_source_snapshot_rejects_unsafe_or_unsupported_inputs(tmp_path) -> None:
    invalid_utf8 = tmp_path / "invalid.txt"
    nul_text = tmp_path / "nul.txt"
    binary = tmp_path / "image.png"
    unsupported = tmp_path / "archive.bin"
    invalid_utf8.write_bytes(b"\xff")
    nul_text.write_bytes(b"before\0after")
    binary.write_bytes(b"PNG")
    unsupported.write_bytes(b"plain text")

    assert load_source_snapshot(str(invalid_utf8), "Text") is None
    assert load_source_snapshot(str(nul_text), "Text") is None
    assert load_source_snapshot(str(binary), "Image") is None
    assert load_source_snapshot(str(unsupported), None) is None


def test_source_snapshot_rejects_read_failure(tmp_path, monkeypatch) -> None:
    source = tmp_path / "blocked.txt"
    source.write_text("content", encoding="utf-8")
    real_open = builtins.open

    def fail_target(path, *args, **kwargs):
        if str(path) == str(source):
            raise PermissionError("blocked")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fail_target)

    assert load_source_snapshot(str(source), "Text") is None


def test_summarize_repo_reports_nested_directory_levels(tmp_path) -> None:
    deep = tmp_path / "alpha" / "beta" / "gamma"
    deep.mkdir(parents=True)
    (deep / "deep.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "root.py").write_text("VALUE = 2\n", encoding="utf-8")
    connection = load_cache(str(tmp_path))

    summary = summarize_repo(str(tmp_path), connection)
    connection.close()

    assert [(item["name"], item["type"], item["level"]) for item in summary] == [
        ("alpha", "directory", 0),
        ("beta", "directory", 1),
        ("gamma", "directory", 2),
        ("deep.py", "file", 3),
        ("root.py", "file", 0),
    ]


def test_summarize_repo_prunes_default_ignored_directories_and_subtrees(
    tmp_path,
) -> None:
    (tmp_path / "build" / "reports").mkdir(parents=True)
    (tmp_path / "build" / "artifact.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "build" / "reports" / "summary.md").write_text(
        "# Report\n", encoding="utf-8"
    )
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text(
        "var a = 1;\n", encoding="utf-8"
    )
    (tmp_path / "keep.py").write_text("VALUE = 2\n", encoding="utf-8")
    connection = load_cache(str(tmp_path))

    summary = summarize_repo(str(tmp_path), connection)
    connection.close()

    assert [(item["type"], item["name"]) for item in summary] == [("file", "keep.py")]


def test_summarize_repo_never_scans_pruned_directories(tmp_path, monkeypatch) -> None:
    (tmp_path / "node_modules" / "pkg" / "deep").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text(
        "var a = 1;\n", encoding="utf-8"
    )
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "nested.py").write_text("VALUE = 1\n", encoding="utf-8")
    connection = load_cache(str(tmp_path))
    scanned: list[str] = []
    real_scandir = file_scanner.os.scandir

    def recording_scandir(path):
        scanned.append(os.fspath(path))
        return real_scandir(path)

    monkeypatch.setattr(file_scanner.os, "scandir", recording_scandir)
    summarize_repo(str(tmp_path), connection)
    connection.close()

    ignored_root = str(tmp_path / "node_modules")
    assert [path for path in scanned if path.startswith(ignored_root)] == []
    assert str(tmp_path / "alpha") in scanned


def test_summarize_repo_prunes_gitignore_directory_pattern_not_partial_name(
    tmp_path,
) -> None:
    (tmp_path / ".gitignore").write_text("generated/\n", encoding="utf-8")
    (tmp_path / "generated" / "inner").mkdir(parents=True)
    (tmp_path / "generated" / "inner" / "output.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    (tmp_path / "generated_docs").mkdir()
    (tmp_path / "generated_docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (tmp_path / "builder").mkdir()
    (tmp_path / "builder" / "tool.py").write_text("VALUE = 2\n", encoding="utf-8")
    connection = load_cache(str(tmp_path))

    summary = summarize_repo(str(tmp_path), connection)
    connection.close()

    assert [(item["name"], item["level"]) for item in summary] == [
        ("builder", 0),
        ("tool.py", 1),
        ("generated_docs", 0),
        ("guide.md", 1),
        (".gitignore", 0),
    ]


def test_summarize_repo_excludes_credential_bearing_formats_by_default(
    tmp_path,
) -> None:
    excluded = [
        ".env",
        ".envrc",
        "terraform.tfvars",
        "prod.tfstate",
        "credentials.ini",
        "app.conf",
        "boto.cfg",
    ]
    retained = ["main.tf", "main.tfstate.backup", "compose.yml", "notes.txt"]
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "secrets.ini").write_text(
        "token = PLACEHOLDER\n", encoding="utf-8"
    )
    (tmp_path / "config" / "settings.json").write_text("{}\n", encoding="utf-8")
    for name in [*excluded, *retained]:
        (tmp_path / name).write_text("token = PLACEHOLDER\n", encoding="utf-8")
    connection = load_cache(str(tmp_path))

    summary = summarize_repo(str(tmp_path), connection)
    connection.close()
    files = {item["path"]: item for item in summary if item["type"] == "file"}

    assert {os.path.basename(path) for path in files}.isdisjoint(excluded)
    assert str(tmp_path / "config" / "secrets.ini") not in files
    assert files[str(tmp_path / "config" / "settings.json")]["source_eligible"] is True
    assert all(files[str(tmp_path / name)]["source_eligible"] for name in retained)


def test_summarize_repo_honors_gitignore_negation_for_ignored_format(tmp_path) -> None:
    (tmp_path / ".gitignore").write_text("!app.conf\n", encoding="utf-8")
    (tmp_path / "app.conf").write_text("token = PLACEHOLDER\n", encoding="utf-8")
    (tmp_path / "other.conf").write_text("token = PLACEHOLDER\n", encoding="utf-8")
    connection = load_cache(str(tmp_path))

    summary = summarize_repo(str(tmp_path), connection)
    connection.close()
    files = {item["name"]: item for item in summary if item["type"] == "file"}

    assert files["app.conf"]["source_eligible"] is True
    assert "other.conf" not in files


def test_summarize_repo_skips_ineligible_files_without_stopping_siblings(
    tmp_path,
) -> None:
    (tmp_path / "valid.md").write_text("# Valid\n", encoding="utf-8")
    (tmp_path / "empty.json").write_text("", encoding="utf-8")
    (tmp_path / "invalid.py").write_bytes(b"\xff")
    (tmp_path / "nul.txt").write_bytes(b"text\0data")
    (tmp_path / "image.png").write_bytes(b"PNG")
    connection = load_cache(str(tmp_path))

    summary = summarize_repo(str(tmp_path), connection)
    connection.close()
    files = {item["name"]: item for item in summary if item["type"] == "file"}

    assert files["valid.md"]["source_eligible"] is True
    assert files["empty.json"]["source_eligible"] is True
    assert files["invalid.py"]["source_eligible"] is False
    assert files["nul.txt"]["source_eligible"] is False
    assert files["image.png"]["source_eligible"] is False
    assert all("source" not in item for item in files.values())
