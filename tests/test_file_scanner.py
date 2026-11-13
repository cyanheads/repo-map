"""Tests for repository traversal and file metadata."""

import builtins
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from repo_map import file_scanner
from repo_map.cache_manager import load_cache
from repo_map.file_scanner import (
    MAX_SOURCE_BYTES,
    load_source_snapshot,
    summarize_repo,
)
from repo_map.models import SUPPORTED_TEXT_LANGUAGES

# PEM fixtures are assembled from parts so this module's own text carries no key
# header: repo-map analyzes its own repository, and a literal header here would
# make this file ineligible for upload.
_PEM_BEGIN = "-----BEGIN "
_PEM_END = "-----END "
_PEM_DASHES = "-----"
_PRIVATE_KEY_LABELS = [
    "PRIVATE KEY",
    "RSA PRIVATE KEY",
    "EC PRIVATE KEY",
    "OPENSSH PRIVATE KEY",
    "ENCRYPTED PRIVATE KEY",
]


def _pem_lines(label: str) -> list[str]:
    """A PEM block's lines around a placeholder body."""
    return [
        _PEM_BEGIN + label + _PEM_DASHES,
        "PLACEHOLDER",
        _PEM_END + label + _PEM_DASHES,
    ]


def _key_as_json(label: str) -> str:
    """A service-account style key file: the block in a string, line breaks escaped."""
    key = "".join(f"{line}\n" for line in _pem_lines(label))
    return json.dumps({"type": "service_account", "private_key": key}) + "\n"


def _key_as_yaml(label: str) -> str:
    """The block in a YAML literal block scalar."""
    return "private_key: |\n" + "".join(f"  {line}\n" for line in _pem_lines(label))


def _key_as_text(label: str) -> str:
    """The bare block."""
    return "".join(f"{line}\n" for line in _pem_lines(label))


_KEY_FILES = {
    "service-account.json": ("JSON", _key_as_json),
    "deploy.yml": ("YAML", _key_as_yaml),
    "key.txt": ("Text", _key_as_text),
}


def _write_tree(root: Path, files: dict[str, str]) -> None:
    """Write each file, creating its parent directories."""
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _scan_files(root: Path) -> dict[str, dict[str, Any]]:
    """Scan ``root`` and return its file entries keyed by relative path."""
    connection = load_cache(str(root))
    try:
        summary = summarize_repo(str(root), connection)
    finally:
        connection.close()
    return {item["rel_path"]: item for item in summary if item["type"] == "file"}


def _make_git_dir(root: Path, info_exclude: str | None = None) -> None:
    """Create a hand-made ``.git`` directory, optionally with ``info/exclude``."""
    (root / ".git" / "info").mkdir(parents=True)
    if info_exclude is not None:
        (root / ".git" / "info" / "exclude").write_text(info_exclude, encoding="utf-8")


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
    """`.tfstate.backup` wins over `.tfstate` -- issue #5.

    The backup is a default exclusion, so the root `.gitignore` re-includes it
    to keep the compound suffix observable through the scan.
    """
    expected = {
        ".gitignore": "Git",
        "Dockerfile": "Docker",
        "container.dockerfile": "Docker",
        "main.py": "Python",
        "main.tfstate.backup": "Terraform",
    }
    for name in [*expected, ".env", ".envrc"]:
        (tmp_path / name).write_text("content\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("!main.tfstate.backup\n", encoding="utf-8")
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


def test_summarize_repo_hydrates_a_row_that_predates_the_identity_columns(
    tmp_path,
) -> None:
    """The scanner's hash-only lookup reads NULL identity columns fine -- issue #13.

    Whether that hydrated analysis is still valid for this run is decided by
    `cli_handler.plan_upload()`, not here.
    """
    source = tmp_path / "sample.py"
    source.write_text("def run():\n    pass\n", encoding="utf-8")
    connection = load_cache(str(tmp_path))
    scanned = summarize_repo(str(tmp_path), connection)
    file_data = next(item for item in scanned if item["name"] == "sample.py")
    _seed_cache_row(connection, "sample.py", file_data["hash"], "Cached description")
    identity = connection.execute(
        "SELECT model, contract_version FROM cache WHERE path = ?", ("sample.py",)
    ).fetchone()

    summary = summarize_repo(str(tmp_path), connection)
    connection.close()

    cached_file = next(item for item in summary if item["name"] == "sample.py")
    assert identity == (None, None)
    assert cached_file["description"] == "Cached description"


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
        "main.tfstate.backup",
        "credentials.ini",
        "app.conf",
        "boto.cfg",
    ]
    retained = ["main.tf", "compose.yml", "notes.txt"]
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


# Private keys inside eligible text files -- issue #29.


@pytest.mark.parametrize("label", _PRIVATE_KEY_LABELS)
@pytest.mark.parametrize("name", sorted(_KEY_FILES))
def test_source_snapshot_rejects_a_private_key_header_with_a_warning(
    tmp_path, caplog, name, label
) -> None:
    language, render = _KEY_FILES[name]
    source = tmp_path / name
    source.write_text(render(label), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="repo_map.file_scanner"):
        snapshot = load_source_snapshot(str(source), language)

    assert snapshot is None
    assert str(source) in caplog.text


def test_summarize_repo_marks_key_bearing_files_ineligible(tmp_path) -> None:
    for name, (_, render) in _KEY_FILES.items():
        (tmp_path / name).write_text(render("PRIVATE KEY"), encoding="utf-8")

    files = _scan_files(tmp_path)

    assert {
        name: (entry["language"], entry["source_eligible"])
        for name, entry in files.items()
    } == {
        "deploy.yml": ("YAML", False),
        "key.txt": ("Text", False),
        "service-account.json": ("JSON", False),
    }


def test_source_snapshot_keeps_public_keys_and_certificates_eligible(tmp_path) -> None:
    for label in ["PUBLIC KEY", "RSA PUBLIC KEY", "CERTIFICATE"]:
        source = tmp_path / f"{label.replace(' ', '-').lower()}.txt"
        source.write_text(_key_as_text(label), encoding="utf-8")

        snapshot = load_source_snapshot(str(source), "Text")

        assert snapshot is not None, label
        assert snapshot.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()


def test_only_the_private_key_header_makes_an_otherwise_identical_file_ineligible(
    tmp_path,
) -> None:
    body = "\nPLACEHOLDER\n"
    private = tmp_path / "private.txt"
    public = tmp_path / "public.txt"
    private.write_text(
        _PEM_BEGIN + "PRIVATE KEY" + _PEM_DASHES + body, encoding="utf-8"
    )
    public.write_text(_PEM_BEGIN + "PUBLIC KEY" + _PEM_DASHES + body, encoding="utf-8")

    public_snapshot = load_source_snapshot(str(public), "Text")

    assert load_source_snapshot(str(private), "Text") is None
    assert public_snapshot is not None
    assert public_snapshot.sha256 == hashlib.sha256(public.read_bytes()).hexdigest()


def test_source_snapshot_reads_a_key_header_between_escaped_line_breaks(
    tmp_path,
) -> None:
    """Encoders that escape the block's line breaks leave the header line whole."""
    header = _PEM_BEGIN + "EC PRIVATE KEY" + _PEM_DASHES
    sources = {
        "crlf.json": '{"key": "\\r\\n' + header + '\\r\\nPLACEHOLDER\\r\\n"}\n',
        "unseparated.json": '{"key": "' + header + 'PLACEHOLDER"}\n',
        "single-line.env.json": '{"KEY": "' + header + '\\nPLACEHOLDER"}\n',
    }
    for name, text in sources.items():
        (tmp_path / name).write_text(text, encoding="utf-8")

    assert [
        name
        for name in sources
        if load_source_snapshot(str(tmp_path / name), "JSON") is not None
    ] == []


def test_source_snapshot_needs_the_header_label_on_one_line(tmp_path) -> None:
    """The match is the header line as PEM encoders write it, label unbroken.

    A label split by an escaped line break is not a key header under the
    decided pattern, so the file stays eligible.
    """
    source = tmp_path / "split.json"
    source.write_text(
        '{"key": "' + _PEM_BEGIN + "RSA\\nPRIVATE KEY" + _PEM_DASHES + '"}\n',
        encoding="utf-8",
    )

    assert load_source_snapshot(str(source), "JSON") is not None


def test_summarize_repo_hydrates_nothing_for_a_cached_file_that_holds_a_key(
    tmp_path,
) -> None:
    source = tmp_path / "service-account.json"
    source.write_text(_key_as_json("PRIVATE KEY"), encoding="utf-8")
    connection = load_cache(str(tmp_path))
    _seed_cache_row(
        connection,
        "service-account.json",
        hashlib.sha256(source.read_bytes()).hexdigest(),
        "Cached description",
    )

    summary = summarize_repo(str(tmp_path), connection)
    connection.close()

    entry = next(item for item in summary if item["name"] == "service-account.json")
    assert entry["source_eligible"] is False
    assert {"hash", "description", "imports", "size_bytes"}.isdisjoint(entry)


def test_gitignore_negation_leaves_a_key_bearing_file_ineligible(tmp_path) -> None:
    _write_tree(
        tmp_path,
        {".gitignore": "!app.conf\n", "app.conf": _key_as_text("PRIVATE KEY")},
    )

    files = _scan_files(tmp_path)

    assert files["app.conf"]["source_eligible"] is False


# Credential suffixes under any prefix -- issue #32.


def test_summarize_repo_excludes_prefixed_dotenv_and_terraform_state_backups(
    tmp_path,
) -> None:
    excluded = [
        "prod.env",
        "dev.env",
        "local.envrc",
        "terraform.tfstate.backup",
        "prod.tfstate.backup",
        "terraform.tfvars.json",
        "a.auto.tfvars.json",
        ".env",
        "sub/.env",
        "terraform.tfstate",
    ]
    retained = ["settings.json", "environment.py"]
    _write_tree(
        tmp_path, dict.fromkeys([*excluded, *retained], "token = PLACEHOLDER\n")
    )

    files = _scan_files(tmp_path)

    assert sorted(files) == sorted(retained)
    assert all(files[name]["source_eligible"] for name in retained)


@pytest.mark.parametrize(
    "suffix",
    [
        ".env",
        ".envrc",
        ".tfvars",
        ".tfstate",
        ".tfstate.backup",
        ".ini",
        ".conf",
        ".cfg",
    ],
)
def test_summarize_repo_excludes_each_credential_suffix_under_any_prefix(
    tmp_path, suffix
) -> None:
    name = f"x{suffix}"
    (tmp_path / name).write_text("token = PLACEHOLDER\n", encoding="utf-8")

    files = _scan_files(tmp_path)

    assert file_scanner._detect_language(name) in SUPPORTED_TEXT_LANGUAGES
    assert name not in files


def test_summarize_repo_negation_re_includes_prefixed_credential_files(
    tmp_path,
) -> None:
    _write_tree(
        tmp_path,
        {
            ".gitignore": "!prod.env\n!terraform.tfstate.backup\n",
            "prod.env": "token = PLACEHOLDER\n",
            "terraform.tfstate.backup": "{}\n",
            "other.env": "token = PLACEHOLDER\n",
        },
    )

    files = _scan_files(tmp_path)

    assert (files["prod.env"]["language"], files["prod.env"]["source_eligible"]) == (
        "Config",
        True,
    )
    assert files["terraform.tfstate.backup"]["source_eligible"] is True
    assert "other.env" not in files


# Rust, C, TOML, Make, and ES module files -- issue #30.


def test_summarize_repo_recognizes_rust_c_toml_make_and_es_module_files(
    tmp_path,
) -> None:
    expected = {
        "main.rs": "Rust",
        "util.c": "C",
        "pyproject.toml": "TOML",
        "index.mjs": "JavaScript",
        "server.cjs": "JavaScript",
        "mod.mts": "TypeScript",
        "mod.cts": "TypeScript",
        "Makefile": "Make",
        "tools/Makefile": "Make",
    }
    unrecognized = ["GNUmakefile", "rules.mk", "Cargo.lock"]
    _write_tree(tmp_path, dict.fromkeys([*expected, *unrecognized], "x\n"))

    files = _scan_files(tmp_path)

    assert {
        name: (files[name]["language"], files[name]["source_eligible"])
        for name in expected
    } == {name: (language, True) for name, language in expected.items()}
    assert {
        name: (files[name]["language"], files[name]["source_eligible"])
        for name in unrecognized
    } == dict.fromkeys(unrecognized, (None, False))


def test_summarize_repo_extracts_imports_from_es_module_files(tmp_path) -> None:
    _write_tree(
        tmp_path,
        {
            "index.mjs": "import x from './util.mjs'\nimport {y} from 'pkg'\n",
            "mod.mts": "import type { T } from './t.mjs'\n",
            "server.cjs": "const a = require('./a.cjs')\nimport b from './b.cjs'\n",
        },
    )

    files = _scan_files(tmp_path)

    assert {name: entry.get("imports") for name, entry in files.items()} == {
        "index.mjs": ["./util.mjs", "pkg"],
        "mod.mts": ["./t.mjs"],
        "server.cjs": ["./b.cjs"],
    }


def test_summarize_repo_excludes_toml_credential_files(tmp_path) -> None:
    excluded = [
        "secrets.toml",
        ".secrets.toml",
        ".streamlit/secrets.toml",
        ".cargo/credentials.toml",
    ]
    retained = ["pyproject.toml", "Cargo.toml"]
    _write_tree(tmp_path, dict.fromkeys([*excluded, *retained], "token = 'x'\n"))

    files = _scan_files(tmp_path)

    assert sorted(files) == sorted(retained)
    assert all(files[name]["source_eligible"] for name in retained)


def test_summarize_repo_negation_re_includes_secrets_toml_only(tmp_path) -> None:
    _write_tree(
        tmp_path,
        {
            ".gitignore": "!secrets.toml\n",
            "secrets.toml": "token = 'x'\n",
            ".secrets.toml": "token = 'x'\n",
        },
    )

    files = _scan_files(tmp_path)

    assert (
        files["secrets.toml"]["language"],
        files["secrets.toml"]["source_eligible"],
    ) == (
        "TOML",
        True,
    )
    assert ".secrets.toml" not in files


# Nested .gitignore files and .git/info/exclude -- issue #35.


def test_summarize_repo_applies_a_subdirectory_gitignore(tmp_path) -> None:
    """The reproduction from issue #35, with no `.git` anywhere."""
    _write_tree(
        tmp_path,
        {
            "config/.gitignore": "local.json\n",
            "config/local.json": '{"token": "PLACEHOLDER"}\n',
        },
    )

    files = _scan_files(tmp_path)

    assert "config/local.json" not in files
    assert files["config/.gitignore"]["source_eligible"] is True


def test_nested_gitignore_patterns_are_relative_to_their_own_directory(
    tmp_path,
) -> None:
    _write_tree(
        tmp_path,
        {
            "config/.gitignore": "/local.json\n",
            "config/local.json": "{}\n",
            "config/deep/local.json": "{}\n",
            "local.json": "{}\n",
        },
    )

    files = _scan_files(tmp_path)

    assert "config/local.json" not in files
    assert files["config/deep/local.json"]["source_eligible"] is True
    assert files["local.json"]["source_eligible"] is True


def test_a_deeper_gitignore_overrides_a_shallower_one(tmp_path) -> None:
    _write_tree(
        tmp_path,
        {
            "a/.gitignore": "*.txt\n",
            "a/b/.gitignore": "!keep.txt\n",
            "a/top.txt": "x\n",
            "a/b/keep.txt": "x\n",
            "a/b/drop.txt": "x\n",
        },
    )

    files = _scan_files(tmp_path)

    assert files["a/b/keep.txt"]["source_eligible"] is True
    assert "a/b/drop.txt" not in files
    assert "a/top.txt" not in files


def test_nested_gitignore_rules_stack_three_levels_deep(tmp_path) -> None:
    """A negation in the middle level reaches below it; each level's rule only reaches down."""
    _write_tree(
        tmp_path,
        {
            "a/.gitignore": "*.txt\n",
            "a/b/.gitignore": "!keep.txt\n",
            "a/b/c/.gitignore": "secret.md\n",
            "a/b/c/keep.txt": "x\n",
            "a/b/c/drop.txt": "x\n",
            "a/b/c/secret.md": "x\n",
            "a/b/secret.md": "x\n",
            "a/b/keep.txt": "x\n",
            "keep.txt": "x\n",
        },
    )

    files = _scan_files(tmp_path)

    assert sorted(path for path in files if not path.endswith(".gitignore")) == [
        "a/b/c/keep.txt",
        "a/b/keep.txt",
        "a/b/secret.md",
        "keep.txt",
    ]


def test_summarize_repo_applies_git_info_exclude(tmp_path) -> None:
    _make_git_dir(tmp_path, info_exclude="scratch.md\n/local/\n")
    _write_tree(
        tmp_path,
        {
            "scratch.md": "x\n",
            "docs/scratch.md": "x\n",
            "local/notes.md": "x\n",
            "readme.md": "x\n",
        },
    )

    files = _scan_files(tmp_path)

    assert sorted(files) == ["readme.md"]


def test_a_gitignore_overrides_git_info_exclude(tmp_path) -> None:
    """Git ranks every `.gitignore` above `info/exclude`."""
    _make_git_dir(tmp_path, info_exclude="*.md\n")
    _write_tree(
        tmp_path,
        {
            "docs/.gitignore": "!guide.md\n",
            "docs/guide.md": "x\n",
            "docs/other.md": "x\n",
        },
    )

    files = _scan_files(tmp_path)

    assert files["docs/guide.md"]["source_eligible"] is True
    assert "docs/other.md" not in files


def test_a_git_file_is_a_work_tree_without_info_exclude(tmp_path, caplog) -> None:
    """A linked worktree's `.git` is a file; it is never followed or an error."""
    (tmp_path / ".git").write_text(
        "gitdir: /nonexistent/worktrees/x\n", encoding="utf-8"
    )
    _write_tree(
        tmp_path,
        {
            "config/.gitignore": "local.json\n",
            "config/local.json": "{}\n",
            "app.py": "x\n",
        },
    )

    with caplog.at_level(logging.WARNING, logger="repo_map.file_scanner"):
        files = _scan_files(tmp_path)

    assert "config/local.json" not in files
    assert files["app.py"]["source_eligible"] is True
    assert caplog.records == []


def test_a_target_below_a_git_ancestor_honors_the_ancestors_rules(tmp_path) -> None:
    outer = tmp_path / "outer"
    _make_git_dir(outer, info_exclude="*.draft.md\n")
    _write_tree(
        outer,
        {
            ".gitignore": "private.json\n",
            "pkg/.gitignore": "/target/local.md\n",
            "pkg/target/private.json": "{}\n",
            "pkg/target/local.md": "x\n",
            "pkg/target/plan.draft.md": "x\n",
            "pkg/target/public.json": "{}\n",
            "pkg/target/sub/local.md": "x\n",
        },
    )

    files = _scan_files(outer / "pkg" / "target")

    assert sorted(files) == ["public.json", "sub/local.md"]


def test_a_target_its_work_tree_ignores_maps_nothing_and_says_why(
    tmp_path, caplog
) -> None:
    outer = tmp_path / "outer"
    _make_git_dir(outer)
    _write_tree(
        outer,
        {".gitignore": "vendor/\n", "vendor/lib/a.py": "x = 1\n", "src/b.py": "y\n"},
    )

    with caplog.at_level(logging.WARNING, logger="repo_map.file_scanner"):
        ignored = _scan_files(outer / "vendor" / "lib")
        kept = _scan_files(outer / "src")

    assert ignored == {}
    assert sorted(kept) == ["b.py"]
    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 1
    assert str(outer / "vendor" / "lib") in warnings[0]
    assert str(outer) in warnings[0]


def test_a_parent_gitignore_is_not_read_outside_a_work_tree(tmp_path) -> None:
    assert not any(
        os.path.lexists(path / ".git") for path in [tmp_path, *tmp_path.parents]
    )
    _write_tree(tmp_path, {".gitignore": "*.json\n", "project/data.json": "{}\n"})

    files = _scan_files(tmp_path / "project")

    assert files["data.json"]["source_eligible"] is True


def test_a_nested_repository_applies_its_own_rules_inside_it(tmp_path) -> None:
    inner = tmp_path / "vendor" / "inner"
    _make_git_dir(inner, info_exclude="notes.md\n")
    _write_tree(
        tmp_path,
        {
            "vendor/inner/.gitignore": "*.gen.json\n",
            "vendor/inner/x.gen.json": "{}\n",
            "vendor/inner/notes.md": "x\n",
            "vendor/inner/kept.json": "{}\n",
            "x.gen.json": "{}\n",
            "notes.md": "x\n",
        },
    )

    files = _scan_files(tmp_path)

    assert sorted(path for path in files if not path.endswith(".gitignore")) == [
        "notes.md",
        "vendor/inner/kept.json",
        "x.gen.json",
    ]


def test_an_ignored_directory_is_neither_emitted_scanned_nor_read(
    tmp_path, monkeypatch
) -> None:
    _write_tree(
        tmp_path,
        {
            ".gitignore": "vendor/\n",
            "vendor/.gitignore": "!kept.md\n",
            "vendor/kept.md": "x\n",
            "node_modules/.gitignore": "!index.js\n",
            "node_modules/index.js": "x\n",
            "config/.gitignore": "secret/\n",
            "config/secret/.gitignore": "!x.md\n",
            "config/secret/x.md": "x\n",
            "config/app.md": "x\n",
        },
    )
    pruned = [
        tmp_path / "vendor",
        tmp_path / "node_modules",
        tmp_path / "config" / "secret",
    ]
    scanned: list[str] = []
    opened: list[str] = []
    real_scandir = file_scanner.os.scandir
    real_open = builtins.open

    def recording_scandir(path):
        scanned.append(os.fspath(path))
        return real_scandir(path)

    def recording_open(path, *args, **kwargs):
        if isinstance(path, str | os.PathLike):
            opened.append(os.fspath(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(file_scanner.os, "scandir", recording_scandir)
    monkeypatch.setattr(builtins, "open", recording_open)
    connection = load_cache(str(tmp_path))
    summary = summarize_repo(str(tmp_path), connection)
    connection.close()

    touched = [
        path for path in [*scanned, *opened] if path.startswith(tuple(map(str, pruned)))
    ]
    assert touched == []
    assert [item.get("rel_path", item["name"]) for item in summary] == [
        "config",
        "config/.gitignore",
        "config/app.md",
        ".gitignore",
    ]
    assert str(tmp_path / "config") in scanned


def test_a_nested_negation_cannot_re_include_a_default_or_root_exclusion(
    tmp_path,
) -> None:
    _write_tree(
        tmp_path,
        {
            ".gitignore": "root-only.json\n",
            "sub/.gitignore": "!app.conf\n!root-only.json\n",
            "sub/app.conf": "token = PLACEHOLDER\n",
            "sub/root-only.json": "{}\n",
            "sub/kept.json": "{}\n",
        },
    )

    files = _scan_files(tmp_path)

    assert "sub/app.conf" not in files
    assert "sub/root-only.json" not in files
    assert files["sub/kept.json"]["source_eligible"] is True


def test_summarize_repo_reads_no_global_excludes_and_spawns_no_process(
    tmp_path, monkeypatch
) -> None:
    """Only the repository's own rule files are read, and nothing is executed.

    `tracked.json` stands for a file git tracks: the scanner never reads the
    index, so a pattern excludes a path whatever its tracked state.
    """
    home = tmp_path / "home"
    _write_tree(
        home,
        {
            "global-ignore": "*.py\n",
            "gitconfig": f"[core]\n\texcludesFile = {home / 'global-ignore'}\n",
            ".config/git/ignore": "*.md\n",
        },
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(home / "gitconfig"))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    repository = tmp_path / "repository"
    _make_git_dir(repository, info_exclude="tracked.json\n")
    _write_tree(
        repository,
        {
            ".git/config": "[core]\n\tfsmonitor = touch fsmonitor-ran\n",
            ".git/index": "DIRC placeholder\n",
            "app.py": "x\n",
            "readme.md": "x\n",
            "tracked.json": "{}\n",
        },
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("the scan started a process")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    for name in ["system", "fork", "posix_spawn", "posix_spawnp"]:
        monkeypatch.setattr(os, name, forbidden)
    opened: list[str] = []
    real_open = builtins.open

    def recording_open(path, *args, **kwargs):
        if isinstance(path, str | os.PathLike):
            opened.append(os.fspath(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", recording_open)
    files = _scan_files(repository)

    assert sorted(files) == ["app.py", "readme.md"]
    assert all(entry["source_eligible"] for entry in files.values())
    assert [path for path in opened if f"{os.sep}.git{os.sep}" in path] == [
        str(repository / ".git" / "info" / "exclude")
    ]


@pytest.mark.parametrize("rules_file", [".gitignore", "config/.gitignore"])
def test_a_bad_rules_file_line_is_dropped_with_a_warning(
    tmp_path, caplog, rules_file
) -> None:
    rules_path = tmp_path / rules_file
    directory = rules_path.parent
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    rules_path.write_bytes(b"\xff\xfe-not-utf8\n[z-a]\nsecret.json\n")
    _write_tree(directory, {"secret.json": "{}\n", "kept.json": "{}\n"})
    prefix = "" if directory == tmp_path else "config/"

    with caplog.at_level(logging.WARNING, logger="repo_map.file_scanner"):
        files = _scan_files(tmp_path)

    assert f"{prefix}secret.json" not in files
    assert files[f"{prefix}kept.json"]["source_eligible"] is True
    assert [
        record.getMessage()
        for record in caplog.records
        if str(rules_path) in record.getMessage()
    ] != []


@pytest.mark.parametrize("rules_file", [".gitignore", "config/.gitignore"])
def test_a_rules_file_byte_order_mark_is_not_part_of_the_first_pattern(
    tmp_path, rules_file
) -> None:
    rules_path = tmp_path / rules_file
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    rules_path.write_text("\ufeffsecret.json\n", encoding="utf-8")
    _write_tree(rules_path.parent, {"secret.json": "{}\n"})

    files = _scan_files(tmp_path)

    assert [path for path in files if path.endswith("secret.json")] == []


def test_a_rules_file_that_is_not_a_regular_file_is_skipped_with_a_warning(
    tmp_path, caplog
) -> None:
    (tmp_path / "config").mkdir()
    os.mkfifo(tmp_path / "config" / ".gitignore")
    _write_tree(tmp_path, {"config/app.json": "{}\n"})

    with caplog.at_level(logging.WARNING, logger="repo_map.file_scanner"):
        files = _scan_files(tmp_path)

    assert files["config/app.json"]["source_eligible"] is True
    assert str(tmp_path / "config" / ".gitignore") in caplog.text


def test_an_empty_root_gitignore_leaves_the_default_exclusions_alone(tmp_path) -> None:
    _write_tree(tmp_path, {".gitignore": "", "app.conf": "x\n", "app.json": "{}\n"})

    files = _scan_files(tmp_path)

    assert sorted(files) == [".gitignore", "app.json"]


def test_a_root_gitignore_that_is_a_directory_is_skipped_with_a_warning(
    tmp_path, caplog
) -> None:
    _write_tree(
        tmp_path,
        {".gitignore/inner.json": "{}\n", "app.conf": "x\n", "app.json": "{}\n"},
    )

    with caplog.at_level(logging.WARNING, logger="repo_map.file_scanner"):
        files = _scan_files(tmp_path)

    assert sorted(files) == [".gitignore/inner.json", "app.json"]
    assert str(tmp_path / ".gitignore") in caplog.text


def test_an_unreadable_root_gitignore_is_skipped_with_a_warning(
    tmp_path, caplog
) -> None:
    rules = tmp_path / ".gitignore"
    rules.write_text("app.json\n", encoding="utf-8")
    rules.chmod(0)
    _write_tree(tmp_path, {"app.conf": "x\n", "app.json": "{}\n"})

    try:
        with caplog.at_level(logging.WARNING, logger="repo_map.file_scanner"):
            files = _scan_files(tmp_path)
    finally:
        rules.chmod(0o644)

    assert files["app.json"]["source_eligible"] is True
    assert "app.conf" not in files
    assert str(rules) in caplog.text
