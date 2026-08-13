"""Tests for console and Markdown tree rendering."""

from repo_map.report_generator import (
    format_tree_lines,
    save_json_map,
    save_markdown_map,
)


def test_format_tree_lines_renders_metadata() -> None:
    structure = [
        {"name": "src", "path": "/repo/src", "level": 0, "type": "directory"},
        {
            "name": "module.py",
            "path": "/repo/src/module.py",
            "level": 1,
            "type": "file",
            "language": "Python",
            "description": "Runs the application.",
            "maintenance_flag": "Stable",
            "critical_dependencies": '{"aiohttp": "HTTP client"}',
        },
    ]

    output = "\n".join(format_tree_lines(structure))

    assert "src/" in output
    assert "module.py (Python)" in output
    assert "Description: Runs the application." in output
    assert "aiohttp: HTTP client" in output


def test_save_report_files(tmp_path) -> None:
    structure = [
        {
            "name": "README.md",
            "path": str(tmp_path / "README.md"),
            "level": 0,
            "type": "file",
            "language": "Markdown",
        }
    ]
    markdown_path = tmp_path / "map.md"
    json_path = tmp_path / "map.json"

    save_markdown_map(structure, str(tmp_path), str(markdown_path))
    save_json_map(structure, str(json_path))

    assert "# Repository Map" in markdown_path.read_text(encoding="utf-8")
    assert '"name": "README.md"' in json_path.read_text(encoding="utf-8")
