"""Tests for console and Markdown tree rendering."""

import json
import time
from typing import Any

from repo_map.report_generator import (
    format_tree_lines,
    save_json_map,
    save_markdown_map,
)


def _directory(name: str, level: int) -> dict[str, Any]:
    """Build a directory entry for a tree fixture."""
    return {
        "name": name,
        "path": f"/repo/{name}",
        "level": level,
        "type": "directory",
    }


def _file(name: str, level: int, **extra: Any) -> dict[str, Any]:
    """Build a file entry for a tree fixture."""
    item: dict[str, Any] = {
        "name": name,
        "path": f"/repo/{name}",
        "level": level,
        "type": "file",
        "language": "Python",
        "critical_dependencies": "{}",
    }
    item.update(extra)
    return item


def _wide_fixture(groups: int) -> list[dict[str, Any]]:
    """Build a fixed-depth tree of ``groups`` six-entry branches."""
    structure: list[dict[str, Any]] = []
    for group in range(groups):
        structure.append(_directory(f"pkg{group}", 0))
        structure.append(_directory(f"pkg{group}/sub", 1))
        structure.append(_directory(f"pkg{group}/sub/inner", 2))
        structure.append(_file(f"deep{group}.py", 3))
        structure.append(_file(f"mid{group}.py", 2))
        structure.append(_file(f"top{group}.py", 1))
    return structure


def _elapsed(structure: list[dict[str, Any]], runs: int = 3) -> float:
    """Return the fastest wall-clock time of a full render of ``structure``."""
    best = float("inf")
    for _ in range(runs):
        start = time.perf_counter()
        for _line in format_tree_lines(structure):
            pass
        best = min(best, time.perf_counter() - start)
    return best


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


def test_format_tree_lines_renders_metadata_exact_lines() -> None:
    """Pin the exact rendering of the single-branch metadata fixture."""
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

    assert list(format_tree_lines(structure)) == [
        "└── src/",
        "    └── module.py (Python)",
        "        ├── Description: Runs the application.",
        "        ├── Maintenance Flag: Stable",
        "        ├── Critical Dependencies:",
        "        └──   - aiohttp: HTTP client",
    ]


def test_format_tree_lines_single_entry_has_no_prefix_cells() -> None:
    """A lone entry renders with a bare connector and no indentation."""
    assert list(format_tree_lines([_file("only.py", 0)])) == ["└── only.py (Python)"]


def test_format_tree_lines_draws_ancestor_bars_to_depth_three() -> None:
    """Every prefix cell tracks whether that depth's ancestor has a later sibling."""
    structure = [
        _directory("a", 0),
        _directory("b", 1),
        _directory("c", 2),
        _file("d.py", 3),
        _file("e.py", 2),
        _file("f.py", 1),
        _directory("g", 0),
        _file("h.py", 1),
    ]

    assert list(format_tree_lines(structure)) == [
        "├── a/",
        "│   ├── b/",
        "│   │   ├── c/",
        "│   │   │   └── d.py (Python)",
        "│   │   └── e.py (Python)",
        "│   └── f.py (Python)",
        "└── g/",
        "    └── h.py (Python)",
    ]


def test_format_tree_lines_renders_all_metadata_under_nested_prefix() -> None:
    """Metadata lines inherit the corrected prefix of their nested parent."""
    structure = [
        _directory("a", 0),
        _directory("b", 1),
        _file(
            "deep.py",
            2,
            description="Runs the application.",
            developer_consideration="Import order matters.",
            maintenance_flag="Stable",
            architectural_role="Service",
            refactoring_suggestions="Split the client out.",
            security_assessment="Validates every payload.",
            critical_dependencies='{"aiohttp": "HTTP client"}',
        ),
        _file("sibling.py", 1),
        _directory("z", 0),
    ]

    assert list(format_tree_lines(structure)) == [
        "├── a/",
        "│   ├── b/",
        "│   │   └── deep.py (Python)",
        "│   │       ├── Description: Runs the application.",
        "│   │       ├── Developer Consideration: Import order matters.",
        "│   │       ├── Maintenance Flag: Stable",
        "│   │       ├── Architectural Role: Service",
        "│   │       ├── Refactoring Suggestions: Split the client out.",
        "│   │       ├── Security Assessment: Validates every payload.",
        "│   │       ├── Critical Dependencies:",
        "│   │       └──   - aiohttp: HTTP client",
        "│   └── sibling.py (Python)",
        "└── z/",
    ]


def test_format_tree_lines_matches_issue_18_first_fixture() -> None:
    """A child of a non-last directory keeps its parent's vertical bar."""
    structure = [_directory("a", 0), _file("f.py", 1), _directory("b", 0)]

    assert list(format_tree_lines(structure)) == [
        "├── a/",
        "│   └── f.py (Python)",
        "└── b/",
    ]


def test_format_tree_lines_matches_issue_18_second_fixture() -> None:
    """A grandchild is connected to the branch its grandparent still holds open."""
    structure = [
        _directory("a", 0),
        _directory("b", 1),
        _file("f.py", 2),
        _directory("c", 1),
    ]

    assert list(format_tree_lines(structure)) == [
        "└── a/",
        "    ├── b/",
        "    │   └── f.py (Python)",
        "    └── c/",
    ]


def test_format_tree_lines_collapses_embedded_whitespace() -> None:
    """Newlines, carriage returns, and tabs never survive into a yielded line."""
    structure = [
        _file(
            "mod.py",
            0,
            description="Line one\nLine two",
            developer_consideration="Tabbed\there\r\nand wrapped",
            refactoring_suggestions="Split\n\n  this",
            security_assessment="Careful\rnow",
            critical_dependencies='{"aiohttp": "HTTP\\nclient"}',
        )
    ]

    lines = list(format_tree_lines(structure))

    for line in lines:
        assert "\n" not in line
        assert "\r" not in line
        assert "\t" not in line
    assert lines == [
        "└── mod.py (Python)",
        "    ├── Description: Line one Line two",
        "    ├── Developer Consideration: Tabbed here and wrapped",
        "    ├── Refactoring Suggestions: Split this",
        "    ├── Security Assessment: Careful now",
        "    ├── Critical Dependencies:",
        "    └──   - aiohttp: HTTP client",
    ]


def test_format_tree_lines_defuses_fences() -> None:
    """A fence in model text cannot survive as three consecutive backticks."""
    structure = [
        _file(
            "mod.py",
            0,
            description="Uses a fence:\n```python\nprint(1)\n```\nand then continues.",
        )
    ]

    lines = list(format_tree_lines(structure))

    assert lines == [
        "└── mod.py (Python)",
        "    └── Description: Uses a fence: ` ` `python print(1) ` ` ` "
        "and then continues.",
    ]
    assert "```" not in "\n".join(lines)


def test_format_tree_lines_leaves_single_spaced_text_intact() -> None:
    """Ordinary single-spaced text renders unchanged."""
    structure = [_file("mod.py", 0, description="A plain one line description.")]

    assert list(format_tree_lines(structure)) == [
        "└── mod.py (Python)",
        "    └── Description: A plain one line description.",
    ]


def test_format_tree_lines_does_not_mutate_structure() -> None:
    """Normalization is local to the rendered lines."""
    description = "Line one\nLine two"
    structure = [_file("mod.py", 0, description=description)]

    list(format_tree_lines(structure))

    assert structure[0]["description"] == description


def test_save_markdown_map_keeps_fenced_description_inside_the_block(tmp_path) -> None:
    """Issue #22's reproduction renders entirely inside one Markdown block."""
    structure = [
        _file(
            "mod.py",
            0,
            description="Uses a fence:\n```python\nprint(1)\n```\nand then continues.",
        ),
        _file("other.py", 0, description="Second file."),
    ]
    markdown_path = tmp_path / "report.md"

    save_markdown_map(structure, "/repo", str(markdown_path))
    lines = markdown_path.read_text(encoding="utf-8").splitlines()

    fences = [
        index for index, line in enumerate(lines) if line.lstrip().startswith("```")
    ]
    assert len(fences) == 2
    assert lines[fences[0]] == "```markdown"
    assert lines[fences[1]] == "```"

    body = lines[fences[0] + 1 : fences[1]]
    assert "├── mod.py (Python)" in body
    assert "└── other.py (Python)" in body
    assert body.index("├── mod.py (Python)") < body.index("└── other.py (Python)")
    assert body[-1].startswith("└──────────────")


def test_save_json_map_preserves_raw_text(tmp_path) -> None:
    """Rendering normalization never reaches the JSON structure output."""
    description = "Line one\nLine two\twith a tab"
    structure = [_file("mod.py", 0, description=description)]
    json_path = tmp_path / "map.json"

    list(format_tree_lines(structure))
    save_json_map(structure, str(json_path))

    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert written[0]["description"] == description


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


def test_format_tree_lines_scales_linearly() -> None:
    """Quadrupling the entry count must not quadruple the per-entry cost."""
    small = _elapsed(_wide_fixture(100))
    large = _elapsed(_wide_fixture(400))

    # Linear growth lands near 4x, quadratic near 16x; 8x is the midpoint.
    assert large < small * 8
