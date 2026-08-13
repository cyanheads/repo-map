"""Tests for language-specific structural extraction."""

from repo_map.code_parser import (
    get_imports,
    get_module_docstring,
    get_structure,
)


def test_python_extraction_reports_top_level_structure(tmp_path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        '''"""Module summary."""

import os
from pathlib import Path

LIMIT = 3

class Runner:
    def run(self):
        return Path(os.getcwd())

def main():
    return Runner().run()
''',
        encoding="utf-8",
    )

    classes, functions, constants = get_structure(str(source), "Python")

    assert classes == {"Runner": ["run"]}
    assert functions == ["main"]
    assert constants == ["LIMIT"]
    assert get_imports(str(source), "Python") == ["os", "pathlib.Path"]
    assert get_module_docstring(str(source), "Python") == "Module summary."


def test_javascript_extraction_reports_simple_class_and_function(tmp_path) -> None:
    source = tmp_path / "sample.js"
    source.write_text(
        "class Runner {\n  run() {}\n}\nconst LIMIT = 3;\n",
        encoding="utf-8",
    )

    classes, functions, constants = get_structure(str(source), "JavaScript")

    assert classes == {"Runner": ["run"]}
    assert functions == []
    assert constants == ["LIMIT"]
