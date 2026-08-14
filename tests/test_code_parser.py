"""Tests for language-specific structural extraction."""

from pathlib import Path

from repo_map.code_parser import (
    get_imports,
    get_module_docstring,
    get_structure,
)

JAVA_MAIN_SOURCE = """import java.util.List;
import static java.util.Arrays.asList;

public class Main {
    public static void main(String[] args) { }
    public List<String> getNames() { return null; }
    public Main() {}
    private int count(int x) { return x; }
}
"""


def _write(tmp_path: Path, name: str, source: str) -> str:
    """Write a fixture file and return its path as a string."""
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return str(path)


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


def test_java_extraction_reports_class_method_and_constant(tmp_path) -> None:
    source = _write(
        tmp_path,
        "Counter.java",
        """import java.util.List;

public class Counter {
    public static final int LIMIT = 3;

    private int count(int x) { return x; }
}
""",
    )

    classes, functions, constants = get_structure(source, "Java")

    assert classes == {"Counter": ["count"]}
    assert functions == []
    assert constants == ["LIMIT"]
    assert get_imports(source, "Java") == ["java.util.List"]


def test_csharp_extraction_reports_class_method_and_constant(tmp_path) -> None:
    source = _write(
        tmp_path,
        "Counter.cs",
        """using System.Text;

public class Counter
{
    public const int LIMIT = 3;

    private int Count(int x) { return x; }
}
""",
    )

    classes, functions, constants = get_structure(source, "C#")

    assert classes == {"Counter": ["Count"]}
    assert functions == []
    assert constants == ["LIMIT"]
    assert get_imports(source, "C#") == ["System.Text"]


def test_typescript_extraction_reports_class_and_relative_import(tmp_path) -> None:
    source = _write(
        tmp_path,
        "service.ts",
        """import { Local } from "./local";

export class Service {
  fetch() {}
}

const LIMIT = 3;
""",
    )

    classes, functions, constants = get_structure(source, "TypeScript")

    assert classes == {"Service": ["fetch"]}
    assert functions == []
    assert constants == ["LIMIT"]
    assert get_imports(source, "TypeScript") == ["./local"]


def test_ruby_module_docstring_is_empty(tmp_path) -> None:
    source = _write(
        tmp_path,
        "runner.rb",
        "# Runner for the batch job.\nclass Runner\nend\n",
    )

    assert get_module_docstring(source, "Ruby") == ""


def test_python_extraction_reports_module_level_async_function(tmp_path) -> None:
    source = _write(tmp_path, "sample.py", "async def run():\n    pass\n")

    classes, functions, constants = get_structure(source, "Python")

    assert classes == {}
    assert functions == ["run"]
    assert constants == []


def test_python_extraction_reports_async_method_in_class(tmp_path) -> None:
    source = _write(
        tmp_path,
        "sample3.py",
        "class Runner:\n    async def run(self):\n        pass\n",
    )

    classes, _, _ = get_structure(source, "Python")

    assert classes == {"Runner": ["run"]}


def test_javascript_class_scope_closes_at_end_of_class_body(tmp_path) -> None:
    source = _write(
        tmp_path,
        "sample.js",
        "class A {\n  method() {}\n}\nfunction outside() {}\n",
    )

    classes, functions, _ = get_structure(source, "JavaScript")

    assert classes == {"A": ["method"]}
    assert functions == ["outside"]


def test_javascript_call_expression_is_not_reported_as_method(tmp_path) -> None:
    source = _write(
        tmp_path,
        "sample2.js",
        "class B {\n  init() {\n    helper();\n  }\n}\n",
    )

    classes, functions, _ = get_structure(source, "JavaScript")

    assert classes == {"B": ["init"]}
    assert functions == []


def test_javascript_control_flow_is_not_reported_as_method(tmp_path) -> None:
    source = _write(
        tmp_path,
        "flow.js",
        """class C {
  run(items) {
    for (const item of items) {
      if (item) {
        handle(item);
      }
    }
  }
}
""",
    )

    classes, functions, _ = get_structure(source, "JavaScript")

    assert classes == {"C": ["run"]}
    assert functions == []


def test_typescript_annotated_methods_are_reported(tmp_path) -> None:
    source = _write(
        tmp_path,
        "Queue.ts",
        """export class Queue {
  private readonly items: string[] = [];

  constructor(private ttl: number) {}

  private get size(): number {
    return this.items.length;
  }

  async enqueue(item: string): Promise<void> {
    this.items.push(item);
  }

  map<T>(fn: (value: string) => T): T[] {
    return this.items.map(fn);
  }
}
""",
    )

    classes, functions, _ = get_structure(source, "TypeScript")

    assert classes == {"Queue": ["constructor", "size", "enqueue", "map"]}
    assert functions == []


def test_typescript_module_docstring_returns_only_jsdoc_header(tmp_path) -> None:
    source = _write(
        tmp_path,
        "Service.ts",
        """/**
 * Service for fetching users.
 */
import { Local } from "./local";

// helper for retries
export const retry = () => 1;
// another note
""",
    )

    assert get_module_docstring(source, "TypeScript") == "Service for fetching users."


def test_typescript_module_docstring_returns_leading_line_comment_run(
    tmp_path,
) -> None:
    source = _write(
        tmp_path,
        "Utils.ts",
        """
// Utility helpers.
// Shared across services.
import { Local } from "./local";
// unrelated trailing note
""",
    )

    docstring = get_module_docstring(source, "TypeScript")

    assert docstring == "Utility helpers. Shared across services."


def test_typescript_module_docstring_excludes_comments_after_header(tmp_path) -> None:
    source = _write(
        tmp_path,
        "Late.ts",
        """import { Local } from "./local";

// helper for retries
export const retry = () => 1;
""",
    )

    assert get_module_docstring(source, "TypeScript") == ""


def test_java_imports_include_static_targets(tmp_path) -> None:
    source = _write(tmp_path, "Main.java", JAVA_MAIN_SOURCE)

    assert get_imports(source, "Java") == [
        "java.util.List",
        "java.util.Arrays.asList",
    ]


def test_java_imports_keep_wildcard_specifier_intact(tmp_path) -> None:
    source = _write(tmp_path, "Wild.java", "import java.util.*;\n")

    assert get_imports(source, "Java") == ["java.util.*"]


def test_java_structure_reports_static_generic_and_constructor_methods(
    tmp_path,
) -> None:
    source = _write(tmp_path, "Main.java", JAVA_MAIN_SOURCE)

    assert get_structure(source, "Java") == (
        {"Main": ["main", "getNames", "Main", "count"]},
        [],
        [],
    )


def test_java_structure_ignores_field_initializers_and_calls(tmp_path) -> None:
    source = _write(
        tmp_path,
        "Widget.java",
        """public class Widget {
    public static final int LIMIT = compute(3);

    private void render() {
        helper();
    }
}
""",
    )

    classes, functions, constants = get_structure(source, "Java")

    assert classes == {"Widget": ["render"]}
    assert functions == []
    assert constants == ["LIMIT"]


def test_csharp_structure_reports_static_generic_and_constructor_methods(
    tmp_path,
) -> None:
    source = _write(
        tmp_path,
        "Main.cs",
        """using System.Collections.Generic;

public class Main
{
    public static void Run(string[] args) { }
    public List<string> GetNames() { return null; }
    public Main() {}
    private int Count(int x) { return x; }
}
""",
    )

    assert get_structure(source, "C#") == (
        {"Main": ["Run", "GetNames", "Main", "Count"]},
        [],
        [],
    )


def test_typescript_imports_accept_scoped_and_hyphenated_specifiers(tmp_path) -> None:
    source = _write(
        tmp_path,
        "Service.ts",
        """import { useQuery } from "@tanstack/react-query";
import { format } from "date-fns";
import { Local } from "./local";
""",
    )

    assert get_imports(source, "TypeScript") == [
        "@tanstack/react-query",
        "date-fns",
        "./local",
    ]


def test_typescript_imports_accept_side_effect_and_reexport_forms(tmp_path) -> None:
    source = _write(
        tmp_path,
        "Extra.ts",
        """import "./polyfills";
export { format } from "date-fns";
import Default from "lodash";
""",
    )

    assert get_imports(source, "TypeScript") == [
        "./polyfills",
        "date-fns",
        "lodash",
    ]


def test_javascript_imports_accept_omitted_semicolons(tmp_path) -> None:
    source = _write(
        tmp_path,
        "loose.js",
        """import { format } from 'date-fns'
import Default from "lodash"
export * from "./local"
""",
    )

    assert get_imports(source, "JavaScript") == ["date-fns", "lodash", "./local"]


def test_typescript_imports_ignore_non_import_statements(tmp_path) -> None:
    source = _write(
        tmp_path,
        "Noise.ts",
        """const label = "import { a } from 'b';";
export const NAME = "date-fns";
export default class Service {}
""",
    )

    assert get_imports(source, "TypeScript") == []
