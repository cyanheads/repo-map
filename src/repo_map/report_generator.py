"""Functions for generating and saving the repository map reports."""

import json
import logging
import os
import re
from collections.abc import Generator
from typing import Any

logger = logging.getLogger(__name__)

_FENCE_PATTERN = re.compile(r"`{3,}")


def _render_text(value: Any) -> str:
    """Flatten model-authored text into a single safe tree line.

    Collapses every whitespace run — newlines, carriage returns, and tabs
    included — to one space, then spaces out any run of three or more
    backticks so model output cannot close the Markdown report's own fence.
    The caller's ``structure`` is never touched: only the rendered line is
    normalized, so the raw text still reaches the cache and the JSON map.
    """
    collapsed = " ".join(str(value).split())
    return _FENCE_PATTERN.sub(lambda match: " ".join(match.group()), collapsed)


def format_tree_lines(structure: list[dict[str, Any]]) -> Generator[str, None, None]:
    """Yield lines that represent the repository tree with metadata."""
    # Index ``d`` records whether the ancestor at depth ``d`` has a later
    # sibling, which is what decides between a vertical bar and blank
    # indentation for that prefix cell.
    open_branches: list[bool] = []
    for i, item in enumerate(structure):
        level = item["level"]

        is_last = True
        for j in range(i + 1, len(structure)):
            if structure[j]["level"] == level:
                is_last = False
                break
            if structure[j]["level"] < level:
                break

        del open_branches[level:]
        prefix = "".join("│   " if branch else "    " for branch in open_branches)
        open_branches.append(not is_last)

        connector = "└── " if is_last else "├── "
        if item["type"] == "directory":
            yield f"{prefix}{connector}{item['name']}/"
            continue

        language = item.get("language", "None")
        yield f"{prefix}{connector}{item['name']} ({language})"

        details_prefix = prefix + ("    " if is_last else "│   ")
        detail_lines: list[str] = []
        if item.get("description"):
            detail_lines.append(f"Description: {_render_text(item['description'])}")
        if item.get("developer_consideration"):
            detail_lines.append(
                "Developer Consideration: "
                f"{_render_text(item['developer_consideration'])}"
            )

        maintenance_flag = item.get("maintenance_flag")
        if maintenance_flag and maintenance_flag != "Unknown":
            detail_lines.append(f"Maintenance Flag: {maintenance_flag}")

        architectural_role = item.get("architectural_role")
        if architectural_role and architectural_role != "Unknown":
            detail_lines.append(f"Architectural Role: {architectural_role}")

        refactoring_suggestions = item.get("refactoring_suggestions")
        if refactoring_suggestions and refactoring_suggestions != "None":
            detail_lines.append(
                f"Refactoring Suggestions: {_render_text(refactoring_suggestions)}"
            )

        security_assessment = item.get("security_assessment")
        if security_assessment and security_assessment != "None":
            detail_lines.append(
                f"Security Assessment: {_render_text(security_assessment)}"
            )

        try:
            deps_str = item.get("critical_dependencies", "{}")
            deps = json.loads(deps_str)
            if deps:
                detail_lines.append("Critical Dependencies:")
                for dep, reason in deps.items():
                    detail_lines.append(
                        f"  - {_render_text(dep)}: {_render_text(reason)}"
                    )
        except json.JSONDecodeError:
            detail_lines.append("Critical Dependencies: (invalid JSON)")

        for idx, detail in enumerate(detail_lines):
            detail_connector = "└── " if idx == len(detail_lines) - 1 else "├── "
            yield f"{details_prefix}{detail_connector}{detail}"


def print_tree(structure: list[dict[str, Any]]) -> None:
    """Log the repository tree to the console."""
    logger.info("/ (Root Directory)")
    for line in format_tree_lines(structure):
        logger.info(line)
    logger.info("└────────────── ")


def save_markdown_map(
    structure: list[dict[str, Any]], repo_root: str, output_path: str
) -> None:
    """Persist the repository map to a Markdown file."""
    repo_name = os.path.basename(os.path.normpath(repo_root))
    try:
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("# Repository Map\n\n")
            handle.write("```markdown\n")
            handle.write(f"/ ({repo_name})\n")
            for line in format_tree_lines(structure):
                handle.write(f"{line}\n")
            handle.write("└────────────── \n")
            handle.write("```\n")
    except OSError as exc:
        logger.error("Error saving repository map: %s", exc)


def save_json_map(structure: list[dict[str, Any]], output_path: str) -> None:
    """Persist the raw structure to JSON."""
    try:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(structure, handle, indent=4)
        logger.info("Pre-enhancement structure saved to '%s'.", output_path)
    except OSError as exc:
        logger.error("Error saving JSON structure map: %s", exc)
