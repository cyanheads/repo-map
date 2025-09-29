"""
Main script for the repo-map tool.

This script provides a CLI for generating a structured summary of a software
repository, enhanced with AI-generated descriptions.
"""

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
from collections.abc import Generator
from typing import Any, Optional

from tqdm import tqdm

from repo_map.cache_manager import load_cache
from repo_map.config import settings
from repo_map.file_scanner import summarize_repo
from repo_map.llm_service import get_llm_descriptions, update_api_semaphore_limit

# --- Tqdm-Friendly Logging ---

logger = logging.getLogger(__name__)


class TqdmLoggingHandler(logging.Handler):
    """Redirects logging output through tqdm.write."""

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (OSError, ValueError, TypeError):
            self.handleError(record)


def setup_logging(level=logging.INFO):
    """Configures logging for the application."""
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[TqdmLoggingHandler()],
    )


# --- Core Application Logic ---


class RepoMapApp:
    """Encapsulates the logic for the repo-map CLI application."""

    def __init__(self) -> None:
        self.args: Optional[argparse.Namespace] = None
        self.cache_conn: Optional[sqlite3.Connection] = None

    async def run(self):
        """Main execution flow for the application."""
        setup_logging()
        self.args = self._parse_args()

        if not settings.has_api_key():
            logger.error("Error: OPENROUTER_API_KEY environment variable not set.")
            sys.exit(1)

        assert self.args is not None
        try:
            update_api_semaphore_limit(self.args.concurrency)
        except ValueError as exc:
            logger.error("Invalid concurrency value: %s", exc)
            sys.exit(1)

        if not os.path.isdir(self.args.repository_path):
            logger.error("Error: %s is not a valid directory", self.args.repository_path)
            sys.exit(1)

        if not self.args.yes and not self._confirm_disclaimer():
            logger.warning("Operation cancelled by the user.")
            sys.exit(0)

        self.cache_conn = load_cache(self.args.repository_path)

        try:
            await self._process_repository()
        finally:
            if self.cache_conn:
                self.cache_conn.close()

    async def _process_repository(self):
        """Orchestrates scanning, enhancing, and saving the repository map."""
        assert self.args is not None
        logger.info("Generating repository summary...")
        summary = summarize_repo(self.args.repository_path, self.cache_conn)

        pre_enhanced_path = os.path.join(
            self.args.repository_path, ".repo_map_structure.json"
        )
        self._save_json_map(summary, pre_enhanced_path)

        logger.info(
            "Enhancing repository summary with descriptions using OpenRouter LLM..."
        )
        model_name = self.args.model or settings.openrouter_model_name
        await self._enhance_summary_with_llm(summary, model_name)

        logger.info("\nUpdated Repository Map:")
        self._print_tree(summary)

        output_path = self._get_output_path()
        self._save_markdown_map(summary, self.args.repository_path, output_path)
        logger.info(
            "Your repo-map has been saved to '%s'.", os.path.basename(output_path)
        )

    async def _enhance_summary_with_llm(
        self, structure: list[dict[str, Any]], model_name: str
    ) -> None:
        """Enhances the repository structure with descriptions using LLM."""
        files_to_process = self._get_files_to_process(structure)

        tasks = [
            (index, file)
            for index, file in enumerate(structure)
            if file["path"] in files_to_process
        ]

        if not tasks:
            logger.info("No new or modified files to enhance. All up to date.")
            return

        for index, file in tqdm(tasks, desc="Enhancing files", ncols=100):
            tqdm.write(f"Processing: {file['name']}")
            await get_llm_descriptions(structure, index, file, model=model_name)
            self._update_cache_for_file(file)

    def _get_files_to_process(self, structure: list[dict[str, Any]]) -> set[str]:
        """Determines which files need LLM enhancement based on cache status."""
        assert self.cache_conn is not None
        cursor = self.cache_conn.cursor()
        files_to_process = set()
        for item in structure:
            if item["type"] == "file" and (item.get("imports") or item.get("functions")):
                cursor.execute("SELECT hash FROM cache WHERE path = ?", (item["path"],))
                row = cursor.fetchone()
                if not row or row[0] != item.get("hash", ""):
                    files_to_process.add(item["path"])
        return files_to_process

    def _update_cache_for_file(self, file_data: dict[str, Any]):
        """Updates the cache with the new data for a single file."""
        assert self.cache_conn is not None
        cursor = self.cache_conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO cache (path, hash, description, developer_consideration, imports, functions)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                file_data["path"],
                file_data.get("hash", ""),
                file_data.get("description", ""),
                file_data.get("developer_consideration", ""),
                json.dumps(file_data.get("imports", [])),
                json.dumps(file_data.get("functions", [])),
            ),
        )
        self.cache_conn.commit()

    def _format_tree_lines(
        self, structure: list[dict[str, Any]]
    ) -> Generator[str, None, None]:
        """Yields formatted lines for the repository tree."""
        for i, item in enumerate(structure):
            prefix = ""
            for level in range(item["level"]):
                # Check if the parent at this level is the last one
                is_parent_last = True
                # Find the parent of the current item at `level`
                parent_of_item_at_level_idx = -1
                for k in range(i - 1, -1, -1):
                    if structure[k]["level"] == level -1:
                        parent_of_item_at_level_idx = k
                        break

                # Check if this parent is the last among its siblings
                if parent_of_item_at_level_idx != -1:
                    is_parent_last = True # Assume last
                    for j in range(parent_of_item_at_level_idx + 1, len(structure)):
                        if structure[j]["level"] == level -1:
                            is_parent_last = False
                            break
                        if structure[j]["level"] < level -1:
                            break
                if is_parent_last:
                    prefix += "    "
                else:
                    prefix += "│   "


            is_last = True
            for j in range(i + 1, len(structure)):
                if structure[j]["level"] == item["level"]:
                    is_last = False
                    break
                if structure[j]["level"] < item["level"]:
                    break

            connector = "└── " if is_last else "├── "
            if item["type"] == "directory":
                yield f"{prefix}{connector}{item['name']}/"
            else:
                language = item.get("language", "None")
                yield f"{prefix}{connector}{item['name']} ({language})"

                details_prefix = prefix + ("    " if is_last else "│   ")
                details = []
                if item.get("description"):
                    details.append(f"Description: {item['description']}")
                if item.get("developer_consideration"):
                    details.append(
                        f'Developer Consideration: "{item["developer_consideration"]}"'
                    )
                for k, detail in enumerate(details):
                    detail_connector = "└── " if k == len(details) - 1 else "├── "
                    yield f"{details_prefix}{detail_connector}{detail}"

    def _print_tree(self, structure: list[dict[str, Any]]):
        """Prints the repository structure to the console."""
        logger.info("/ (Root Directory)")
        for line in self._format_tree_lines(structure):
            logger.info(line)
        logger.info("└────────────── ")

    def _save_markdown_map(
        self, structure: list[dict[str, Any]], repo_root: str, output_path: str
    ):
        """Saves the repository map to a Markdown file."""
        repo_name = os.path.basename(os.path.normpath(repo_root))
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("# Repository Map\n\n")
                f.write("```markdown\n")
                f.write(f"/ ({repo_name})\n")
                for line in self._format_tree_lines(structure):
                    f.write(f"{line}\n")
                f.write("└────────────── \n")
                f.write("```\n")
        except OSError as e:
            logger.error("Error saving repository map: %s", e)

    def _save_json_map(self, structure: list[dict[str, Any]], output_path: str):
        """Saves the structure to a JSON file."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(structure, f, indent=4)
            logger.info("Pre-enhancement structure saved to '%s'.", output_path)
        except OSError as e:
            logger.error("Error saving JSON structure map: %s", e)

    def _confirm_disclaimer(self) -> bool:
        """Prompts the user to acknowledge the disclaimer."""
        disclaimer_message = (
            "repo-map: Generates a summary of a repository, enhanced with AI.\n"
            "This tool uses .gitignore to exclude files.\n"
            "DISCLAIMER: Files will be sent to the OpenRouter LLM for processing.\n"
            "Proceed? [y/n]: "
        )
        while True:
            try:
                user_input = input(disclaimer_message).strip().lower()
                if user_input in ("y", "yes", ""):
                    return True
                if user_input in ("n", "no"):
                    return False
                print("Invalid input. Please enter 'y' or 'n'.")
            except (EOFError, KeyboardInterrupt):
                print("\nOperation cancelled.")
                return False

    def _get_output_path(self) -> str:
        """Determines the full path for the output Markdown file."""
        assert (
            self.args is not None
        ), "Arguments must be parsed before calling _get_output_path"
        directory_name = os.path.basename(os.path.normpath(self.args.repository_path))
        output_file_name = f"{directory_name}_repo_map.md"
        return os.path.join(self.args.repository_path, output_file_name)

    def _parse_args(self) -> argparse.Namespace:
        """Parses command-line arguments."""
        parser = argparse.ArgumentParser(
            description="repo-map: Generates a structured summary of a software repository, enhanced with AI.",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        parser.add_argument(
            "repository_path", type=str, help="Path to the repository to be summarized."
        )
        parser.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help="Automatically accept the disclaimer without prompting.",
        )
        parser.add_argument(
            "--model",
            type=str,
            default=None,
            help="OpenRouter LLM model name. Overrides .env settings.",
        )
        parser.add_argument(
            "--concurrency",
            type=int,
            default=settings.api_semaphore_limit,
            help="Number of concurrent API calls.",
        )
        return parser.parse_args()


def run_main() -> None:
    """Runs the main async function and handles top-level exceptions."""
    try:
        app: RepoMapApp = RepoMapApp()
        asyncio.run(app.run())
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user. Exiting.")
        sys.exit(130)
    except RuntimeError:
        logger.error("An unexpected runtime error occurred:", exc_info=True)
        sys.exit(1)
