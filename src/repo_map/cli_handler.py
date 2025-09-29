"""CLI handler for the repo-map tool."""

import argparse
import json
import logging
import os
import sqlite3
import sys
from typing import Any, Optional

from tqdm import tqdm

from repo_map.cache_manager import load_cache
from repo_map.config import settings
from repo_map.file_scanner import summarize_repo
from repo_map.llm_service import get_llm_descriptions, update_api_semaphore_limit
from repo_map.logging_utils import setup_logging
from repo_map.report_generator import (
    print_tree,
    save_json_map,
    save_markdown_map,
)

logger = logging.getLogger(__name__)


class RepoMapApp:
    """Encapsulates the CLI application lifecycle."""

    def __init__(self) -> None:
        self.args: Optional[argparse.Namespace] = None
        self.cache_conn: Optional[sqlite3.Connection] = None

    async def run(self) -> None:
        """Execute the CLI flow."""
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

    async def _process_repository(self) -> None:
        """Orchestrate scanning, LLM enhancement, and persistence."""
        assert self.args is not None
        assert self.cache_conn is not None
        logger.info("Generating repository summary...")
        summary = summarize_repo(self.args.repository_path, self.cache_conn)

        pre_enhanced_path = os.path.join(
            self.args.repository_path, ".repo_map_structure.json"
        )
        save_json_map(summary, pre_enhanced_path)

        logger.info(
            "Enhancing repository summary with descriptions using OpenRouter LLM..."
        )
        model_name = self.args.model or settings.openrouter_model_name
        await self._enhance_summary_with_llm(summary, model_name)

        logger.info("\nUpdated Repository Map:")
        print_tree(summary)

        output_path = self._get_output_path()
        save_markdown_map(summary, self.args.repository_path, output_path)
        logger.info(
            "Your repo-map has been saved to '%s'.", os.path.basename(output_path)
        )

    async def _enhance_summary_with_llm(
        self, structure: list[dict[str, Any]], model_name: str
    ) -> None:
        """Enhance file entries with LLM-produced metadata."""
        files_to_process = self._get_files_to_process(structure)

        tasks = [
            file for file in structure if file["path"] in files_to_process
        ]

        if not tasks:
            logger.info("No new or modified files to enhance. All up to date.")
            return

        for file in tqdm(tasks, desc="Enhancing files", ncols=100):
            tqdm.write(f"Processing: {file['name']}")
            await get_llm_descriptions(structure, file, model=model_name)
            self._update_cache_for_file(file)

    def _get_files_to_process(self, structure: list[dict[str, Any]]) -> set[str]:
        """Determine which files need LLM enhancement based on cache state."""
        assert self.cache_conn is not None
        cursor = self.cache_conn.cursor()
        files_to_process: set[str] = set()
        for item in structure:
            if item["type"] == "file" and (item.get("imports") or item.get("functions")):
                cursor.execute("SELECT hash FROM cache WHERE path = ?", (item["path"],))
                row = cursor.fetchone()
                if not row or row[0] != item.get("hash", ""):
                    files_to_process.add(item["path"])
        return files_to_process

    def _update_cache_for_file(self, file_data: dict[str, Any]) -> None:
        """Persist the latest LLM metadata for a file in the cache."""
        assert self.cache_conn is not None
        cursor = self.cache_conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO cache (
                path,
                hash,
                description,
                developer_consideration,
                imports,
                functions,
                maintenance_flag,
                critical_dependencies,
                architectural_role,
                code_quality_score,
                refactoring_suggestions,
                security_assessment
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_data["path"],
                file_data.get("hash", ""),
                file_data.get("description", ""),
                file_data.get("developer_consideration", ""),
                json.dumps(file_data.get("imports", [])),
                json.dumps(file_data.get("functions", [])),
                file_data.get("maintenance_flag", "Unknown"),
                file_data.get("critical_dependencies", "{}"),
                file_data.get("architectural_role", "Unknown"),
                file_data.get("code_quality_score", 0),
                file_data.get("refactoring_suggestions", "None"),
                file_data.get("security_assessment", "None"),
            ),
        )
        self.cache_conn.commit()

    def _confirm_disclaimer(self) -> bool:
        """Prompt the user to acknowledge the LLM disclaimer."""
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
        """Resolve the output path for the Markdown report."""
        assert self.args is not None
        directory_name = os.path.basename(os.path.normpath(self.args.repository_path))
        output_file_name = f"{directory_name}_repo_map.md"
        return os.path.join(self.args.repository_path, output_file_name)

    def _parse_args(self) -> argparse.Namespace:
        """Parse CLI arguments."""
        parser = argparse.ArgumentParser(
            description=(
                "repo-map: Generates a structured summary of a software repository, "
                "enhanced with AI."
            ),
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
