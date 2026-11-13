"""CLI handler for the repo-map tool."""

import argparse
import asyncio
import json
import logging
import os
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from repo_map.cache_manager import (
    CacheError,
    load_cache,
    load_cache_snapshot,
    prune_cache,
)
from repo_map.config import settings
from repo_map.file_scanner import summarize_repo
from repo_map.llm_service import (
    ANALYSIS_CONTRACT_VERSION,
    AnalysisOutcome,
    FatalOpenRouterError,
    get_llm_descriptions,
    resolve_response_format,
    update_api_semaphore_limit,
)
from repo_map.logging_utils import setup_logging
from repo_map.report_generator import (
    print_tree,
    save_json_map,
    save_markdown_map,
)

logger = logging.getLogger(__name__)

# The consent prompt, filled with `UploadPlan.summary()`. Its claims are made
# true elsewhere and change with that code: what each request carries is
# `llm_service._build_messages()` (the repository layout, notes on related
# files, and the target file's complete source), and which ignore files apply
# is `file_scanner._IgnoreRules` (the defaults, every `.gitignore` from the
# work-tree top down, `.git/info/exclude`, never the global excludes file).
# README.md's "What gets sent" section says the same; keep all three in step.
DISCLOSURE_PROMPT = (
    "{summary}\n"
    "Ignore rules come from repo-map's defaults, the repository's .gitignore files at every level, and .git/info/exclude; git's global excludes file is not read.\n"
    "Run with --dry-run to list the files. Proceed? [y/N]: "
)


def _files(count: int) -> str:
    return "1 file" if count == 1 else f"{count} files"


@dataclass(frozen=True, slots=True)
class UploadPlan:
    """What a run sends: the files with no reusable analysis, and how many keep one.

    ``pending`` holds the scanned entries themselves, sorted by
    repository-relative path, so the analysis pass fills in the same entries
    the report renders, and sends nothing the plan did not count.
    """

    pending: tuple[dict[str, Any], ...]
    kept: int

    @property
    def size_bytes(self) -> int:
        """Source bytes the pending files send: exactly the bytes each hash covers."""
        return sum(item["size_bytes"] for item in self.pending)

    def summary(self, model: str) -> str:
        """State what leaves the machine, in the one line the prompt, ``-y``, and ``--dry-run`` share."""
        kept = (
            "1 file with a current cached analysis keeps it; its source is not sent."
            if self.kept == 1
            else f"{self.kept} files with a current cached analysis keep it; their source is not sent."
        )
        return f"repo-map will send the full text of {_files(len(self.pending))} ({self.size_bytes / 1024:.1f} KiB) to OpenRouter ({model}), plus the repository's file and directory names and short notes on related files. {kept}"


def plan_upload(
    connection: sqlite3.Connection, structure: list[dict[str, Any]], model_name: str
) -> UploadPlan:
    """Split the eligible files into those to send and those whose analysis is reused.

    A stored analysis is reusable only when the inputs that produced it still
    hold: the source hash, the model, and the analysis contract revision. The
    same unchanged file is therefore reanalyzed when the run selects another
    model or the contract is bumped, and a row written before those columns
    existed carries `NULL` for both and can never match. `model_name` is the
    model this run analyzes with, and is required: defaulting it would compare
    a row against a model the run never selected.

    Eligibility is the scanner's shared source policy, not parser output, so
    class-only modules and data or configuration files are candidates too.
    Rows are looked up by relative key in ``connection``, which before consent
    is the in-memory copy from `load_cache_snapshot()`.
    """
    analysis_inputs = (model_name, ANALYSIS_CONTRACT_VERSION)
    cursor = connection.cursor()
    pending: list[dict[str, Any]] = []
    kept = 0
    for item in structure:
        if item["type"] != "file" or not item.get("source_eligible"):
            continue
        cursor.execute(
            "SELECT hash, model, contract_version FROM cache WHERE path = ?",
            (item["rel_path"],),
        )
        if cursor.fetchone() == (item.get("hash", ""), *analysis_inputs):
            kept += 1
        else:
            pending.append(item)
    pending.sort(key=lambda item: item["rel_path"])
    return UploadPlan(pending=tuple(pending), kept=kept)


class RepoMapApp:
    """Encapsulates the CLI application lifecycle."""

    def __init__(self) -> None:
        self.args: argparse.Namespace | None = None
        self.cache_conn: sqlite3.Connection | None = None

    async def run(self) -> int:
        """Execute the CLI flow and return the process exit status.

        The repository is scanned before anything is sent, against an in-memory
        copy of its cache, so the consent prompt states this run's numbers and
        nothing reaches the target before the user agrees. The real cache opens
        after consent, and the analysis pass sends exactly the files the prompt
        counted. ``--dry-run`` lists those files and stops.
        """
        setup_logging()
        self.args = args = self._parse_args()

        if not args.dry_run and not settings.has_api_key():
            logger.error("Error: OPENROUTER_API_KEY environment variable not set.")
            return 1

        try:
            update_api_semaphore_limit(args.concurrency)
        except ValueError as exc:
            logger.error("Invalid concurrency value: %s", exc)
            return 1

        if not os.path.isdir(args.repository_path):
            logger.error("Error: %s is not a valid directory", args.repository_path)
            return 1

        model_name = args.model or settings.openrouter_model_name
        try:
            structure, plan = self._scan(model_name)
        except CacheError as exc:
            logger.error("%s", exc)
            return 1

        if args.dry_run:
            print(plan.summary(model_name))
            for item in plan.pending:
                print(f"{item['size_bytes']}\t{item['rel_path']}")
            return 0

        if plan.pending:
            summary = plan.summary(model_name)
            if args.yes:
                logger.info("%s", summary)
            elif not self._confirm_disclaimer(
                DISCLOSURE_PROMPT.format(summary=summary)
            ):
                logger.warning("Operation cancelled by the user.")
                return 0

        try:
            cache_conn = self.cache_conn = load_cache(args.repository_path)
        except CacheError as exc:
            logger.error("%s", exc)
            return 1

        try:
            return await self._process_repository(structure, plan, model_name)
        finally:
            cache_conn.close()

    def _scan(self, model_name: str) -> tuple[list[dict[str, Any]], UploadPlan]:
        """Scan the target and plan its upload against an in-memory copy of the cache.

        Raises:
            CacheError: The cache file exists but cannot be read.
        """
        assert self.args is not None
        # Under --dry-run, stdout is the listing alone.
        if not self.args.dry_run:
            logger.info("Generating repository summary...")
        with closing(load_cache_snapshot(self.args.repository_path)) as snapshot:
            structure = summarize_repo(self.args.repository_path, snapshot)
            return structure, plan_upload(snapshot, structure, model_name)

    async def _process_repository(
        self, structure: list[dict[str, Any]], plan: UploadPlan, model_name: str
    ) -> int:
        """Analyze the planned files, then write the tree and the report.

        Returns 0 only when every pending file was analyzed and the report was
        written. A run-fatal OpenRouter error writes neither tree nor report;
        the analyses that finished before it are already cached.
        """
        assert self.args is not None
        pre_enhanced_path = os.path.join(
            self.args.repository_path, ".repo_map_structure.json"
        )
        save_json_map(structure, pre_enhanced_path)

        logger.info(
            "Enhancing repository summary with descriptions using OpenRouter LLM..."
        )
        try:
            unanalyzed = await self._enhance_summary_with_llm(
                structure, plan.pending, model_name
            )
        except FatalOpenRouterError as exc:
            logger.error("Run stopped: %s", exc)
            return 1
        self._prune_orphaned_cache_entries(structure)

        logger.info("\nUpdated Repository Map:")
        print_tree(structure)

        status = 0
        output_path = self._get_output_path()
        try:
            save_markdown_map(structure, self.args.repository_path, output_path)
        except OSError as exc:
            logger.error(
                "Cannot write the repository map to '%s': %s", output_path, exc
            )
            status = 1
        else:
            logger.info(
                "Your repo-map has been saved to '%s'.", os.path.basename(output_path)
            )

        if unanalyzed:
            logger.error(
                "%s",
                "1 file was not analyzed; rerun repo-map to retry it."
                if unanalyzed == 1
                else f"{unanalyzed} files were not analyzed; rerun repo-map to retry them.",
            )
            status = 1
        return status

    async def _enhance_summary_with_llm(
        self,
        structure: list[dict[str, Any]],
        pending: Sequence[dict[str, Any]],
        model_name: str,
    ) -> int:
        """Analyze each pending file concurrently, caching each success as it lands.

        Returns how many files were not analyzed. A failure in one file's
        analysis stays with that file, whatever raised it, so its peers still
        finish and cache. `FatalOpenRouterError` is the exception: every
        remaining request would hit it too, so the first one cancels the
        analyses still in flight and propagates, and the successes recorded
        before it stay cached.
        """
        if not pending:
            logger.info("No new or modified files to enhance. All up to date.")
            return 0

        response_format = await resolve_response_format(model_name)

        async def enhance_file(
            file_data: dict[str, Any],
        ) -> tuple[dict[str, Any], AnalysisOutcome]:
            """Analyze one file; only a run-fatal error escapes."""
            try:
                outcome = await get_llm_descriptions(
                    structure,
                    file_data,
                    model=model_name,
                    response_format=response_format,
                )
            except FatalOpenRouterError:
                raise
            except Exception:
                logger.error(
                    "Unexpected error analyzing %s:", file_data["path"], exc_info=True
                )
                return file_data, AnalysisOutcome.FAILURE
            return file_data, outcome

        in_flight = {asyncio.create_task(enhance_file(file)) for file in pending}
        unanalyzed = 0
        with tqdm(total=len(pending), desc="Enhancing files", ncols=100) as progress:
            try:
                while in_flight:
                    done, in_flight = await asyncio.wait(
                        in_flight, return_when=asyncio.FIRST_COMPLETED
                    )
                    fatal: FatalOpenRouterError | None = None
                    for task in done:
                        try:
                            file, outcome = task.result()
                        except FatalOpenRouterError as exc:
                            fatal = fatal or exc
                            continue
                        if outcome is AnalysisOutcome.SUCCESS:
                            tqdm.write(f"Processed: {file['name']}")
                            self._update_cache_for_file(file)
                        else:
                            tqdm.write(f"Not analyzed: {file['name']}")
                            unanalyzed += 1
                        progress.update()
                    if fatal is not None:
                        raise fatal
            finally:
                # Tasks remain only when the pass is ending early; none may
                # outlive it.
                for task in in_flight:
                    task.cancel()
                await asyncio.gather(*in_flight, return_exceptions=True)
        return unanalyzed

    def _prune_orphaned_cache_entries(self, structure: list[dict[str, Any]]) -> None:
        """Drop cache rows that no file in this scan claims.

        Bounds the database to the current tree: rows for deleted files go, and
        so do rows written under the pre-portable absolute-path key, which no
        relative key can match.
        """
        assert self.cache_conn is not None
        removed = prune_cache(
            self.cache_conn,
            {item["rel_path"] for item in structure if item["type"] == "file"},
        )
        if removed:
            logger.info("Removed %s cache entries with no matching file.", removed)

    def _update_cache_for_file(self, file_data: dict[str, Any]) -> None:
        """Persist the latest LLM metadata for a file in the cache.

        `INSERT OR REPLACE` on the relative key also supersedes whatever the row
        held before, so an upgrade never has to rewrite a key in place.

        `model` and `contract_version` are read without a fallback: only a
        validated analysis reaches here, and `get_llm_descriptions()` stamps
        both onto the file before reporting that success. A default would write
        an identity no run produced and pin the row against reanalysis.
        """
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
                refactoring_suggestions,
                security_assessment,
                model,
                contract_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_data["rel_path"],
                file_data.get("hash", ""),
                file_data.get("description", ""),
                file_data.get("developer_consideration", ""),
                json.dumps(file_data.get("imports", [])),
                json.dumps(file_data.get("functions", [])),
                file_data.get("maintenance_flag", "Unknown"),
                file_data.get("critical_dependencies", "{}"),
                file_data.get("architectural_role", "Unknown"),
                file_data.get("refactoring_suggestions", "None"),
                file_data.get("security_assessment", "None"),
                file_data["model"],
                file_data["contract_version"],
            ),
        )
        self.cache_conn.commit()

    def _confirm_disclaimer(self, prompt: str) -> bool:
        """Show the rendered upload disclosure and ask for consent.

        Consent must be explicit: an empty answer takes the `[y/N]` default and
        declines, so a stray newline on stdin cannot approve an upload.
        """
        while True:
            try:
                user_input = input(prompt).strip().lower()
                if user_input in ("y", "yes"):
                    return True
                if user_input in ("", "n", "no"):
                    return False
                print("Invalid input. Please enter 'y' or 'n'.")
            except (EOFError, KeyboardInterrupt):
                print("\nOperation cancelled.")
                return False

    def _get_output_path(self) -> str:
        """Resolve the output path for the Markdown report."""
        assert self.args is not None
        directory_name = os.path.basename(os.path.abspath(self.args.repository_path))
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
            help="Send without asking for confirmation; the upload summary is still logged.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="List the files a run would send, with their sizes in bytes, then exit\nwithout sending anything or writing to the repository. Needs no API key.",
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
