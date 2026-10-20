---
name: code-simplifier
description: >
  Behavior-preserving cleanup pass over repo-map code: the uncommitted diff by
  default, or a named module or the whole of src/repo_map/ and tests/. Modernizes
  Python 3.12 syntax, cuts dead code and needless complexity, consolidates
  duplicated work, and catches efficiency issues. Defects are reported, not
  fixed. Use after a substantive change, before git-wrapup, or when asked to
  simplify, clean up, tighten, or de-slop.
metadata:
  author: cyanheads
  version: "2.0"
  audience: project
  type: workflow
---

# Code Simplifier

Every change must earn its keep: clearer, more correct, or more consistent with the rest of `src/repo_map/`. "I would have written it differently" is not a reason to edit working code.

## 1. Set the scope

- **Diff** (the default when nothing is named): `git status`, then `git diff HEAD` for staged and unstaged changes. List new files with `git ls-files --others --exclude-standard` and read them directly, since no diff shows them. An empty diff with no new files means review the last commit (`git show HEAD`). Don't go looking for other files to improve.
- **Target** (a named module, or "the whole codebase"): the named files, whatever their git state. A whole-codebase pass covers `src/repo_map/` and `tests/`, one module at a time, with `make check` after each. `scripts/build-changelog.ts` is changelog tooling and stays out of scope.

The tree may hold another writer's work. Changes you don't recognize that were modified in the last few minutes (check mtimes) stay out of scope; list them in the report.

## 2. Understand before editing

1. Read each in-scope file in full, not just the hunks.
2. Read `CLAUDE.md` / `AGENTS.md` for conventions, then the sibling modules that define the local pattern (`models.py` for language tables, `cache_manager.py` for persistence, `logging_utils.py` for output).
3. Run `make check` once as the baseline. If it is already red, say so in the report and don't attribute the failure to the pass.

## 3. Review

**Cohesion**
- **Reuse**: an existing helper already does what new code re-implements.
- **Consolidation**: near-duplicate blocks. Unify only when the shared version is simpler than the duplication.
- **Consistency**: naming, error handling, and typing that match the surrounding modules.

**Quality**
- **Dead code**: unreachable branches, unused names, commented-out code, leftover `print` calls that aren't the CLI's real output.
- **Speculative generality**: parameters, options, or branches no caller exercises.
- **Pass-through layers**: a wrapper whose deletion makes the complexity vanish gets inlined.
- **Impossible-state guards**: checks the types or upstream validation already rule out.
- **Type escapes**: `Any` where a concrete type is known, `cast()`, `# type: ignore`. Narrow them; keep one only when it documents a real third-party typing gap.
- **Swallowed errors and masking defaults**: `except Exception: pass`, log-and-continue, `.get(key, {})` standing in for a value that must exist. Let it raise, or raise with a message naming what is missing; preserve chains with `raise X from err`. A per-file failure in `llm_service.py` that logs and returns `AnalysisOutcome.FAILURE` is the design: the file is reported and left uncached, and the run continues.
- **Comment noise**: comments that restate the code or describe removed behavior. Keep docstrings and any comment carrying a why.
- **Weakened tests**: skips, deleted or loosened assertions, removed tests. Report each with `file:line`; don't fix them here.

**Efficiency**
- Repeated `ast.parse()`, file reads, or hashing of the same file.
- Independent async work run sequentially, but confirm independence first: requests in `llm_service.py` are deliberately bounded by the semaphore and rate limiter.
- Unbounded `asyncio.gather` over a caller-sized list.
- TOCTOU existence checks before an operation that can handle the error itself.

## 4. Apply

1. Filter hard. Skip false positives and anything not worth the churn. A finding that would change a contract listed under "Never a simplification" goes in the report instead.
2. Edit only in-scope files, plus the exact lines a finding requires elsewhere (an import of an existing helper, a private name the diff orphaned).
3. One category of change at a time, then re-run `make check`. A simplification that breaks the gate is worse than the verbosity it removed.
4. Formatting belongs to ruff. Never hand-adjust whitespace, quotes, or import order. Hunks `make format` writes outside the scope stay, with a line in the report.
5. Never stage, commit, tag, push, or stash. The pass ends with a dirty tree and a report.

## Never a simplification

These are repo-map's contracts. Changing one is a feature or a fix with its own changelog entry, never a cleanup:

- **CLI surface**: `repository_path`, `-y/--yes`, `--model`, `--concurrency`, exit behavior.
- **Settings**: `OPENROUTER_API_KEY`, `OPENROUTER_MODEL_NAME`, `API_SEMAPHORE_LIMIT`, and `.env` loading.
- **Output**: the `<dir>_repo_map.md` report format and `.repo_map_structure.json`.
- **Cache**: the `.repo-map-cache.db` schema and the reuse rule (hash, model, `ANALYSIS_CONTRACT_VERSION`).
- **Analysis contract**: `SYSTEM_PROMPT`, the message content `_build_messages()` assembles, the request payload in `rate_limited_api_call()`, and the fields `_first_invalid_field()` validates. Any edit here changes what cached analyses mean and requires bumping `ANALYSIS_CONTRACT_VERSION`.
- **Deliberate hardening**: transport constants kept off `Settings` in `config.py`, digest-tagged source markers, and `_printable()` escaping all look removable and are not. Read the comment before touching them.

## Python 3.12 transformations

| Before | After | Why |
| --- | --- | --- |
| `Optional[str]`, `List[str]`, `Dict[str, int]` | `str \| None`, `list[str]`, `dict[str, int]` | Built-in generics and union syntax |
| `T = TypeVar("T")` + `def f(x: T) -> T` | `def f[T](x: T) -> T` | PEP 695 generics |
| `Alias = Union[A, B]` | `type Alias = A \| B` | `type` statement |
| `class C: def __init__(self, a, b): self.a = a ...` | `@dataclass(slots=True) class C: a: str; b: int` | Less boilerplate; `frozen=True` when instances shouldn't mutate |
| `out = []; for x in xs: out.append(f(x))` | `out = [f(x) for x in xs]` | Comprehension |
| `[xs[i:i + n] for i in range(0, len(xs), n)]` | `itertools.batched(xs, n)` | Works on any iterable; yields tuples |
| `os.path.join(d, n)`, `open(p).read()` | `Path(d) / n`, `p.read_text()` | `pathlib` |
| `datetime.utcnow()` | `datetime.now(UTC)` | Deprecated in 3.12; returns naive datetimes |
| `zip(a, b)` where lengths must match | `zip(a, b, strict=True)` | Mismatch raises instead of truncating |
| `m = rx.match(s)` then `if m:` | `if (m := rx.match(s)):` | Removes a throwaway assignment |
| `isinstance` chains that pull attributes | `match x: case Foo(a=a):` | Only where it destructures, never for flat equality chains |
| Sequential `await` on independent I/O | `async with asyncio.TaskGroup() as tg:` | Structured concurrency; keep `gather(..., return_exceptions=True)` where every result is wanted |
| `except Exception as e: log(e)` | `except SpecificError:` with real handling | Catch only what you can handle |

## Leave alone

- Code that works and reads clearly.
- Cosmetic renames.
- Lazy logging arguments (`logger.info("%s", x)`); don't turn them into f-strings.
- Test setup duplicated for isolation; don't DRY tests.
- An abstraction for two similar blocks. Wait for a third, and only if it is simpler.
- `uv.lock`, `CHANGELOG.md`, and `examples/example.md`, which are generated.

## Report

- **Gate**: `make check` before and after.
- **Fixed**: changes grouped by category.
- **Skipped**: findings left alone, with the reason, plus other writers' changes excluded from scope.
- **Defects and recommendations**: bugs and out-of-scope changes, each with `file:line`.

When nothing earned a change, say the code was already clean.
