---
name: release-and-publish
description: >
  Publish a repo-map release that git-wrapup has already committed and tagged:
  re-run the gate, build and smoke-test the distributions, push main and the
  vX.Y.Z tag, upload to PyPI with uv publish, verify the version installs,
  create the GitHub release from the tag, then comment on and close the issues
  it fixed. Use only when the maintainer explicitly asks to release, publish,
  or push a tagged version. Halts with a partial-state report on any failure it
  cannot retry.
metadata:
  author: cyanheads
  version: "1.0"
  audience: project
  type: workflow
---

# Release and Publish

This skill starts where `git-wrapup` stops. It never authors commits, changelog entries, or tags; if one is missing or wrong, halt and send the work back to `git-wrapup`.

## Preconditions

- The maintainer explicitly asked for the release.
- On `main`, working tree clean.
- `pyproject.toml` `[project] version` is `X.Y.Z`, and `changelog/<major.minor>.x/X.Y.Z.md` exists.
- HEAD is `chore(release): X.Y.Z — <theme>`, and the annotated tag `vX.Y.Z` points at it.

## Failure protocol

Network steps (push, upload, PyPI checks, `gh` calls) retry a transient failure twice, about 5 s and then 15 s apart. Transient means connection reset or refused, a timeout, DNS failure, or HTTP 502, 503, or 504. Every other failure halts on the first non-zero exit.

These mean a step already succeeded on an earlier run; treat them as success and move on:

- `git push`: `Everything up-to-date`
- `uv publish`: PyPI accepts a re-upload of a byte-identical file without error
- `gh release create`: an `already_exists` validation error, once `gh release view vX.Y.Z` confirms the release
- An issue that is already closed

When halting, report the failed step, its exact error output, the retry count, and the partial state: whether `main` and the tag are pushed, whether PyPI has the files, whether the GitHub release exists, and which issues are closed. A re-run then skips what already succeeded.

Never, under any failure:
- force-push, or rebase commits a tag points at;
- delete, move, or recreate a tag that has been pushed;
- delete or yank a PyPI release. A broken release is fixed forward with a new patch version.

## Steps

### 1. Sanity check

```bash
uv version --short                                 # X.Y.Z
git branch --show-current                          # main
git status --porcelain                             # empty
git log -1 --format=%s                             # chore(release): X.Y.Z — ...
git cat-file -t vX.Y.Z                             # tag (annotated, not lightweight)
test "$(git rev-parse 'vX.Y.Z^{commit}')" = "$(git rev-parse HEAD)" && echo tag-at-head
git fetch origin
git merge-base --is-ancestor origin/main HEAD && echo fast-forward
```

Any mismatch halts. If `origin/main` is not an ancestor of HEAD, `main` moved upstream after the wrap-up; report it rather than rebasing, since a rebase would leave the tag on a commit that is no longer on the branch.

### 2. Gate

```bash
make check
```

Read the full output. Any failure or warning halts.

### 3. Build and smoke-test

`dist/` can hold older builds, so every release builds into its own directory, and nothing ever publishes `dist/*`.

```bash
curl -s -m 15 -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/repo-map/X.Y.Z/json
```

- `404`: nothing is on PyPI yet. Build fresh: `uv build --out-dir dist/X.Y.Z --clear`.
- `200`: an earlier run already uploaded files. Reuse `dist/X.Y.Z/` unchanged, because a rebuild isn't guaranteed byte-identical and PyPI rejects a different file under a name it already has. If `dist/X.Y.Z/` is missing, halt and report.

`dist/X.Y.Z/` must hold exactly `repo_map-X.Y.Z.tar.gz` and `repo_map-X.Y.Z-py3-none-any.whl`, plus the `.gitignore` uv writes there. Then run the built wheel, not the source tree:

```bash
uvx --from dist/X.Y.Z/repo_map-X.Y.Z-py3-none-any.whl repo-map --help
```

A failure here halts with nothing public yet.

### 4. Push

```bash
git push origin main
git push origin vX.Y.Z
```

`main` first, then the tag. A rejected push halts.

### 5. Upload to PyPI

The token lives in the maintainer's macOS Keychain. Confirm it exists without printing it:

```bash
security find-generic-password -s poetry-repository-pypi -a __token__ >/dev/null 2>&1 && echo token-present
```

If that prints nothing, stop and ask the maintainer how to provide credentials. Never prompt for the token, paste it, echo it, export it in a separate command, or write it to a file. Read it inline, scoped to the one command:

```bash
UV_PUBLISH_TOKEN="$(security find-generic-password -s poetry-repository-pypi -a __token__ -w)" uv publish dist/X.Y.Z/*
```

`400 File already exists` means PyPI holds a different file under that name. Halt; that version can't be replaced.

### 6. Verify on PyPI

Run each loop as one foreground command, with a command timeout above its total wait:

```bash
for i in $(seq 1 8); do
  [ "$(curl -s -m 15 -o /dev/null -w '%{http_code}' https://pypi.org/pypi/repo-map/X.Y.Z/json)" = 200 ] && echo served && break
  sleep 15
done

for i in $(seq 1 6); do
  uvx --refresh-package repo-map repo-map@X.Y.Z --help >/dev/null && echo installs && break
  sleep 15
done
```

The JSON API usually answers first; the install index can trail it by a minute or two. If either loop ends without its success line, run it once more before halting. A successful upload exit code alone is not proof that the version installs.

### 7. GitHub release

```bash
gh release create vX.Y.Z --title "repo-map X.Y.Z" --notes-from-tag --verify-tag
```

The tag message becomes the release notes. Attach no assets; PyPI hosts the distributions.

### 8. Close fixed issues

Collect the candidates from the commit stack and the changelog entry:

```bash
git log "$(git describe --tags --abbrev=0 'vX.Y.Z^')"..vX.Y.Z --format='%h %s%n%b'
```

along with the issue links in `changelog/<major.minor>.x/X.Y.Z.md`. Close only issues this release fully fixes; one it only touches gets a comment and stays open.

For each fixed issue, write the comment with the file-writing tool to a file outside the repository: what changed in one or two sentences, the commit's short SHA, and a link to `https://github.com/cyanheads/repo-map/releases/tag/vX.Y.Z`. On an issue someone other than the maintainer opened, start with a greeting and thank them for the report. Don't ask the reporter to test or confirm anything.

```bash
gh issue comment <N> --repo cyanheads/repo-map --body-file <comment-file>
gh issue close <N> --repo cyanheads/repo-map --reason completed
```

### 9. Report

- PyPI: `https://pypi.org/project/repo-map/X.Y.Z/`
- GitHub release: `https://github.com/cyanheads/repo-map/releases/tag/vX.Y.Z`
- The `uvx repo-map@X.Y.Z --help` result
- Issues closed, and issues commented but left open

## Checklist

- [ ] Release explicitly requested; on `main`, tree clean, release commit and annotated tag at HEAD, `origin/main` an ancestor
- [ ] `make check` green
- [ ] Distributions built into `dist/X.Y.Z/` (or reused after a partial upload); built wheel runs `--help`
- [ ] `main` pushed, then `vX.Y.Z`; no force-push
- [ ] `uv publish dist/X.Y.Z/*` with the Keychain token read inline, never printed or stored
- [ ] PyPI JSON returns 200 and `uvx repo-map@X.Y.Z --help` runs
- [ ] GitHub release `repo-map X.Y.Z` created from the tag, no assets
- [ ] Fixed issues commented and closed; partial fixes commented only
- [ ] Report delivered, or a partial-state report on halt
