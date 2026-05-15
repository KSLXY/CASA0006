---
name: casa0006-public-release
description: Use before staging, committing, pushing, or publishing the CASA0006 STATS19 project to GitHub. Enforces public-release hygiene, privacy checks, large-file policy, generated-artifact boundaries, and history-cleanup requirements for this repository.
---

# CASA0006 Public Release

Use this skill before any GitHub sync, branch publication, release, or PR for this project.

## Required Gate

Run the audit before staging and again after staging:

```bash
python3 scripts/publication_audit.py
git status --short --branch
git diff --cached --stat
```

Do not push if the audit fails. Fix the listed files first.

## Public Boundary

Publish:
- source code in `src/`, `app/`, `scripts/`, `tests/`, and `configs/`
- README, license, runtime/dependency files, public data dictionary
- lightweight aggregate metrics and figures in `artifacts/` and `reports/figures/`

Do not publish:
- `data/raw/`, `data/interim/`, `data/processed/`, or `data/sample/`
- trained models such as `artifacts/model.joblib`
- row-level outputs such as `artifacts/error_cases.csv`
- root-level course practicals, solution notebooks, PDFs, ZIP archives, and local images
- secrets, credentials, tokens, API keys, `.env` files, private keys, or local database files

## GitHub Sync Procedure

1. Review changed files with `git status --short --branch`.
2. Run `python3 scripts/publication_audit.py`.
3. Stage explicit files only; avoid `git add .`.
4. Review `git diff --cached --stat` and `git diff --cached --name-only`.
5. Run `pytest`.
6. Run the audit again after staging.
7. If generated/private files were ever committed, clean history before pushing.
8. Push only after the branch is clean and the audit passes.

## History Cleanup

This repository previously tracked model/data artifacts. Before force-pushing a public release, remove these paths from all refs:

```bash
git-filter-repo --force --invert-paths \
  --path artifacts/model.joblib \
  --path artifacts/error_cases.csv \
  --path data/sample/merged_sample.csv
```

If `git-filter-repo` is unavailable, install it or use an equivalent history rewrite. After rewriting, verify with:

```bash
git log --all -- artifacts/model.joblib artifacts/error_cases.csv data/sample/merged_sample.csv
git count-objects -vH
python3 scripts/publication_audit.py
```

History rewriting changes commit hashes and requires a force push. Coordinate before rewriting shared branches.

## Public Project Standards

- GitHub blocks regular Git files over 100 MiB and warns over 50 MiB; keep tracked files below 50 MiB.
- If a real secret was committed, rotate or revoke it before history cleanup.
- Keep README accurate about data provenance, omitted artifacts, and reproduction commands.
- Keep a root `LICENSE` file for public open-source use.
- Prefer GitHub Releases or external storage for large binaries if model distribution is needed later.
