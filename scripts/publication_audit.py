from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


WARN_SIZE_BYTES = 25 * 1024 * 1024
FAIL_SIZE_BYTES = 50 * 1024 * 1024

SECRET_RE = re.compile(
    r"(?i)("
    r"(api[_-]?key|secret|token|password|passwd|authorization|credential|private[_-]?key)"
    r"\s*[:=]\s*['\"][^'\"]{8,}['\"]|"
    r"bearer\s+[A-Za-z0-9._~+/=-]{20,}|"
    r"aws_access_key_id\s*[:=]\s*['\"]?[A-Z0-9]{16,}|"
    r"postgres://[^\s]+|mysql://[^\s]+|mongodb://[^\s]+|"
    r"-----BEGIN (RSA|OPENSSH|DSA|EC|PRIVATE) KEY-----|"
    r"sk-[A-Za-z0-9]{20,}"
    r")"
)

FORBIDDEN_EXACT = {
    "artifacts/model.joblib": "trained model artifact must not be tracked",
    "artifacts/error_cases.csv": "row-level error cases must not be tracked",
    "data/sample/merged_sample.csv": "sample data is not approved for public tracking",
}

FORBIDDEN_SUFFIXES = {
    ".env",
    ".pem",
    ".key",
    ".p12",
    ".sqlite",
    ".sqlite3",
}

COURSE_PATTERNS = (
    re.compile(r"^Practical", re.IGNORECASE),
    re.compile(r"SOLUTION", re.IGNORECASE),
    re.compile(r"^Template_submission", re.IGNORECASE),
)

SECRET_SCAN_EXEMPT = {
    "scripts/publication_audit.py",
    "tests/test_publication_audit.py",
}


@dataclass(frozen=True)
class Finding:
    level: str
    path: str
    message: str


def run_git(args: list[str], root: Path) -> str:
    result = subprocess.run(["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout


def git_paths(args: list[str], root: Path) -> list[str]:
    output = run_git(args, root)
    if "\0" in output:
        return [p for p in output.split("\0") if p]
    return [p for p in output.splitlines() if p]


def is_placeholder_allowed(path: str) -> bool:
    return path in {"data/raw/.gitkeep", "data/interim/.gitkeep", "data/processed/.gitkeep"}


def forbidden_reason(path: str) -> str | None:
    normalized = path.replace(os.sep, "/")
    name = Path(normalized).name
    suffix = Path(normalized).suffix.lower()
    if normalized in FORBIDDEN_EXACT:
        return FORBIDDEN_EXACT[normalized]
    if suffix in FORBIDDEN_SUFFIXES or name.startswith(".env"):
        return "secret, private key, or local database file must not be tracked"
    if normalized.startswith(("data/raw/", "data/interim/", "data/processed/")) and not is_placeholder_allowed(normalized):
        return "raw/interim/processed data must stay local"
    if normalized.startswith("data/sample/"):
        return "sample data requires explicit anonymization approval before tracking"
    if normalized.startswith("artifacts/") and suffix in {".joblib", ".pkl", ".pickle"}:
        return "model binary must not be tracked in regular Git"
    if normalized.endswith(".zip") and "/" not in normalized:
        return "root-level archive is treated as local/private coursework or source data"
    if normalized.endswith(".pdf") and "/" not in normalized:
        return "root-level PDF is treated as local/private coursework"
    if normalized.endswith(".ipynb") and "/" not in normalized and any(pattern.search(name) for pattern in COURSE_PATTERNS):
        return "coursework notebook is not part of the public release"
    if name in {"Cluster-House-Prices-Raw.png", "biplot_2d.png", "casa0006_individual_work.ipynb"}:
        return "local coursework/reference file must not be tracked"
    return None


def size_findings(paths: list[str], root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        full_path = root / path
        if not full_path.exists() or not full_path.is_file():
            continue
        size = full_path.stat().st_size
        if size >= FAIL_SIZE_BYTES:
            findings.append(Finding("FAIL", path, f"tracked file is {size / (1024 * 1024):.1f} MiB; public repo limit is 50 MiB"))
        elif size >= WARN_SIZE_BYTES:
            findings.append(Finding("WARN", path, f"tracked file is {size / (1024 * 1024):.1f} MiB; consider release storage"))
    return findings


def forbidden_findings(paths: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        reason = forbidden_reason(path)
        if reason:
            findings.append(Finding("FAIL", path, reason))
    return findings


def text_secret_findings(paths: list[str], root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        if path in SECRET_SCAN_EXEMPT:
            continue
        full_path = root / path
        if not full_path.exists() or not full_path.is_file():
            continue
        if full_path.stat().st_size > 2 * 1024 * 1024:
            continue
        try:
            data = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(data.splitlines(), start=1):
            if SECRET_RE.search(line):
                findings.append(Finding("FAIL", path, f"possible secret-like text at line {line_no}"))
                break
    return findings


def untracked_findings(root: Path) -> list[Finding]:
    output = run_git(["status", "--porcelain=v1", "--untracked-files=all"], root)
    findings: list[Finding] = []
    for line in output.splitlines():
        if not line.startswith("?? "):
            continue
        path = line[3:].strip()
        reason = forbidden_reason(path)
        if reason:
            findings.append(Finding("FAIL", path, f"untracked private/publication-risk file is visible to Git: {reason}"))
    return findings


def required_file_findings(root: Path) -> list[Finding]:
    required = {
        "README.md": "README is required for public GitHub use",
        "LICENSE": "LICENSE is required for open-source publication",
        ".codex/skills/casa0006-public-release/SKILL.md": "project release skill is required",
    }
    return [Finding("FAIL", path, message) for path, message in required.items() if not (root / path).exists()]


def audit(root: Path) -> list[Finding]:
    tracked = git_paths(["ls-files", "-z"], root)
    findings: list[Finding] = []
    findings.extend(required_file_findings(root))
    findings.extend(forbidden_findings(tracked))
    findings.extend(size_findings(tracked, root))
    findings.extend(text_secret_findings(tracked, root))
    findings.extend(untracked_findings(root))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CASA0006 repository before public GitHub sync.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root.")
    args = parser.parse_args()
    root = args.root.resolve()

    findings = audit(root)
    failures = [finding for finding in findings if finding.level == "FAIL"]
    warnings = [finding for finding in findings if finding.level == "WARN"]

    for finding in findings:
        print(f"{finding.level}: {finding.path}: {finding.message}")

    if failures:
        print(f"\nPublication audit failed: {len(failures)} failure(s), {len(warnings)} warning(s).")
        return 1
    print(f"Publication audit passed: 0 failures, {len(warnings)} warning(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
