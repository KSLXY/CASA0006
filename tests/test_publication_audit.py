from pathlib import Path

from scripts.publication_audit import forbidden_reason, size_findings, text_secret_findings


def test_forbidden_reason_blocks_private_release_files():
    assert forbidden_reason("artifacts/model.joblib")
    assert forbidden_reason("artifacts/error_cases.csv")
    assert forbidden_reason("data/processed/processed_master.parquet")
    assert forbidden_reason("Practical-01-Getting_Started.ipynb")
    assert forbidden_reason("Practical-01-Getting_Started_SOLUTION.ipynb")


def test_size_findings_fail_large_tracked_file(tmp_path: Path):
    large_file = tmp_path / "large.bin"
    large_file.write_bytes(b"0" * (51 * 1024 * 1024))

    findings = size_findings(["large.bin"], tmp_path)

    assert findings
    assert findings[0].level == "FAIL"


def test_text_secret_findings_detects_secret_like_text(tmp_path: Path):
    secret_file = tmp_path / "settings.py"
    secret_file.write_text("API_KEY = 'not-a-real-value-for-test'\n", encoding="utf-8")

    findings = text_secret_findings(["settings.py"], tmp_path)

    assert findings
    assert findings[0].path == "settings.py"
