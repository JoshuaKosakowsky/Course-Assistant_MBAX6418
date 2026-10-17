"""Repository hygiene tests — enforce the class security policy and scaffold.

Offline and fast (stdlib only, no network, no keys). Mirrors the secret
scan in .github/workflows/ci.yml; these run locally with plain `pytest`
(unit tier, included in the default run).
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Detects OpenAI-style and GitHub token shapes. The literal regex text does
# not match itself (bracket '[' is not in the character classes). The
# tracked .env.example holds only dummy placeholders and is excluded.
KEY_PATTERN = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{16,}|gh[pous]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
)
SCAN_EXCLUDED = {".env.example"}


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


@pytest.mark.unit
def test_env_is_not_tracked():
    assert ".env" not in _tracked_files(), ".env must stay local and gitignored"


@pytest.mark.unit
def test_class_endpoint_file_stays_local():
    assert not any(
        Path(rel).name == "class-endpoints.txt" for rel in _tracked_files()
    ), "class-endpoints.txt may contain credentials and must never be tracked"
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--",
         "class-endpoints.txt", "notes/class-endpoints.txt"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert set(result.stdout.splitlines()) == {
        "class-endpoints.txt", "notes/class-endpoints.txt"
    }


@pytest.mark.unit
def test_no_secret_patterns_in_tracked_files():
    offenders = []
    for rel in _tracked_files():
        if rel in SCAN_EXCLUDED:
            continue
        content = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        if KEY_PATTERN.search(content):
            offenders.append(rel)
    assert not offenders, f"Possible secret pattern in tracked files: {offenders}"


@pytest.mark.unit
def test_env_example_is_placeholder_only():
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "6418" not in text, ".env.example must not contain the real class key"
    assert "cyankiwi" not in text, ".env.example must not contain the real vision model"


@pytest.mark.unit
def test_guidelines_scaffold_present():
    required = [
        "README.md",
        "PLAN.md",
        "AGENTS.md",
        "CONTRIBUTING.md",
        "docs/COMMITS.md",
        "docs/TESTING.md",
        "pyproject.toml",
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/ISSUE_TEMPLATE/bug_report.md",
        ".github/ISSUE_TEMPLATE/feature_request.md",
    ]
    missing = [p for p in required if not (REPO_ROOT / p).exists()]
    assert not missing, f"Missing guideline files: {missing}"
