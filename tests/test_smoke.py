"""P0 smoke tests. Prove the package imports and the layout matches PROJECT.md.

Replaced by real tests as each phase lands. The placeholder integration test
that used to live here is gone: `tests/test_integration.py` is the 5-question
end-to-end run the Jenkins `integration` stage was waiting for.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import verisearch

REPO_ROOT = Path(__file__).resolve().parent.parent

# PROJECT.md section 0 ownership split, encoded so a rename is caught in CI.
DHRUBO_PACKAGES = ("env", "rewards", "train")
AYESHA_PACKAGES = ("retrieval", "faithfulness", "data", "eval", "serve")


def test_version_is_set() -> None:
    assert verisearch.__version__ == "0.1.0"


@pytest.mark.parametrize("name", [*DHRUBO_PACKAGES, *AYESHA_PACKAGES])
def test_subpackage_imports(name: str) -> None:
    module = importlib.import_module(f"verisearch.{name}")
    assert module.__doc__, f"verisearch.{name} must document its owner"


def test_package_is_typed() -> None:
    assert (REPO_ROOT / "verisearch" / "py.typed").is_file()


@pytest.mark.parametrize(
    "relpath",
    [
        "pyproject.toml",
        "Jenkinsfile",
        "PROJECT.md",
        ".github/pull_request_template.md",
        "configs/base.yaml",
        "scripts/00_setup_check.py",
    ],
)
def test_required_file_exists(relpath: str) -> None:
    assert (REPO_ROOT / relpath).is_file(), f"missing {relpath}"
