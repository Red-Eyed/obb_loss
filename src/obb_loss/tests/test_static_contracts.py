"""Exercise real checker rejection instead of assuming annotations enforce usage."""

import subprocess
from pathlib import Path


def test_pyrefly_rejects_unhandled_result_and_wrong_callback(tmp_path: Path) -> None:
    """Strict checking rejects unchecked success access and incompatible functions."""
    source = tmp_path / "contract.py"
    source.write_text(
        '"""Deliberately invalid checker input."""\n'
        "from torch import Tensor\n"
        "from obb_loss.result import Result\n"
        "from obb_loss.protocols import BoxFunction\n"
        "def unchecked(outcome: Result[int, str]) -> int:\n"
        "    return outcome.value\n"
        "def wrong(preds: int, targets: int) -> str:\n"
        '    return "wrong"\n'
        "callback: BoxFunction = wrong\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["pyrefly", "check", "--config", "pyrefly.toml", str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    diagnostics = result.stdout + result.stderr
    assert result.returncode != 0
    assert "missing-attribute" in diagnostics
    assert "bad-assignment" in diagnostics


def test_strict_configs_reject_any_and_missing_annotations(tmp_path: Path) -> None:
    """Ensure strict Pyrefly settings apply to explicit files beyond default discovery."""
    source = tmp_path / "strictness.py"
    source.write_text(
        "from typing import Any\ndef bad(value: Any):\n    return value\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["pyrefly", "check", "--config", "pyrefly.toml", str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    diagnostics = result.stdout + result.stderr
    assert result.returncode != 0
    assert "explicit-any" in diagnostics
    assert "unannotated-return" in diagnostics


def test_ruff_rejects_unannotated_code(tmp_path: Path) -> None:
    """Strict linting reports annotations and undefined names on representative code."""
    source = tmp_path / "lint_violation.py"
    source.write_text("def bad(value):\n    return missing + value\n", encoding="utf-8")
    result = subprocess.run(
        ["ruff", "check", "--config", "ruff.toml", str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "ANN001" in result.stdout
    assert "F821" in result.stdout


def test_pytest_rejects_unknown_markers(tmp_path: Path) -> None:
    """Strict markers reject an unregistered tag rather than silently dropping coverage."""
    source = tmp_path / "test_unknown_marker.py"
    source.write_text(
        "import pytest\n@pytest.mark.not_registered\ndef test_bad():\n    pass\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["pytest", "--collect-only", "-c", "pytest.ini", str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "not_registered" in result.stdout
