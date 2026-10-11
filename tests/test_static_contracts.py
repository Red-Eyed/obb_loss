"""Exercise real checker rejection instead of assuming annotations enforce usage."""

import subprocess
import sys
from pathlib import Path


def test_pyrefly_rejects_unhandled_result_and_wrong_callback(tmp_path: Path) -> None:
    """Strict checking rejects unchecked success access and incompatible functions."""
    source = tmp_path / "contract.py"
    source.write_text(
        '"""Deliberately invalid checker input."""\n'
        "from torch import Tensor\n"
        "from collections.abc import Callable\n"
        "from obb_loss import Result\n"
        "def unchecked(outcome: Result[int, str]) -> int:\n"
        "    return outcome.value\n"
        "def wrong(preds: int, targets: int) -> str:\n"
        '    return "wrong"\n'
        "callback: Callable[[Tensor, Tensor], Tensor] = wrong\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.typecheck",
            "--config",
            "pyrefly.toml",
            str(source),
        ],
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
        [
            sys.executable,
            "-m",
            "tests.typecheck",
            "--config",
            "pyrefly.toml",
            str(source),
        ],
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


def test_native_shapes_and_module_callbacks(tmp_path: Path) -> None:
    """Check module calls and protocol conformance using the real native stubs."""
    source = tmp_path / "native_contract.py"
    valid = (
        "from __future__ import annotations\n"
        "from typing import TYPE_CHECKING, assert_type\n"
        "from torch import Tensor, nn\n"
        "from typing_extensions import override\n"
        "from obb_loss.losses.l1 import box_l1_loss\n"
        "from obb_loss import Ok, Result, InputError\n"
        "if TYPE_CHECKING:\n"
        "    from shape_extensions import IntVar\n"
        '    N = IntVar("N")\n'
        "class Custom(nn.Module):\n"
        "    @override\n"
        "    def forward(self, preds: Tensor[[N, 6]], targets: Tensor[[N, 6]])"
        " -> Tensor[[N]]:  # pyrefly: ignore[bad-override]\n"
        "        result = box_l1_loss(preds, targets)\n"
        "        assert isinstance(result, Ok)\n"
        "        return result.value\n"
        "def correct(x: Tensor[[3, 6]]) -> Tensor[[3]]:\n"
        "    assert_type(Custom()(x, x), Tensor[[3]])\n"
        "    return Custom()(x, x)\n"
    )
    source.write_text(valid, encoding="utf-8")
    command = [sys.executable, "-m", "tests.typecheck", str(source)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    source.write_text(
        valid
        + "def wrong_shape(x: Tensor[[3, 5]]) -> Result[Tensor, InputError]:\n"
        + "    return box_l1_loss(x, x)\n"
        + "def wrong_rows(x: Tensor[[3, 6]], y: Tensor[[4, 6]]) -> Result[Tensor, InputError]:\n"
        + "    return box_l1_loss(x, y)\n"
        + "def wrong_call(module: Custom) -> Tensor:\n"
        + '    return module("bad", "bad")\n',
        encoding="utf-8",
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    diagnostics = result.stdout + result.stderr
    assert result.returncode != 0
    assert "expected Int[6], got Int[5]" in diagnostics
    assert "expected Int[3], got Int[4]" in diagnostics
    assert "Custom.forward" in diagnostics
