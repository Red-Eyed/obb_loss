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
        [
            sys.executable,
            "-m",
            "tools.typecheck",
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
            "tools.typecheck",
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


def test_set_criterion_requires_records_and_result_narrowing(tmp_path: Path) -> None:
    """Reject raw tensor arguments and unchecked loss access in the high-level criterion."""
    source = tmp_path / "set_contract.py"
    source.write_text(
        "import obb_loss as obb\n"
        "from torch import Tensor\n"
        "def misuse(criterion: obb.SetCriterion, preds: obb.Predictions, "
        "targets: obb.Targets, boxes: Tensor) -> Tensor:\n"
        "    criterion.forward(boxes, boxes)\n"
        "    return criterion.forward(preds, targets).total\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.typecheck",
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
    assert "bad-argument-type" in diagnostics
    assert "missing-attribute" in diagnostics


def test_native_shapes_and_module_callbacks(tmp_path: Path) -> None:
    """Check module calls and protocol conformance using the real native stubs."""
    source = tmp_path / "native_contract.py"
    valid = (
        "from __future__ import annotations\n"
        "from typing import TYPE_CHECKING, assert_type\n"
        "from torch import Tensor, nn\n"
        "from typing_extensions import override\n"
        "from obb_loss.protocols import BoxFunction\n"
        "from obb_loss.losses.l1 import box_l1_loss\n"
        "if TYPE_CHECKING:\n"
        "    from shape_extensions import IntVar\n"
        '    N = IntVar("N")\n'
        "class Custom(nn.Module):\n"
        "    @override\n"
        "    def forward(self, preds: Tensor[[N, 6]], targets: Tensor[[N, 6]])"
        " -> Tensor[[N]]:  # pyrefly: ignore[bad-override]\n"
        "        return box_l1_loss(preds, targets)\n"
        "callback: BoxFunction = Custom()\n"
        "def correct(x: Tensor[[3, 6]]) -> Tensor[[3]]:\n"
        "    assert_type(Custom()(x, x), Tensor[[3]])\n"
        "    return callback(x, x)\n"
    )
    source.write_text(valid, encoding="utf-8")
    command = [sys.executable, "-m", "tools.typecheck", str(source)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    source.write_text(
        valid
        + "def wrong_shape(x: Tensor[[3, 5]]) -> Tensor:\n"
        + "    return box_l1_loss(x, x)\n"
        + "def wrong_rows(x: Tensor[[3, 6]], y: Tensor[[4, 6]]) -> Tensor:\n"
        + "    return box_l1_loss(x, y)\n"
        + "def wrong_call(module: Custom) -> Tensor:\n"
        + '    return module("bad", "bad")\n'
        + "class Wrong(nn.Module):\n"
        + "    @override\n"
        + "    def forward(self, value: int) -> str:  # pyrefly: ignore[bad-override]\n"
        + "        return str(value)\n"
        + "bad_callback: BoxFunction = Wrong()\n",
        encoding="utf-8",
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    diagnostics = result.stdout + result.stderr
    assert result.returncode != 0
    assert "expected Int[6], got Int[5]" in diagnostics
    assert "expected Int[3], got Int[4]" in diagnostics
    assert "Custom.forward" in diagnostics
    assert "bad-assignment" in diagnostics


def test_set_subclass_contracts(tmp_path: Path) -> None:
    """Accept typed subclass methods and reject incompatible overrides."""
    source = tmp_path / "set_subclasses.py"
    valid = (
        "import obb_loss as obb\n"
        "from torch import Tensor\n"
        "from typing_extensions import override\n"
        "class Custom(obb.SetCriterion):\n"
        "    @override\n"
        "    def classification_quality(self, preds: obb.Predictions, targets: obb.Targets, assignment: obb.Assignment) -> obb.Result[Tensor, obb.InputError]:\n"
        "        return obb.Ok(preds.boxes.values[:, 0])\n"
        "criterion = Custom(obb.SetCriterionConfig())\n"
    )
    source.write_text(valid, encoding="utf-8")
    command = [sys.executable, "-m", "tools.typecheck", str(source)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    source.write_text(
        valid + "class Wrong(obb.SetCriterion):\n"
        "    @override\n"
        "    def classification_quality(self, preds: int, targets: int, assignment: int) -> Tensor:\n"
        "        return Tensor()\n",
        encoding="utf-8",
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "bad-override" in result.stdout + result.stderr
