"""Run native shape checks in Python 3.12 while checking Python 3.11 syntax."""

import importlib.util
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Provision the locked shape environment and forward arguments to Pyrefly.

    Returns:
        The checker exit status; dependency provisioning failures retain their status.
        Command-line arguments are forwarded unchanged as Pyrefly check arguments.
    """
    if importlib.util.find_spec("shape_extensions") is None:
        command = [
            "uv",
            "run",
            "--locked",
            "--isolated",
            "--python",
            "3.12",
            "--group",
            "shapes",
            "python",
            "-m",
            "tests.typecheck",
            *sys.argv[1:],
        ]
    else:
        config = Path(__file__).resolve().parents[1] / "pyrefly.toml"
        arguments = sys.argv[1:]
        if "--config" not in arguments and "-c" not in arguments:
            arguments = ["--config", str(config), *arguments]
        command = [
            "pyrefly",
            "check",
            "--python-interpreter-path",
            sys.executable,
            *arguments,
        ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
