"""The shipped examples must keep showing what the README says they show."""

import subprocess
import sys
from pathlib import Path

import pytest

import nezaret

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_faulty_bundle_findings():
    r = nezaret.lint(EXAMPLES / "faulty_bundle")
    got = sorted((f.file, f.rule, f.severity) for f in r.findings)
    assert got == sorted([
        ("configs/inference.json", "NZ104", "error"),
        ("configs/inference.json", "NZ106", "warning"),
        ("configs/inference.json", "NZ109", "warning"),
        ("configs/train.json", "NZ101", "error"),
        ("configs/train.json", "NZ102", "error"),
        ("configs/train.json", "NZ103", "error"),
        ("configs/train.json", "NZ107", "error"),
        ("configs/train.json", "NZ109", "warning"),
        ("configs/train.json", "NZ111", "info"),
    ])


def test_lint_compose_example_runs():
    pytest.importorskip("monai")
    out = subprocess.run([sys.executable, str(EXAMPLES / "lint_compose_example.py")],
                         capture_output=True, text=True, check=True).stdout
    for rule in ("NZ101", "NZ107", "NZ105"):
        assert rule in out
