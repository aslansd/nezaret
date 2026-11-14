"""Running the rules over files, bundles and live pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .__version__ import __version__
from .config import ConfigFile, from_dict, load_bundle
from .knowledge import monai_components
from .pipeline import Pipeline, all_targets, pipeline_from_compose, pipelines_from_config
from .rules import (
    PIPELINE_RULES, RULES, Context, Finding, infer_label_keys, nz104_file, nz106,
    nz109, nz111, rank,
)

CONFIG_SUFFIXES = (".json", ".yaml", ".yml")
_SKIP_NAMES = {"metadata.json", "logging.conf", "large_files.yml", "large_files.yaml",
               "large_files.json"}


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    pipelines: int = 0
    notes: list[str] = field(default_factory=list)
    cases: int = 0                                   # set by check_data
    stats: dict = field(default_factory=dict)        # set by check_data / check_results
    tables: str = ""                                 # set by check_results

    def counts(self) -> dict[str, int]:
        c = {"error": 0, "warning": 0, "info": 0}
        for f in self.findings:
            c[f.severity] += 1
        return c

    @property
    def ok(self) -> bool:
        return self.counts()["error"] == 0

    def at_least(self, severity: str) -> list[Finding]:
        return [f for f in self.findings if rank(f.severity) <= rank(severity)]

    def extend(self, other: "Report"):
        self.findings.extend(other.findings)
        self.files.extend(other.files)
        self.pipelines += other.pipelines
        self.notes.extend(other.notes)
        self.cases += other.cases

    def to_dict(self) -> dict:
        version, _ = monai_components()
        return {
            "nezaret": __version__,
            "checked_against_monai": version,
            "files": self.files,
            "pipelines": self.pipelines,
            "cases": self.cases,
            "stats": self.stats,
            "summary": self.counts(),
            "findings": [f.to_dict() for f in self.findings],
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        from .report import text
        return text(self)

    def to_markdown(self) -> str:
        from .report import markdown
        return markdown(self)

    def __str__(self) -> str:
        return self.to_text()


def _finish(findings: Iterable[Finding], ignore: Iterable[str]) -> list[Finding]:
    ignore = {i.upper() for i in ignore or ()}
    ignore |= {r.id for r in RULES.values() if r.slug in {i.lower() for i in ignore}}
    seen, out = set(), []
    for f in findings:
        if f.rule in ignore:
            continue
        key = (f.rule, f.file, f.location, f.message)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    out.sort(key=lambda f: (f.file, rank(f.severity), f.rule, f.location))
    return out


def _run(files: list[ConfigFile], label_keys, ignore, extra_pipelines=(),
         context_only=()) -> Report:
    report = Report(files=[f.name for f in files])
    ctx = Context()
    findings: list[Finding] = []
    for cf in files:
        for p in pipelines_from_config(cf):
            p.label_keys = infer_label_keys(p, label_keys)
            ctx.pipelines.append(p)
        findings.extend(nz104_file(cf, all_targets(cf)))
    for p in list(extra_pipelines) + list(context_only):
        p.label_keys = infer_label_keys(p, label_keys)
        ctx.pipelines.append(p)
    linted = [p for p in ctx.pipelines if not any(p is c for c in context_only)]
    report.pipelines = len(linted)
    for p in linted:
        for rule in PIPELINE_RULES:
            findings.extend(rule(p) or ())
        findings.extend(nz106(p, ctx) or ())
    findings.extend(nz109(ctx) or ())
    findings.extend(nz111(ctx, files) or ())
    report.findings = _finish(findings, ignore)
    return report


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def find_bundles(path: Path) -> list[list[Path]]:
    """Group config files into bundles.

    * a file is a bundle of one
    * a directory with a ``configs/`` folder is one bundle
    * a directory containing config files is one bundle
    * otherwise, every ``configs/`` folder below it is a bundle
    """
    path = Path(path)
    if path.is_file():
        return [[path]]

    def configs_in(d: Path) -> list[Path]:
        return sorted(p for p in d.iterdir()
                      if p.is_file() and p.suffix.lower() in CONFIG_SUFFIXES
                      and p.name not in _SKIP_NAMES)

    if (path / "configs").is_dir():
        return [configs_in(path / "configs")]
    direct = [p for p in configs_in(path) if "_target_" in p.read_text(errors="ignore")]
    if direct:
        return [direct]
    return [configs_in(d) for d in sorted(path.rglob("configs")) if d.is_dir() and configs_in(d)]


def lint(paths, label_keys: Iterable[str] | None = None,
         ignore: Iterable[str] = ()) -> Report:
    """Lint MONAI Bundle config files or bundle directories.

    ``paths`` may be one path or a list. Each directory is treated as one
    bundle, so cross-file rules (training vs evaluation preprocessing)
    compare files of the same bundle only.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    total = Report()
    for path in paths:
        path = Path(path)
        if not path.exists():
            total.notes.append(f"{path}: does not exist")
            continue
        for group in find_bundles(path):
            files, notes = load_bundle(group)
            r = _run(files, label_keys, ignore)
            base = path if path.is_dir() else path.parent
            shown = {cf.name: str(cf.path.relative_to(base)) for cf in files}
            for f in r.findings:
                f.file = shown.get(f.file, f.file)
                if f.pipeline:
                    name = f.pipeline.split(":", 1)[0]
                    if name in shown:
                        f.pipeline = shown[name] + f.pipeline[len(name):]
            r.files = list(shown.values())
            r.notes.extend(notes)
            total.extend(r)
    return total


def lint_config(config: dict, name: str = "config.json",
                label_keys: Iterable[str] | None = None,
                ignore: Iterable[str] = ()) -> Report:
    """Lint one config already loaded as a dict."""
    return _run([from_dict(config, name)], label_keys, ignore)


def lint_compose(compose, role: str = "train", label_keys: Iterable[str] | None = None,
                 preprocessing=None, ignore: Iterable[str] = (),
                 name: str = "<compose>") -> Report:
    """Lint a live ``monai.transforms.Compose``.

    ``role`` is one of ``"train"``, ``"eval"``, ``"post"``: nezaret cannot
    tell from an object what it is used for. Pass the preprocessing pipeline
    as ``preprocessing=`` when linting a postprocessing one, so the
    saved-not-inverted check has something to compare against.
    """
    pipes = [pipeline_from_compose(compose, role=role, name=name)]
    ctx = []
    if preprocessing is not None:
        # context for NZ106 only; lint it separately if you want it checked
        ctx.append(pipeline_from_compose(preprocessing, role="eval", name=name))
    report = _run([], label_keys, ignore, extra_pipelines=pipes, context_only=ctx)
    report.files = [name]
    return report
