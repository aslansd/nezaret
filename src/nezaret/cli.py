"""Command line: ``nezaret lint|rules|explain|doctor``."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from .__version__ import __version__
from .knowledge import MONAI_REFERENCE_VERSION, monai_components
from .dataset import check_data
from .lint import lint
from .report import explain, rules_table
from .rules import SEVERITIES, rank


def _split(values):
    out = []
    for v in values or []:
        out.extend(x.strip() for x in v.split(",") if x.strip())
    return out


def _emit(a, report) -> None:
    if a.format == "json":
        out = report.to_json()
    elif a.format == "markdown":
        out = report.to_markdown()
    else:
        out = report.to_text()
    if a.output:
        Path(a.output).write_text(out + ("\n" if not out.endswith("\n") else ""), encoding="utf-8")
        print(f"wrote {a.output}: {report.counts()}")
    else:
        print(out)


def cmd_lint(a) -> int:
    report = lint(a.paths, label_keys=_split(a.label_keys) or None, ignore=_split(a.ignore))
    _emit(a, report)
    if not report.files:
        why = "none could be read (see notes above)" if report.notes else "none found"
        print(f"nezaret: no config files checked: {why}", file=sys.stderr)
        return 2
    if a.fail_on == "never":
        return 0
    return 1 if report.at_least(a.fail_on) else 0


def cmd_data(a) -> int:
    report = check_data(a.datalist, base_dir=a.base_dir,
                        label_keys=_split(a.label_keys) or None,
                        patient_key=a.patient_key, patient_id_regex=a.patient_id_regex,
                        site_regex=a.site_regex, pixels=not a.no_pixels,
                        near_duplicate=a.near_duplicate, ignore=_split(a.ignore))
    _emit(a, report)
    if not report.cases:
        print("nezaret: no cases checked (see notes above)", file=sys.stderr)
        return 2
    if a.fail_on == "never":
        return 0
    return 1 if report.at_least(a.fail_on) else 0


def cmd_rules(a) -> int:
    print(rules_table())
    return 0


def cmd_explain(a) -> int:
    text = explain(a.rule)
    if text is None:
        print(f"unknown rule: {a.rule}", file=sys.stderr)
        return 2
    print(text)
    return 0


def cmd_doctor(a) -> int:
    version, names = monai_components()
    print(f"nezaret      {__version__}")
    print(f"python       {platform.python_version()} ({sys.executable})")
    print(f"rules        checked against MONAI {version} ({len(names)} component names)")
    try:
        import yaml
        print(f"pyyaml       {yaml.__version__} (YAML configs supported)")
    except ImportError:
        print("pyyaml       not installed (YAML configs are skipped; pip install nezaret[yaml])")
    for mod, purpose in (("nibabel", "NIfTI for nezaret data"), ("pydicom", "DICOM for nezaret data")):
        try:
            m = __import__(mod)
            print(f"{mod:<12} {m.__version__} ({purpose})")
        except ImportError:
            print(f"{mod:<12} not installed ({purpose}; pip install nezaret[data])")
    try:
        import monai
        mv = monai.__version__
        flag = "" if mv.split("+")[0] == MONAI_REFERENCE_VERSION else \
            f"  <- differs from {MONAI_REFERENCE_VERSION}; run the live tests (see TESTING.md)"
        print(f"monai        {mv} (live Compose linting available){flag}")
    except Exception as exc:  # ImportError, or a broken torch underneath
        kind = "not installed" if isinstance(exc, ImportError) else f"BROKEN: {exc!r}"
        print(f"monai        {kind} (only needed for lint_compose)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nezaret",
                                 description="Find MONAI pipeline faults that do not raise errors.")
    ap.add_argument("--version", action="version", version=f"nezaret {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("lint", help="lint bundle configs or bundle directories")
    p.add_argument("paths", nargs="+")
    p.add_argument("--format", choices=["text", "markdown", "json"], default="text")
    p.add_argument("-o", "--output", help="write the report to a file")
    p.add_argument("--label-keys", action="append",
                   help="comma-separated keys that hold label maps (default: inferred)")
    p.add_argument("--ignore", action="append", help="comma-separated rule ids or slugs")
    p.add_argument("--fail-on", choices=list(SEVERITIES) + ["never"], default="error",
                   help="exit 1 if a finding at this severity or worse exists (default: error)")
    p.set_defaults(func=cmd_lint)

    p = sub.add_parser("data", help="check a datalist and the files it points at")
    p.add_argument("datalist", help="a MONAI / Decathlon datalist JSON")
    p.add_argument("--base-dir", help="folder relative paths start from (default: the datalist's)")
    p.add_argument("--format", choices=["text", "markdown", "json"], default="text")
    p.add_argument("-o", "--output", help="write the report to a file")
    p.add_argument("--label-keys", action="append",
                   help="comma-separated keys that hold label maps (default: inferred)")
    p.add_argument("--patient-key", help="datalist field holding the patient id")
    p.add_argument("--patient-id-regex",
                   help="regex on file paths; group 1 (or the match) is the patient id")
    p.add_argument("--site-regex", help="regex on file paths; group 1 (or the match) is the site")
    p.add_argument("--no-pixels", action="store_true",
                   help="read headers only (fast; skips label, class and duplicate checks)")
    p.add_argument("--near-duplicate", type=float, default=0.995, metavar="R",
                   help="fingerprint correlation that counts as a near-duplicate (default 0.995)")
    p.add_argument("--ignore", action="append", help="comma-separated rule ids or slugs")
    p.add_argument("--fail-on", choices=list(SEVERITIES) + ["never"], default="error")
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("rules", help="list the rules")
    p.set_defaults(func=cmd_rules)

    p = sub.add_parser("explain", help="explain one rule")
    p.add_argument("rule")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("doctor", help="report versions and optional dependencies")
    p.set_defaults(func=cmd_doctor)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
