"""Turning a Report into something to read."""

from __future__ import annotations

from collections import defaultdict

from .rules import RULES

_MARK = {"error": "E", "warning": "W", "info": "I"}


def _summary(report) -> str:
    c = report.counts()
    return (f"{len(report.files)} file(s), {report.pipelines} pipeline(s): "
            f"{c['error']} error(s), {c['warning']} warning(s), {c['info']} info")


def text(report) -> str:
    lines = []
    by_file = defaultdict(list)
    for f in report.findings:
        by_file[f.file].append(f)
    for file, fs in by_file.items():
        lines.append(file)
        for f in fs:
            lines.append(f"  {_MARK[f.severity]} {f.rule} {f.slug:<24} {f.location}")
            lines.append(f"      {f.message}")
        lines.append("")
    for n in report.notes:
        lines.append(f"note: {n}")
    lines.append(_summary(report))
    if report.findings:
        lines.append("Run `nezaret explain <rule>` for why a rule matters and how to fix it.")
    return "\n".join(lines)


def markdown(report) -> str:
    out = ["# nezaret report", "", _summary(report), ""]
    if report.findings:
        out += ["| | Rule | File | Location | Finding |", "|---|---|---|---|---|"]
        for f in report.findings:
            msg = f.message.replace("|", "\\|")
            out.append(f"| {f.severity} | {f.rule} `{f.slug}` | `{f.file}` | `{f.location}` | {msg} |")
        out.append("")
        out.append("## Rules that fired")
        out.append("")
        for rid in sorted({f.rule for f in report.findings}):
            r = RULES[rid]
            out += [f"### {r.id} `{r.slug}`: {r.title}", "", r.why, "", f"**Fix:** {r.fix}", ""]
    else:
        out.append("No findings.")
    if report.notes:
        out += ["", "## Notes", ""] + [f"- {n}" for n in report.notes]
    return "\n".join(out) + "\n"


def rules_table() -> str:
    lines = [f"{'Rule':<6} {'Severity':<8} {'Slug':<24} Title"]
    for r in RULES.values():
        lines.append(f"{r.id:<6} {r.severity:<8} {r.slug:<24} {r.title}")
    return "\n".join(lines)


def explain(rule: str) -> str | None:
    key = rule.strip()
    r = RULES.get(key.upper()) or next((x for x in RULES.values() if x.slug == key.lower()), None)
    if r is None:
        return None
    return f"{r.id} {r.slug} ({r.severity})\n{r.title}\n\nWhy it matters:\n  {r.why}\n\nFix:\n  {r.fix}\n"
