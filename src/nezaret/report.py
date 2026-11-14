"""Turning a Report into something to read."""

from __future__ import annotations

from collections import defaultdict

from .rules import RULES

_MARK = {"error": "E", "warning": "W", "info": "I"}


def _summary(report) -> str:
    c = report.counts()
    tail = f"{c['error']} error(s), {c['warning']} warning(s), {c['info']} info"
    if report.cases and "methods" in (report.stats or {}):
        st = report.stats
        return (f"{report.cases} labelled case(s), {len(st['methods'])} method(s), "
                f"{len(st['classes'])} class(es): {tail}")
    if report.cases:
        st = report.stats or {}
        splits = st.get("splits", {})
        parts = ", ".join(f"{k} {v}" for k, v in splits.items())
        pats = f", {st['patients']} patient id(s)" if st.get("patients") else ""
        return (f"{report.cases} case(s) [{parts}], {st.get('files', 0)} file(s) read{pats}: {tail}")
    return f"{len(report.files)} file(s), {report.pipelines} pipeline(s): {tail}"


def text(report) -> str:
    lines = []
    by_file = defaultdict(list)
    for f in report.findings:
        by_file[f.file].append(f)
    for file, fs in by_file.items():
        lines.append(file)
        for f in fs:
            lines.append(f"  {_MARK[f.severity]} {f.rule} {f.slug:<33} {f.location}")
            lines.append(f"      {f.message}")
        lines.append("")
    if getattr(report, "tables", ""):
        lines.append(report.tables.lstrip("\n"))
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
    if getattr(report, "tables", ""):
        out += ["", "## Scores", "", "```text", report.tables.strip("\n"), "```"]
    if report.notes:
        out += ["", "## Notes", ""] + [f"- {n}" for n in report.notes]
    return "\n".join(out) + "\n"


def rules_table() -> str:
    lines = [f"{'Rule':<6} {'Severity':<8} {'Slug':<33} Title"]
    for r in RULES.values():
        lines.append(f"{r.id:<6} {r.severity:<8} {r.slug:<33} {r.title}")
    return "\n".join(lines)


def explain(rule: str) -> str | None:
    key = rule.strip()
    r = RULES.get(key.upper()) or next((x for x in RULES.values() if x.slug == key.lower()), None)
    if r is None:
        return None
    return f"{r.id} {r.slug} ({r.severity})\n{r.title}\n\nWhy it matters:\n  {r.why}\n\nFix:\n  {r.fix}\n"
