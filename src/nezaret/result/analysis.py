"""Scoring every method under every defensible variant, and the checks on that.

The question is the one ``ayar`` asks of spike sorting: does the conclusion
survive the choices the analyst could reasonably have made differently?
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations, product
from pathlib import Path

from ..rules import RULES, Finding, Rule
from . import metrics as M


def _rule(id, slug, severity, title, why, fix):
    RULES[id] = Rule(id, slug, severity, title, why, fix)


_rule("NZ301", "unmatched-cases", "error",
      "Predictions and labels do not cover the same cases",
      "A mean over 48 cases and a mean over 50 are not comparable, and a "
      "method missing its hardest cases looks better than it is. Nothing in a "
      "scoring loop that skips missing files says so.",
      "Produce a prediction for every labelled case, or score every method on "
      "the cases all of them have (nezaret compares pairs on their common cases).")

_rule("NZ302", "prediction-geometry", "error",
      "A prediction and its label disagree on shape, spacing, origin or direction",
      "Usually a prediction saved in the preprocessed space (no Invertd, see "
      "NZ106). Scored voxel by voxel against an original-space label it gives "
      "a Dice that measures misalignment, not segmentation. The case is not "
      "scored.",
      "Save predictions in the label's space (Invertd before SaveImaged, or "
      "SaveImaged resample=true).")

_rule("NZ303", "prediction-values", "error",
      "A prediction is not a class-index map",
      "Probabilities saved instead of argmax/threshold output, or a one-hot "
      "map, are thresholded somewhere downstream by whatever rule that code "
      "happens to use. Classes the labels never contain are scored as false "
      "positives in no class at all.",
      "Save AsDiscreted output, one integer class per voxel.")

_rule("NZ304", "empty-handling-sensitive", "warning",
      "The mean Dice depends on how empty ground truths are scored",
      "MONAI's DiceMetric drops a case whose ground truth is empty "
      "(ignore_empty=True), ignore_empty=False scores it 1 if the prediction is "
      "also empty and 0 otherwise, and a smoothed formula scores it 0. Papers "
      "rarely say which. When the spread is larger than the differences being "
      "reported, the reported ranking is a choice of convention.",
      "Report the convention and the number of empty cases per split, or "
      "report results on non-empty cases and detection on empty ones separately.")

_rule("NZ305", "distance-implementation-sensitive", "warning",
      "HD95 or ASSD depends on which implementation computed it",
      "MONAI takes the larger of the two directed 95th percentiles; MedPy "
      "takes one percentile of both directions pooled. Both return voxel units "
      "unless spacing is passed. Edge extraction with 6- or 26-connectivity "
      "moves the surface. Two papers quoting 'HD95' from different code are "
      "not comparing the same number.",
      "Name the implementation and its version, pass spacing, and say how "
      "cases with an empty mask were handled. Reported as a warning when "
      "voxel units would differ from mm (spacing not passed is a real "
      "mistake), as info when only the implementations differ.")

_rule("NZ306", "ranking-not-robust", "warning",
      "Which method is better depends on a scoring choice",
      "If method A wins under one defensible convention and B under another, "
      "the data do not say which is better; the convention does.",
      "Report the ranking under each convention, or the one the field's "
      "guidelines (Metrics Reloaded) recommend for the task, and say so.")

_rule("NZ307", "difference-unresolved", "warning",
      "The difference between two methods is within its bootstrap interval of zero",
      "A mean Dice higher by 0.01 on 30 cases is often inside case-to-case "
      "variation. Resampling cases with replacement (paired, so each case is "
      "compared with itself) gives the interval of differences the same "
      "evaluation could have produced.",
      "Report the paired interval, and treat an interval that includes zero "
      "as no demonstrated difference.")


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------

DICE_VARIANTS = [(e, a) for a in ("case-first", "class-first")
                 for e in ("exclude", "both-empty-1", "zero")] + [("-", "pooled")]
DICE_DEFAULT = ("exclude", "case-first")   # MONAI DiceMetric() defaults

DIST_VARIANTS = list(product(("monai", "medpy"), ("mm", "voxel"), (1, 3), ("exclude", "worst")))
DIST_DEFAULT = ("monai", "mm", 1, "exclude")  # MONAI with spacing passed
ASSD_VARIANTS = list(product(("pooled", "mean-of-means"), ("mm", "voxel"), (1, 3), ("exclude", "worst")))
ASSD_DEFAULT = ("pooled", "mm", 1, "exclude")


def dice_label(v) -> str:
    e, a = v
    if a == "pooled":
        return "pooled over cases"
    tag = " (MONAI default)" if v == DICE_DEFAULT else ""
    tag = " (MONAI ignore_empty=False)" if v == ("both-empty-1", "case-first") else tag
    return f"{a}, empty GT {e}{tag}"


def dist_label(v) -> str:
    style, units, conn, empty = v
    nb = {1: 6, 3: 26}[conn]
    return f"{style}, {units}, {nb}-neighbour edges, empty {empty}"


@dataclass
class CaseResult:
    key: str
    classes: dict[int, M.ClassScore] = field(default_factory=dict)
    shape: tuple = ()
    spacing: tuple = ()


@dataclass
class MethodResult:
    name: str
    cases: dict[str, CaseResult] = field(default_factory=dict)


def _nanmean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def per_case_dice(m: MethodResult, keys, classes, empty: str) -> dict[str, float]:
    return {k: _nanmean([M.dice(m.cases[k].classes[c], empty) for c in classes])
            for k in keys if k in m.cases}


def aggregate_dice(m: MethodResult, keys, classes, variant) -> float:
    empty, agg = variant
    keys = [k for k in keys if k in m.cases]
    if agg == "case-first":
        return _nanmean(per_case_dice(m, keys, classes, empty).values())
    if agg == "class-first":
        return _nanmean([_nanmean([M.dice(m.cases[k].classes[c], empty) for k in keys])
                         for c in classes])
    if agg == "pooled":
        vals = []
        for c in classes:
            tp = sum(m.cases[k].classes[c].tp for k in keys)
            den = sum(m.cases[k].classes[c].gt + m.cases[k].classes[c].pred for k in keys)
            if den:
                vals.append(2.0 * tp / den)
        return _nanmean(vals)
    raise ValueError(agg)


def _dist_one(s: M.ClassScore, v, kind: str) -> float:
    style, units, conn, empty = v
    if s.gt_empty and s.pred_empty:
        return float("nan") if empty == "exclude" else 0.0
    if s.gt_empty or s.pred_empty:
        return float("nan") if empty == "exclude" else s.worst[units]
    d = s.dist.get((conn, units))
    if d is None:
        return float("nan")
    return M.hd95(*d, style) if kind == "hd95" else M.assd(*d, style)


def per_case_dist(m: MethodResult, keys, classes, v, kind) -> dict[str, float]:
    return {k: _nanmean([_dist_one(m.cases[k].classes[c], v, kind) for c in classes])
            for k in keys if k in m.cases}


def aggregate_dist(m, keys, classes, v, kind) -> float:
    return _nanmean(per_case_dist(m, keys, classes, v, kind).values())


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------

def bootstrap_mean(values, n_boot: int, seed: int):
    import numpy as np
    x = np.array([v for v in values if not math.isnan(v)], dtype=float)
    if x.size < 2:
        return None
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, (n_boot, x.size))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_paired(a: dict, b: dict, n_boot: int, seed: int):
    import numpy as np
    keys = [k for k in a if k in b and not math.isnan(a[k]) and not math.isnan(b[k])]
    if len(keys) < 2:
        return None
    d = np.array([a[k] - b[k] for k in keys])
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, (n_boot, d.size))].mean(axis=1)
    return {"n": len(keys), "diff": float(d.mean()),
            "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5)),
            "p_a_better": float((means > 0).mean())}


# ---------------------------------------------------------------------------
# the checks on the scores
# ---------------------------------------------------------------------------

def _f(rule_id, location, message, severity=None) -> Finding:
    r = RULES[rule_id]
    return Finding(r.id, r.slug, severity or r.severity, "", location, message, "")


def analyse(methods: list[MethodResult], label_keys: list[str], classes: list[int],
            distances: bool, tolerance: float, rel_tolerance: float,
            n_boot: int, seed: int, min_spacing: float = 1.0) -> tuple[list[Finding], dict]:
    findings: list[Finding] = []
    stats: dict = {"classes": classes, "methods": {}, "pairs": []}
    n_empty_gt = {}
    for m in methods:
        keys = [k for k in label_keys if k in m.cases]
        ent = {"cases": len(keys), "dice": {}, "hd95": {}, "assd": {}}
        for v in DICE_VARIANTS:
            ent["dice"][dice_label(v)] = aggregate_dice(m, keys, classes, v)
        if distances:
            for v in DIST_VARIANTS:
                ent["hd95"][dist_label(v)] = aggregate_dist(m, keys, classes, v, "hd95")
            for v in ASSD_VARIANTS:
                ent["assd"][dist_label(v)] = aggregate_dist(m, keys, classes, v, "assd")
        ent["dice_default"] = ent["dice"][dice_label(DICE_DEFAULT)]
        ent["dice_ci"] = bootstrap_mean(per_case_dice(m, keys, classes, "exclude").values(),
                                        n_boot, seed)
        if distances:
            ent["hd95_default"] = ent["hd95"][dist_label(DIST_DEFAULT)]
            ent["assd_default"] = ent["assd"][dist_label(ASSD_DEFAULT)]
        empties = sum(1 for k in keys for c in classes if m.cases[k].classes[c].gt_empty)
        n_empty_gt[m.name] = empties
        stats["methods"][m.name] = ent

        # NZ304: spread across empty-GT policies, same aggregation
        for agg in ("case-first", "class-first"):
            vals = {e: aggregate_dice(m, keys, classes, (e, agg))
                    for e in ("exclude", "both-empty-1", "zero")}
            fin = [v for v in vals.values() if not math.isnan(v)]
            if len(fin) >= 2 and max(fin) - min(fin) > tolerance:
                findings.append(_f("NZ304", m.name,
                                   f"{m.name}: mean Dice ({agg}) is "
                                   + ", ".join(f"{v:.4f} ({e})" for e, v in vals.items())
                                   + f" over {empties} empty (case, class) pair(s); "
                                   f"spread {max(fin) - min(fin):.4f} > {tolerance}."))
                break

        # NZ305: spread across distance implementations
        if distances:
            for kind, default in (("hd95", DIST_DEFAULT), ("assd", ASSD_DEFAULT)):
                vals = ent[kind]
                d0 = vals[dist_label(default)]
                if math.isnan(d0) or d0 == 0:
                    continue
                mm_vals = {k: v for k, v in vals.items() if ", mm," in k and "exclude" in k}
                vox = vals[dist_label((default[0], "voxel", 1, "exclude"))]
                fin = [v for v in mm_vals.values() if not math.isnan(v)]
                spread_mm = (max(fin) - min(fin)) if fin else 0.0
                vox_mm = abs(vox - d0) if not math.isnan(vox) else 0.0
                # below half a voxel a difference is not worth a warning
                big = lambda x: x / d0 > rel_tolerance and x > 0.5 * min_spacing  # noqa: E731
                if not (big(spread_mm) or big(vox_mm)):
                    continue
                alt_style = "medpy" if kind == "hd95" else "mean-of-means"
                alt = vals[dist_label((alt_style, "mm", 1, "exclude"))]
                c26 = vals[dist_label((default[0], "mm", 3, "exclude"))]
                worst = vals[dist_label((default[0], "mm", 1, "worst"))]
                name = "HD95" if kind == "hd95" else "ASSD"
                spread_txt = (f"{d0:.3f} mm ({default[0]}-style) vs {alt:.3f} mm "
                              f"({alt_style}-style), {c26:.3f} mm with 26-neighbour edges, "
                              f"{worst:.3f} mm with empty masks scored as the image diagonal")
                if big(vox_mm):
                    # anisotropic or non-1 mm data: forgetting `spacing` changes the number
                    findings.append(_f("NZ305", m.name, (
                        f"{m.name}: {name} is {vox:.3f} in voxel units, the default of both "
                        f"MONAI and MedPy when spacing is not passed, but {d0:.3f} mm with "
                        f"spacing. Across implementations: {spread_txt}.")))
                else:
                    findings.append(_f("NZ305", m.name,
                                       f"{m.name}: {name} {spread_txt}.", severity="info"))

    # pairs
    for a, b in combinations(methods, 2):
        common = [k for k in label_keys if k in a.cases and k in b.cases]
        if len(common) < 2:
            continue
        pair = {"a": a.name, "b": b.name, "cases": len(common)}
        # NZ306 across Dice variants (higher is better)
        wins = {}
        for v in DICE_VARIANTS:
            da, db = aggregate_dice(a, common, classes, v), aggregate_dice(b, common, classes, v)
            if not (math.isnan(da) or math.isnan(db)) and da != db:
                wins[dice_label(v)] = a.name if da > db else b.name
        pair["dice_winners"] = wins
        if len(set(wins.values())) > 1:
            by = {w: [v for v, x in wins.items() if x == w] for w in set(wins.values())}
            findings.append(_f("NZ306", f"{a.name} vs {b.name}", (
                f"Dice: {a.name} better under {len(by.get(a.name, []))} of {len(wins)} "
                f"conventions, {b.name} under {len(by.get(b.name, []))}; e.g. {a.name} wins with "
                f"'{by.get(a.name, ['-'])[0]}', {b.name} with '{by.get(b.name, ['-'])[0]}'.")))
        if distances:
            hw = {}
            for v in DIST_VARIANTS:
                ha = aggregate_dist(a, common, classes, v, "hd95")
                hb = aggregate_dist(b, common, classes, v, "hd95")
                if not (math.isnan(ha) or math.isnan(hb)) and ha != hb:
                    hw[dist_label(v)] = a.name if ha < hb else b.name   # lower is better
            pair["hd95_winners"] = hw
            if len(set(hw.values())) > 1:
                by = {w: [v for v, x in hw.items() if x == w] for w in set(hw.values())}
                findings.append(_f("NZ306", f"{a.name} vs {b.name}", (
                    f"HD95: {a.name} better under {len(by.get(a.name, []))} of {len(hw)} "
                    f"implementations, {b.name} under {len(by.get(b.name, []))}; e.g. {a.name} "
                    f"wins with '{by.get(a.name, ['-'])[0]}', {b.name} with "
                    f"'{by.get(b.name, ['-'])[0]}'.")))
        # NZ307 paired bootstrap on the default Dice
        pb = bootstrap_paired(per_case_dice(a, common, classes, "exclude"),
                              per_case_dice(b, common, classes, "exclude"), n_boot, seed)
        pair["bootstrap"] = pb
        if pb and pb["lo"] <= 0 <= pb["hi"]:
            findings.append(_f("NZ307", f"{a.name} vs {b.name}", (
                f"mean Dice difference {a.name} - {b.name} = {pb['diff']:+.4f}, paired 95% "
                f"bootstrap interval [{pb['lo']:+.4f}, {pb['hi']:+.4f}] over {pb['n']} cases "
                f"includes 0 (P({a.name} > {b.name}) = {pb['p_a_better']:.2f}).")))
        stats["pairs"].append(pair)
    stats["empty_gt_pairs"] = n_empty_gt
    return findings, stats
