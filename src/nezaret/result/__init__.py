"""``nezaret result``: does a segmentation result survive the scoring choices?

    >>> import nezaret
    >>> r = nezaret.check_results("labelsTs/", {"unet": "preds/unet/", "swin": "preds/swin/"})
    >>> print(r)

Needs numpy and nibabel; surface distances (HD95, ASSD) also need scipy:
``pip install nezaret[result]``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from ..lint import Report, _finish
from . import metrics as M
from .analysis import (DICE_DEFAULT, DIST_DEFAULT, CaseResult, MethodResult, analyse,
                       dice_label, dist_label)
from .analysis import _f as _finding
from .match import index, labels_from_datalist, nifti_files

__all__ = ["check_results"]


def _load(path: Path):
    """(array in canonical orientation, spacing in that orientation, header Volume)."""
    import nibabel as nib
    import numpy as np
    from ..dataset.readers import read
    vol = read(path)
    canon = nib.as_closest_canonical(nib.load(str(path)))
    arr = np.asanyarray(canon.dataobj)
    spacing = tuple(float(z) for z in canon.header.get_zooms()[:3])
    return arr, spacing, vol


def _geometry_diff(a, b) -> list[str]:
    diffs = []
    if a.spatial_shape != b.spatial_shape:
        diffs.append(f"shape {a.spatial_shape} vs {b.spatial_shape}")
    if any(abs(x - y) > 1e-3 * max(min(a.spacing), 1e-3) for x, y in zip(a.spacing, b.spacing)):
        diffs.append("spacing " + ", ".join(f"{x:.4g}" for x in a.spacing) + " vs "
                     + ", ".join(f"{x:.4g}" for x in b.spacing))
    if any(abs(x - y) > 1e-2 for x, y in zip(a.origin, b.origin)):
        diffs.append("origin differs")
    da = [v for r in a.direction for v in r]
    db = [v for r in b.direction for v in r]
    if any(abs(x - y) > 1e-3 for x, y in zip(da, db)):
        diffs.append("direction differs")
    return diffs


def _values(a):
    """Distinct values of an integer array, fast for class maps.

    A class map holds a handful of small non-negative integers; one ``==``
    pass per candidate value is much cheaper than sorting the whole volume.
    Anything else (negative or large values: an image saved as a label)
    falls back to ``np.unique``.
    """
    import numpy as np
    mn, mx = int(a.min()), int(a.max())
    if 0 <= mn and mx < 64:
        return np.array([v for v in range(mn, mx + 1) if v in (mn, mx) or (a == v).any()])
    return np.unique(a)


def _class_map(arr):
    """Integer class values of a prediction, or (None, reason)."""
    import numpy as np
    a = np.asarray(arr)
    if a.ndim > 3 and any(s > 1 for s in a.shape[3:]):
        return None, f"has {a.ndim} dimensions {a.shape}; expected one class index per voxel"
    a = a.reshape(a.shape[:3]) if a.ndim > 3 else a
    if a.dtype.kind == "f":
        finite = a[np.isfinite(a)]
        if finite.size and np.any(finite != np.round(finite)):
            return None, "has fractional values (probabilities saved instead of classes?)"
        a = np.round(np.nan_to_num(a)).astype(np.int32)
    elif a.dtype.kind not in "iub":
        return None, f"has dtype {a.dtype}; expected integer classes"
    vals = _values(a)
    if vals.size >= 256:
        return None, f"has {vals.size} distinct values"
    return a, None


def _fmt(x, nd=4):
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


def _table(stats, distances: bool) -> str:
    lines = ["", "Scores", "------"]
    w = max([len(n) for n in stats["methods"]] + [6])
    head = f"  {'method':<{w}}  cases  Dice (MONAI default)   95% CI            Dice range over conventions"
    if distances:
        head += "   HD95 mm   HD95 range         ASSD mm"
    lines.append(head)
    for name, e in stats["methods"].items():
        ci = e.get("dice_ci")
        ci_s = f"[{ci[1]:.4f}, {ci[2]:.4f}]" if ci else "n/a"
        dv = [v for v in e["dice"].values() if v == v]
        rng = f"{min(dv):.4f} - {max(dv):.4f}" if dv else "n/a"
        row = f"  {name:<{w}}  {e['cases']:>5}  {_fmt(e['dice_default']):>20}   {ci_s:<16}  {rng:<27}"
        if distances:
            hv = [v for k, v in e["hd95"].items() if v == v and ", mm," in k]
            hr = f"{min(hv):.2f} - {max(hv):.2f}" if hv else "n/a"
            row += f"   {_fmt(e.get('hd95_default'), 2):>7}   {hr:<17}  {_fmt(e.get('assd_default'), 2):>7}"
        lines.append(row)
    for p in stats["pairs"]:
        b = p.get("bootstrap")
        if not b:
            continue
        lines.append(f"  {p['a']} - {p['b']}: Dice {b['diff']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}] "
                     f"over {b['n']} cases, P({p['a']} better) = {b['p_a_better']:.2f}; "
                     f"{p['a']} wins {sum(w == p['a'] for w in p['dice_winners'].values())} of "
                     f"{len(p['dice_winners'])} Dice conventions"
                     + (f", {sum(w == p['a'] for w in p.get('hd95_winners', {}).values())} of "
                        f"{len(p.get('hd95_winners', {}))} HD95 implementations" if distances else ""))
    lines.append(f"  classes scored: {stats['classes']} (background excluded); "
                 f"Dice default = {dice_label(DICE_DEFAULT)}; HD95 default = {dist_label(DIST_DEFAULT)}")
    return "\n".join(lines)


def check_results(labels, predictions: dict, datalist=None, split: str | None = None,
                  base_dir=None, classes: Iterable[int] | None = None,
                  case_regex: str | None = None, distances: bool = True,
                  tolerance: float = 0.01, rel_tolerance: float = 0.05,
                  n_boot: int = 2000, seed: int = 0, ignore: Iterable[str] = ()) -> Report:
    """Score saved predictions against labels under every defensible convention.

    ``labels`` is a folder of label files (or None with ``datalist`` and
    ``split``); ``predictions`` maps a method name to a folder of its
    predictions. Files are paired by case name, ignoring extensions and
    postfixes such as ``_seg`` that MONAI's SaveImaged adds; ``case_regex``
    (group 1) overrides that. Background (class 0) is never scored.
    """
    report = Report()
    rx = re.compile(case_regex) if case_regex else None
    if datalist is not None:
        label_paths = labels_from_datalist(Path(datalist), split or "validation", base_dir)
    else:
        label_paths = nifti_files(Path(labels))
    lab_idx = index(label_paths, rx)
    report.files = [str(p) for p in lab_idx.files.values()]
    for k, ps in lab_idx.ambiguous.items():
        report.notes.append(f"labels: {len(ps)} files map to case '{k}'; none of them scored")
    if not lab_idx.files:
        report.notes.append("no label files found")
        return report
    try:
        import numpy  # noqa: F401
        import nibabel  # noqa: F401
    except ImportError as exc:
        report.notes.append(f"scoring needs numpy and nibabel (pip install nezaret[result]): {exc}")
        return report
    if distances:
        try:
            M._ndimage()
        except M.MetricsUnavailable as exc:
            distances = False
            report.notes.append(f"{exc}; Dice only")

    pred_idx = {name: index(nifti_files(Path(folder)), rx) for name, folder in predictions.items()}
    findings = []
    for name, idx in pred_idx.items():
        for k, ps in idx.ambiguous.items():
            report.notes.append(f"{name}: {len(ps)} files map to case '{k}'; none of them scored")
        missing = sorted(set(lab_idx.files) - set(idx.files))
        extra = sorted(set(idx.files) - set(lab_idx.files))
        if missing:
            findings.append(_finding("NZ301", name,
                                     f"{name}: no prediction for {len(missing)} of {len(lab_idx.files)} "
                                     f"labelled case(s): {', '.join(missing[:5])}"
                                     + (" ..." if len(missing) > 5 else "") + "."))
        if extra:
            findings.append(_finding("NZ301", name,
                                     f"{name}: {len(extra)} prediction(s) with no label: "
                                     f"{', '.join(extra[:5])}" + (" ..." if len(extra) > 5 else "") + ".",
                                     severity="warning"))

    # one case at a time: a label, then every method's prediction of it
    import numpy as np
    label_keys = sorted(lab_idx.files)
    fixed = sorted(int(c) for c in classes) if classes else None
    methods = [MethodResult(name) for name in predictions]
    label_classes: set[int] = set()
    pred_classes: dict[tuple, set] = {}
    loaded: dict[str, tuple] = {}     # case -> spacing (for the half-voxel floor)
    for k in label_keys:
        try:
            lab, sp, lvol = _load(lab_idx.files[k])
        except Exception as exc:
            report.notes.append(f"label '{k}' not read: {str(exc).splitlines()[0]}")
            continue
        lab_map, why = _class_map(lab)
        del lab
        if lab_map is None:
            report.notes.append(f"label '{k}' {why}; not scored")
            continue
        here = {int(v) for v in _values(lab_map) if v > 0}
        label_classes |= here
        loaded[k] = sp
        for m in methods:
            p = pred_idx[m.name].files.get(k)
            if p is None:
                continue
            try:
                pred, _, pvol = _load(p)
            except Exception as exc:
                report.notes.append(f"{m.name} '{k}' not read: {str(exc).splitlines()[0]}")
                continue
            diffs = _geometry_diff(pvol, lvol) if pvol.format == lvol.format else []
            if diffs:
                findings.append(_finding("NZ302", f"{m.name}/{k}",
                                         f"{m.name} '{p.name}' vs label: " + "; ".join(diffs)
                                         + ". Not scored."))
                continue
            pmap, why = _class_map(pred)
            del pred
            if pmap is None:
                findings.append(_finding("NZ303", f"{m.name}/{k}", f"{m.name} '{p.name}' {why}. Not scored."))
                continue
            pvals = {int(v) for v in _values(pmap) if v > 0}
            pred_classes[(m.name, k)] = pvals
            cr = CaseResult(k)
            for c in (fixed if fixed is not None else sorted(here | pvals)):
                cr.classes[c] = M.score_class(lab_map == c, pmap == c, sp, distances=distances)
            cr.shape = lab_map.shape
            cr.spacing = sp
            m.cases[k] = cr
        del lab_map
    cls = fixed if fixed is not None else sorted(label_classes)
    # classes a case never contained in label or prediction: both empty, no distances
    for m in methods:
        for cr in m.cases.values():
            for c in cls:
                if c not in cr.classes:
                    empty = np.zeros(1, dtype=bool)
                    cr.classes[c] = M.score_class(empty, empty, (1.0,), distances=False)
                    cr.classes[c].worst = M.score_class(np.zeros(cr.shape, bool), np.zeros(cr.shape, bool),
                                                        cr.spacing, distances=False).worst
            for c in list(cr.classes):
                if c not in cls:
                    del cr.classes[c]
    for (name, k), pvals in pred_classes.items():
        unknown = sorted(pvals - set(cls))
        if unknown:
            findings.append(_finding("NZ303", f"{name}/{k}",
                                     f"{name} '{k}' predicts class(es) {unknown} that no label "
                                     f"contains; they are not scored in any class.",
                                     severity="warning"))

    scored = [k for k in label_keys if k in loaded]
    min_sp = min((min(v) for v in loaded.values()), default=1.0)
    more, stats = analyse(methods, scored, cls, distances, tolerance, rel_tolerance, n_boot, seed,
                          min_spacing=min_sp)
    findings.extend(more)
    for f in findings:
        f.file = "result"
    report.findings = _finish(findings, ignore)
    report.cases = len(scored)
    stats["files"] = len(scored) * (1 + len(methods))
    report.stats = stats
    report.tables = _table(stats, distances)
    return report

