"""The data checks, NZ201-NZ208.

Like the config rules, each names a way a dataset produces a wrong result
without raising an error. Unlike them, these read files: headers always,
voxels unless ``pixels=False``.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..rules import RULES, Finding, Rule
from .datalist import Case, is_eval_split
from .readers import ReadError, Volume, exact_hash, read, signature


def _rule(id, slug, severity, title, why, fix):
    RULES[id] = Rule(id, slug, severity, title, why, fix)


_rule("NZ201", "missing-file", "error",
      "A datalist entry points at a file that does not exist",
      "MONAI raises when it reaches the entry, but often only when a random "
      "crop or a later epoch reaches it, and a cached dataset may have been "
      "built before the file went missing. Every other data check skips the "
      "case, so a missing file also hides whatever else is wrong with it.",
      "Fix the path, or set --base-dir if the paths are relative to another folder.")

_rule("NZ202", "geometry-mismatch", "error",
      "An image and its label disagree on shape, spacing, origin or direction",
      "MONAI resamples each array with its own header. If a label was saved "
      "with the wrong affine (an identity matrix, a flipped axis, the origin "
      "of another scan), Orientationd and Spacingd move it somewhere else "
      "than the image. If the shapes still agree after that, nothing raises.",
      "Re-export the label with the image's header (e.g. nibabel: "
      "nib.save(nib.Nifti1Image(label, image.affine), path)), after checking "
      "which of the two headers is right.")

_rule("NZ203", "label-values", "error",
      "A label map holds values that are not class indices",
      "Fractional values come from a label that was resampled with "
      "interpolation before it was saved; negative values and hundreds of "
      "distinct values usually mean an image was saved under a label path. "
      "One-hot conversion and Dice still run on them.",
      "Regenerate the label with nearest-neighbour resampling, or check the "
      "datalist entry points at the right file.")

_rule("NZ204", "empty-labels", "info",
      "Cases whose label map is empty",
      "An empty ground truth has no defined Dice. MONAI's DiceMetric drops "
      "such cases by default (ignore_empty=True), other tools score them 1 or "
      "0, so the same predictions give different reported means depending on "
      "how many empty cases a split holds.",
      "Report how empty cases were scored, and the count per split.")

_rule("NZ205", "class-coverage", "warning",
      "A class appears in evaluation but never in training, or the reverse",
      "A class missing from training cannot be learned; a class missing from "
      "an evaluation split is never scored, so a mean over classes silently "
      "averages fewer of them than the model predicts.",
      "Rebalance the split so every class appears in each part, or report "
      "per-class results with the classes each split actually contains.")

_rule("NZ206", "patient-leakage", "error",
      "The same patient, or the same file, appears in more than one split",
      "Two scans of one patient share anatomy, scanner and pathology. With one "
      "in training and one in evaluation, the model is scored on a patient it "
      "has already seen, and the score says more about memory than about "
      "generalisation.",
      "Split by patient, not by scan. nezaret reads patient identity from "
      "DICOM PatientID, a patient field in the datalist, or --patient-id-regex.")

_rule("NZ207", "duplicate-image", "error",
      "The same image content appears in more than one split",
      "Identical voxels in training and evaluation is leakage that no ID "
      "check catches: the same scan exported twice under two names, or with "
      "a different datatype. nezaret compares exact content (error) and a "
      "16x16x16 block-mean fingerprint (warning at correlation >= the "
      "threshold), which matches re-saved copies but not cropped or "
      "re-registered ones.",
      "Remove one copy, or keep both in the same split.")

_rule("NZ208", "site-confounding", "warning",
      "A scanner or site appears in only one split",
      "If every GE scan is in training and every Philips scan in testing, the "
      "test score measures transfer between scanners as much as the model. "
      "Read from DICOM Manufacturer, ManufacturerModelName, "
      "MagneticFieldStrength and InstitutionName, or from --site-regex.",
      "Stratify the split by scanner and site, or report the shift as part "
      "of the result.")


@dataclass
class Options:
    pixels: bool = True
    patient_id_regex: str | None = None
    site_regex: str | None = None
    near_duplicate: float = 0.995
    min_group: int = 3        # NZ208: smallest group worth reporting


@dataclass
class Scan:
    case: Case
    key: str
    is_label: bool
    vol: Volume | None = None
    values: set | None = None   # label classes
    fractional: bool = False
    negative: bool = False
    distinct: int = 0
    exact: str | None = None
    sig: object = None


@dataclass
class State:
    scans: list[Scan] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)


def _f(rule_id, file, location, message, severity=None) -> Finding:
    r = RULES[rule_id]
    return Finding(r.id, r.slug, severity or r.severity, file, location, message, "")


def _short(p: Path, base: Path) -> str:
    try:
        return str(p.relative_to(base))
    except ValueError:
        return str(p)


def _pid_hash(pid: str) -> str:
    # never print a patient identifier; a short digest is enough to match
    return hashlib.sha256(pid.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------

def load_scans(cases: list[Case], datalist: str, base: Path, opt: Options) -> State:
    st = State()
    unreadable: Counter = Counter()
    for c in cases:
        for key, paths in list(c.images.items()) + list(c.labels.items()):
            is_label = key in c.labels
            for j, p in enumerate(paths):
                loc = f"{c.location}#{key}" + (f"#{j}" if len(paths) > 1 else "")
                if not p.exists():
                    st.findings.append(_f("NZ201", datalist, loc,
                                          f"'{_short(p, base)}' does not exist."))
                    continue
                s = Scan(case=c, key=key, is_label=is_label)
                try:
                    s.vol = read(p)
                except ReadError as exc:
                    unreadable[str(exc).splitlines()[0]] += 1
                    continue
                if opt.pixels:
                    try:
                        _pixels(s, opt)
                    except ReadError as exc:
                        unreadable[str(exc).splitlines()[0]] += 1
                st.scans.append(s)
    for why, n in unreadable.most_common():
        st.notes.append(f"{n} file(s) not checked: {why}")
    return st


def _pixels(s: Scan, opt: Options):
    import numpy as np
    a = s.vol.array()
    if s.is_label:
        a = np.asarray(a)
        if a.dtype.kind == "f":
            finite = a[np.isfinite(a)]
            s.fractional = bool(np.any(finite != np.round(finite)))
            vals = np.unique(np.round(finite).astype(np.int64)) if finite.size else np.array([])
        else:
            vals = np.unique(a)
        s.distinct = int(vals.size)
        s.negative = bool(vals.size and vals.min() < 0)
        s.values = {int(v) for v in vals[:1000]}
    else:
        s.exact = exact_hash(a)
        s.sig = signature(a)


# ---------------------------------------------------------------------------

def _close(a, b, atol):
    return len(a) == len(b) and all(abs(x - y) <= atol for x, y in zip(a, b))


def nz202(st: State, datalist: str, base: Path):
    by_case = defaultdict(list)
    for s in st.scans:
        by_case[id(s.case)].append(s)
    for scans in by_case.values():
        labels = [s for s in scans if s.is_label]
        images = [s for s in scans if not s.is_label]
        for lab in labels:
            for img in images:
                if img.vol.format != lab.vol.format:
                    continue
                diffs = []
                a, b = img.vol, lab.vol
                if a.spatial_shape != b.spatial_shape:
                    diffs.append(f"shape {a.spatial_shape} vs {b.spatial_shape}")
                if not _close(a.spacing, b.spacing, 1e-3 * max(min(a.spacing or (1,)), 1e-3)):
                    diffs.append(f"spacing {_r(a.spacing)} vs {_r(b.spacing)}")
                if not _close(a.origin, b.origin, 1e-2):
                    diffs.append(f"origin {_r(a.origin)} vs {_r(b.origin)}")
                da = [v for row in a.direction for v in row]
                db = [v for row in b.direction for v in row]
                if not _close(da, db, 1e-3):
                    diffs.append("direction differs")
                if diffs:
                    st.findings.append(_f(
                        "NZ202", datalist, f"{lab.case.location}#{lab.key}",
                        f"label '{_short(lab.vol.path, base)}' vs image "
                        f"'{_short(img.vol.path, base)}': " + "; ".join(diffs) + "."))
                    break  # one finding per label is enough


def _r(t):
    return "(" + ", ".join(f"{x:.4g}" for x in t) + ")"


def nz203(st: State, datalist: str, base: Path):
    for s in st.scans:
        if not s.is_label or s.values is None:
            continue
        why = []
        if s.fractional:
            why.append("fractional values (resampled with interpolation?)")
        if s.negative:
            why.append("negative values")
        if s.distinct >= 256:
            why.append(f"{s.distinct} distinct values (an image saved as a label?)")
        if why:
            st.findings.append(_f("NZ203", datalist, f"{s.case.location}#{s.key}",
                                  f"'{_short(s.vol.path, base)}' has " + ", ".join(why) + "."))


def nz204(st: State, datalist: str):
    empty, total = Counter(), Counter()
    first = {}
    for s in st.scans:
        if s.is_label and s.values is not None:
            total[s.case.split] += 1
            if s.values <= {0}:
                empty[s.case.split] += 1
                first.setdefault(s.case.split, f"{s.case.location}#{s.key}")
    for split in sorted(empty):
        st.findings.append(_f("NZ204", datalist, first[split],
                              f"{empty[split]} of {total[split]} label maps in '{split}' are empty."))


def nz205(st: State, datalist: str):
    classes = defaultdict(set)
    for s in st.scans:
        if s.is_label and s.values is not None and not s.fractional and s.distinct < 256:
            classes[s.case.split] |= s.values - {0}
    if len(classes) < 2:
        return
    train_splits = [k for k in classes if not is_eval_split(k)]
    eval_splits = [k for k in classes if is_eval_split(k)]
    folds = [k for k in classes if "[fold=" in k]
    for ev in eval_splits:
        others = set().union(*(classes[k] for k in classes if k != ev and
                               (k in train_splits or (ev in folds and k in folds))))
        if not others:
            continue
        unseen = sorted(classes[ev] - others)
        unscored = sorted(others - classes[ev])
        if unseen:
            st.findings.append(_f("NZ205", datalist, ev,
                                  f"class(es) {unseen} appear in '{ev}' but in no training data."))
        if unscored:
            st.findings.append(_f("NZ205", datalist, ev,
                                  f"class(es) {unscored} are in training but never in '{ev}', "
                                  f"so they are never scored there."))


def _patient(case: Case, scans: list[Scan], regex) -> str | None:
    if case.patient_field:
        return case.patient_field
    for s in scans:
        if s.vol is not None and s.vol.patient_id:
            return s.vol.patient_id
    if regex is not None:
        for _, p in case.all_paths():
            m = regex.search(str(p))
            if m:
                return m.group(1) if m.groups() else m.group(0)
    return None


def nz206(st: State, cases: list[Case], datalist: str, base: Path, opt: Options):
    regex = re.compile(opt.patient_id_regex) if opt.patient_id_regex else None
    scans_of = defaultdict(list)
    for s in st.scans:
        scans_of[id(s.case)].append(s)
    # 1. the very same file in two splits
    where = defaultdict(set)
    for c in cases:
        for _, p in c.all_paths():
            where[p].add((c.split, c.location))
    reported: set[str] = set()
    shared = defaultdict(list)   # the entries involved -> files they share
    for p, uses in where.items():
        if len({u[0] for u in uses}) > 1:
            shared[tuple(sorted(uses))].append(p)
    for uses, files in shared.items():
        order = {c.location: n for n, c in enumerate(cases)}
        uses = sorted(uses, key=lambda u: order.get(u[1], 0))
        locs = [u[1] for u in uses]
        splits = list(dict.fromkeys(u[0] for u in uses))
        reported |= set(locs)
        names = ", ".join(f"'{_short(p, base)}'" for p in files)
        st.findings.append(_f("NZ206", datalist, locs[0],
                              f"{names} listed in {len(splits)} splits "
                              f"({', '.join(splits)}): {', '.join(locs)}."))
    # 2. the same patient in two splits
    by_patient = defaultdict(list)
    for c in cases:
        pid = _patient(c, scans_of[id(c)], regex)
        if pid:
            by_patient[pid].append(c)
    for pid, cs in by_patient.items():
        splits = list(dict.fromkeys(c.split for c in cs))   # datalist order
        if len(splits) < 2 or all(c.location in reported for c in cs):
            continue  # one split, or already reported as the same file
        per = ", ".join(f"{s} ({sum(c.split == s for c in cs)})" for s in splits)
        st.findings.append(_f("NZ206", datalist, cs[0].location,
                              f"patient #{_pid_hash(pid)} has cases in {per}: "
                              f"{', '.join(c.location for c in cs)}."))
    return len(by_patient)


def nz207(st: State, datalist: str, base: Path, opt: Options):
    imgs = [s for s in st.scans if not s.is_label and s.exact]
    if not imgs:
        return  # nothing hashed (e.g. no reader installed): numpy may not be there
    by_hash = defaultdict(list)
    for s in imgs:
        by_hash[s.exact].append(s)
    exact_pairs = set()
    for group in by_hash.values():
        splits = {s.case.split for s in group}
        if len(splits) > 1 and len({s.vol.path for s in group}) > 1:
            locs = [f"{s.case.location}#{s.key}" for s in group]
            st.findings.append(_f("NZ207", datalist, locs[0],
                                  f"identical voxels in {len(splits)} splits: "
                                  + ", ".join(f"'{_short(s.vol.path, base)}' ({s.case.split})"
                                              for s in group) + "."))
            exact_pairs |= {(id(a), id(b)) for a in group for b in group}
    with_sig = [s for s in imgs if s.sig is not None]
    if len(with_sig) < 2:
        return
    import numpy as np
    M = np.stack([s.sig for s in with_sig])
    C = (M @ M.T) / M.shape[1]
    ii, jj = np.where(np.triu(C, 1) >= opt.near_duplicate)
    for i, j in zip(ii, jj):
        a, b = with_sig[i], with_sig[j]
        if a.case.split == b.case.split or (id(a), id(b)) in exact_pairs \
                or a.vol.path == b.vol.path:
            continue
        st.findings.append(_f(
            "NZ207", datalist, f"{a.case.location}#{a.key}",
            f"near-duplicate (fingerprint r={C[i, j]:.4f}): '{_short(a.vol.path, base)}' "
            f"({a.case.split}) and '{_short(b.vol.path, base)}' ({b.case.split}).",
            severity="warning"))


def nz208(st: State, cases: list[Case], datalist: str, opt: Options):
    regex = re.compile(opt.site_regex) if opt.site_regex else None
    attrs = defaultdict(dict)  # case id -> field -> value
    for s in st.scans:
        if s.vol is not None and s.vol.scanner and not s.is_label:
            for k, v in s.vol.scanner.items():
                attrs[id(s.case)].setdefault(k, v)
    if regex is not None:
        for c in cases:
            for _, p in c.all_paths():
                m = regex.search(str(p))
                if m:
                    attrs[id(c)]["site"] = m.group(1) if m.groups() else m.group(0)
                    break
    fields = sorted({k for d in attrs.values() for k in d})
    for fld in fields:
        counts = defaultdict(Counter)  # value -> split -> n
        split_total = Counter()
        for c in cases:
            v = attrs.get(id(c), {}).get(fld)
            if v is not None:
                counts[v][c.split] += 1
                split_total[c.split] += 1
        if len(split_total) < 2 or len(counts) < 2:
            continue
        only = []
        for v, per in sorted(counts.items()):
            if len(per) == 1:
                (split, n), = per.items()
                if n >= opt.min_group:
                    only.append(f"'{v}' only in {split} ({n})")
        if only:
            st.findings.append(_f("NZ208", datalist, fld, f"{fld}: " + "; ".join(only) + "."))


def run(cases: list[Case], datalist: str, base: Path, opt: Options) -> tuple[State, dict]:
    st = load_scans(cases, datalist, base, opt)
    nz202(st, datalist, base)
    if opt.pixels:
        nz203(st, datalist, base)
        nz204(st, datalist)
        nz205(st, datalist)
        nz207(st, datalist, base, opt)
    patients = nz206(st, cases, datalist, base, opt)
    nz208(st, cases, datalist, opt)
    stats = {"cases": len(cases), "files": len(st.scans), "patients": patients,
             "splits": dict(Counter(c.split for c in cases))}
    return st, stats
