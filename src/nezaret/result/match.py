"""Pairing prediction files with label files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

NIFTI = (".nii.gz", ".nii")
# what MONAI's SaveImaged and common tools append to a case name
POSTFIXES = ("_seg", "_pred", "_prediction", "_label", "_labels", "_mask", "_gt",
             "_trans", "_inferred", "_output")


def case_key(path: Path, regex: re.Pattern | None = None) -> str:
    if regex is not None:
        m = regex.search(str(path))
        if m:
            return m.group(1) if m.groups() else m.group(0)
    name = path.name
    for ext in NIFTI:
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    low = name.lower()
    for post in POSTFIXES:
        if low.endswith(post) and len(low) > len(post):
            name = name[: -len(post)]
            break
    return name


def nifti_files(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*")
                  if p.is_file() and p.name.lower().endswith(NIFTI))


@dataclass
class Index:
    files: dict[str, Path]
    ambiguous: dict[str, list[Path]]


def index(paths: list[Path], regex=None) -> Index:
    files: dict[str, Path] = {}
    amb: dict[str, list[Path]] = {}
    for p in paths:
        k = case_key(p, regex)
        if k in files or k in amb:
            amb.setdefault(k, [files.pop(k)] if k in files else []).append(p)
        else:
            files[k] = p
    return Index(files, amb)


def labels_from_datalist(datalist: Path, split: str, base_dir=None, label_key=None) -> list[Path]:
    from ..dataset.datalist import load_datalist
    cases = load_datalist(datalist, base_dir)
    out = []
    for c in cases:
        if c.list_key != split and c.split != split:
            continue
        keys = [label_key] if label_key else list(c.labels)
        for k in keys:
            out.extend(c.labels.get(k, []))
    return out
