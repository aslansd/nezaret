"""Reading a MONAI datalist (the Decathlon format ``load_decathlon_datalist`` reads).

    {"training":   [{"image": "imagesTr/a.nii.gz", "label": "labelsTr/a.nii.gz", "fold": 0}, ...],
     "validation": [...],
     "testing":    ["imagesTs/b.nii.gz", ...]}

Each top-level list is a split. If training entries carry a ``fold`` number,
each fold is its own split (``training[fold=2]``), because in k-fold
cross-validation every fold is someone's validation set.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..knowledge import looks_like_label

PATIENT_FIELDS = ("patient_id", "patient", "PatientID", "subject", "subject_id")
_EVAL = re.compile(r"(val|test|eval|infer|holdout|hold_out)", re.IGNORECASE)


@dataclass
class Case:
    split: str
    index: int                        # position inside its datalist list
    list_key: str                     # the datalist key it came from
    images: dict[str, list[Path]] = field(default_factory=dict)
    labels: dict[str, list[Path]] = field(default_factory=dict)
    patient_field: str | None = None

    @property
    def location(self) -> str:
        return f"{self.list_key}#{self.index}"

    @property
    def is_eval(self) -> bool:
        return is_eval_split(self.split)

    def all_paths(self):
        for key, paths in list(self.images.items()) + list(self.labels.items()):
            for p in paths:
                yield key, p


def is_eval_split(split: str) -> bool:
    return bool(_EVAL.search(split)) or "[fold=" in split


class DatalistError(Exception):
    pass


def _paths(value, base: Path) -> list[Path] | None:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
        return None
    out = [(base / v).resolve() if not Path(v).is_absolute() else Path(v) for v in value]
    # a value is a path if it looks like one, or if it exists (DICOM folders
    # such as "case_017" look like neither); "male" or "T1" in a metadata field is not
    if not all(("/" in v or "\\" in v or "." in v) or p.exists() for v, p in zip(value, out)):
        return None
    return out


def load_datalist(path: Path, base_dir: Path | None = None,
                  label_keys=None, patient_key: str | None = None) -> list[Case]:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatalistError(f"{path.name}: cannot read datalist ({exc})") from exc
    if not isinstance(data, dict):
        raise DatalistError(f"{path.name}: a datalist is a JSON object of lists")
    base = Path(base_dir) if base_dir else path.parent
    cases: list[Case] = []
    for list_key, entries in data.items():
        if not isinstance(entries, list):
            continue  # "name", "description", "labels", "numTraining"...
        for i, entry in enumerate(entries):
            if isinstance(entry, str):
                entry = {"image": entry}
            if not isinstance(entry, dict):
                continue
            split = list_key
            if "fold" in entry and isinstance(entry["fold"], int) and not is_eval_split(list_key):
                split = f"{list_key}[fold={entry['fold']}]"
            case = Case(split=split, index=i, list_key=list_key)
            for key, value in entry.items():
                if key == "fold":
                    continue
                if patient_key and key == patient_key or (not patient_key and key in PATIENT_FIELDS):
                    case.patient_field = str(value)
                    continue
                paths = _paths(value, base)
                if paths is None:
                    continue
                is_label = key in label_keys if label_keys else looks_like_label(key)
                (case.labels if is_label else case.images)[key] = paths
            if case.images or case.labels:
                cases.append(case)
    if not cases:
        raise DatalistError(f"{path.name}: no entries with file paths found")
    return cases
