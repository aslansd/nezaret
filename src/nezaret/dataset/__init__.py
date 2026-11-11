"""``nezaret data``: checks on a datalist and the files it points at.

    >>> import nezaret
    >>> report = nezaret.check_data("dataset_0.json", base_dir="/data/Task09_Spleen")

Needs nibabel (NIfTI) and/or pydicom (DICOM) for anything beyond the path
checks: ``pip install nezaret[data]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..lint import Report, _finish
from .checks import Options, run
from .datalist import DatalistError, load_datalist

__all__ = ["check_data"]


def check_data(datalist, base_dir=None, label_keys: Iterable[str] | None = None,
               patient_key: str | None = None, patient_id_regex: str | None = None,
               site_regex: str | None = None, pixels: bool = True,
               near_duplicate: float = 0.995, ignore: Iterable[str] = ()) -> Report:
    """Check a MONAI (Decathlon-format) datalist and the files it lists.

    ``base_dir`` is what relative paths are relative to (default: the
    datalist's folder, as in ``load_decathlon_datalist``). ``pixels=False``
    reads headers only: missing files, geometry, patient identity and scanner
    balance, without loading voxels.
    """
    path = Path(datalist)
    base = Path(base_dir) if base_dir else path.parent
    report = Report(files=[path.name])
    try:
        cases = load_datalist(path, base, label_keys=label_keys, patient_key=patient_key)
    except DatalistError as exc:
        report.files = []
        report.notes.append(str(exc))
        return report
    opt = Options(pixels=pixels, patient_id_regex=patient_id_regex, site_regex=site_regex,
                  near_duplicate=near_duplicate)
    state, stats = run(cases, path.name, base.resolve(), opt)
    report.findings = _finish(state.findings, ignore)
    report.notes.extend(state.notes)
    report.cases = stats["cases"]
    report.stats = stats
    if not pixels:
        report.notes.append("headers only (--no-pixels): NZ203, NZ204, NZ205 and NZ207 not run")
    return report
