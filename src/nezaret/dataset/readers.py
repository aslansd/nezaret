"""Reading image headers and voxels, for NIfTI (nibabel) and DICOM (pydicom).

Both libraries are optional (``pip install nezaret[data]``). A file nezaret
cannot read is reported as a note, never treated as clean.

Every reader returns a :class:`Volume` with geometry in the file's own
convention. Geometry is only ever compared between files of the same format:
NIfTI affines are RAS and DICOM positions are LPS, and converting between
them is exactly the kind of step this tool should not get subtly wrong.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NIFTI_SUFFIXES = (".nii", ".nii.gz", ".hdr", ".img")
DICOM_SUFFIXES = (".dcm", ".dicom", ".ima")

# DICOM attributes that describe where an image came from (NZ208)
SCANNER_FIELDS = ("Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
                  "InstitutionName")


class ReadError(Exception):
    """A file exists but cannot be read; the message says why."""


@dataclass
class Volume:
    path: Path
    format: str                       # "nifti" or "dicom"
    shape: tuple[int, ...]            # spatial shape first, extra dims after
    spacing: tuple[float, ...]        # spatial only
    origin: tuple[float, ...]
    direction: tuple[tuple[float, ...], ...]   # unit columns, spatial only
    patient_id: str | None = None
    scanner: dict[str, str] = field(default_factory=dict)
    _loader: Any = None

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return tuple(self.shape[: len(self.spacing)])

    def array(self):
        """Voxel data, loaded on first use, in canonical orientation for NIfTI."""
        if self._loader is None:
            raise ReadError(f"{self.path.name}: no voxel data")
        return self._loader()


def _need(module: str):
    try:
        return __import__(module)
    except ImportError as exc:
        raise ReadError(f"reading this needs {module} (pip install nezaret[data])") from exc


def kind_of(path: Path) -> str | None:
    name = path.name.lower()
    if path.is_dir():
        return "dicom"
    if name.endswith(NIFTI_SUFFIXES):
        return "nifti"
    if name.endswith(DICOM_SUFFIXES):
        return "dicom"
    return None


def read(path: Path) -> Volume:
    kind = kind_of(path)
    if kind == "nifti":
        return _read_nifti(path)
    if kind == "dicom":
        return _read_dicom(path)
    raise ReadError(f"unsupported format '{''.join(path.suffixes) or path.name}' "
                    f"(0.2.0 reads NIfTI and DICOM)")


# ---------------------------------------------------------------------------
# NIfTI
# ---------------------------------------------------------------------------

def _read_nifti(path: Path) -> Volume:
    nib = _need("nibabel")
    np = _need("numpy")
    try:
        img = nib.load(str(path))
    except Exception as exc:  # nibabel raises several types
        raise ReadError(f"nibabel could not read it ({exc})") from exc
    shape = tuple(int(s) for s in img.shape)
    nd = min(3, len(shape))
    aff = np.asarray(img.affine, dtype=float)
    lin = aff[:nd, :nd]
    spacing = tuple(float(x) for x in np.linalg.norm(lin, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        direction = lin / np.where(np.array(spacing) == 0, 1.0, np.array(spacing))
    origin = tuple(float(x) for x in aff[:nd, 3])

    def load():
        canon = nib.as_closest_canonical(img)
        return np.asanyarray(canon.dataobj)

    return Volume(path=path, format="nifti", shape=shape, spacing=spacing, origin=origin,
                  direction=tuple(tuple(float(v) for v in row) for row in direction),
                  _loader=load)


# ---------------------------------------------------------------------------
# DICOM: a single file (2D or multi-frame) or a directory holding one series
# ---------------------------------------------------------------------------

def _scanner(ds) -> dict[str, str]:
    out = {}
    for f in SCANNER_FIELDS:
        v = ds.get(f)
        if v not in (None, ""):
            out[f] = str(v).strip()
    return out


def _read_dicom(path: Path) -> Volume:
    pydicom = _need("pydicom")
    np = _need("numpy")
    files = sorted(p for p in path.iterdir() if p.is_file()) if path.is_dir() else [path]
    headers = []
    for f in files:
        try:
            headers.append((f, pydicom.dcmread(str(f), stop_before_pixels=True)))
        except Exception:
            continue  # DICOMDIR, stray files
    headers = [(f, ds) for f, ds in headers if "Rows" in ds]
    if not headers:
        raise ReadError("no readable DICOM image found")
    series = {str(ds.get("SeriesInstanceUID", "")) for _, ds in headers}
    if len(series) > 1:
        raise ReadError(f"{len(series)} series in one folder; give one series per entry")
    first = headers[0][1]
    rows, cols = int(first.Rows), int(first.Columns)
    ps = [float(x) for x in first.get("PixelSpacing", [1.0, 1.0])]
    iop = [float(x) for x in first.get("ImageOrientationPatient", [1, 0, 0, 0, 1, 0])]
    r, c = np.array(iop[:3]), np.array(iop[3:])
    normal = np.cross(r, c)

    def pos(ds):
        return np.array([float(x) for x in ds.get("ImagePositionPatient", [0, 0, 0])])

    headers.sort(key=lambda fd: float(pos(fd[1]) @ normal))
    n = len(headers)
    frames = int(first.get("NumberOfFrames", 1) or 1)
    if n > 1:
        gaps = np.diff([float(pos(ds) @ normal) for _, ds in headers])
        dz = float(np.median(gaps)) if len(gaps) else 1.0
        shape = (cols, rows, n)
    else:
        dz = float(first.get("SpacingBetweenSlices", first.get("SliceThickness", 1.0)) or 1.0)
        shape = (cols, rows, frames) if frames > 1 else (cols, rows)
    spacing = (ps[1], ps[0], dz)[: len(shape)]
    direction = tuple(tuple(float(v) for v in col) for col in np.stack([r, c, normal], axis=1))
    pid = str(first.get("PatientID", "")).strip() or None

    def load():
        slices = []
        for f, _ in headers:
            try:
                slices.append(pydicom.dcmread(str(f)).pixel_array)
            except Exception as exc:  # compressed syntax without a decoder, etc.
                raise ReadError(f"pixel data not decodable ({exc})") from exc
        a = np.stack(slices, axis=-1) if len(slices) > 1 else slices[0]
        if a.ndim == 3 and n == 1:   # multi-frame: (frames, rows, cols)
            a = np.moveaxis(a, 0, -1)
        return np.swapaxes(a, 0, 1) if a.ndim >= 2 else a

    return Volume(path=path, format="dicom", shape=tuple(int(s) for s in shape),
                  spacing=tuple(float(s) for s in spacing),
                  origin=tuple(float(x) for x in pos(first)),
                  direction=direction[: len(shape)] if len(shape) == 3 else direction[:2],
                  patient_id=pid, scanner=_scanner(first), _loader=load)


# ---------------------------------------------------------------------------
# pixel fingerprints
# ---------------------------------------------------------------------------

SIGNATURE_SIZE = 16


def exact_hash(a) -> str:
    np = _need("numpy")
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _shrink_axis(a, axis: int, n: int):
    np = _need("numpy")
    size = a.shape[axis]
    if size >= n:
        parts = np.array_split(a, n, axis=axis)
        return np.stack([p.mean(axis=axis) for p in parts], axis=axis)
    idx = (np.arange(n) * size / n).astype(int)
    return np.take(a, idx, axis=axis)


def signature(a, n: int = SIGNATURE_SIZE):
    """A z-scored n**3 vector of block means, or None for a constant image.

    Spatial axes only (a 4D image contributes its first volume). Two arrays
    with the same content at a different dtype, scaling, size or orientation
    give signatures with a correlation close to 1.
    """
    np = _need("numpy")
    a = np.asarray(a, dtype=np.float64)
    while a.ndim > 3:
        a = a[..., 0]
    while a.ndim < 3:
        a = a[..., None]
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    for ax in range(3):
        a = _shrink_axis(a, ax, n)
    v = a.ravel()
    sd = v.std()
    if not np.isfinite(sd) or sd == 0:
        return None
    return (v - v.mean()) / sd
