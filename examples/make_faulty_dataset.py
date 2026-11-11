"""Write a small synthetic dataset with planted faults, then check it.

    pip install "nezaret[data]"
    python examples/make_faulty_dataset.py /tmp/faulty_dataset
    nezaret data /tmp/faulty_dataset/dataset.json --patient-id-regex "(p\\d{3})" --site-regex "(site[A-Z])"

``build(root, faults=True)`` is also what the test suite uses; with
``faults=False`` it writes the same dataset without any of them, and nezaret
must report nothing.

Every phantom is 24x24x16 voxels of seeded noise texture with one or two
spheres as labels: small enough to write in a second, real enough that
nibabel, the fingerprint and the label checks run exactly as on a real scan.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

SHAPE = (24, 24, 16)
AFFINE = np.array([[-1.5, 0, 0, 18.0],
                   [0, 1.5, 0, -20.0],
                   [0, 0, 2.5, -12.0],
                   [0, 0, 0, 1]])


def phantom(seed: int, classes=(1, 2), empty=False):
    rng = np.random.default_rng(seed)
    img = rng.normal(100, 20, SHAPE)
    img = (img + np.roll(img, 1, 0) + np.roll(img, 1, 1)) / 3       # some texture
    lab = np.zeros(SHAPE, dtype=np.uint8)
    if not empty:
        z, y, x = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
        for k, cls in enumerate(classes):
            c = rng.integers(6, 18, 3)
            c[2] = rng.integers(4, 12)
            r = 3 + k
            m = (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 <= r * r
            lab[m] = cls
            img[m] += 60 * cls
    return img.astype(np.int16), lab


def save(arr, path: Path, affine=AFFINE):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr, affine), str(path))


def build(root, faults: bool = True) -> Path:
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    tr, va, ts = [], [], []

    def case(split, pid, scan, seed, site="siteA", classes=(1, 2), empty=False):
        name = f"{site}_{pid}_scan{scan}.nii.gz"
        img, lab = phantom(seed, classes=classes, empty=empty)
        save(img, root / "images" / name)
        entry = {"image": f"images/{name}"}
        if split is not ts:
            save(lab, root / "labels" / name)
            entry["label"] = f"labels/{name}"
        split.append(entry)
        return img, lab, entry

    for i in range(1, 9):
        case(tr, f"p{i:03d}", 1, seed=i)
    for i in range(11, 14):
        case(va, f"p{i:03d}", 1, seed=i)
    for i in range(21, 24):
        case(ts, f"p{i:03d}", 1, seed=i)

    if faults:
        # NZ202: label of p003 saved with an identity affine
        _, lab, e = case(tr, "p031", 1, seed=31)
        save(lab, root / e["label"], affine=np.eye(4))
        # NZ203: label of p032 resampled with interpolation before saving
        _, lab, e = case(tr, "p032", 1, seed=32)
        smooth = (lab + np.roll(lab, 1, 0) + np.roll(lab, 1, 1)) / 3.0
        save(smooth.astype(np.float32), root / e["label"])
        # NZ204: two empty label maps in training
        case(tr, "p033", 1, seed=33, empty=True)
        case(tr, "p034", 1, seed=34, empty=True)
        # NZ205: validation never contains class 2
        for e in va:
            img, lab = phantom(int(e["image"].split("_p0")[1][:2]), classes=(1,))
            save(lab, root / e["label"])
        # NZ206: patient p005 scanned again, second scan in testing
        case(ts, "p005", 2, seed=105)
        # NZ206: the same entry listed in training and testing
        ts.append(dict(tr[2]))
        # NZ207: p001's image exported again under another name, into testing
        shutil.copy(root / tr[0]["image"], root / "images" / "siteB_p041_scan1.nii.gz")
        ts.append({"image": "images/siteB_p041_scan1.nii.gz"})
        # NZ207 (near): p002's image rescaled to float and saved into testing
        img = np.asanyarray(nib.load(str(root / tr[1]["image"])).dataobj).astype(np.float32)
        save(img * 0.5 + 3, root / "images" / "siteB_p042_scan1.nii.gz")
        ts.append({"image": "images/siteB_p042_scan1.nii.gz"})
        # NZ208: testing comes from a site training never saw
        for i in range(43, 46):
            case(ts, f"p{i:03d}", 1, seed=i, site="siteB")
        # NZ201: an entry whose file is gone
        tr.append({"image": "images/siteA_p099_scan1.nii.gz",
                   "label": "labels/siteA_p099_scan1.nii.gz"})

    (root / "dataset.json").write_text(json.dumps(
        {"name": "faulty phantom" if faults else "clean phantom",
         "training": tr, "validation": va, "testing": ts}, indent=1))
    return root / "dataset.json"


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "faulty_dataset")
    clean = "--clean" in sys.argv
    print(build(out, faults=not clean))
