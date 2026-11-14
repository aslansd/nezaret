"""Write labels and two methods' predictions with planted faults, then score them.

    pip install "nezaret[result]"
    python examples/make_faulty_results.py /tmp/faulty_results
    nezaret result --labels /tmp/faulty_results/labels \\
        --pred unet=/tmp/faulty_results/unet --pred swin=/tmp/faulty_results/swin

``build(root, faults=True)`` is also what the test suite uses. With
``faults=False`` it writes a clean evaluation (complete, aligned, class maps,
no empty ground truths, one method clearly better), and nezaret must report
nothing.

The story the faulty version tells: ``swin`` is a little better on every case
that contains the organ, ``unet`` is right on the cases that do not (it
predicts nothing, ``swin`` predicts a spurious blob). Which one is "better"
then depends on how empty ground truths are scored, which papers rarely
state.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

SHAPE = (48, 48, 20)


def affine(spacing):
    a = np.diag(list(spacing) + [1.0])
    a[:3, 3] = [-20.0, -25.0, -30.0]
    return a


def organ(rng, empty=False):
    lab = np.zeros(SHAPE, np.uint8)
    if empty:
        return lab
    z, y, x = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    c = rng.integers([14, 14, 6], [34, 34, 14])
    r = rng.uniform(6, 9)
    lab[((z - c[0]) / r) ** 2 + ((y - c[1]) / r) ** 2 + ((x - c[2]) / (r / 2.5)) ** 2 <= 1] = 1
    # a second, smaller structure
    c2 = c + rng.integers([-8, 6, -2], [-4, 9, 2])
    lab[((z - c2[0]) / 3.5) ** 2 + ((y - c2[1]) / 3.5) ** 2 + ((x - c2[2]) / 1.5) ** 2 <= 1] = 2
    return lab


def predict(lab, rng, quality: float, blob_on_empty: bool):
    """A plausible prediction: the label with a jittered boundary, worse when quality is low."""
    pred = np.zeros_like(lab)
    for c in (1, 2):
        m = lab == c
        if not m.any():
            continue
        noise = ndimage.gaussian_filter(rng.normal(size=lab.shape), 2)
        noise /= noise.std()                       # unit boundary jitter, in voxels
        dist_in = ndimage.distance_transform_edt(m) - ndimage.distance_transform_edt(~m)
        pred[(dist_in + (1.0 - quality) * 8 * noise) > (1.0 - quality) * 2] = c
    if not lab.any() and blob_on_empty:
        z, y, x = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
        pred[((z - 24) / 4) ** 2 + ((y - 24) / 4) ** 2 + ((x - 10) / 2) ** 2 <= 1] = 1
    return pred


def save(arr, path: Path, aff):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr, aff), str(path))


def build(root, faults: bool = True) -> Path:
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    rng = np.random.default_rng(7)
    spacing = (0.7, 0.7, 3.0) if faults else (1.0, 1.0, 1.0)
    aff = affine(spacing)
    n_cases, n_empty = (24, 6) if faults else (20, 0)
    q_unet, q_swin = (0.78, 0.80) if faults else (0.85, 0.55)
    for i in range(n_cases):
        name = f"case_{i:03d}"
        lab = organ(rng, empty=i < n_empty)
        save(lab, root / "labels" / f"{name}.nii.gz", aff)
        pu = predict(lab, rng, q_unet, blob_on_empty=False)
        ps = predict(lab, rng, q_swin, blob_on_empty=True)
        # MONAI SaveImaged-style layout: <name>/<name>_seg.nii.gz
        save(pu, root / "unet" / name / f"{name}_seg.nii.gz", aff)
        save(ps, root / "swin" / name / f"{name}_seg.nii.gz", aff)
    if faults:
        # NZ301: unet never produced case_023
        shutil.rmtree(root / "unet" / "case_023")
        # NZ302: swin's case_010 saved on the resampled 1.5 mm grid (no Invertd)
        p = root / "swin" / "case_010" / "case_010_seg.nii.gz"
        small = np.asanyarray(nib.load(str(p)).dataobj)[::2, ::2, :]
        save(small, p, affine((1.4, 1.4, 3.0)))
        # NZ303: swin's case_011 saved as probabilities, not classes
        p = root / "swin" / "case_011" / "case_011_seg.nii.gz"
        prob = ndimage.gaussian_filter(np.asanyarray(nib.load(str(p)).dataobj).astype(np.float32), 1)
        save(prob / max(prob.max(), 1e-6), p, aff)
    return root


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "faulty_results")
    print(build(out, faults="--clean" not in sys.argv))
