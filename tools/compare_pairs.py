"""Check whether pairs of MSD cases are the same scan, voxel by voxel.

    python compare_pairs.py DATASET_DIR CASE_A CASE_B [CASE_A CASE_B ...]

e.g. python compare_pairs.py Task01_BrainTumour BRATS_277 BRATS_009 BRATS_243 BRATS_720

Images are looked up in imagesTr and then imagesTs, so a training case can be
compared with a test case; test cases have no label, and the label line then
says so.

For each pair, compares the images (every channel, e.g. the four BraTS MRI
sequences) and the labels, and prints whether they are identical, how many
voxels differ, the correlation of each channel, and whether the headers agree.
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def load(path):
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj), img.affine


def describe(a, b, what):
    if a.shape != b.shape:
        return f"{what}: different shapes {a.shape} vs {b.shape}"
    same = np.array_equal(a, b)
    out = f"{what}: {'IDENTICAL' if same else f'{int((a != b).sum())} of {a.size} voxels differ'}"
    if not same and a.ndim == 4:
        rs = [np.corrcoef(a[..., c].ravel(), b[..., c].ravel())[0, 1] for c in range(a.shape[-1])]
        out += "; channel correlations " + ", ".join(f"{r:.4f}" for r in rs)
    elif not same:
        out += f"; correlation {np.corrcoef(a.ravel().astype(float), b.ravel().astype(float))[0, 1]:.4f}"
    return out


def main(root, cases):
    root = Path(root)
    def find(sub, case):
        for d in ((sub, sub.replace("Tr", "Ts")) if sub.endswith("Tr") else (sub,)):
            p = root / d / f"{case}.nii.gz"
            if p.exists():
                return p
        return None

    for a, b in zip(cases[::2], cases[1::2]):
        print(f"\n{a} vs {b}")
        for sub, what in (("imagesTr", "image"), ("labelsTr", "label")):
            pa, pb = find(sub, a), find(sub, b)
            if pa is None or pb is None:
                missing = ", ".join(c for c, p in ((a, pa), (b, pb)) if p is None)
                print(f"  {what}: not found for {missing}" + (" (test cases have no label)" if what == "label" else ""))
                continue
            (xa, aa), (xb, ab) = load(pa), load(pb)
            print("  " + describe(xa, xb, what))
            print(f"  {what} headers (affine): {'same' if np.allclose(aa, ab) else 'differ'}")


if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) % 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
