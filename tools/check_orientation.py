"""Load one image the way a bundle's training config does and the way its
inference config does, and report whether the network sees the same array.

Written for the renalStructures_CECT_segmentation report: training loads with
``LoadImaged(as_closest_canonical=True)`` (array reordered to RAS) and adds no
Orientationd, while inference.json loads plainly and applies
``Orientationd(axcodes="LPS")``. Whether that is a bug depends on what the
files' affines say, which this script prints.

Usage:
    python tools/check_orientation.py path/to/case_artery.nii.gz [more files...]
    python tools/check_orientation.py FILE --train-axcodes RAS --infer-axcodes LPS

Verdicts:
    SAME            both pipelines give the same array: no issue to file.
    FLIPPED x,y     same voxels, mirrored along the listed axes: the network is
                    run on mirrored input at inference. File the issue.
    DIFFERENT       shapes or contents differ in some other way: look closer.
"""

from __future__ import annotations

import argparse
import itertools
import sys


def _load(path: str, canonical: bool, axcodes: str | None):
    from monai.transforms import Compose, EnsureChannelFirst, LoadImage, Orientation

    steps = [LoadImage(image_only=True, as_closest_canonical=canonical), EnsureChannelFirst()]
    if axcodes:
        steps.append(Orientation(axcodes=axcodes))
    return Compose(steps)(path)


def compare(path: str, train_canonical: bool, train_axcodes: str | None, infer_axcodes: str | None) -> str:
    import nibabel as nib
    import numpy as np

    img = nib.load(path)
    print(f"\n{path}")
    print(f"  stored shape      {img.shape}")
    print(f"  stored axcodes    {''.join(nib.aff2axcodes(img.affine))}")
    print("  affine            " + np.array2string(img.affine, precision=3, suppress_small=True).replace("\n", "\n                    "))

    a = np.asarray(_load(path, train_canonical, train_axcodes))
    b = np.asarray(_load(path, False, infer_axcodes))
    print(f"  training array    {a.shape}")
    print(f"  inference array   {b.shape}")

    if a.shape == b.shape and np.array_equal(a, b):
        verdict = "SAME"
    else:
        verdict = "DIFFERENT"
        spatial = range(1, a.ndim)  # axis 0 is the channel
        if a.shape == b.shape:
            for k in range(1, len(spatial) + 1):
                for axes in itertools.combinations(spatial, k):
                    if np.array_equal(np.flip(a, axis=axes), b):
                        verdict = "FLIPPED " + ",".join("xyz"[ax - 1] for ax in axes)
                        break
                if verdict != "DIFFERENT":
                    break
    print(f"  verdict           {verdict}")
    return verdict


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="NIfTI files (one channel of a case is enough)")
    p.add_argument("--no-train-canonical", action="store_true",
                   help="training LoadImaged does NOT use as_closest_canonical")
    p.add_argument("--train-axcodes", default=None, help="Orientationd axcodes in training, if any")
    p.add_argument("--infer-axcodes", default="LPS", help="Orientationd axcodes in inference (default LPS)")
    args = p.parse_args(argv)

    verdicts = [compare(f, not args.no_train_canonical, args.train_axcodes, args.infer_axcodes) for f in args.files]
    print("\nsummary: " + ", ".join(sorted(set(verdicts))))
    return 0 if all(v == "SAME" for v in verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
