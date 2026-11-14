"""Prepare the spleen comparison for `nezaret result` (ROADMAP §3, step 1).

After running the two bundles' inference on the last 9 MSD Task09 training
cases (the spleen bundle's own validation fold), this script:

1. copies those 9 labels into <out>/labels, so nezaret scores exactly them;
2. copies the spleen_ct_segmentation predictions unchanged into <out>/spleen_unet;
3. keeps only class 1 (spleen) of the swin_unetr_btcv_segmentation predictions,
   which label 13 abdominal organs, into <out>/swin_unetr.

    python prepare_spleen_eval.py DATASET_DIR UNET_PRED_DIR SWIN_PRED_DIR OUT_DIR

Nothing is resampled: each file keeps its own header, so a geometry problem
still shows up in nezaret as NZ302.
"""

import glob
import os
import shutil
import sys

import nibabel as nib
import numpy as np


def main(dataset_dir, unet_dir, swin_dir, out):
    found = sorted(glob.glob(os.path.join(dataset_dir, "labelsTr", "*.nii.gz")))
    if len(found) != 41:
        print(f"warning: {len(found)} labels in labelsTr; the MSD Task09 download has 41")
    labels = found[-9:]   # the same text-sorted last 9 the bundle validated on
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    for p in labels:
        shutil.copy(p, os.path.join(out, "labels", os.path.basename(p)))
    print("labels:", ", ".join(os.path.basename(p) for p in labels))

    for src in sorted(glob.glob(os.path.join(unet_dir, "**", "*.nii.gz"), recursive=True)):
        dst = os.path.join(out, "spleen_unet", os.path.relpath(src, unet_dir))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    n = 0
    for src in sorted(glob.glob(os.path.join(swin_dir, "**", "*.nii.gz"), recursive=True)):
        img = nib.load(src)
        spleen = (np.asanyarray(img.dataobj) == 1).astype(np.uint8)   # BTCV class 1 = spleen
        dst = os.path.join(out, "swin_unetr", os.path.relpath(src, swin_dir))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        nib.save(nib.Nifti1Image(spleen, img.affine, img.header), dst)
        n += 1
    print(f"swin_unetr: {n} prediction(s) reduced to class 1 (spleen)")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
