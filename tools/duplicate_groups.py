"""Sort every fingerprint near-duplicate in a dataset into what it really is.

    python duplicate_groups.py DATASET.json [SPLIT] [--threshold 0.995] [--csv pairs.csv]

For every pair of images whose nezaret fingerprint correlates at or above the
threshold, loads both at full resolution and measures:

* per-channel correlation of the full volumes (min over channels),
* the share of voxels that differ by more than 1 % of the intensity range,
* Dice between the two label maps (foreground vs background), if labels exist.

Then groups pairs into connected sets (a scan stored three times is one group
of three) and prints how many pairs fall in each fingerprint band, so the
threshold can be judged: if pairs in the lowest band are clearly different
patients, the threshold is too low for this kind of data.
"""

import csv
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from nezaret.dataset.readers import read, signature


def arr(path):
    return np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32)


def full_res(a, b):
    if a.shape != b.shape:
        return float("nan"), float("nan")
    a4 = a.reshape(a.shape[:3] + (-1,))
    b4 = b.reshape(b.shape[:3] + (-1,))
    rs = [np.corrcoef(a4[..., c].ravel(), b4[..., c].ravel())[0, 1] for c in range(a4.shape[-1])]
    rng = float(max(a.max() - a.min(), 1e-6))
    frac = float((np.abs(a - b) > 0.01 * rng).mean())
    return float(min(rs)), frac


def label_dice(pa, pb):
    if pa is None or pb is None or not (pa.exists() and pb.exists()):
        return float("nan")
    x, y = arr(pa) > 0, arr(pb) > 0
    if x.shape != y.shape:
        return float("nan")
    s = x.sum() + y.sum()
    return float(2 * (x & y).sum() / s) if s else 1.0


def groups(n, edges):
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in edges:
        parent[find(i)] = find(j)
    out = {}
    for i in {k for e in edges for k in e}:
        out.setdefault(find(i), []).append(i)
    return sorted((sorted(g) for g in out.values()), key=len, reverse=True)


def main(datalist, split="training", threshold=0.995, csv_path=None):
    datalist = Path(datalist)
    entries = json.loads(datalist.read_text())[split]
    imgs, labs, sigs = [], [], []
    t0 = time.time()
    for e in entries:
        img = e["image"] if isinstance(e, dict) else e
        img = img[0] if isinstance(img, list) else img
        lab = e.get("label") if isinstance(e, dict) else None
        p = (datalist.parent / img).resolve()
        s = signature(read(p).array())
        if s is None:
            continue
        imgs.append(p)
        labs.append((datalist.parent / lab).resolve() if lab else None)
        sigs.append(s)
    print(f"{len(imgs)} fingerprints in {time.time() - t0:.0f} s")
    m = np.stack(sigs)
    c = (m @ m.T) / m.shape[1]
    ii, jj = np.where(np.triu(c, 1) >= threshold)
    print(f"{len(ii)} pairs at or above {threshold}; measuring each at full resolution...")
    rows = []
    for n, (i, j) in enumerate(zip(ii, jj)):
        r_full, frac = full_res(arr(imgs[i]), arr(imgs[j]))
        rows.append({"a": imgs[i].name, "b": imgs[j].name, "fingerprint_r": round(float(c[i, j]), 5),
                     "full_res_min_channel_r": round(r_full, 5),
                     "voxels_differing_over_1pct": round(frac, 4),
                     "label_dice": round(label_dice(labs[i], labs[j]), 4)})
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(ii)}", flush=True)
    rows.sort(key=lambda r: -r["fingerprint_r"])

    bands = [(0.9999, 1.01), (0.999, 0.9999), (0.995, 0.999)]
    print(f"\n{'fingerprint band':<18}{'pairs':>6}  {'full-res r (min..max)':<24}{'label Dice (median)':<20}")
    for lo, hi in bands:
        sel = [r for r in rows if lo <= r["fingerprint_r"] < hi]
        if not sel:
            print(f"[{lo}, {min(hi, 1)}]".ljust(18) + f"{0:>6}")
            continue
        fr = [r["full_res_min_channel_r"] for r in sel]
        ld = [r["label_dice"] for r in sel if r["label_dice"] == r["label_dice"]]
        print(f"[{lo}, {min(hi, 1)}]".ljust(18) + f"{len(sel):>6}  "
              f"{min(fr):.4f} .. {max(fr):.4f}".ljust(24) + "  "
              + (f"{np.median(ld):.3f}" if ld else "n/a"))

    gs = groups(len(imgs), list(zip(ii.tolist(), jj.tolist())))
    sizes = {}
    for g in gs:
        sizes[len(g)] = sizes.get(len(g), 0) + 1
    print(f"\n{sum(len(g) for g in gs)} images in {len(gs)} group(s); group sizes: "
          + ", ".join(f"{k} images x {v}" for k, v in sorted(sizes.items(), reverse=True)))
    print("largest groups:")
    for g in gs[:5]:
        print("  " + ", ".join(imgs[k].name.replace(".nii.gz", "") for k in g))
    print("\nlowest-fingerprint pairs (the ones that decide the threshold):")
    for r in rows[-8:]:
        print(f"  {r['fingerprint_r']:.4f}  {r['a']} {r['b']}  full-res r {r['full_res_min_channel_r']:.4f}, "
              f"{r['voxels_differing_over_1pct']:.1%} voxels differ, label Dice {r['label_dice']:.3f}")
    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nall pairs written to {csv_path}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    thr, out = 0.995, None
    if "--threshold" in a:
        k = a.index("--threshold"); thr = float(a[k + 1]); del a[k:k + 2]
    if "--csv" in a:
        k = a.index("--csv"); out = a[k + 1]; del a[k:k + 2]
    main(*a, threshold=thr, csv_path=out)
