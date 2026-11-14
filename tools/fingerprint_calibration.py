"""Calibrate NZ207's near-duplicate threshold on a real cohort (ROADMAP §2, step 3).

Computes nezaret's own fingerprint for every image in a datalist split and
reports how alike *different* subjects get. NZ207 fires at a correlation of
0.995; if different subjects reach that, the threshold is too low for this
kind of data.

    python fingerprint_calibration.py DATASET.json [SPLIT] [--limit N]

SPLIT defaults to "training". Takes a few seconds per case (decompression
dominates). Prints the highest between-subject correlations, a few
percentiles, and how many pairs would fire at 0.995. Writes nothing; copy
the summary into the changelog.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from nezaret.dataset.readers import read, signature


def main(datalist, split="training", limit=None):
    datalist = Path(datalist)
    entries = json.loads(datalist.read_text())[split]
    if limit:
        entries = entries[:limit]
    names, sigs = [], []
    t0 = time.time()
    for i, e in enumerate(entries):
        img = e["image"] if isinstance(e, dict) else e
        img = img[0] if isinstance(img, list) else img
        path = (datalist.parent / img).resolve()
        s = signature(read(path).array())
        if s is not None:
            names.append(path.name)
            sigs.append(s)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(entries)} read, {time.time() - t0:.0f} s", flush=True)
    m = np.stack(sigs)
    c = (m @ m.T) / m.shape[1]
    iu = np.triu_indices(len(names), 1)
    vals = c[iu]
    order = np.argsort(vals)[::-1]
    print(f"\n{len(names)} images, {vals.size} pairs of different entries")
    print("highest correlations (look at these pairs: a real duplicate is possible):")
    for k in order[:10]:
        i, j = iu[0][k], iu[1][k]
        print(f"  {vals[k]:.4f}  {names[i]}  {names[j]}")
    for q in (50, 99, 99.9):
        print(f"  {q:>5}th percentile: {np.percentile(vals, q):.4f}")
    print(f"pairs at or above 0.995 (NZ207's threshold): {int((vals >= 0.995).sum())}")
    print(f"highest between-subject correlation: {vals.max():.4f}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
        args = [a for a in args if a != str(limit)]
    if not args:
        sys.exit(__doc__)
    main(*args, limit=limit)
