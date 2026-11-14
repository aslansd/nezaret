"""Per-case, per-class overlap and surface-distance numbers.

Everything the scoring variants need is computed once per (case, class) and
kept as plain numbers, so trying another variant costs nothing.

The surface-distance code follows the two implementations it imitates, read
from their source:

* MONAI 1.6.1 ``compute_hausdorff_distance``: edges are
  ``binary_erosion(mask) ^ mask`` (scipy's default 6-neighbour structure in
  3D); HD95 is ``max(p95(d(pred->gt)), p95(d(gt->pred)))``; spacing ``None``
  means voxel units.
* MedPy 0.5.2 ``hd95``: the same edges (``connectivity=1``), but
  ``percentile(hstack(d1, d2), 95)``: one percentile over both directions
  pooled; ``voxelspacing=None`` also means voxel units.
* ASSD: both libraries pool the two directions before averaging.

``tests/test_live_metrics.py`` checks both against the real libraries.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class MetricsUnavailable(Exception):
    pass


def _np():
    try:
        import numpy as np
        return np
    except ImportError as exc:  # pragma: no cover
        raise MetricsUnavailable("needs numpy (pip install nezaret[result])") from exc


def _ndimage():
    try:
        from scipy import ndimage
        return ndimage
    except ImportError as exc:
        raise MetricsUnavailable("surface distances need scipy (pip install nezaret[result])") from exc


@dataclass
class ClassScore:
    """Numbers for one class in one case."""
    gt: int                    # voxels in ground truth
    pred: int                  # voxels in prediction
    tp: int                    # voxels in both
    # directed surface distances, keyed by (connectivity, units):
    # {(1, "mm"): (d_pred_to_gt, d_gt_to_pred), ...}; empty when either mask is empty
    dist: dict = field(default_factory=dict)
    worst: dict = field(default_factory=dict)   # units -> distance used for "worst"

    @property
    def gt_empty(self) -> bool:
        return self.gt == 0

    @property
    def pred_empty(self) -> bool:
        return self.pred == 0


def _edges(mask, connectivity: int):
    nd = _ndimage()
    structure = nd.generate_binary_structure(mask.ndim, connectivity)
    return nd.binary_erosion(mask, structure=structure) ^ mask


def _directed(a_pts, b_tree):
    """Distances from every edge point of a to the nearest edge point of b.

    Equal to ``distance_transform_edt(~b_edges, sampling)[a_edges]``, the
    way MONAI and MedPy compute it: the EDT is the exact Euclidean distance
    to the nearest zero voxel, and the zeros are b's edge voxels. A KD-tree
    over those voxels' coordinates (scaled by spacing) finds the same
    nearest voxel without visiting every other voxel in the volume.
    """
    d, _ = b_tree.query(a_pts, k=1)
    return d


def score_class(gt, pred, spacing, distances: bool = True,
                connectivities=(1, 3)) -> ClassScore:
    np = _np()
    gt = np.asarray(gt, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    s = ClassScore(gt=int(gt.sum()), pred=int(pred.sum()), tp=int((gt & pred).sum()))
    diag_vox = float(np.sqrt(sum(n * n for n in gt.shape)))
    diag_mm = float(np.sqrt(sum((n * sp) ** 2 for n, sp in zip(gt.shape, spacing))))
    s.worst = {"mm": diag_mm, "voxel": diag_vox}
    if not distances or s.gt_empty or s.pred_empty:
        return s
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise MetricsUnavailable("surface distances need scipy (pip install nezaret[result])") from exc
    # crop to the union's bounding box, padded by one zero voxel so erosion
    # treats the outside as background exactly as on the full array
    union = gt | pred
    sl = []
    for ax in range(union.ndim):
        other = tuple(i for i in range(union.ndim) if i != ax)
        nz = np.flatnonzero(union.any(axis=other))
        sl.append(slice(int(nz[0]), int(nz[-1]) + 1))
    sl = tuple(sl)
    g, p = np.pad(gt[sl], 1), np.pad(pred[sl], 1)
    scale = np.asarray([float(x) for x in spacing][: g.ndim])
    for conn in connectivities:
        if conn > g.ndim:
            continue
        gi = np.argwhere(_edges(g, conn)).astype(float)
        pi = np.argwhere(_edges(p, conn)).astype(float)
        for units, f in (("mm", scale), ("voxel", 1.0)):
            gp, pp = gi * f, pi * f
            s.dist[(conn, units)] = (_directed(pp, cKDTree(gp)), _directed(gp, cKDTree(pp)))
    return s


def hd95(d_pg, d_gp, style: str) -> float:
    np = _np()
    if style == "monai":
        return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))
    if style == "medpy":
        return float(np.percentile(np.hstack((d_pg, d_gp)), 95))
    raise ValueError(style)


def assd(d_pg, d_gp, style: str = "pooled") -> float:
    """Average symmetric surface distance.

    * ``pooled``: mean over both directions' distances together. Both MONAI 1.6.1
      (``torch.cat(distances).mean()``) and MedPy 0.5.2 (``concatenate(...).mean()``)
      compute it this way.
    * ``mean-of-means``: mean of the two directed means, the other definition in
      use; it weights the smaller surface as much as the larger one.
    """
    np = _np()
    if style == "pooled":
        return float(np.mean(np.hstack((d_pg, d_gp))))
    if style == "mean-of-means":
        return float((np.mean(d_pg) + np.mean(d_gp)) / 2.0)
    raise ValueError(style)


def dice(s: ClassScore, empty: str) -> float:
    """Dice for one class under one empty-ground-truth policy.

    * ``exclude``: NaN when the ground truth is empty (MONAI ignore_empty=True, the default)
    * ``both-empty-1``: 1 if both empty, 0 if only the truth is (MONAI ignore_empty=False)
    * ``zero``: 0 whenever the ground truth is empty, as a smoothed 2|A∩B|/(|A|+|B|+eps) gives
    """
    if not s.gt_empty:
        return 2.0 * s.tp / (s.gt + s.pred)
    if empty == "exclude":
        return float("nan")
    if empty == "both-empty-1":
        return 1.0 if s.pred_empty else 0.0
    if empty == "zero":
        return 0.0
    raise ValueError(empty)
