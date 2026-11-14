"""nezaret's metrics against the libraries they imitate. Skips without them.

Every convention nezaret names after a library must give that library's
number: per case, and for the aggregate over cases.
"""

import math

import pytest

np = pytest.importorskip("numpy")
ndimage = pytest.importorskip("scipy.ndimage")

from nezaret.result import metrics as M  # noqa: E402
from nezaret.result.analysis import CaseResult, MethodResult, aggregate_dice  # noqa: E402


def blobs(n=20, seed=0, empty_every=0):
    rng = np.random.default_rng(seed)
    out = []
    for t in range(n):
        shape = tuple(int(x) for x in rng.integers(18, 40, 3))
        sp = tuple(float(x) for x in np.round(rng.uniform(0.5, 3, 3), 3))
        g = ndimage.gaussian_filter(rng.random(shape), 3) > 0.5
        p = ndimage.gaussian_filter(rng.random(shape), 3) > 0.5
        if t % 4 == 0:
            g[0] = True          # a mask touching the volume border
        if empty_every and t % empty_every == 0:
            g[:] = False
        out.append((g, p, sp))
    return out


# --- MONAI ----------------------------------------------------------------------------

def test_per_case_against_monai():
    torch = pytest.importorskip("torch")
    from monai.metrics import (compute_average_surface_distance, compute_dice,
                               compute_hausdorff_distance)
    for g, p, sp in blobs():
        if not g.any() or not p.any():
            continue
        s = M.score_class(g, p, sp)
        G, P = torch.tensor(g[None, None]), torch.tensor(p[None, None])
        assert M.dice(s, "exclude") == pytest.approx(float(compute_dice(P, G)[0, 0]), abs=1e-6)
        mm = float(compute_hausdorff_distance(P, G, percentile=95, spacing=list(sp))[0, 0])
        vox = float(compute_hausdorff_distance(P, G, percentile=95)[0, 0])
        assd = float(compute_average_surface_distance(P, G, symmetric=True, spacing=list(sp))[0, 0])
        assert M.hd95(*s.dist[(1, "mm")], "monai") == pytest.approx(mm, abs=1e-4)
        assert M.hd95(*s.dist[(1, "voxel")], "monai") == pytest.approx(vox, abs=1e-4)
        assert M.assd(*s.dist[(1, "mm")], "pooled") == pytest.approx(assd, abs=1e-4)


def _method(cases):
    m = MethodResult("m")
    for i, (g, p) in enumerate(cases):
        cr = CaseResult(str(i))
        for c in (1, 2):
            cr.classes[c] = M.score_class(g == c, p == c, (1, 1, 1), distances=False)
        m.cases[str(i)] = cr
    return m


def _batch(seed=3):
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(9):
        g = np.zeros((12, 12, 6), np.uint8)
        p = np.zeros_like(g)
        if i % 3:                                   # every third case has no class 2
            g[2:8, 2:8, 1:4] = 2
        if i % 4:
            g[6:11, 5:11, 2:5] = 1
        p[rng.integers(1, 4):9, 2:8, 1:4] = 2 if i % 2 else 0
        p[6:11, rng.integers(4, 7):11, 2:5] = 1
        cases.append((g, p))
    return cases


def _onehot(a):
    import torch
    return torch.tensor(np.stack([a == 1, a == 2])[None].astype(np.float32))


@pytest.mark.parametrize("ignore_empty,empty", [(True, "exclude"), (False, "both-empty-1")])
def test_aggregates_against_monai_dice_metric(ignore_empty, empty):
    """case-first == DiceMetric(reduction='mean'); class-first == mean of 'mean_batch'."""
    pytest.importorskip("torch")
    import torch
    from monai.metrics import DiceMetric
    cases = _batch()
    m = _method(cases)
    keys = list(m.cases)
    for reduction, agg in (("mean", "case-first"), ("mean_batch", "class-first")):
        dm = DiceMetric(include_background=True, reduction=reduction, ignore_empty=ignore_empty)
        for g, p in cases:
            dm(_onehot(p), _onehot(g))
        got = dm.aggregate()
        want = float(got) if reduction == "mean" else float(torch.nanmean(got))
        assert aggregate_dice(m, keys, [1, 2], (empty, agg)) == pytest.approx(want, abs=1e-6), reduction


def test_the_conventions_really_differ_on_that_batch():
    m = _method(_batch())
    keys = list(m.cases)
    vals = {(e, a): aggregate_dice(m, keys, [1, 2], (e, a))
            for e in ("exclude", "both-empty-1", "zero") for a in ("case-first", "class-first")}
    assert len({round(v, 6) for v in vals.values()}) >= 4


# --- MedPy ----------------------------------------------------------------------------

def test_per_case_against_medpy():
    binary = pytest.importorskip("medpy.metric.binary")
    for g, p, sp in blobs(seed=5):
        if not g.any() or not p.any():
            continue
        s = M.score_class(g, p, sp)
        for conn in (1, 3):
            assert M.hd95(*s.dist[(conn, "mm")], "medpy") == pytest.approx(
                binary.hd95(p, g, voxelspacing=sp, connectivity=conn), abs=1e-9)
            assert M.assd(*s.dist[(conn, "mm")], "pooled") == pytest.approx(
                binary.assd(p, g, voxelspacing=sp, connectivity=conn), abs=1e-9)
        assert M.hd95(*s.dist[(1, "voxel")], "medpy") == pytest.approx(binary.hd95(p, g), abs=1e-9)
        assert M.dice(s, "exclude") == pytest.approx(binary.dc(p, g), abs=1e-12)


def test_monai_and_medpy_hd95_really_differ():
    vals = []
    for g, p, sp in blobs(seed=9):
        if g.any() and p.any():
            s = M.score_class(g, p, sp)
            a, b = (M.hd95(*s.dist[(1, "mm")], st) for st in ("monai", "medpy"))
            assert a >= b - 1e-9          # max of two percentiles >= percentile of the pool
            vals.append(a - b)
    assert max(vals) > 0.5


def test_kdtree_equals_distance_transform():
    """The shortcut in _directed equals the EDT lookup the libraries do."""
    for g, p, sp in blobs(n=8, seed=11):
        if not (g.any() and p.any()):
            continue
        s = M.score_class(g, p, sp)
        ge = ndimage.binary_erosion(g) ^ g
        pe = ndimage.binary_erosion(p) ^ p
        d_pg = ndimage.distance_transform_edt(~ge, sampling=sp)[pe]
        d_gp = ndimage.distance_transform_edt(~pe, sampling=sp)[ge]
        a, b = s.dist[(1, "mm")]
        assert np.allclose(np.sort(a), np.sort(d_pg)) and np.allclose(np.sort(b), np.sort(d_gp))
