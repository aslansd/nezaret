"""nezaret result: matching, every rule with its clean twin, the CLI."""

import json
import math
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
nib = pytest.importorskip("nibabel")
pytest.importorskip("scipy")

import nezaret  # noqa: E402
from nezaret.cli import main  # noqa: E402
from nezaret.result.match import case_key, index  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import make_faulty_results as mfr  # noqa: E402


def rules_of(r, severity=None):
    return sorted(f.rule for f in r.findings if severity is None or f.severity == severity)


def save(arr, path, spacing=(1.0, 1.0, 1.0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr, np.diag(list(spacing) + [1.0])), str(path))


def cube(lo, hi, shape=(20, 20, 10), cls=1):
    a = np.zeros(shape, np.uint8)
    a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = cls
    return a


# --- the example, and its clean twin ----------------------------------------------------

@pytest.fixture(scope="module")
def faulty(tmp_path_factory):
    root = mfr.build(tmp_path_factory.mktemp("fr"))
    return nezaret.check_results(root / "labels", {"unet": root / "unet", "swin": root / "swin"})


def test_faulty_example_rules(faulty):
    assert rules_of(faulty) == sorted(["NZ301", "NZ302", "NZ303", "NZ304", "NZ304", "NZ305",
                                       "NZ305", "NZ306", "NZ306", "NZ307"])
    assert faulty.counts() == {"error": 3, "warning": 7, "info": 0}


def test_faulty_example_story(faulty):
    st = faulty.stats
    unet, swin = st["methods"]["unet"], st["methods"]["swin"]
    assert swin["dice_default"] > unet["dice_default"]               # MONAI default: swin
    both = "case-first, empty GT both-empty-1 (MONAI ignore_empty=False)"
    assert unet["dice"][both] > swin["dice"][both]                   # ignore_empty=False: unet
    [pair] = st["pairs"]
    assert pair["bootstrap"]["lo"] < 0 < pair["bootstrap"]["hi"]
    assert unet["cases"] == 23 and swin["cases"] == 22               # 1 missing; 2 not scored


def test_clean_twin(tmp_path):
    root = mfr.build(tmp_path / "c", faults=False)
    r = nezaret.check_results(root / "labels", {"unet": root / "unet", "swin": root / "swin"})
    assert r.counts()["error"] == 0 and r.counts()["warning"] == 0
    assert all(f.rule == "NZ305" for f in r.findings)   # the implementation spread, as info
    assert r.notes == []


# --- matching -------------------------------------------------------------------------------

def test_case_keys():
    assert case_key(Path("pred/case_001/case_001_seg.nii.gz")) == "case_001"
    assert case_key(Path("labelsTs/spleen_12.nii.gz")) == "spleen_12"
    assert case_key(Path("x/spleen_12_trans.nii")) == "spleen_12"
    assert case_key(Path("x/_seg.nii.gz")) == "_seg"
    import re
    assert case_key(Path("a/sub-07/anat/seg.nii.gz"), re.compile(r"(sub-\d+)")) == "sub-07"


def test_ambiguous_keys_are_notes(tmp_path):
    lab = cube((2, 2, 2), (8, 8, 6))
    save(lab, tmp_path / "lab" / "c1.nii.gz")
    save(lab, tmp_path / "p" / "a" / "c1_seg.nii.gz")
    save(lab, tmp_path / "p" / "b" / "c1_seg.nii.gz")
    r = nezaret.check_results(tmp_path / "lab", {"m": tmp_path / "p"})
    assert any("2 files map to case 'c1'" in n for n in r.notes)
    assert rules_of(r) == ["NZ301"]


def test_labels_from_datalist(tmp_path):
    save(cube((2, 2, 2), (8, 8, 6)), tmp_path / "labelsTr" / "c1.nii.gz")
    save(cube((2, 2, 2), (8, 8, 6)), tmp_path / "labelsTr" / "c2.nii.gz")
    save(cube((2, 2, 2), (8, 8, 6)), tmp_path / "p" / "c2.nii.gz")
    (tmp_path / "d.json").write_text(json.dumps({
        "training": [{"image": "i/c1.nii.gz", "label": "labelsTr/c1.nii.gz"}],
        "validation": [{"image": "i/c2.nii.gz", "label": "labelsTr/c2.nii.gz"}]}))
    r = nezaret.check_results(None, {"m": tmp_path / "p"}, datalist=tmp_path / "d.json",
                              split="validation", distances=False)
    assert r.cases == 1 and r.findings == []


# --- per-case problems -------------------------------------------------------------------------

def test_nz301_missing_and_extra(tmp_path):
    for k in ("a", "b"):
        save(cube((2, 2, 2), (8, 8, 6)), tmp_path / "lab" / f"{k}.nii.gz")
    save(cube((2, 2, 2), (8, 8, 6)), tmp_path / "p" / "a.nii.gz")
    save(cube((2, 2, 2), (8, 8, 6)), tmp_path / "p" / "z.nii.gz")
    r = nezaret.check_results(tmp_path / "lab", {"m": tmp_path / "p"}, distances=False)
    assert sorted((f.rule, f.severity) for f in r.findings) == [("NZ301", "error"), ("NZ301", "warning")]


def test_nz302_prediction_in_resampled_space(tmp_path):
    save(cube((2, 2, 2), (8, 8, 6)), tmp_path / "lab" / "a.nii.gz", (0.7, 0.7, 3))
    save(cube((1, 1, 2), (4, 4, 6), shape=(10, 10, 10)), tmp_path / "p" / "a.nii.gz", (1.4, 1.4, 3))
    r = nezaret.check_results(tmp_path / "lab", {"m": tmp_path / "p"})
    assert rules_of(r) == ["NZ302"] and r.stats["methods"]["m"]["cases"] == 0


def test_nz303_probabilities_onehot_and_unknown_class(tmp_path):
    lab = cube((2, 2, 2), (8, 8, 6))
    for k in ("a", "b", "c"):
        save(lab, tmp_path / "lab" / f"{k}.nii.gz")
    save(lab.astype(np.float32) * 0.7, tmp_path / "p" / "a.nii.gz")
    save(np.stack([lab == 0, lab == 1], -1).astype(np.uint8), tmp_path / "p" / "b.nii.gz")
    save(lab + cube((10, 10, 2), (14, 14, 6), cls=7), tmp_path / "p" / "c.nii.gz")
    r = nezaret.check_results(tmp_path / "lab", {"m": tmp_path / "p"}, distances=False)
    got = sorted((f.severity, f.message.split("' ", 1)[1][:15]) for f in r.findings if f.rule == "NZ303")
    assert [s for s, _ in got] == ["error", "error", "warning"]
    assert any("fractional" in f.message for f in r.findings)
    assert any("dimensions" in f.message for f in r.findings)
    assert any("[7]" in f.message for f in r.findings)


def test_float_prediction_holding_integers_is_fine(tmp_path):
    lab = cube((2, 2, 2), (8, 8, 6))
    save(lab, tmp_path / "lab" / "a.nii.gz")
    save(lab.astype(np.float32), tmp_path / "p" / "a.nii.gz")
    r = nezaret.check_results(tmp_path / "lab", {"m": tmp_path / "p"})
    assert r.findings == [] and r.stats["methods"]["m"]["dice_default"] == 1.0


# --- the numbers -------------------------------------------------------------------------------

def test_dice_conventions_by_hand(tmp_path):
    # case a: perfect; case b: empty truth, prediction has a false blob
    save(cube((2, 2, 2), (8, 8, 6)), tmp_path / "lab" / "a.nii.gz")
    save(cube((2, 2, 2), (8, 8, 6)), tmp_path / "p" / "a.nii.gz")
    save(np.zeros((20, 20, 10), np.uint8), tmp_path / "lab" / "b.nii.gz")
    save(cube((10, 10, 2), (12, 12, 4)), tmp_path / "p" / "b.nii.gz")
    r = nezaret.check_results(tmp_path / "lab", {"m": tmp_path / "p"}, distances=False)
    d = r.stats["methods"]["m"]["dice"]
    assert d["case-first, empty GT exclude (MONAI default)"] == 1.0
    assert d["case-first, empty GT both-empty-1 (MONAI ignore_empty=False)"] == 0.5
    assert d["case-first, empty GT zero"] == 0.5
    tp, gt, pr = 6 * 6 * 4, 6 * 6 * 4, 6 * 6 * 4 + 2 * 2 * 2
    assert d["pooled over cases"] == pytest.approx(2 * tp / (gt + pr))
    assert rules_of(r) == ["NZ304"]


def test_hd95_by_hand_and_voxel_units(tmp_path):
    # two cubes 2 voxels apart along z, spacing 3 mm there: every distance is 6 mm or 2 voxels
    lab = cube((4, 4, 2), (12, 12, 6))
    pred = cube((4, 4, 4), (12, 12, 8))
    save(lab, tmp_path / "lab" / "a.nii.gz", (1, 1, 3))
    save(pred, tmp_path / "p" / "a.nii.gz", (1, 1, 3))
    r = nezaret.check_results(tmp_path / "lab", {"m": tmp_path / "p"})
    m = r.stats["methods"]["m"]
    assert m["hd95_default"] == pytest.approx(6.0)
    assert m["hd95"]["monai, voxel, 6-neighbour edges, empty exclude"] == pytest.approx(2.0)
    [f] = [f for f in r.findings if f.rule == "NZ305" and "HD95" in f.message]
    assert f.severity == "warning" and "voxel units" in f.message


def test_isotropic_1mm_voxel_units_do_not_warn(tmp_path):
    save(cube((4, 4, 2), (12, 12, 6)), tmp_path / "lab" / "a.nii.gz")
    save(cube((4, 4, 3), (12, 12, 7)), tmp_path / "p" / "a.nii.gz")
    r = nezaret.check_results(tmp_path / "lab", {"m": tmp_path / "p"})
    assert all(f.severity == "info" for f in r.findings if f.rule == "NZ305")


def test_worst_case_uses_the_image_diagonal(tmp_path):
    save(cube((4, 4, 2), (12, 12, 6)), tmp_path / "lab" / "a.nii.gz", (1, 1, 2))
    save(np.zeros((20, 20, 10), np.uint8), tmp_path / "p" / "a.nii.gz", (1, 1, 2))
    r = nezaret.check_results(tmp_path / "lab", {"m": tmp_path / "p"})
    hd = r.stats["methods"]["m"]["hd95"]
    assert math.isnan(hd["monai, mm, 6-neighbour edges, empty exclude"])
    assert hd["monai, mm, 6-neighbour edges, empty worst"] == pytest.approx(math.sqrt(20**2 + 20**2 + 20**2))


def test_classes_option_and_no_distances(tmp_path):
    lab = cube((2, 2, 2), (8, 8, 6)) + cube((10, 10, 2), (14, 14, 6), cls=2)
    save(lab, tmp_path / "lab" / "a.nii.gz")
    save(lab, tmp_path / "p" / "a.nii.gz")
    r = nezaret.check_results(tmp_path / "lab", {"m": tmp_path / "p"}, classes=[2], distances=False)
    assert r.stats["classes"] == [2] and r.stats["methods"]["m"]["hd95"] == {}


def test_ranking_robust_and_resolved(tmp_path):
    """A clearly better method on enough cases: no NZ306, no NZ307."""
    rng = np.random.default_rng(1)
    for i in range(12):
        lab = cube((3, 3, 2), (13, 13, 8))
        save(lab, tmp_path / "lab" / f"c{i}.nii.gz")
        save(np.roll(lab, int(rng.integers(0, 2)), 0), tmp_path / "good" / f"c{i}.nii.gz")
        save(np.roll(lab, int(rng.integers(3, 5)), 0), tmp_path / "bad" / f"c{i}.nii.gz")
    r = nezaret.check_results(tmp_path / "lab", {"good": tmp_path / "good", "bad": tmp_path / "bad"})
    assert "NZ306" not in rules_of(r) and "NZ307" not in rules_of(r)
    [p] = r.stats["pairs"]
    assert p["bootstrap"]["lo"] > 0 and p["bootstrap"]["p_a_better"] == 1.0


def test_bootstrap_is_seeded(tmp_path):
    root = mfr.build(tmp_path / "f")
    args = (root / "labels", {"unet": root / "unet", "swin": root / "swin"})
    a = nezaret.check_results(*args, distances=False).stats["pairs"][0]["bootstrap"]
    b = nezaret.check_results(*args, distances=False).stats["pairs"][0]["bootstrap"]
    c = nezaret.check_results(*args, distances=False, seed=5).stats["pairs"][0]["bootstrap"]
    assert a == b and a != c


# --- CLI ---------------------------------------------------------------------------------------

def test_cli(tmp_path, capsys):
    root = mfr.build(tmp_path / "f")
    base = ["result", "--labels", str(root / "labels"), "--pred", f"unet={root / 'unet'}",
            "--pred", f"swin={root / 'swin'}", "--no-distances"]
    assert main(base) == 1
    assert main(base + ["--ignore", "NZ301,NZ302,NZ303"]) == 0
    assert main(base + ["--ignore", "NZ301,NZ302,NZ303", "--fail-on", "warning"]) == 1
    out = capsys.readouterr().out
    assert "Scores" in out and "P(unet better)" in out
    main(base + ["--format", "json"])
    data = json.loads(capsys.readouterr().out)
    assert set(data["stats"]["methods"]) == {"unet", "swin"} and data["cases"] == 24
    main(base + ["--format", "markdown"])
    assert "## Scores" in capsys.readouterr().out
    assert main(["result", "--labels", str(root / "labels"), "--pred", "nonsense"]) == 2
    assert main(["result", "--pred", f"u={root / 'unet'}"]) == 2
    assert main(["result", "--labels", str(tmp_path / "empty"), "--pred", f"u={root / 'unet'}"]) == 2
