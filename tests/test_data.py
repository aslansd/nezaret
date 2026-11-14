"""nezaret data on real files: NIfTI written with nibabel, the nibabel sample
scans, and DICOM series written with pydicom. Skips without them."""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
nib = pytest.importorskip("nibabel")

import nezaret  # noqa: E402
from nezaret.dataset.readers import read, signature  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES))
import make_faulty_dataset as mfd  # noqa: E402

AFF = mfd.AFFINE
REGEX = dict(patient_id_regex=r"(p\d{3})", site_regex=r"(site[A-Z])")


def rules_of(r, severity=None):
    return sorted(f.rule for f in r.findings if severity is None or f.severity == severity)


def save(arr, path, affine=AFF):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr, affine), str(path))


def datalist(tmp_path, data):
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(data))
    return p


def pair(tmp_path, name, img=None, lab=None, img_aff=AFF, lab_aff=AFF, seed=0):
    i, l = mfd.phantom(seed)
    save(i if img is None else img, tmp_path / "img" / name, img_aff)
    save(l if lab is None else lab, tmp_path / "lab" / name, lab_aff)
    return {"image": f"img/{name}", "label": f"lab/{name}"}


# --- the example, and its clean twin ------------------------------------------------

def test_faulty_example_every_rule(tmp_path):
    r = nezaret.check_data(mfd.build(tmp_path / "f"), **REGEX)
    assert rules_of(r) == sorted(["NZ201", "NZ201", "NZ202", "NZ203", "NZ204", "NZ205",
                                  "NZ206", "NZ206", "NZ207", "NZ207", "NZ208"])
    assert r.counts() == {"error": 8, "warning": 2, "info": 1}
    # both planted copies are the same scan (one exact, one rescaled), so both are errors
    assert [f.severity for f in r.findings if f.rule == "NZ207"] == ["error", "error"]


def test_faulty_example_without_path_hints(tmp_path):
    r = nezaret.check_data(mfd.build(tmp_path / "f"))
    got = rules_of(r)
    assert "NZ208" not in got                 # no scanner metadata in NIfTI, no --site-regex
    assert got.count("NZ206") == 1            # only the shared entry; p005 needs the regex


def test_clean_twin_is_clean(tmp_path):
    r = nezaret.check_data(mfd.build(tmp_path / "c", faults=False), **REGEX)
    assert r.findings == [] and r.notes == []
    assert r.cases == 14 and r.stats["patients"] == 14


# --- NZ202 geometry -------------------------------------------------------------------

@pytest.mark.parametrize("lab_aff,expect", [
    (np.eye(4), "spacing"),
    (AFF @ np.diag([1, 1, 1, 1]) + np.array([[0, 0, 0, 5], [0] * 4, [0] * 4, [0] * 4]), "origin"),
    (np.diag([1.5, 1.5, 2.5, 1]) @ np.eye(4) + np.array([[0, 0, 0, 18], [0, 0, 0, -20],
                                                         [0, 0, 0, -12], [0, 0, 0, 0]]), "direction"),
])
def test_nz202_each_part(tmp_path, lab_aff, expect):
    dl = datalist(tmp_path, {"training": [pair(tmp_path, "a.nii.gz", lab_aff=lab_aff)]})
    [f] = nezaret.check_data(dl).findings
    assert f.rule == "NZ202" and expect in f.message


def test_nz202_shape(tmp_path):
    i, l = mfd.phantom(1)
    dl = datalist(tmp_path, {"training": [pair(tmp_path, "a.nii.gz", lab=l[:, :, :-2])]})
    [f] = nezaret.check_data(dl).findings
    assert "shape (24, 24, 16) vs (24, 24, 14)" in f.message


def test_nz202_float32_header_noise_is_tolerated(tmp_path):
    aff = AFF.copy()
    aff[0, 3] += 1e-4
    aff[1, 1] *= 1 + 1e-6
    dl = datalist(tmp_path, {"training": [pair(tmp_path, "a.nii.gz", lab_aff=aff)]})
    assert nezaret.check_data(dl).findings == []


def test_nz202_multichannel_image_against_3d_label(tmp_path):
    i, l = mfd.phantom(2)
    dl = datalist(tmp_path, {"training": [pair(tmp_path, "a.nii.gz", img=np.stack([i, i], -1), lab=l)]})
    assert nezaret.check_data(dl).findings == []


def test_nz202_real_reoriented_scan(tmp_path):
    d = Path(os.path.dirname(nib.__file__)) / "tests" / "data"
    shutil.copy(d / "anatomical.nii", tmp_path / "anat.nii")
    shutil.copy(d / "reoriented_anat_moved.nii", tmp_path / "moved_seg.nii")
    dl = datalist(tmp_path, {"training": [{"image": "anat.nii", "label": "moved_seg.nii"}]})
    r = nezaret.check_data(dl)
    nz202 = [f for f in r.findings if f.rule == "NZ202"]
    assert nz202 and "shape (33, 41, 25) vs (21, 26, 22)" in nz202[0].message
    assert "NZ203" in rules_of(r)  # and a float scan is not a label map


# --- NZ203 / NZ204 / NZ205 labels -----------------------------------------------------------

def test_nz203_negative_and_image_as_label(tmp_path):
    i, l = mfd.phantom(3)
    neg = l.astype(np.int16)
    neg[0, 0, 0] = -1
    dl = datalist(tmp_path, {"training": [pair(tmp_path, "a.nii.gz", lab=neg, seed=3),
                                          pair(tmp_path, "b.nii.gz", lab=np.random.default_rng(4).integers(0, 3000, i.shape).astype(np.int16), seed=4)]})
    msgs = [f.message for f in nezaret.check_data(dl).findings if f.rule == "NZ203"]
    assert any("negative" in m for m in msgs)
    assert any("distinct values" in m for m in msgs)


def test_nz203_float_label_with_integer_values_is_clean(tmp_path):
    _, l = mfd.phantom(5)
    dl = datalist(tmp_path, {"training": [pair(tmp_path, "a.nii.gz", lab=l.astype(np.float32))]})
    assert nezaret.check_data(dl).findings == []


def test_nz204_counts_per_split(tmp_path):
    _, empty = mfd.phantom(6, empty=True)
    dl = datalist(tmp_path, {"training": [pair(tmp_path, "a.nii.gz", seed=1),
                                          pair(tmp_path, "b.nii.gz", lab=empty, seed=2)]})
    [f] = nezaret.check_data(dl).findings
    assert (f.rule, f.severity) == ("NZ204", "info")
    assert "1 of 2 label maps in 'training'" in f.message


def test_nz205_class_only_in_evaluation(tmp_path):
    i, l1 = mfd.phantom(7, classes=(1,))
    _, l3 = mfd.phantom(8, classes=(1, 3))
    dl = datalist(tmp_path, {"training": [pair(tmp_path, "a.nii.gz", lab=l1, seed=7)],
                             "validation": [pair(tmp_path, "b.nii.gz", lab=l3, seed=8)]})
    [f] = nezaret.check_data(dl).findings
    assert f.rule == "NZ205" and "[3] appear in 'validation' but in no training data" in f.message


def test_nz205_per_fold(tmp_path):
    entries = []
    for k, classes in enumerate([(1, 2), (1, 2), (1,)]):
        _, l = mfd.phantom(10 + k, classes=classes)
        e = pair(tmp_path, f"f{k}.nii.gz", lab=l, seed=10 + k)
        e["fold"] = k
        entries.append(e)
    r = nezaret.check_data(datalist(tmp_path, {"training": entries}))
    [f] = r.findings
    assert f.location == "training[fold=2]" and "[2]" in f.message


# --- NZ207 duplicates -------------------------------------------------------------------------

def test_nz207_exact_and_rescaled_copies_form_one_group(tmp_path):
    i, _ = mfd.phantom(20)
    save(i, tmp_path / "a.nii.gz")
    save(i, tmp_path / "a_copy.nii.gz")
    save(i.astype(np.float32) * 2.5 - 7, tmp_path / "a_rescaled.nii.gz")
    dl = datalist(tmp_path, {"training": [{"image": "a.nii.gz"}],
                             "testing": [{"image": "a_copy.nii.gz"}, {"image": "a_rescaled.nii.gz"}]})
    r = nezaret.check_data(dl)
    [f] = r.findings
    assert (f.rule, f.severity) == ("NZ207", "error") and "identical voxels" in f.message
    [g] = r.stats["duplicate_groups"]
    assert len(g["images"]) == 3 and g["kind"] == "exact" and g["splits"] == ["training", "testing"]


def test_nz207_within_one_split_is_a_summary_warning(tmp_path):
    entries = []
    for k in range(3):
        i, _ = mfd.phantom(40 + k)
        save(i, tmp_path / f"s{k}.nii.gz")
        save(i.astype(np.float32) * 1.7, tmp_path / f"s{k}_again.nii.gz")
        entries += [{"image": f"s{k}.nii.gz"}, {"image": f"s{k}_again.nii.gz"}]
    i, _ = mfd.phantom(50)
    save(i, tmp_path / "unique.nii.gz")
    entries.append({"image": "unique.nii.gz"})
    r = nezaret.check_data(datalist(tmp_path, {"training": entries}))
    [f] = r.findings                                   # one summary, not one per pair
    assert (f.rule, f.severity) == ("NZ207", "warning")
    assert "'training' holds 3 group(s)" in f.message and "6 images; 3 group(s) are the same scan" in f.message
    assert len(r.stats["duplicate_groups"]) == 3
    assert "split by group" in f.message


def test_nz207_duplicates_inside_a_test_split_say_why_that_matters(tmp_path):
    i, _ = mfd.phantom(60)
    save(i, tmp_path / "t1.nii.gz")
    save(i.astype(np.float32) * 2, tmp_path / "t1_again.nii.gz")
    r = nezaret.check_data(datalist(tmp_path, {"test": [{"image": "t1.nii.gz"},
                                                        {"image": "t1_again.nii.gz"}]}))
    [f] = r.findings
    assert "counts twice in the mean score" in f.message and "split by group" not in f.message


def _atlas_cohort(tmp_path, rng):
    """BraTS-like: every subject aligned to one template, so different subjects correlate highly."""
    from scipy import ndimage
    template = ndimage.gaussian_filter(rng.normal(size=(32, 32, 24)), 3) * 40 + 200
    def subject(sd):
        return template + ndimage.gaussian_filter(rng.normal(size=template.shape), 2) * sd
    return template, subject


def test_nz207_brats_like_patterns(tmp_path):
    """The three patterns MSD Task01 showed, plus unrelated subjects that must not group."""
    rng = np.random.default_rng(3)
    _, subject = _atlas_cohort(tmp_path, rng)
    tr, ts = [], []
    def add(lst, name, img):
        save(img.astype(np.float32), tmp_path / f"{name}.nii.gz")
        lst.append({"image": f"{name}.nii.gz"})
    base = [subject(10) for _ in range(6)]           # different subjects correlate ~0.9, as in BraTS
    for k, img in enumerate(base):
        add(tr, f"sub{k}", img)                      # unrelated subjects: no group
    add(ts, "sub0_reexport", base[0] * 1.3 + 5)      # the same scan re-normalised, in test
    from scipy import ndimage
    follow = base[1] + ndimage.gaussian_filter(np.random.default_rng(9).normal(size=base[1].shape), 2) * 2
    add(ts, "sub1_followup", follow)
    r = nezaret.check_data(datalist(tmp_path, {"training": tr, "testing": ts}))
    kinds = sorted((f.severity, f.message.split(" across")[0][:20]) for f in r.findings if f.rule == "NZ207")
    assert [k[0] for k in kinds] == ["error", "warning"]
    assert {len(g["images"]) for g in r.stats["duplicate_groups"]} == {2}
    assert len(r.stats["duplicate_groups"]) == 2   # the six unrelated subjects stay apart


def test_nz207_threshold_and_constant_images(tmp_path):
    i, _ = mfd.phantom(22)
    noisy = i + np.random.default_rng(0).normal(0, 40, i.shape)
    save(i, tmp_path / "a.nii.gz")
    save(noisy.astype(np.float32), tmp_path / "b.nii.gz")
    save(np.zeros(mfd.SHAPE, np.int16), tmp_path / "z1.nii.gz")
    save(np.ones(mfd.SHAPE, np.int16), tmp_path / "z2.nii.gz")
    dl = datalist(tmp_path, {"training": [{"image": "a.nii.gz"}, {"image": "z1.nii.gz"}],
                             "testing": [{"image": "b.nii.gz"}, {"image": "z2.nii.gz"}]})
    r_default = nezaret.check_data(dl)
    r_loose = nezaret.check_data(dl, near_duplicate=0.5)
    assert r_default.findings == []
    assert [f.rule for f in r_loose.findings] == ["NZ207"]


def test_real_scans_are_not_near_duplicates_of_each_other(tmp_path):
    """Calibration on nibabel's real scans (see the 0.2.0 changelog)."""
    d = Path(os.path.dirname(nib.__file__)) / "tests" / "data"
    for f in ("anatomical.nii", "reoriented_anat_moved.nii", "resampled_anat_moved.nii",
              "functional.nii", "example4d.nii.gz"):
        shutil.copy(d / f, tmp_path / f)
    dl = datalist(tmp_path, {"training": [{"image": "anatomical.nii"}, {"image": "functional.nii"}],
                             "testing": [{"image": "reoriented_anat_moved.nii"},
                                         {"image": "resampled_anat_moved.nii"},
                                         {"image": "example4d.nii.gz"}]})
    assert nezaret.check_data(dl).findings == []


def test_signature_invariances():
    i, _ = mfd.phantom(30)
    s = signature(i)
    assert float(s @ signature(i.astype(np.float64) * 3 + 100)) / s.size > 0.9999
    with_nan = i.astype(np.float32)
    with_nan[0, 0, 0] = np.nan
    assert signature(with_nan) is not None
    assert signature(np.zeros((4, 4, 4))) is None
    assert signature(np.arange(10.0)).size == 16 ** 3   # thin inputs are stretched


# --- headers only ---------------------------------------------------------------------------

def test_no_pixels(tmp_path):
    r_full = nezaret.check_data(mfd.build(tmp_path / "f"), **REGEX)
    r_head = nezaret.check_data(mfd.build(tmp_path / "f"), pixels=False, **REGEX)
    assert set(rules_of(r_head)) == {"NZ201", "NZ202", "NZ206", "NZ208"}
    assert set(rules_of(r_head)) < set(rules_of(r_full))
    assert any("headers only" in n for n in r_head.notes)


# --- DICOM ------------------------------------------------------------------------------------

def test_dicom_series_leakage_and_scanners(tmp_path):
    pytest.importorskip("pydicom")
    from dicom_util import write_series
    for k in range(3):
        write_series(tmp_path / f"tr{k}", patient_id=f"PAT{k}", seed=k, manufacturer="SIEMENS")
    write_series(tmp_path / "te0", patient_id="PAT1", seed=50, manufacturer="GE")   # PAT1 again
    write_series(tmp_path / "te1", patient_id="PAT9", seed=51, manufacturer="GE")
    write_series(tmp_path / "te2", patient_id="PAT8", seed=52, manufacturer="GE")
    dl = datalist(tmp_path, {"training": [{"image": f"tr{k}"} for k in range(3)],
                             "testing": [{"image": f"te{k}"} for k in range(3)]})
    r = nezaret.check_data(dl)
    assert rules_of(r) == ["NZ206", "NZ208"]   # one NZ208 per DICOM attribute
    text = r.to_text() + r.to_json()
    assert "PAT1" not in text                     # never print a patient id
    nz208 = " ".join(f.message for f in r.findings if f.rule == "NZ208")
    assert "'GE' only in testing (3)" in nz208 and "'SIEMENS' only in training (3)" in nz208


def test_dicom_series_geometry_and_duplicates(tmp_path):
    pytest.importorskip("pydicom")
    from dicom_util import write_series
    vol = write_series(tmp_path / "a", patient_id="A", seed=1)
    write_series(tmp_path / "b", patient_id="B", seed=2, pixels=vol)        # same voxels
    v = read(tmp_path / "a")
    assert v.shape == (20, 16, 4) and v.spacing == (0.8, 0.8, 2.0)
    assert v.array().shape == (20, 16, 4)
    assert np.array_equal(v.array()[:, :, 2], vol[2].T)  # slices sorted by position, not filename
    dl = datalist(tmp_path, {"training": [{"image": "a"}], "testing": [{"image": "b"}]})
    assert [(f.rule, f.severity) for f in nezaret.check_data(dl).findings] == [("NZ207", "error")]


def test_real_dicom_files(tmp_path):
    pytest.importorskip("pydicom")
    from pydicom.data import get_testdata_file
    for f in ("CT_small.dcm", "MR_small.dcm"):
        p = get_testdata_file(f)
        if p is None:
            pytest.skip("pydicom test files not available")
        shutil.copy(p, tmp_path / f)
    v = read(tmp_path / "CT_small.dcm")
    assert v.shape == (128, 128) and v.patient_id == "1CT1"
    assert v.scanner["Manufacturer"] == "GE MEDICAL SYSTEMS"
    dl = datalist(tmp_path, {"training": [{"image": "CT_small.dcm"}],
                             "testing": [{"image": "MR_small.dcm"}]})
    assert nezaret.check_data(dl).findings == []


def test_two_series_in_one_folder_is_a_note(tmp_path):
    pytest.importorskip("pydicom")
    from dicom_util import write_series
    write_series(tmp_path / "mixed", patient_id="A", seed=1)
    write_series(tmp_path / "other", patient_id="A", seed=2)
    for f in (tmp_path / "other").iterdir():
        f.rename(tmp_path / "mixed" / ("b_" + f.name))
    dl = datalist(tmp_path, {"training": [{"image": "mixed"}]})
    r = nezaret.check_data(dl)
    assert r.findings == [] and "2 series in one folder" in r.notes[0]
