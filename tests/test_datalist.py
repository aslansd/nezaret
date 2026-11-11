"""nezaret data checks that need no imaging library: datalist parsing, missing
files, the same file in two splits, patient ids from paths and fields."""

import json

import pytest

import nezaret
from nezaret.cli import main
from nezaret.dataset import readers
from nezaret.dataset.datalist import load_datalist


def write(tmp_path, data, files=()):
    for f in files:
        p = tmp_path / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"not really an image")
    dl = tmp_path / "dataset.json"
    dl.write_text(json.dumps(data))
    return dl


def rules_of(r):
    return sorted(f.rule for f in r.findings)


def test_decathlon_parsing(tmp_path):
    dl = write(tmp_path, {
        "name": "x", "labels": {"0": "bg"}, "numTraining": 3,
        "training": [{"image": "./imagesTr/a.nii.gz", "label": "./labelsTr/a.nii.gz", "fold": 0},
                     {"image": "imagesTr/b.nii.gz", "label": "labelsTr/b.nii.gz", "fold": 1},
                     {"image": ["t1/c.nii.gz", "t2/c.nii.gz"], "seg": "labelsTr/c.nii.gz", "fold": 1}],
        "validation": [{"image": "v.nii.gz", "label": "vl.nii.gz", "patient_id": "P7"}],
        "testing": ["imagesTs/d.nii.gz", {"image": "/abs/e.nii.gz"}],
    })
    cases = load_datalist(dl)
    assert [c.split for c in cases] == ["training[fold=0]", "training[fold=1]", "training[fold=1]",
                                        "validation", "testing", "testing"]
    assert set(cases[2].labels) == {"seg"} and len(cases[2].images["image"]) == 2
    assert cases[0].images["image"][0] == (tmp_path / "imagesTr/a.nii.gz").resolve()
    assert cases[3].patient_field == "P7"
    assert str(cases[5].images["image"][0]) == "/abs/e.nii.gz"
    assert all(c.is_eval for c in cases)  # every fold is someone's validation set


def test_metadata_strings_are_not_paths(tmp_path):
    dl = write(tmp_path, {"training": [{"image": "a.nii", "sex": "F", "modality": "T1"}]},
               files=["a.nii"])
    [c] = load_datalist(dl)
    assert set(c.images) == {"image"}


def test_base_dir_and_label_keys(tmp_path):
    dl = write(tmp_path, {"training": [{"ct": "a.nii", "organs": "b.nii"}]})
    [c] = load_datalist(dl, base_dir=tmp_path / "elsewhere", label_keys=["organs"])
    assert set(c.labels) == {"organs"} and set(c.images) == {"ct"}
    assert c.images["ct"][0].parent == (tmp_path / "elsewhere").resolve()


def test_missing_files(tmp_path):
    dl = write(tmp_path, {"training": [{"image": "i/a.nii.gz", "label": "l/a.nii.gz"}]},
               files=["i/a.nii.gz"])
    r = nezaret.check_data(dl)
    [f] = r.findings
    assert (f.rule, f.location) == ("NZ201", "training#0#label")


def test_same_entry_in_two_splits(tmp_path):
    e = {"image": "i/a.nii.gz", "label": "l/a.nii.gz"}
    dl = write(tmp_path, {"training": [e], "validation": [e]}, files=["i/a.nii.gz", "l/a.nii.gz"])
    r = nezaret.check_data(dl)
    nz206 = [f for f in r.findings if f.rule == "NZ206"]
    assert len(nz206) == 1  # one finding for the entry, not one per file
    assert "training, validation" in nz206[0].message


def test_patient_from_path_regex(tmp_path):
    dl = write(tmp_path, {"training": [{"image": "sub-01_ses-1.nii"}, {"image": "sub-02_ses-1.nii"}],
                          "testing": [{"image": "sub-01_ses-2.nii"}]},
               files=["sub-01_ses-1.nii", "sub-02_ses-1.nii", "sub-01_ses-2.nii"])
    assert rules_of(nezaret.check_data(dl)) == []
    r = nezaret.check_data(dl, patient_id_regex=r"(sub-\d+)")
    [f] = r.findings
    assert f.rule == "NZ206" and "training (1), testing (1)" in f.message
    assert "sub-01" not in f.message  # identifiers are never printed


def test_patient_field_and_same_split_is_fine(tmp_path):
    dl = write(tmp_path, {"training": [{"image": "a.nii", "patient_id": "X"},
                                       {"image": "b.nii", "patient_id": "X"}],
                          "validation": [{"image": "c.nii", "patient_id": "Y"}]},
               files=["a.nii", "b.nii", "c.nii"])
    assert rules_of(nezaret.check_data(dl)) == []


def test_leakage_between_folds(tmp_path):
    dl = write(tmp_path, {"training": [{"image": "a.nii", "subject": "S1", "fold": 0},
                                       {"image": "b.nii", "subject": "S1", "fold": 1},
                                       {"image": "c.nii", "subject": "S2", "fold": 1}]},
               files=["a.nii", "b.nii", "c.nii"])
    r = nezaret.check_data(dl)
    assert rules_of(r) == ["NZ206"]
    assert "training[fold=0] (1), training[fold=1] (1)" in r.findings[0].message


def test_unsupported_format_is_a_note_not_a_pass(tmp_path):
    dl = write(tmp_path, {"training": [{"image": "a.nrrd", "label": "a_seg.nrrd"}]},
               files=["a.nrrd", "a_seg.nrrd"])
    r = nezaret.check_data(dl)
    assert r.findings == []
    assert any("unsupported format '.nrrd'" in n for n in r.notes)


def test_missing_reader_library_is_a_note(tmp_path, monkeypatch):
    def no(module):
        raise readers.ReadError(f"reading this needs {module} (pip install nezaret[data])")
    monkeypatch.setattr(readers, "_need", no)
    dl = write(tmp_path, {"training": [{"image": "a.nii.gz"}]}, files=["a.nii.gz"])
    r = nezaret.check_data(dl)
    assert r.findings == []
    assert r.notes == ["1 file(s) not checked: reading this needs nibabel (pip install nezaret[data])"]


def test_bad_datalists(tmp_path, capsys):
    (tmp_path / "list.json").write_text("[1, 2]")
    (tmp_path / "broken.json").write_text("{")
    (tmp_path / "nopaths.json").write_text(json.dumps({"training": [{"fold": 0}]}))
    for name in ("list.json", "broken.json", "nopaths.json", "absent.json"):
        assert main(["data", str(tmp_path / name)]) == 2


def test_cli_exit_codes_and_json(tmp_path, capsys):
    dl = write(tmp_path, {"training": [{"image": "gone.nii.gz"}]})
    assert main(["data", str(dl)]) == 1
    assert main(["data", str(dl), "--ignore", "missing-file"]) == 0
    capsys.readouterr()
    main(["data", str(dl), "--format", "json"])
    out = json.loads(capsys.readouterr().out)
    assert out["cases"] == 1 and out["stats"]["splits"] == {"training": 1}
    assert out["findings"][0]["rule"] == "NZ201"


def test_data_rules_are_listed_and_explained(capsys):
    assert main(["rules"]) == 0
    assert main(["explain", "patient-leakage"]) == 0
    out = capsys.readouterr().out
    for rid in ("NZ201", "NZ204", "NZ208"):
        assert rid in out
    assert "Split by patient" in out
