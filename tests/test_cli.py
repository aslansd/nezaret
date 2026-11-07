import json

import pytest

from nezaret.cli import main
from conftest import IL, compose, good_train_config, load, t


def bad_train():
    cfg = good_train_config()
    cfg["train"]["random_transforms"].append(t("RandZoomd", IL, prob=0.2))
    return cfg


def test_lint_exit_codes(write_bundle, capsys):
    good = write_bundle({"train.json": good_train_config()}, name="good")
    bad = write_bundle({"train.json": bad_train()}, name="bad")
    assert main(["lint", str(good)]) == 0
    assert main(["lint", str(bad)]) == 1
    assert main(["lint", str(bad), "--fail-on", "never"]) == 0
    assert main(["lint", str(bad), "--ignore", "NZ101"]) == 0
    out = capsys.readouterr().out
    assert "NZ101" in out and "label-interpolation" in out


def test_fail_on_warning(write_bundle):
    cfg = good_train_config()
    cfg["validate"]["preprocessing"] = compose(load(IL, ensure_channel_first=True),
                                               t("RandFlipd", IL, prob=0.5))
    root = write_bundle({"train.json": cfg})
    assert main(["lint", str(root)]) == 0
    assert main(["lint", str(root), "--fail-on", "warning"]) == 1


def test_json_output(write_bundle, tmp_path, capsys):
    root = write_bundle({"train.json": bad_train()})
    out = tmp_path / "r.json"
    main(["lint", str(root), "--format", "json", "-o", str(out)])
    data = json.loads(out.read_text())
    assert data["summary"]["error"] == 1
    assert data["files"] == ["configs/train.json"]
    assert data["findings"][0]["location"] == "train#random_transforms#3"


def test_markdown_output(write_bundle, capsys):
    root = write_bundle({"train.json": bad_train()})
    main(["lint", str(root), "--format", "markdown"])
    assert "# nezaret report" in capsys.readouterr().out


def test_many_bundles_under_one_root(write_bundle, tmp_path, capsys):
    write_bundle({"train.json": good_train_config()}, name="zoo/a")
    write_bundle({"train.json": bad_train()}, name="zoo/b")
    assert main(["lint", str(tmp_path / "zoo"), "--format", "json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert sorted(data["files"]) == ["a/configs/train.json", "b/configs/train.json"]
    assert [f["file"] for f in data["findings"]] == ["b/configs/train.json"]


def test_single_file(write_bundle):
    root = write_bundle({"train.json": bad_train()})
    assert main(["lint", str(root / "configs" / "train.json")]) == 1


def test_no_configs(tmp_path, capsys):
    assert main(["lint", str(tmp_path)]) == 2


def test_yaml(tmp_path):
    yaml = pytest.importorskip("yaml")
    d = tmp_path / "b" / "configs"
    d.mkdir(parents=True)
    (d / "train.yaml").write_text(yaml.safe_dump(bad_train()))
    assert main(["lint", str(tmp_path / "b")]) == 1


def test_rules_explain_doctor(capsys):
    assert main(["rules"]) == 0
    assert main(["explain", "NZ107"]) == 0
    assert main(["explain", "label-guided-eval"]) == 0
    assert main(["explain", "NZ999"]) == 2
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "NZ111" in out and "ground truth" in out and "checked against MONAI 1.6.1" in out


def test_installed_version_matches_source():
    from importlib.metadata import version
    import nezaret
    assert version("nezaret") == nezaret.__version__
