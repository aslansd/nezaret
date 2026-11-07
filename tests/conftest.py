import json

import pytest


def load(keys, **kw):
    return {"_target_": "LoadImaged", "keys": keys, **kw}


def t(name, keys, **kw):
    return {"_target_": name, "keys": keys, **kw}


def compose(*transforms):
    return {"_target_": "Compose", "transforms": list(transforms)}


IL = ["image", "label"]

GOOD_DET = [
    load(IL),
    t("EnsureChannelFirstd", IL),
    t("Orientationd", IL, axcodes="RAS"),
    t("Spacingd", IL, pixdim=[1.5, 1.5, 2.0], mode=["bilinear", "nearest"]),
    t("ScaleIntensityRanged", "image", a_min=-57, a_max=164, b_min=0, b_max=1, clip=True),
]


def good_train_config():
    """A config modelled on MONAI's spleen bundle. nezaret must find nothing."""
    return {
        "initialize": ["$monai.utils.set_determinism(seed=0)"],
        "train": {
            "deterministic_transforms": [dict(x) for x in GOOD_DET],
            "random_transforms": [
                t("RandCropByPosNegLabeld", IL, label_key="label",
                  spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4),
                t("RandFlipd", IL, prob=0.5, spatial_axis=0),
                t("RandShiftIntensityd", "image", offsets=0.1, prob=0.5),
            ],
            "preprocessing": {"_target_": "Compose",
                              "transforms": "$@train#deterministic_transforms + @train#random_transforms"},
        },
        "validate": {
            "preprocessing": {"_target_": "Compose", "transforms": "%train#deterministic_transforms"},
        },
    }


@pytest.fixture
def good_config():
    return good_train_config()


@pytest.fixture
def write_bundle(tmp_path):
    def _write(files: dict, name="bundle"):
        root = tmp_path / name
        (root / "configs").mkdir(parents=True)
        for fname, data in files.items():
            (root / "configs" / fname).write_text(json.dumps(data, indent=1))
        (root / "configs" / "metadata.json").write_text(json.dumps({"version": "0.1"}))
        return root
    return _write
