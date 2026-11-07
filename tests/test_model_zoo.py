"""Regression test against a checkout of MONAI's model zoo. Opt-in.

    git clone --depth 1 https://github.com/Project-MONAI/model-zoo.git
    NEZARET_MODEL_ZOO=model-zoo/models pytest tests/test_model_zoo.py -v

The expected findings were read by hand from the bundle sources at model-zoo
commit feb2e8c (20 October 2026). If the zoo fixes a bundle, the test fails
and the list below should shrink; it should never need to grow without a
reason someone has read the config for.
"""

import os
from pathlib import Path

import pytest

import nezaret

ZOO = os.environ.get("NEZARET_MODEL_ZOO")
pytestmark = pytest.mark.skipif(not ZOO, reason="set NEZARET_MODEL_ZOO to a model-zoo/models checkout")

# (bundle, rule, location) for every error-level finding
EXPECTED_ERRORS = {
    ("mednist_reg/configs/train.yaml", "NZ104", "loss_metric#metric_handler"),
    ("prostate_mri_anatomy/configs/inference.json", "NZ105", "postprocessing#transforms#2"),
    ("prostate_mri_anatomy/configs/train.json", "NZ101", "train#random_transforms#2"),
    ("prostate_mri_anatomy/configs/train.json", "NZ101", "train#random_transforms#4"),
    ("prostate_mri_anatomy/configs/train.json", "NZ101", "train#random_transforms#5"),
    ("renalStructures_CECT_segmentation/configs/train.json", "NZ101", "train#random_transforms#0"),
}

EXPECTED_WARNINGS = {
    ("wholeBrainSeg_Large_UNEST_segmentation/configs/train.json", "NZ109"),
    ("renalStructures_CECT_segmentation/configs/inference.json", "NZ109"),
    ("lung_nodule_ct_detection/configs/inference.json", "NZ109"),
}


@pytest.fixture(scope="module")
def report():
    return nezaret.lint(Path(ZOO))


def test_errors_are_exactly_the_known_ones(report):
    got = {(f.file, f.rule, f.location) for f in report.findings if f.severity == "error"}
    assert got == EXPECTED_ERRORS


def test_known_warnings_present(report):
    got = {(f.file, f.rule) for f in report.findings if f.severity == "warning"}
    assert EXPECTED_WARNINGS <= got


def test_reference_bundles_clean(report):
    for bundle in ("spleen_ct_segmentation", "swin_unetr_btcv_segmentation"):
        assert [f for f in report.findings if f.file.startswith(bundle)] == []
