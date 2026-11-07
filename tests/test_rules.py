"""Every rule, with a planted fault and the clean version of the same config.

None of these tests needs MONAI: they read configs as data.
"""

import copy

import pytest

import nezaret
from conftest import GOOD_DET, IL, compose, good_train_config, load, t


def rules_of(report, severity=None):
    return sorted(f.rule for f in report.findings if severity is None or f.severity == severity)


def lint_train(*transforms, extra=None, **kw):
    cfg = {"initialize": "$monai.utils.set_determinism(seed=1)",
           "train": {"preprocessing": compose(*transforms)}}
    cfg.update(extra or {})
    return nezaret.lint_config(cfg, name="train.json", **kw)


def lint_validate(*transforms):
    return nezaret.lint_config({"validate": {"preprocessing": compose(*transforms)}},
                               name="train.json")


# --- the clean baseline ------------------------------------------------------

def test_spleen_like_config_is_clean(good_config):
    r = nezaret.lint_config(good_config, name="train.json")
    assert r.findings == []
    assert r.pipelines == 2
    assert r.ok


# --- NZ101 label-interpolation --------------------------------------------------

def test_nz101_default_mode_on_label():
    r = lint_train(load(IL), t("EnsureChannelFirstd", IL), t("Spacingd", IL, pixdim=[1, 1, 1]))
    [f] = r.findings
    assert f.rule == "NZ101" and f.severity == "error"
    assert "'bilinear'" in f.message and "default" in f.message
    assert f.location == "train#preprocessing#transforms#2"


@pytest.mark.parametrize("name,default", [("Resized", "area"), ("Zoomd", "area"),
                                          ("RandAffined", "bilinear"), ("RandZoomd", "area"),
                                          ("Rand3DElasticd", "bilinear")])
def test_nz101_each_resampler_default(name, default):
    r = lint_train(load(IL), t("EnsureChannelFirstd", IL), t(name, IL, prob=0.5))
    assert rules_of(r) == ["NZ101"]
    assert repr(default) in r.findings[0].message


def test_nz101_explicit_wrong_mode():
    r = lint_train(load(IL), t("EnsureChannelFirstd", IL),
                   t("Spacingd", IL, pixdim=1, mode=["bilinear", "trilinear"]))
    assert rules_of(r) == ["NZ101"]
    assert "set explicitly" in r.findings[0].message


@pytest.mark.parametrize("mode", [["bilinear", "nearest"], ["bilinear", 0],
                                  ["bilinear", "nearest-exact"]])
def test_nz101_clean_modes(mode):
    r = lint_train(load(IL), t("EnsureChannelFirstd", IL), t("Spacingd", IL, pixdim=1, mode=mode))
    assert r.findings == []


def test_nz101_all_nearest_does_not_make_image_a_label():
    # some bundles resample every key with nearest; that is not evidence
    # that "image" holds labels, so the intensity transform must not fire NZ102
    r = lint_train(load(IL), t("EnsureChannelFirstd", IL),
                   t("RandAffined", IL, prob=1.0, mode="nearest"),
                   t("RandGaussianNoised", "image", prob=0.2))
    assert r.findings == []


def test_nz101_keys_through_references():
    cfg = {"image_key": "image", "label_key": "label",
           "initialize": "$monai.utils.set_determinism(seed=1)",
           "train": {"preprocessing": compose(
               load(["@image_key", "@label_key"]),
               t("EnsureChannelFirstd", ["@image_key", "@label_key"]),
               t("Spacingd", ["@image_key", "@label_key"], pixdim=1))}}
    r = nezaret.lint_config(cfg, name="train.json")
    assert rules_of(r) == ["NZ101"]


def test_nz101_unresolvable_mode_is_skipped():
    r = lint_train(load(IL), t("EnsureChannelFirstd", IL),
                   t("Spacingd", IL, pixdim=1, mode="$@modes"))
    assert r.findings == []


def test_label_keys_can_be_given():
    r = lint_train(load(["ct", "organs"]), t("EnsureChannelFirstd", ["ct", "organs"]),
                   t("Spacingd", ["ct", "organs"], pixdim=1), label_keys=["organs"])
    assert rules_of(r) == ["NZ101"]
    assert "'organs'" in r.findings[0].message


# --- NZ102 label-intensity -------------------------------------------------------

def test_nz102_noise_on_label_is_error():
    r = lint_train(*GOOD_DET, t("RandGaussianNoised", IL, prob=0.1))
    [f] = r.findings
    assert (f.rule, f.severity) == ("NZ102", "error")


def test_nz102_minmax_on_label_is_warning():
    r = lint_train(*GOOD_DET, t("ScaleIntensityd", IL))
    [f] = r.findings
    assert (f.rule, f.severity) == ("NZ102", "warning")
    assert "0/255" in f.message


def test_nz102_image_only_is_clean():
    assert lint_train(*GOOD_DET, t("RandGaussianNoised", "image", prob=0.1)).findings == []


# --- NZ103 geometry-split --------------------------------------------------------

def test_nz103_partial_geometry():
    r = lint_train(load(IL), t("EnsureChannelFirstd", IL), t("Orientationd", "image", axcodes="RAS"),
                   t("Spacingd", IL, pixdim=1, mode=["bilinear", "nearest"]))
    [f] = r.findings
    assert (f.rule, f.severity) == ("NZ103", "error")
    assert f.location.endswith("#2")


def test_nz103_label_never_moved_in_training_is_warning():
    r = lint_train(load(IL), t("EnsureChannelFirstd", IL), t("Orientationd", "image", axcodes="RAS"))
    [f] = r.findings
    assert (f.rule, f.severity) == ("NZ103", "warning")


def test_nz103_label_kept_in_original_space_for_eval_is_clean():
    # vista3d scores in original space after Invertd; that is deliberate
    r = lint_validate(load(IL, ensure_channel_first=True), t("Orientationd", "image", axcodes="RAS"),
                      t("Spacingd", "image", pixdim=1))
    assert r.findings == []


def test_nz103_centred_crops_of_different_size_are_clean():
    # HoVerNet crops the label smaller than the input, both centred
    r = lint_train(load(IL), t("EnsureChannelFirstd", IL),
                   t("CenterSpatialCropd", "image", roi_size=[270, 270]),
                   t("RandFlipd", IL, prob=0.5),
                   t("CenterSpatialCropd", "label", roi_size=[80, 80]))
    assert r.findings == []


def test_nz103_crop_guide_key_may_be_left_behind():
    r = lint_train(load(IL), t("EnsureChannelFirstd", IL),
                   t("CopyItemsd", "label", names=["label4crop"], times=1),
                   t("RandCropByLabelClassesd", IL, label_key="label4crop", spatial_size=[96, 96, 96]),
                   t("RandFlipd", IL, prob=0.5))
    assert r.findings == []


def test_nz103_custom_step_stops_tracking():
    r = lint_train(load(["a", "b", "label"]), t("EnsureChannelFirstd", ["a", "b", "label"]),
                   {"_target_": "scripts.transforms.Merge", "keys_merge": ["a", "b"], "keys_out": "image"},
                   t("RandFlipd", IL, prob=0.5))
    assert r.findings == []


def test_nz103_delete_items_is_respected():
    r = lint_train(load(IL), t("EnsureChannelFirstd", IL), t("DeleteItemsd", "label"),
                   t("Orientationd", "image", axcodes="RAS"))
    assert r.findings == []


# --- NZ104 unknown-target ---------------------------------------------------------

def test_nz104_removed_transform_names_replacement():
    r = lint_train(load(IL), t("AddChanneld", IL))
    [f] = [x for x in r.findings if x.rule == "NZ104"]
    assert "EnsureChannelFirstd" in f.message and 'no_channel' in f.message


def test_nz104_renamed_handler():
    r = nezaret.lint_config({"h": {"_target_": "IgniteMetric"}}, name="train.json")
    assert "IgniteMetricHandler" in r.findings[0].message


def test_nz104_typo_and_qualified_names():
    cfg = {"a": {"_target_": "Spacingdd"},
           "b": {"_target_": "torch.optim.Adam"},
           "c": {"_target_": "scripts.net.MyNet"},
           "d": {"_target_": "$@x"}}
    r = nezaret.lint_config(cfg, name="train.json")
    assert [(f.rule, f.location) for f in r.findings] == [("NZ104", "a")]


# --- NZ105 discrete-interpolation -------------------------------------------------

def _inference(post, pre=None):
    pre = pre or [load("image", ensure_channel_first=True), t("Spacingd", "image", pixdim=1)]
    return nezaret.lint_config({"preprocessing": compose(*pre), "postprocessing": compose(*post)},
                               name="inference.json")


INV_FALSE = t("Invertd", "pred", transform="@preprocessing", orig_keys="image", nearest_interp=False)


def test_nz105_invert_after_discrete():
    r = _inference([t("AsDiscreted", "pred", argmax=True), INV_FALSE, t("SaveImaged", "pred")])
    assert rules_of(r) == ["NZ105"]


def test_nz105_invert_before_discrete_is_clean():
    r = _inference([t("Activationsd", "pred", softmax=True), INV_FALSE,
                    t("AsDiscreted", "pred", argmax=True), t("SaveImaged", "pred")])
    assert r.findings == []


def test_nz105_default_nearest_interp_is_clean():
    inv = t("Invertd", "pred", transform="@preprocessing", orig_keys="image")
    r = _inference([t("AsDiscreted", "pred", threshold=0.5), inv, t("SaveImaged", "pred")])
    assert r.findings == []


# --- NZ106 saved-not-inverted ------------------------------------------------------

def test_nz106_save_without_invert():
    r = _inference([t("AsDiscreted", "pred", argmax=True), t("SaveImaged", "pred")])
    [f] = r.findings
    assert f.rule == "NZ106" and "Spacingd" in f.message


def test_nz106_resample_true_is_clean():
    r = _inference([t("AsDiscreted", "pred", argmax=True), t("SaveImaged", "pred", resample=True)])
    assert r.findings == []


def test_nz106_saving_an_unmoved_input_is_clean():
    pre = [load(["image", "label"], ensure_channel_first=True), t("Spacingd", "image", pixdim=1)]
    r = _inference([INV_FALSE, t("AsDiscreted", "pred", argmax=True), t("SaveImaged", "pred"),
                    t("SaveImaged", "label")], pre=pre)
    assert r.findings == []


def test_nz106_no_grid_change_is_clean():
    pre = [load("image", ensure_channel_first=True), t("ScaleIntensityd", "image")]
    r = _inference([t("AsDiscreted", "pred", argmax=True), t("SaveImaged", "pred")], pre=pre)
    assert r.findings == []


# --- NZ107 label-guided-eval / NZ108 random-in-eval --------------------------------

def test_nz107_crop_foreground_from_label_in_validation():
    r = lint_validate(load(IL, ensure_channel_first=True),
                      t("CropForegroundd", IL, source_key="label"))
    assert rules_of(r) == ["NZ107"]


def test_nz107_crop_from_image_is_clean():
    r = lint_validate(load(IL, ensure_channel_first=True),
                      t("CropForegroundd", IL, source_key="image"))
    assert r.findings == []


def test_nz107_label_guided_crop_in_validation_not_double_reported():
    r = lint_validate(load(IL, ensure_channel_first=True),
                      t("RandCropByPosNegLabeld", IL, label_key="label", spatial_size=[96, 96, 96]))
    assert rules_of(r) == ["NZ107"]


def test_nz107_is_fine_in_training():
    r = lint_train(load(IL, ensure_channel_first=True), t("CropForegroundd", IL, source_key="label"))
    assert r.findings == []


def test_nz108_random_in_validation():
    r = lint_validate(load(IL, ensure_channel_first=True), t("RandFlipd", IL, prob=0.5))
    assert rules_of(r) == ["NZ108"]


def test_nz108_prob_zero_is_clean():
    r = lint_validate(load(IL, ensure_channel_first=True), t("RandFlipd", IL, prob=0.0))
    assert r.findings == []


# --- NZ109 train-eval-mismatch -----------------------------------------------------

def test_nz109_intensity_window_differs(good_config):
    cfg = copy.deepcopy(good_config)
    val = copy.deepcopy(GOOD_DET)
    val[4]["a_max"] = 250
    cfg["validate"]["preprocessing"] = compose(*val)
    r = nezaret.lint_config(cfg, name="train.json")
    [f] = r.findings
    assert f.rule == "NZ109" and "a_max" in f.message and "164" in f.message


def test_nz109_missing_step_in_validation(good_config):
    cfg = copy.deepcopy(good_config)
    cfg["validate"]["preprocessing"] = compose(*GOOD_DET[:4])
    r = nezaret.lint_config(cfg, name="train.json")
    assert rules_of(r) == ["NZ109"]
    assert "ScaleIntensityRanged only in training" in r.findings[0].message


def test_nz109_across_files(write_bundle, good_config):
    infer = {"preprocessing": compose(load("image"), t("EnsureChannelFirstd", "image"),
                                      t("Orientationd", "image", axcodes="LPS"),
                                      t("Spacingd", "image", pixdim=[1.5, 1.5, 2.0]),
                                      t("ScaleIntensityRanged", "image", a_min=-57, a_max=164,
                                        b_min=0, b_max=1, clip=True))}
    root = write_bundle({"train.json": good_config, "inference.json": infer})
    r = nezaret.lint(root)
    [f] = r.findings
    assert f.rule == "NZ109" and f.file == "configs/inference.json"
    assert "'RAS' vs eval 'LPS'" in f.message


def test_nz109_closest_canonical_counts_as_ras(write_bundle):
    train = {"initialize": "$set_determinism(0)",
             "train": {"preprocessing": compose(load(IL, as_closest_canonical=True),
                                                t("EnsureChannelFirstd", IL))}}
    infer = {"preprocessing": compose(load("image"), t("EnsureChannelFirstd", "image"),
                                      t("Orientationd", "image", axcodes="LPS"))}
    r = nezaret.lint(write_bundle({"train.json": train, "inference.json": infer}))
    assert rules_of(r) == ["NZ109"]
    assert "as_closest_canonical" in r.findings[0].message


def test_nz109_string_booleans_compare_equal(good_config):
    cfg = copy.deepcopy(good_config)
    cfg["train"]["deterministic_transforms"].append(t("NormalizeIntensityd", "image", nonzero=True))
    val = copy.deepcopy(cfg["train"]["deterministic_transforms"])
    val[-1]["nonzero"] = "True"
    cfg["validate"]["preprocessing"] = compose(*val)
    assert nezaret.lint_config(cfg, name="train.json").findings == []


# --- NZ110 / NZ111 info ---------------------------------------------------------------

def test_nz110_no_channel_first_is_info():
    r = lint_train(load(IL), t("Spacingd", IL, pixdim=1, mode=["bilinear", "nearest"]))
    assert {f.rule for f in r.findings} == {"NZ110"}
    assert all(f.severity == "info" for f in r.findings)


def test_nz111_no_seed():
    cfg = good_train_config()
    del cfg["initialize"]
    r = nezaret.lint_config(cfg, name="train.json")
    assert rules_of(r) == ["NZ111"] and r.ok


# --- config resolution ------------------------------------------------------------------

def test_location_points_at_the_list_not_the_reference():
    cfg = good_train_config()
    cfg["train"]["random_transforms"].append(t("RandZoomd", IL, prob=0.2))
    [f] = nezaret.lint_config(cfg, name="train.json").findings
    assert f.location == "train#random_transforms#3"


def test_flat_override_keys_and_fallback(write_bundle, good_config):
    evaluate = {"validate#preprocessing": compose(*GOOD_DET, t("RandFlipd", IL, prob=0.5))}
    r = nezaret.lint(write_bundle({"train.json": good_config, "evaluate.json": evaluate}))
    assert [(f.rule, f.file) for f in r.findings] == [("NZ108", "configs/evaluate.json")]


def test_macro_into_sibling_file(write_bundle, good_config):
    infer = {"preprocessing": {"_target_": "Compose",
                               "transforms": "%train.json#train#deterministic_transforms"}}
    r = nezaret.lint(write_bundle({"train.json": good_config, "inference.json": infer}))
    assert r.findings == []
    assert r.pipelines == 3


def test_nested_containers_are_flattened():
    r = lint_train(*GOOD_DET, {"_target_": "OneOf", "transforms": [
        t("RandGaussianNoised", "label", prob=0.5), t("RandFlipd", IL, prob=0.5)]})
    assert rules_of(r) == ["NZ102"]
    assert r.findings[0].location.endswith("#transforms#5#transforms#0")


def test_unresolvable_pipeline_is_skipped_not_guessed():
    r = nezaret.lint_config({"train": {"preprocessing": {"_target_": "Compose",
                                                         "transforms": "$build_transforms()"}}})
    assert r.findings == [] and r.pipelines == 0


# --- report ----------------------------------------------------------------------------

def test_ignore_by_id_and_slug():
    r1 = lint_train(load(IL), t("EnsureChannelFirstd", IL), t("Spacingd", IL, pixdim=1),
                    ignore=["NZ101"])
    r2 = lint_train(load(IL), t("EnsureChannelFirstd", IL), t("Spacingd", IL, pixdim=1),
                    ignore=["label-interpolation"])
    assert r1.findings == [] and r2.findings == []


def test_report_formats():
    r = lint_train(load(IL), t("EnsureChannelFirstd", IL), t("Spacingd", IL, pixdim=1))
    d = r.to_dict()
    assert d["summary"] == {"error": 1, "warning": 0, "info": 0}
    assert d["checked_against_monai"] == "1.6.1"
    md = r.to_markdown()
    assert "### NZ101 `label-interpolation`" in md and "**Fix:**" in md
    assert "NZ101" in str(r)


def test_every_rule_has_text():
    for rule in nezaret.RULES.values():
        assert rule.why and rule.fix and rule.title
        assert rule.severity in ("error", "warning", "info")
