"""Checks against a real MONAI install. Every test here skips without MONAI.

These are the tests that catch nezaret's knowledge going stale: a changed
default, a renamed transform, or a config reference nezaret resolves
differently from MONAI's own parser.
"""

import copy
import inspect

import pytest

monai = pytest.importorskip("monai")
T = pytest.importorskip("monai.transforms")

import nezaret  # noqa: E402
from nezaret import knowledge as K  # noqa: E402
from conftest import good_train_config  # noqa: E402


def test_resampling_defaults_match_installed_monai():
    for name, expected in K.RESAMPLING_DEFAULT_MODE.items():
        default = inspect.signature(getattr(T, name).__init__).parameters["mode"].default
        assert str(getattr(default, "value", default)) == expected, name


def test_signature_defaults_match_installed_monai():
    for name, params in K.SIGNATURE_PARAMS.items():
        sig = inspect.signature(getattr(T, name).__init__).parameters
        for p, expected in params.items():
            if sig[p].default is inspect.Parameter.empty:
                continue
            assert sig[p].default == expected, f"{name}.{p}"


def test_invertd_nearest_interp_default_is_true():
    assert inspect.signature(T.Invertd.__init__).parameters["nearest_interp"].default is True


def test_every_known_transform_exists():
    names = (K.GEOMETRY | K.INTENSITY | set(K.GUIDE_ARGUMENT) | set(K.SIGNATURE_PARAMS)
             | K.CONTAINERS)
    missing = sorted(n for n in names if not hasattr(T, n))
    assert missing == []


def test_removed_transforms_are_really_gone():
    from monai.bundle.config_item import ComponentLocator
    loc = ComponentLocator()
    present = loc._find_classes_or_functions(loc._find_module_names())
    assert [n for n in K.REMOVED if n in present] == []


def test_shipped_component_list_matches_installed():
    version, shipped = K.monai_components()
    if monai.__version__.split("+")[0] != version:
        pytest.skip(f"shipped list is for MONAI {version}, installed {monai.__version__}")
    from monai.bundle.config_item import ComponentLocator
    loc = ComponentLocator()
    assert set(loc._find_classes_or_functions(loc._find_module_names())) == set(shipped)


def test_resolver_agrees_with_monai_config_parser():
    """nezaret's static resolution of a pipeline == what MONAI builds."""
    from monai.bundle import ConfigParser
    from nezaret.config import from_dict
    from nezaret.pipeline import pipelines_from_config

    cfg = good_train_config()
    cfg["initialize"] = []
    ours = {p.location: [s.name for s in p.steps]
            for p in pipelines_from_config(from_dict(copy.deepcopy(cfg)))}
    parser = ConfigParser(cfg)  # note: MONAI's parser mutates the dict it is given
    built = {
        "train#preprocessing": parser.get_parsed_content("train#preprocessing"),
        "validate#preprocessing": parser.get_parsed_content("validate#preprocessing"),
    }
    for loc, compose in built.items():
        assert ours[loc] == [type(x).__name__ for x in compose.transforms], loc


# --- lint_compose on live objects ---------------------------------------------------

class _NeverWrites:
    """Passed as SaveImaged's writer so MONAI does not look for nibabel/ITK.

    Without it, building SaveImaged raises OptionalImportError when no image
    writer backend is installed. nezaret never saves anything, so the tests
    must not depend on one.
    """


def save(**kw):
    return T.SaveImaged(writer=_NeverWrites, output_dir="/tmp/nezaret-never-written", **kw)


def test_live_compose_default_mode_on_label():
    c = T.Compose([T.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                   T.Spacingd(keys=["image", "label"], pixdim=(1, 1, 1)),
                   T.RandAffined(keys=["image", "label"], prob=0.5,
                                 mode=("bilinear", "nearest"))])
    r = nezaret.lint_compose(c, role="train")
    assert [(f.rule, f.location) for f in r.findings] == [("NZ101", "[1]")]


def test_live_compose_clean():
    c = T.Compose([T.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                   T.Orientationd(keys=["image", "label"], axcodes="RAS"),
                   T.Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")),
                   T.ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0, b_max=1, clip=True),
                   T.OneOf([T.RandFlipd(keys=["image", "label"], prob=0.5),
                            T.RandRotate90d(keys=["image", "label"], prob=0.5)])])
    assert nezaret.lint_compose(c, role="train").findings == []


def test_live_compose_eval_and_post():
    pre = T.Compose([T.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                     T.CropForegroundd(keys=["image", "label"], source_key="label"),
                     T.Spacingd(keys=["image", "label"], pixdim=1, mode=("bilinear", "nearest"))])
    assert [f.rule for f in nezaret.lint_compose(pre, role="eval").findings] == ["NZ107"]

    post = T.Compose([T.AsDiscreted(keys="pred", argmax=True),
                      T.Invertd(keys="pred", transform=pre, orig_keys="image", nearest_interp=False),
                      save(keys="pred")])
    r = nezaret.lint_compose(post, role="post", preprocessing=pre)
    assert "NZ105" in [f.rule for f in r.findings]

    post2 = T.Compose([T.AsDiscreted(keys="pred", argmax=True),
                       save(keys="pred")])
    r2 = nezaret.lint_compose(post2, role="post", preprocessing=pre)
    assert [f.rule for f in r2.findings] == ["NZ106"]
    r3 = nezaret.lint_compose(T.Compose([save(keys="pred", resample=True)]),
                              role="post", preprocessing=pre)
    assert r3.findings == []


def test_live_compose_reads_nested_parameters():
    from nezaret.pipeline import pipeline_from_compose
    c = T.Compose([T.Spacingd(keys="image", pixdim=(1, 1, 2)),
                   T.Orientationd(keys="image", axcodes="LPS"),
                   T.ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0, b_max=1),
                   save(keys="pred", resample=True)])
    steps = pipeline_from_compose(c).steps
    assert steps[0].args["pixdim"] == [1.0, 1.0, 2.0]
    assert steps[1].args["axcodes"] == "LPS"
    assert steps[2].args["a_max"] == 164
    assert steps[3].args["resample"] is True
