# Testing nezaret

## The short version

```bash
pip install -e ".[dev]"
pytest -q
```

**82 tests are collected** when MONAI is installed. What runs depends on what
else is there:

| Environment | Result |
|---|---|
| nezaret alone (no PyYAML, no MONAI) | **66 passed, 6 skipped** |
| `[dev]` (adds PyYAML) | **67 passed, 5 skipped** |
| `[dev]` + MONAI 1.6.1 | **79 passed, 3 skipped** |
| `[dev]` + MONAI 1.6.1 + `NEZARET_MODEL_ZOO` | **82 passed** |

All four were measured on Python 3.13, the first two from the sdist in a fresh
venv. Without MONAI, `test_live_monai.py` skips as **one** module-level skip, so
the totals in the first two rows count 72 tests, not 82.

**`test_rules.py`, `test_cli.py` and `test_examples.py` must always pass.**
They need nothing but the standard library (and PyYAML for one CLI test). Every
skip should be a live MONAI test, the YAML test, or the opt-in model-zoo test.
If anything in `test_rules.py` skips, something is wrong.

---

## The suites

| File | Tests | Needs | What it covers |
|---|---|---|---|
| `tests/test_rules.py` | 56 | nothing | every rule, each with a planted fault **and** the clean version of the same config; reference resolution; report formats |
| `tests/test_cli.py` | 10 | nothing (PyYAML for one) | exit codes, `--fail-on`, `--ignore`, JSON/Markdown output, folders of bundles, `explain`, `doctor`, version agreement |
| `tests/test_examples.py` | 2 | MONAI for one | the shipped examples still show what the README says |
| `tests/test_live_monai.py` | 11 | MONAI | nezaret's knowledge against the installed MONAI |
| `tests/test_model_zoo.py` | 3 | a model-zoo checkout | the findings in the README, as a regression test |

### Why each rule has a clean twin

A linter that fires on everything passes every positive test. Each rule's tests
include the nearest *correct* config, e.g. `mode: ["bilinear", "nearest"]` next
to the missing `mode`, `source_key: "image"` next to `"label"`, `Invertd`
before `AsDiscreted` next to after. Several clean tests exist because a real
bundle produced a false positive during development:

| Test | The bundle that caused it |
|---|---|
| `test_nz101_all_nearest_does_not_make_image_a_label` | `lung_nodule_ct_detection`, `pathology_nuclei_segmentation_classification` resample images with `nearest` |
| `test_nz103_label_kept_in_original_space_for_eval_is_clean` | `vista3d` scores in original space after `Invertd` |
| `test_nz103_centred_crops_of_different_size_are_clean` | HoVerNet crops its labels smaller than its input |
| `test_nz103_crop_guide_key_may_be_left_behind` | `multi_organ_segmentation`'s `label4crop` |
| `test_nz103_custom_step_stops_tracking` | `renalStructures_CECT_segmentation` merges three phases in a `scripts.*` transform |
| `test_nz106_saving_an_unmoved_input_is_clean` | `renalStructures_CECT_segmentation` saves its untouched label next to the prediction |
| `test_nz109_string_booleans_compare_equal` | `wholeBrainSeg_Large_UNEST_segmentation` writes `"True"` in one file, `true` in the other |

---

## The live MONAI tests

```bash
pytest tests/test_live_monai.py -v
```

These are the only tests that can catch nezaret going stale. They check:

- every default in `knowledge.py` (`Spacingd` bilinear, `Resized` area,
  `Invertd` `nearest_interp=True`, the intensity defaults NZ109 compares)
  against `inspect.signature` of the installed MONAI;
- that every transform named in nezaret's tables exists, and every transform in
  `REMOVED` really is gone;
- that the shipped component list equals what MONAI's `ComponentLocator`
  finds (skipped when the installed MONAI is a different version);
- that nezaret's static resolution of `$@a + @b` and `%a#b` yields **the same
  sequence of transforms as MONAI's own `ConfigParser`**;
- `lint_compose()` on real `Compose`, `OneOf`, `Invertd` and `SaveImaged`
  objects.

Run them after every MONAI release. If they fail, see "When MONAI releases" in
[PUBLISHING.md](PUBLISHING.md).

---

## The model-zoo regression test

Opt-in, because it needs a clone of another repository:

```bash
git clone --depth 1 https://github.com/Project-MONAI/model-zoo.git /tmp/model-zoo
NEZARET_MODEL_ZOO=/tmp/model-zoo/models pytest tests/test_model_zoo.py -v
```

It asserts that the **error-level findings are exactly** the six in the README,
that the known warnings are present, and that the reference bundles are clean.
"Exactly" is deliberate: a new error in the zoo is either a new bug in a bundle
or a new false positive in nezaret, and both need someone to read the config.

The expected list was read at model-zoo commit `feb2e8c`. If the zoo fixes one
of those bundles, the test fails and the list should shrink.

---

## Adding a rule

1. Write the planted-fault test and its clean twin first.
2. Run the rule over the model zoo (`nezaret lint /tmp/model-zoo/models`) and
   **read every finding against the bundle source.** Each false positive
   becomes a clean test.
3. Put any MONAI default the rule depends on in `knowledge.py`, and a check
   for it in `test_live_monai.py`.
4. Give the rule a `why` that says what goes wrong *without an error*. If it
   would raise anyway, it probably does not belong in nezaret.
