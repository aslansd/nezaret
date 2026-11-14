# Testing nezaret

## The short version

```bash
pip install -e ".[dev]"
pytest -q
```

**148 tests are collected** when everything is installed. What runs depends on
what else is there:

| Environment | Result |
|---|---|
| nezaret alone (no PyYAML, numpy, nibabel, pydicom, scipy, MONAI) | **80 passed, 9 skipped** |
| + PyYAML | **81 passed, 8 skipped** |
| `[dev]` (PyYAML, numpy, nibabel, pydicom, scipy) | **129 passed, 9 skipped** |
| `[dev]` + MONAI 1.6.1 | **144 passed, 4 skipped** |
| `[dev]` + MONAI 1.6.1 + MedPy 0.5.2 | **145 passed, 3 skipped** |
| all of the above + `NEZARET_MODEL_ZOO` | **148 passed** |

All six were measured on Python 3.13, the first three from the 0.3.0 sdist in
a fresh venv. A test module whose libraries are missing skips as **one**
module-level skip (`test_data.py` and `test_result.py` without nibabel,
`test_live_metrics.py` without scipy, `test_live_monai.py` without MONAI),
which is why the first rows add up to fewer than 148.

MedPy is not in any extra, because nothing but one test uses it. Install it to
run that test: `pip install medpy`.

**Run the first row before every release.** It is the only one that checks
the zero-dependency promise, and it caught a real bug in 0.2.0 before it
shipped: the duplicate check imported numpy before checking whether there was
anything to compare, so `nezaret data` crashed on a bare install instead of
saying "needs nibabel".

**`test_rules.py`, `test_cli.py`, `test_datalist.py` and `test_examples.py`
must always pass.** They need nothing beyond the standard library (PyYAML for
one CLI test, MONAI for one example). Every skip should be a live MONAI or
MedPy test, the YAML test, a module whose imaging libraries are missing, or the
opt-in model-zoo test.

---

## The suites

| File | Tests | Needs | What it covers |
|---|---|---|---|
| `tests/test_rules.py` | 56 | nothing | every `lint` rule, each with a planted fault **and** the clean version of the same config; reference resolution; report formats |
| `tests/test_cli.py` | 10 | nothing (PyYAML for one) | exit codes, `--fail-on`, `--ignore`, JSON/Markdown output, folders of bundles, `explain`, `doctor`, version agreement |
| `tests/test_datalist.py` | 14 | nothing | datalist parsing (splits, folds, strings, base dir, metadata fields), missing files, shared entries, patient ids from fields and paths, fold leakage, unsupported formats and missing readers as notes, `nezaret data` exit codes; `nezaret result` with no imaging library |
| `tests/test_data.py` | 27 | numpy, nibabel (pydicom for 4) | every data rule on real files, with clean twins; nibabel's real sample scans; synthetic DICOM series; `--no-pixels`; duplicate groups across and within splits, including a BraTS-like cohort |
| `tests/test_result.py` | 18 | numpy, nibabel, scipy | every result rule, with clean twins; Dice and HD95 conventions computed by hand on cubes; matching; seeded bootstrap; the CLI |
| `tests/test_live_metrics.py` | 7 | scipy (MONAI for 3, MedPy for 1) | nezaret's numbers against MONAI and MedPy, per case and aggregated |
| `tests/test_examples.py` | 2 | MONAI for one | the shipped `lint` examples still show what the README says |
| `tests/test_live_monai.py` | 11 | MONAI | nezaret's knowledge against the installed MONAI |
| `tests/test_model_zoo.py` | 3 | a model-zoo checkout | the `lint` findings in the README, as a regression test |

### Why each rule has a clean twin

A linter that fires on everything passes every positive test. Each rule's tests
include the nearest *correct* input, e.g. `mode: ["bilinear", "nearest"]` next
to the missing `mode`, a float label holding integer values next to a
fractional one, a 1e-4 mm float32 header difference next to a 5 mm one.
`examples/make_faulty_dataset.py` builds the whole planted-fault dataset and,
with `faults=False`, the same dataset without them; `test_clean_twin_is_clean`
asserts it gives no findings and no notes.

Several `lint` clean tests exist because a real bundle produced a false
positive during development:

| Test | The bundle that caused it |
|---|---|
| `test_nz101_all_nearest_does_not_make_image_a_label` | `lung_nodule_ct_detection`, `pathology_nuclei_segmentation_classification` resample images with `nearest` |
| `test_nz103_label_kept_in_original_space_for_eval_is_clean` | `vista3d` scores in original space after `Invertd` |
| `test_nz103_centred_crops_of_different_size_are_clean` | HoVerNet crops its labels smaller than its input |
| `test_nz103_crop_guide_key_may_be_left_behind` | `multi_organ_segmentation`'s `label4crop` |
| `test_nz103_custom_step_stops_tracking` | `renalStructures_CECT_segmentation` merges three phases in a `scripts.*` transform |
| `test_nz106_saving_an_unmoved_input_is_clean` | `renalStructures_CECT_segmentation` saves its untouched label next to the prediction |
| `test_nz109_string_booleans_compare_equal` | `wholeBrainSeg_Large_UNEST_segmentation` writes `"True"` in one file, `true` in the other |

### What the data tests read

No scan is shipped in the repository. The data tests use three sources:

- **Phantoms** written with nibabel by `examples/make_faulty_dataset.py`:
  24×24×16 seeded noise texture with spherical labels.
- **nibabel's own sample scans** (`anatomical.nii`, `reoriented_anat_moved.nii`,
  `resampled_anat_moved.nii`, `functional.nii`, `example4d.nii.gz`), installed
  with nibabel. `test_real_scans_are_not_near_duplicates_of_each_other` is the
  near-duplicate calibration: no pair of them may fire NZ207.
- **DICOM** series written by `tests/dicom_util.py` with pydicom (slices
  written out of order, so sorting by position is tested), plus pydicom's
  `CT_small.dcm` and `MR_small.dcm`.

`nezaret result` tests use phantom organs and predictions with a jittered
boundary from `examples/make_faulty_results.py`, plus cubes whose Dice and
HD95 can be worked out by hand.

No real cohort is in the suite, because none can be shipped. The runs on real
data (MSD Task09, MSD Task01, two model-zoo bundles) are recorded in
[VALIDATION.md](VALIDATION.md) with the commands to repeat them, and
`test_nz207_brats_like_patterns` reproduces the three patterns Task01 showed
on a synthetic template-registered cohort.

---

## The metric parity tests

```bash
pip install medpy    # MONAI too, for the MONAI half
pytest tests/test_live_metrics.py -v
```

`nezaret result` names its conventions after the libraries that use them, so
those names must be true. The tests check, on random anisotropic shapes
(including masks touching the volume border):

- per case: Dice, HD95 in mm and in voxels, and ASSD against MONAI 1.6.1
  (to 1e-4, MONAI computes in float32) and MedPy 0.5.2 (exactly, for both
  edge connectivities);
- aggregated: "case-first, exclude" equals `DiceMetric(reduction="mean")`,
  "class-first" equals the mean of `DiceMetric(reduction="mean_batch")`, both
  with `ignore_empty=True` and `False`, on a batch built so the conventions
  disagree;
- that the KD-tree nezaret uses to find nearest edge voxels gives exactly the
  distances of the `distance_transform_edt` lookup both libraries do.

The reading of each library's source that these conventions come from is in
the docstring of `src/nezaret/result/metrics.py`. When MONAI or MedPy
release, re-run these; a failure means a library changed a definition.

---

## The live MONAI tests

```bash
pytest tests/test_live_monai.py -v
```

These are the only tests that can catch nezaret's `lint` knowledge going
stale. They check:

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
  objects. `SaveImaged` is built with a dummy writer, so no nibabel or ITK is
  needed: without one, MONAI refuses to build it at all.

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
2. For a `lint` rule, run it over the model zoo (`nezaret lint
   /tmp/model-zoo/models`) and **read every finding against the bundle
   source.** Each false positive becomes a clean test. For a `data` rule, add
   the fault to `make_faulty_dataset.py` (or `make_faulty_results.py` for a
   result rule) and make sure the clean twin stays clean.
3. Put any MONAI default the rule depends on in `knowledge.py`, and a check
   for it in `test_live_monai.py`.
4. Give the rule a `why` that says what goes wrong *without an error*. If it
   would raise anyway, it probably does not belong in nezaret.
5. Run the bare-environment row of the table above.
