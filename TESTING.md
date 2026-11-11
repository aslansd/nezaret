# Testing nezaret

## The short version

```bash
pip install -e ".[dev]"
pytest -q
```

**120 tests are collected** when everything is installed. What runs depends on
what else is there:

| Environment | Result |
|---|---|
| nezaret alone (no PyYAML, numpy, nibabel, pydicom, MONAI) | **79 passed, 7 skipped** |
| + PyYAML | **80 passed, 6 skipped** |
| `[dev]` (PyYAML, numpy, nibabel, pydicom) | **105 passed, 5 skipped** |
| `[dev]` + MONAI 1.6.1 | **117 passed, 3 skipped** |
| `[dev]` + MONAI 1.6.1 + `NEZARET_MODEL_ZOO` | **120 passed** |

All five were measured on Python 3.13, the first three from the 0.2.0 sdist in
a fresh venv. Without nibabel, `test_data.py` skips as **one** module-level
skip, and without MONAI so does `test_live_monai.py`, which is why the first
rows add up to fewer than 120.

**Run the first row before every release.** It is the only one that checks
the zero-dependency promise, and it caught a real bug in 0.2.0 before it
shipped: the duplicate check imported numpy before checking whether there was
anything to compare, so `nezaret data` crashed on a bare install instead of
saying "needs nibabel".

**`test_rules.py`, `test_cli.py`, `test_datalist.py` and `test_examples.py`
must always pass.** They need nothing beyond the standard library (PyYAML for
one CLI test, MONAI for one example). Every skip should be a live MONAI test,
the YAML test, the nibabel/pydicom module, or the opt-in model-zoo test.

---

## The suites

| File | Tests | Needs | What it covers |
|---|---|---|---|
| `tests/test_rules.py` | 56 | nothing | every `lint` rule, each with a planted fault **and** the clean version of the same config; reference resolution; report formats |
| `tests/test_cli.py` | 10 | nothing (PyYAML for one) | exit codes, `--fail-on`, `--ignore`, JSON/Markdown output, folders of bundles, `explain`, `doctor`, version agreement |
| `tests/test_datalist.py` | 13 | nothing | datalist parsing (splits, folds, strings, base dir, metadata fields), missing files, shared entries, patient ids from fields and paths, fold leakage, unsupported formats and missing readers as notes, `nezaret data` exit codes |
| `tests/test_data.py` | 25 | numpy, nibabel (pydicom for 4) | every data rule on real files, with clean twins; nibabel's real sample scans; synthetic DICOM series; `--no-pixels` |
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

There is no real multi-patient cohort in the suite. The roadmap's check of
downloading an open dataset from Tehran is still open (see ROADMAP §2), and
the near-duplicate threshold is uncalibrated on atlas-registered cohorts; the
README says so.

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
   the fault to `make_faulty_dataset.py` and make sure the clean twin stays
   clean.
3. Put any MONAI default the rule depends on in `knowledge.py`, and a check
   for it in `test_live_monai.py`.
4. Give the rule a `why` that says what goes wrong *without an error*. If it
   would raise anyway, it probably does not belong in nezaret.
5. Run the bare-environment row of the table above.
