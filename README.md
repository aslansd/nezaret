# nezaret

**نظارت**: *oversight, supervision, inspection.* The same word in Persian
(*nezārat*), Turkish (*nezaret*) and Azerbaijani (*nəzarət*).

Find faults in [MONAI](https://monai.io) pipelines and datasets that **do not
raise errors**: a label map resampled with bilinear interpolation, a rotation
applied to the image but not its label, a validation crop chosen from the
ground truth, an inference pipeline that preprocesses differently from
training, a label saved with the wrong header, the same patient in training
and test.

Each of these trains, evaluates and reports a number. The number is wrong, and
nothing in the run says so.

**No dependencies. Runs offline. Reads configs as data; never imports torch, never
runs a transform, never trains anything.**

```bash
pip install nezaret
nezaret lint path/to/bundle                  # the pipeline
pip install "nezaret[data]"
nezaret data path/to/dataset.json            # the data it trains on
```

Python 3.10+. Apache 2.0. Rules are checked against MONAI 1.6.1.

[PyPI](https://pypi.org/project/nezaret/) · [INSTALL](INSTALL.md) ·
[TESTING](TESTING.md) · [PUBLISHING](PUBLISHING.md) ·
[TROUBLESHOOTING](TROUBLESHOOTING.md) · [ROADMAP](ROADMAP.md) ·
[CHANGELOG](CHANGELOG.md)

---

## The thirty-second version

```text
$ nezaret lint examples/faulty_bundle
configs/inference.json
  E NZ104 unknown-target           preprocessing#transforms#1
      'AddChanneld' is not a MONAI 1.6.1 component. It was removed or renamed in MONAI; use EnsureChannelFirstd with channel_dim="no_channel".
  W NZ106 saved-not-inverted       postprocessing#transforms#2
      SaveImaged writes 'pred' without Invertd, but the preprocessing at inference.json:preprocessing changes the grid (Orientationd, Spacingd).
  W NZ109 train-eval-mismatch      preprocessing
      compared with train.json:train#preprocessing: NormalizeIntensityd only in training; Orientationd.axcodes: train 'RAS' vs eval 'LPS'

configs/train.json
  E NZ101 label-interpolation      train#deterministic_transforms#3
      Spacingd resamples label key 'label' with mode 'bilinear' (MONAI's default); label maps need 'nearest'.
  E NZ102 label-intensity          train#random_transforms#2
      RandGaussianNoised changes voxel values of label key(s) 'label'.
  E NZ103 geometry-split           train#random_transforms#1
      RandRotated transforms 'image' but not 'label', which were loaded together and stop being aligned here.
  E NZ107 label-guided-eval        validate#preprocessing#transforms#4
      CropForegroundd in an evaluation pipeline takes its region from ground-truth key 'label' (source_key).
  W NZ109 train-eval-mismatch      validate#preprocessing
      compared with train.json:train#preprocessing: NormalizeIntensityd only in training
  I NZ111 no-determinism           train#preprocessing
      random transforms in training and no set_determinism or seed in this file or the config it overrides.

2 file(s), 4 pipeline(s): 5 error(s), 3 warning(s), 1 info
```

Every location is a MONAI Bundle `#`-path, so `train#deterministic_transforms#3`
is the fourth entry of that list. A finding on a transform reached through
`$@train#deterministic_transforms + @train#random_transforms` points at the list
the transform is written in, not at the expression.

---

## What it checks

| Rule | Severity | Catches |
|---|---|---|
| `NZ101` label-interpolation | error | a label key resampled with a non-nearest mode, **including MONAI's defaults** (`Spacingd`, `RandAffined`: bilinear; `Resized`, `Zoomd`, `RandZoomd`: area) |
| `NZ102` label-intensity | error | intensity transforms (noise, scaling, normalisation, contrast) on a label key; min-max scaling is a warning, since 0/255 → 0/1 masks are a real idiom |
| `NZ103` geometry-split | error | a spatial transform (orientation, spacing, crop, pad, flip, rotate, zoom) applied to some of the arrays loaded together but not the others |
| `NZ104` unknown-target | error | a `_target_` MONAI cannot resolve, with the replacement for removed ones (`AddChanneld`, `AsChannelFirstd`, `IgniteMetric`…) |
| `NZ105` discrete-interpolation | error | `Invertd(nearest_interp=false)` after `AsDiscreted`, which interpolates class indices |
| `NZ106` saved-not-inverted | warning | predictions saved on the resampled grid, without `Invertd` or `resample=true` |
| `NZ107` label-guided-eval | error | validation or inference that crops by the ground truth (`CropForegroundd(source_key=label)`, label-guided random crops) |
| `NZ108` random-in-eval | warning | a random transform in validation or inference |
| `NZ109` train-eval-mismatch | warning | orientation, spacing, size or intensity scaling that differs between training and evaluation, **across files of the same bundle** |
| `NZ110` missing-channel-first | info | a spatial transform before any channel-first step |
| `NZ111` no-determinism | info | random training transforms and no seed anywhere |

`nezaret rules` lists them; `nezaret explain NZ107` says why one matters and
how to fix it.

The defaults nezaret relies on were read from `inspect.signature` of MONAI
1.6.1, not from documentation, and `tests/test_live_monai.py` re-checks them
against whatever MONAI is installed.

---

## What it found in MONAI's own model zoo

Linting all 37 bundles of [Project-MONAI/model-zoo](https://github.com/Project-MONAI/model-zoo)
(commit `feb2e8c`, 20 October 2026; 157 config files, 174 pipelines) gives
**6 errors, 7 warnings and 5 info**. Every error, and every warning below, was read
against the bundle source by hand:

| Bundle | Rule | What the config does |
|---|---|---|
| `prostate_mri_anatomy` train | NZ101 ×3 | `RandAffined`, `Rand3DElasticd` and `RandZoomd` on `["image", "label"]` with no `mode`, so the label is resampled bilinearly (twice) and by area averaging |
| `prostate_mri_anatomy` inference | NZ105 | `AsDiscreted(argmax)`, then `Invertd(nearest_interp=false)` on the class map |
| `renalStructures_CECT_segmentation` train | NZ101 | `RandZoomd` on the label with the default `area` mode |
| `renalStructures_CECT_segmentation` inference | NZ109 | training loads with `as_closest_canonical=true` (RAS); inference reorients to `LPS` |
| `wholeBrainSeg_Large_UNEST_segmentation` train | NZ109 | `NormalizeIntensityd` sits in `random_transforms`; validation copies only the deterministic list, so the checkpoint is selected on unnormalised inputs while training and inference normalise |
| `mednist_reg` train | NZ104 | `IgniteMetric`, renamed `IgniteMetricHandler` |

The reference bundles (`spleen_ct_segmentation`, `swin_unetr_btcv_segmentation`)
come back clean. `tests/test_model_zoo.py` turns this table into a regression
test.

The remaining warnings are judgement calls nezaret cannot settle from the
config alone: e.g. `lung_nodule_ct_detection` resamples only at inference
because its README resamples the training data offline, which the message
says.

---

## `nezaret data`: the dataset

`nezaret lint` reads the pipeline. `nezaret data` reads the datalist the
pipeline trains on (the Decathlon format `load_decathlon_datalist` reads) and
the files it points at:

```text
$ python examples/make_faulty_dataset.py /tmp/faulty_dataset
$ nezaret data /tmp/faulty_dataset/dataset.json --patient-id-regex "(p\d{3})" --site-regex "(site[A-Z])"
dataset.json
  E NZ201 missing-file             training#12#image
      'images/siteA_p099_scan1.nii.gz' does not exist.
  E NZ201 missing-file             training#12#label
      'labels/siteA_p099_scan1.nii.gz' does not exist.
  E NZ202 geometry-mismatch        training#8#label
      label 'labels/siteA_p031_scan1.nii.gz' vs image 'images/siteA_p031_scan1.nii.gz': spacing (1.5, 1.5, 2.5) vs (1, 1, 1); origin (18, -20, -12) vs (0, 0, 0); direction differs.
  E NZ203 label-values             training#9#label
      'labels/siteA_p032_scan1.nii.gz' has fractional values (resampled with interpolation?).
  E NZ206 patient-leakage          training#2
      'images/siteA_p003_scan1.nii.gz', 'labels/siteA_p003_scan1.nii.gz' listed in 2 splits (training, testing): training#2, testing#4.
  E NZ206 patient-leakage          training#4
      patient #dc663a6c has cases in training (1), testing (1): training#4, testing#3.
  E NZ207 duplicate-image          training#0#image
      identical voxels in 2 splits: 'images/siteA_p001_scan1.nii.gz' (training), 'images/siteB_p041_scan1.nii.gz' (testing).
  W NZ205 class-coverage           validation
      class(es) [2] are in training but never in 'validation', so they are never scored there.
  W NZ207 duplicate-image          training#1#image
      near-duplicate (fingerprint r=1.0000): 'images/siteA_p002_scan1.nii.gz' (training) and 'images/siteB_p042_scan1.nii.gz' (testing).
  W NZ208 site-confounding         site
      site: 'siteB' only in testing (5).
  I NZ204 empty-labels             training#10#label
      2 of 12 label maps in 'training' are empty.

26 case(s) [training 13, validation 3, testing 10], 41 file(s) read, 24 patient id(s): 7 error(s), 3 warning(s), 1 info
```

| Rule | Severity | Catches |
|---|---|---|
| `NZ201` missing-file | error | a listed file that does not exist |
| `NZ202` geometry-mismatch | error | image and label disagree on shape, spacing, origin or direction (a label saved with an identity affine, a flipped axis) |
| `NZ203` label-values | error | fractional values (a label resampled with interpolation before saving), negative values, or hundreds of distinct values (an image under a label path) |
| `NZ204` empty-labels | info | how many label maps per split are empty, because that changes mean Dice depending on how empty cases are scored |
| `NZ205` class-coverage | warning | a class that appears in evaluation but never in training, or is never in an evaluation split and so never scored |
| `NZ206` patient-leakage | error | the same file, or the same patient, in two splits; **folds count as splits** |
| `NZ207` duplicate-image | error / warning | identical voxels across splits (error), or the same scan saved again at another dtype or scaling (warning) |
| `NZ208` site-confounding | warning | a scanner, field strength, institution or site that appears in only one split |

**Where patient identity comes from**, in order: a `patient_id` (or
`patient`, `subject`, `subject_id`, `PatientID`) field in the datalist entry,
`--patient-key`, DICOM `PatientID`, then `--patient-id-regex` on the file path.
NIfTI files carry no patient or scanner metadata, so for NIfTI datasets the
regexes are how leakage and site checks see anything. **Identifiers are never
printed**; a finding names a patient by the first eight characters of a SHA-256
digest.

**The near-duplicate fingerprint** is a 16×16×16 block-mean of the volume in
canonical orientation, z-scored. On nibabel's real sample scans, copies of the
same scan at another dtype, scaling or with 5 % noise correlate at 1.000;
different scans, and even a 2-voxel crop (0.60) or a left-right flip (0.83) of
the same one, stay far below the 0.995 threshold. So it finds *re-saved*
copies, not cropped or re-registered ones. It has not been calibrated on a
cohort registered to one atlas, where different subjects may correlate more
closely; raise `--near-duplicate` if it fires there.

**Headers only:** `--no-pixels` skips loading voxels (NZ203, NZ204, NZ205 and
NZ207), which is the difference between 0.4 s per CT-sized case and 0.07 s for
forty of them, measured on a 2-core machine.

Reads NIfTI (nibabel) and DICOM: a file, or a folder holding one series
(pydicom). Anything else, or a compressed DICOM with no decoder installed, is
reported as a note, never as clean. Geometry is only compared between two files
of the same format.

```python
report = nezaret.check_data("dataset.json", base_dir="/data/Task09_Spleen",
                            patient_id_regex=r"(sub-\d+)")
```

---

## Commands

```bash
nezaret lint PATH...                 # a config file, a bundle, or a folder of bundles
nezaret lint PATH --format markdown  # or json; -o report.md to write a file
nezaret lint PATH --fail-on warning  # exit 1 on warnings too (default: errors only)
nezaret lint PATH --ignore NZ108,no-determinism
nezaret lint PATH --label-keys seg,organs
nezaret data DATALIST [--base-dir DIR] [--no-pixels] [--patient-id-regex RE] [--site-regex RE]
nezaret rules
nezaret explain NZ103
nezaret doctor
```

Exit codes: `0` clean, `1` a finding at or above `--fail-on`, `2` nothing
could be checked. That makes it a CI step as it stands:

```yaml
- run: pip install "nezaret[yaml]"
- run: nezaret lint configs/ --fail-on error
```

A directory with a `configs/` folder is one bundle; a directory holding
configs is one bundle; anything else is searched for `configs/` folders.
Cross-file rules only compare files within one bundle.

---

## From Python

```python
import nezaret

report = nezaret.lint("my_bundle/")          # Report
report.ok                                     # no errors
report.findings                               # [Finding(rule, severity, file, location, message, ...)]
report.to_json(); report.to_markdown()

nezaret.lint_config(config_dict, name="train.json")
```

For pipelines built in Python rather than in a config (needs MONAI installed):

```python
report = nezaret.lint_compose(train_transforms, role="train")
report = nezaret.lint_compose(val_transforms, role="eval")
report = nezaret.lint_compose(post_transforms, role="post", preprocessing=val_transforms)
```

nezaret reads the transform objects' attributes; it does not call them, load
data or need a GPU. `role` has to be given because an object does not say what
it is used for. See `examples/lint_compose_example.py`.

---

## How it decides what is a label

In this order:

1. `--label-keys` / `label_keys=`, if given, and nothing else.
2. Key names such as `label`, `seg`, `mask`, `gt`, `annotation`, `target`
   (also `label_inst`, `seg_1`, `lbl`…).
3. A per-key `mode` list that mixes `nearest` with something else: the author
   saying which keys are label maps.
4. The `label_key` of a label-guided crop.

A single `mode: "nearest"` for every key is *not* evidence: some bundles
resample their images that way.

## What it does not do

- **It does not evaluate `$` expressions.** `@` references, `%` macros
  (including `%other.json#…`) and `$@a + @b` concatenation are resolved
  statically; any other expression is treated as unknown and the rules that
  would need it are skipped, never guessed. A pipeline built entirely by an
  expression is not checked.
- **`nezaret lint` does not read your data list or your images.** It cannot
  know that your data list supplies four files per key (which is why `NZ110`
  is info). That is `nezaret data`'s job, below.
- **It does not judge custom transforms.** After a `scripts.*` step that may
  add, merge or drop keys, the key-tracking rules stop rather than guess.

---

## Licence

Apache 2.0. See [LICENSE](LICENSE).
