# nezaret

**نظارت**: *oversight, supervision, inspection.* The same word in Persian
(*nezārat*), Turkish (*nezaret*) and Azerbaijani (*nəzarət*).

Find faults in [MONAI](https://monai.io) pipelines, datasets and results that
**do not raise errors**: a label map resampled with bilinear interpolation, a
rotation applied to the image but not its label, a validation crop chosen from
the ground truth, an inference pipeline that preprocesses differently from
training, a label saved with the wrong header, the same patient in training
and test, a "better" method that is only better under one way of scoring
empty cases.

Each of these trains, evaluates and reports a number. The number is wrong, and
nothing in the run says so.

**No dependencies. Runs offline. Reads configs as data; never imports torch, never
runs a transform, never trains anything.**

```bash
pip install nezaret
nezaret lint path/to/bundle                  # the pipeline
pip install "nezaret[data]"
nezaret data path/to/dataset.json            # the data it trains on
pip install "nezaret[result]"
nezaret result --labels labelsTs/ --pred unet=preds/unet/ --pred swin=preds/swin/   # what it reports
```

Python 3.10+. Apache 2.0. Rules are checked against MONAI 1.6.1.

[PyPI](https://pypi.org/project/nezaret/) · [INSTALL](INSTALL.md) ·
[TESTING](TESTING.md) · [PUBLISHING](PUBLISHING.md) ·
[TROUBLESHOOTING](TROUBLESHOOTING.md) · [VALIDATION](VALIDATION.md) · [ROADMAP](ROADMAP.md) ·
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
  E NZ201 missing-file                      training#12#image
      'images/siteA_p099_scan1.nii.gz' does not exist.
  E NZ201 missing-file                      training#12#label
      'labels/siteA_p099_scan1.nii.gz' does not exist.
  E NZ202 geometry-mismatch                 training#8#label
      label 'labels/siteA_p031_scan1.nii.gz' vs image 'images/siteA_p031_scan1.nii.gz': spacing (1.5, 1.5, 2.5) vs (1, 1, 1); origin (18, -20, -12) vs (0, 0, 0); direction differs.
  E NZ203 label-values                      training#9#label
      'labels/siteA_p032_scan1.nii.gz' has fractional values (resampled with interpolation?).
  E NZ206 patient-leakage                   training#2
      'images/siteA_p003_scan1.nii.gz', 'labels/siteA_p003_scan1.nii.gz' listed in 2 splits (training, testing): training#2, testing#4.
  E NZ206 patient-leakage                   training#4
      patient #dc663a6c has cases in training (1), testing (1): training#4, testing#3.
  E NZ207 duplicate-image                   training#0#image
      identical voxels across 2 splits: 'images/siteA_p001_scan1.nii.gz' (training), 'images/siteB_p041_scan1.nii.gz' (testing).
  E NZ207 duplicate-image                   training#1#image
      the same scan re-saved (fingerprint r >= 0.9999) across 2 splits: 'images/siteA_p002_scan1.nii.gz' (training), 'images/siteB_p042_scan1.nii.gz' (testing).
  W NZ205 class-coverage                    validation
      class(es) [2] are in training but never in 'validation', so they are never scored there.
  W NZ208 site-confounding                  site
      site: 'siteB' only in testing (5).
  I NZ204 empty-labels                      training#10#label
      2 of 12 label maps in 'training' are empty.

26 case(s) [training 13, validation 3, testing 10], 41 file(s) read, 24 patient id(s): 8 error(s), 2 warning(s), 1 info
```

| Rule | Severity | Catches |
|---|---|---|
| `NZ201` missing-file | error | a listed file that does not exist |
| `NZ202` geometry-mismatch | error | image and label disagree on shape, spacing, origin or direction (a label saved with an identity affine, a flipped axis) |
| `NZ203` label-values | error | fractional values (a label resampled with interpolation before saving), negative values, or hundreds of distinct values (an image under a label path) |
| `NZ204` empty-labels | info | how many label maps per split are empty, because that changes mean Dice depending on how empty cases are scored |
| `NZ205` class-coverage | warning | a class that appears in evaluation but never in training, or is never in an evaluation split and so never scored |
| `NZ206` patient-leakage | error | the same file, or the same patient, in two splits; **folds count as splits** |
| `NZ207` duplicate-image | error / warning | the same scan in two splits: identical voxels, or re-saved at another dtype or scaling (error); near-identical images, in practice the same scan processed differently or a repeat scan of one patient (warning); and duplicate groups **inside** one split, which leak into any split made from it later (warning) |
| `NZ208` site-confounding | warning | a scanner, field strength, institution or site that appears in only one split |

**Where patient identity comes from**, in order: a `patient_id` (or
`patient`, `subject`, `subject_id`, `PatientID`) field in the datalist entry,
`--patient-key`, DICOM `PatientID`, then `--patient-id-regex` on the file path.
NIfTI files carry no patient or scanner metadata, so for NIfTI datasets the
regexes are how leakage and site checks see anything. **Identifiers are never
printed**; a finding names a patient by the first eight characters of a SHA-256
digest.

**The near-duplicate fingerprint** is a 16×16×16 block-mean of the volume in
canonical orientation, z-scored. Copies of one scan at another dtype or
intensity scaling correlate at 1.000, and fingerprint r ≥ 0.9999 is treated as
"the same scan"; a 2-voxel crop (0.60) or a left-right flip (0.83) stays far
below the 0.995 threshold, so cropped or re-registered copies are not found.
**It has been calibrated on a real atlas-registered cohort**, MSD Task01
(BraTS, 484 training cases): different subjects correlate at a median of 0.912
(99th percentile 0.971), and every one of the 178 pairs at or above 0.995 was,
on inspection, the same scan stored twice (116 pairs, most with full-resolution
r = 1.0000) or one of a run of consecutive case numbers that look like repeat
scans of one patient (62 pairs). Details in [VALIDATION.md](VALIDATION.md).
Duplicates are reported as groups, and the groups are in the JSON report
(`stats.duplicate_groups`) so a dataset can be split by group.

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

## `nezaret result`: the numbers you report

`nezaret result` re-scores saved predictions against the labels under every
convention a careful author could have chosen, and says whether the
conclusion survives. It is `ayar`'s question asked of segmentation: is the
result in the data, or in the analysis?

```text
$ python examples/make_faulty_results.py /tmp/faulty_results
$ nezaret result --labels /tmp/faulty_results/labels \
      --pred unet=/tmp/faulty_results/unet --pred swin=/tmp/faulty_results/swin
result
  E NZ301 unmatched-cases                   unet
      unet: no prediction for 1 of 24 labelled case(s): case_023.
  E NZ302 prediction-geometry               swin/case_010
      swin 'case_010_seg.nii.gz' vs label: shape (24, 24, 20) vs (48, 48, 20); spacing 1.4, 1.4, 3 vs 0.7, 0.7, 3. Not scored.
  E NZ303 prediction-values                 swin/case_011
      swin 'case_011_seg.nii.gz' has fractional values (probabilities saved instead of classes?). Not scored.
  W NZ304 empty-handling-sensitive          swin
      swin: mean Dice (case-first) is 0.7185 (exclude), 0.6589 (both-empty-1), 0.5226 (zero) over 12 empty (case, class) pair(s); spread 0.1960 > 0.01.
  W NZ304 empty-handling-sensitive          unet
      unet: mean Dice (case-first) is 0.7084 (exclude), 0.7845 (both-empty-1), 0.5236 (zero) over 12 empty (case, class) pair(s); spread 0.2609 > 0.01.
  W NZ305 distance-implementation-sensitive swin
      swin: HD95 is 2.061 in voxel units, the default of both MONAI and MedPy when spacing is not passed, but 3.099 mm with spacing. Across implementations: 3.099 mm (monai-style) vs 2.668 mm (medpy-style), 2.848 mm with 26-neighbour edges, 14.366 mm with empty masks scored as the image diagonal.
  W NZ305 distance-implementation-sensitive unet
      unet: HD95 is 2.247 in voxel units, the default of both MONAI and MedPy when spacing is not passed, but 4.217 mm with spacing. Across implementations: 4.217 mm (monai-style) vs 3.474 mm (medpy-style), 3.935 mm with 26-neighbour edges, 4.650 mm with empty masks scored as the image diagonal.
  W NZ306 ranking-not-robust                unet vs swin
      Dice: unet better under 3 of 7 conventions, swin under 4; e.g. unet wins with 'case-first, empty GT both-empty-1 (MONAI ignore_empty=False)', swin with 'case-first, empty GT exclude (MONAI default)'.
  W NZ306 ranking-not-robust                unet vs swin
      HD95: unet better under 8 of 16 implementations, swin under 8; e.g. unet wins with 'monai, mm, 6-neighbour edges, empty worst', swin with 'monai, mm, 6-neighbour edges, empty exclude'.
  W NZ307 difference-unresolved             unet vs swin
      mean Dice difference unet - swin = -0.0011, paired 95% bootstrap interval [-0.0984, +0.1021] over 15 cases includes 0 (P(unet > swin) = 0.51).

Scores
------
  method  cases  Dice (MONAI default)   95% CI            Dice range over conventions   HD95 mm   HD95 range         ASSD mm
  unet       23                0.7084   [0.6549, 0.7577]  0.5236 - 0.7845                  4.22   2.88 - 4.65           0.86
  swin       22                0.7185   [0.6466, 0.7785]  0.5226 - 0.7185                  3.10   2.54 - 14.37          0.73
  unet - swin: Dice -0.0011 [-0.0984, +0.1021] over 15 cases, P(unet better) = 0.51; unet wins 3 of 7 Dice conventions, 8 of 16 HD95 implementations
  classes scored: [1, 2] (background excluded); Dice default = case-first, empty GT exclude (MONAI default); HD95 default = monai, mm, 6-neighbour edges, empty exclude

24 labelled case(s), 2 method(s), 2 class(es): 3 error(s), 7 warning(s), 0 info
```

Read the table first: under MONAI's default DiceMetric, `swin` is ahead
(0.7185 vs 0.7084). Score empty ground truths the way `ignore_empty=False`
does and `unet` is ahead (0.7845 vs 0.6589), because `unet` correctly predicts
nothing on the six empty cases and `swin` does not. On the cases both methods
were scored on, the paired bootstrap gives `swin` an even chance (P = 0.51).
Neither ranking would raise an error, and neither is reported with its
convention in most papers.

| Rule | Severity | Catches |
|---|---|---|
| `NZ301` unmatched-cases | error | a method with no prediction for some labelled cases (warning for predictions with no label) |
| `NZ302` prediction-geometry | error | a prediction on a different grid from its label, e.g. saved in preprocessed space without `Invertd` (see `NZ106`); not scored |
| `NZ303` prediction-values | error | probabilities or a one-hot map instead of class indices (warning for a class no label contains) |
| `NZ304` empty-handling-sensitive | warning | mean Dice that moves by more than `--tolerance` (0.01) between the three ways of scoring an empty ground truth |
| `NZ305` distance-implementation-sensitive | warning / info | HD95 or ASSD in voxel units differing from mm (**warning**: both MONAI and MedPy default to voxels when `spacing` is not passed), or differing by implementation (info) |
| `NZ306` ranking-not-robust | warning | the better method changes with the Dice convention or the HD95 implementation |
| `NZ307` difference-unresolved | warning | a paired 95 % bootstrap interval of the Dice difference that includes zero |

**The conventions, and where each comes from.** Dice: an empty ground truth
excluded (MONAI `DiceMetric()`, `ignore_empty=True`), scored 1 if the
prediction is also empty and 0 otherwise (`ignore_empty=False`), or scored 0
(a smoothed 2|A∩B|/(|A|+|B|+ε)); averaged over classes within a case first
(MONAI `reduction="mean"`), over cases within a class first
(`reduction="mean_batch"`, then over classes), or pooled over all voxels.
HD95: the larger of the two directed 95th percentiles (MONAI) or one
percentile of both directions pooled (MedPy); mm or voxels; 6- or
26-neighbour edges; empty masks excluded or scored as the image diagonal.
ASSD: pooled over both directions (MONAI and MedPy both do this) or the mean
of the two directed means.

**These numbers are MONAI's and MedPy's numbers, checked.**
`tests/test_live_metrics.py` compares nezaret with MONAI 1.6.1 and MedPy 0.5.2
on random anisotropic shapes, including masks touching the volume border:
Dice, HD95 (mm and voxels) and ASSD agree with MONAI to float32 precision and
with MedPy exactly, for both edge connectivities. The aggregates are checked
too: nezaret's "case-first, exclude" equals `DiceMetric(reduction="mean")`
and its "class-first" equals the mean of `reduction="mean_batch"`, with
`ignore_empty` on and off.

Files are paired by case name, ignoring `.nii`/`.nii.gz` and postfixes such as
`_seg` and `_trans` that MONAI's `SaveImaged` adds, including its
`<case>/<case>_seg.nii.gz` layout; `--case-regex` overrides that. Labels can
also come from a datalist split (`--datalist dataset.json --split
validation`). Background (class 0) is never scored. On 20 cases of 512×512×100
with a liver-sized organ and two methods, scoring takes 3.3 s for Dice only
(`--no-distances`) and 25 s with every distance variant, on a 2-core machine.

```python
r = nezaret.check_results("labelsTs/", {"unet": "preds/unet/", "swin": "preds/swin/"})
r.stats["methods"]["unet"]["dice"]       # every convention's mean
r.stats["pairs"][0]["bootstrap"]         # paired difference, interval, P(a better)
```

---

## Tested on real data

[VALIDATION.md](VALIDATION.md) records every run on public data, with the
commands to repeat it. In short:

- **MSD Task09 (spleen CT), `nezaret data`:** 61 cases, no findings. No false
  positives on a dataset with no planted fault.
- **MSD Task01 (BraTS brain MRI), the duplicate check:** **274 of the 484
  training images belong to 127 groups of duplicates**, among them 116 pairs
  that are the same scan stored twice, mostly with different intensity
  scaling and often with different labels. The pattern is consistent with
  Task01 being assembled from both BraTS 2016 and BraTS 2017. This is what
  made NZ207 report duplicates inside a split. Across splits, **52 of the 266
  test images are a training scan re-saved**, about one test case in five of
  the set the MSD challenge was scored on (three checked at full resolution:
  all four MRI channels correlate at 1.0000).
- **Two model-zoo bundles on 9 spleen cases, `nezaret result`:** a clear
  winner, correctly reported as robust; and HD95 for the same Swin UNETR
  predictions of 5.90 mm with MONAI's formula against 3.45 mm with MedPy's.

---

## Commands

```bash
nezaret lint PATH...                 # a config file, a bundle, or a folder of bundles
nezaret lint PATH --format markdown  # or json; -o report.md to write a file
nezaret lint PATH --fail-on warning  # exit 1 on warnings too (default: errors only)
nezaret lint PATH --ignore NZ108,no-determinism
nezaret lint PATH --label-keys seg,organs
nezaret data DATALIST [--base-dir DIR] [--no-pixels] [--patient-id-regex RE] [--site-regex RE]
nezaret result --labels DIR --pred NAME=DIR [--pred NAME=DIR ...] [--no-distances] [--classes 1,2]
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
