# Changelog

## 0.3.0: `nezaret result`

**`nezaret result` asks whether a segmentation result survives the scoring
choices.** It reads saved predictions for one or more methods and the labels,
scores every case under every convention a careful author could have used,
and says when the conclusion depends on the convention. Seven new rules:

| Rule | Severity |
|---|---|
| `NZ301` unmatched-cases | error (warning for predictions without labels) |
| `NZ302` prediction-geometry | error |
| `NZ303` prediction-values | error (warning for an unknown class) |
| `NZ304` empty-handling-sensitive | warning |
| `NZ305` distance-implementation-sensitive | warning (voxel units differ from mm) / info |
| `NZ306` ranking-not-robust | warning |
| `NZ307` difference-unresolved | warning |

**What the conventions are, read from the source of MONAI 1.6.1 and MedPy 0.5.2:**

- `DiceMetric` defaults to `ignore_empty=True` (an empty ground truth is
  dropped) and `reduction="mean"`, which averages over classes within each
  case before averaging over cases.
- MONAI's HD95 is the larger of the two directed 95th percentiles; MedPy's is
  one percentile of both directions pooled. On the phantom example the two
  differ by 14 % (3.10 vs 2.67 mm).
- **Both compute in voxel units unless spacing is passed.** On the example's
  0.7 × 0.7 × 3 mm grid, that turns 3.10 mm into 2.06.
- Both pool the two directions for ASSD; the mean of the two directed means,
  also in use, is offered as a variant.

**Checked against the libraries.** `tests/test_live_metrics.py` asserts that
nezaret's per-case Dice, HD95 (mm and voxels, both connectivities) and ASSD
equal MONAI's and MedPy's on random anisotropic shapes, and that its
aggregates equal `DiceMetric` with `reduction="mean"` and `"mean_batch"`,
`ignore_empty` on and off.

**Fast enough for real volumes.** Nearest edge voxels are found with a KD-tree
over edge coordinates, which gives exactly the distance-transform lookup the
libraries do (tested) at a fraction of the cost. On 20 cases of 512×512×100
and two methods: 3.3 s for Dice only, 25 s with every distance variant, on a
2-core machine. Development profiling took this from 89 s.

**New optional extra `[result]`**: numpy, nibabel, scipy. Without scipy,
`nezaret result` scores Dice only and says so; without nibabel it says what
to install. The core stays dependency-free.

**Validated on real data before release** ([VALIDATION.md](VALIDATION.md)):
MSD Task09 (spleen CT) gives no findings; on MSD Task01 (BraTS), 274 of 484
training images fall into 127 duplicate groups, 116 pairs being the same scan
stored twice, and 52 of the 266 test images are a training scan re-saved;
two model-zoo bundles scored on 9 spleen cases give a robust
ranking and a 71 % gap between MONAI's and MedPy's HD95 for the same Swin
UNETR predictions.

**NZ207 changed because of that run.**

- It now reports duplicates **inside** one split, as one warning per split,
  saying why it matters there: leakage into any split made later from a
  training list, double counting in an evaluation list.
  Before, on Task01 it would have reported none of the 127 groups, all in
  `training`, which leak into any validation split taken from it.
- It reports **groups**, not pairs, and lists them in the JSON report under
  `stats.duplicate_groups`, so a dataset can be split by group.
- Fingerprint r ≥ 0.9999 now counts as the same scan (an error across
  splits), not a near-duplicate: on Task01, 116 pairs were in that band and
  only 2 had identical voxels.
- The 0.995 threshold stays: on Task01 every pair above it was a re-stored
  scan or one of a run of consecutive cases.

**New tools for repeating the validation**: `tools/fingerprint_calibration.py`,
`tools/duplicate_groups.py`, `tools/compare_pairs.py`,
`tools/prepare_spleen_eval.py`.

**Also in this release**

- `nezaret doctor` reports scipy.
- Rule slugs up to 33 characters line up in text output.
- The shipped list of MONAI component names no longer contains `tqdm` (now
  2445 names). MONAI defines a stand-in `tqdm` in
  `monai.apps.pathology.metrics.lesion_froc` only when the real package is
  missing, so the list depended on whether `tqdm` was installed where it was
  generated, and `test_shipped_component_list_matches_installed` failed on
  machines that had it. Such names are now listed in
  `knowledge.ENV_DEPENDENT_COMPONENTS` and left out.

## 0.2.0: `nezaret data`

**`nezaret data` checks a dataset the way `nezaret lint` checks a pipeline.**
It reads a MONAI datalist (the Decathlon format `load_decathlon_datalist`
reads; each top-level list is a split, and each `fold` of a cross-validation
is a split of its own) and the NIfTI and DICOM files it points at. It still
never trains anything. Eight new rules:

| Rule | Severity |
|---|---|
| `NZ201` missing-file | error |
| `NZ202` geometry-mismatch | error |
| `NZ203` label-values | error |
| `NZ204` empty-labels | info |
| `NZ205` class-coverage | warning |
| `NZ206` patient-leakage | error |
| `NZ207` duplicate-image | error (exact) / warning (near) |
| `NZ208` site-confounding | warning |

**Patient identifiers are never printed.** A finding names a patient by the
first eight characters of a SHA-256 digest, so a report can be sent to someone
else.

**The near-duplicate fingerprint was calibrated on real scans, and the result
is recorded here.** On nibabel's sample scans, the same scan re-saved at
another dtype, rescaled, or with 5 % noise correlates at 1.000; a 2-voxel crop
gives 0.60, a left-right flip 0.83, and other or moved scans at most 0.32.
The threshold is 0.995. It finds re-saved copies, not cropped or re-registered
ones, and it is not yet calibrated on an atlas-registered cohort (ROADMAP §2).

**New optional extra `[data]`**: numpy, nibabel, pydicom. The core is still
dependency-free: without the extra, missing files and entries shared between
splits are still found, and every file that could not be read is listed in a
note.

**`--no-pixels`** reads headers only: 0.07 s for forty 256×256×120 cases,
against about 0.4 s per case with voxels, on a 2-core machine.

**Fixed before release:** the duplicate check imported numpy before checking
whether it had anything to compare, so `nezaret data` crashed on a bare
install. Found by running the test suite with nothing but pytest installed,
which PUBLISHING.md now requires.

**Also in this release**

- `nezaret doctor` reports nibabel and pydicom.
- `tests/test_live_monai.py` builds `SaveImaged` with a dummy writer. Without
  nibabel or ITK installed, MONAI refuses to construct it, which made two
  tests fail on a machine that had MONAI but no image writer.
- INSTALL.md: how to install from source when pip cannot reach PyPI
  (`--no-build-isolation`).
- `src/nezaret.egg-info/` is no longer tracked in git.

## 0.1.0: first release

**`nezaret lint`, for MONAI Bundle configs.** Reads `train.json`,
`inference.json` and their YAML equivalents as data, resolves `@` references,
`%` macros (including `%other.json#…`) and `$@a + @b` concatenation without
evaluating anything, and checks every transform pipeline it finds against
eleven rules. Each rule names a way a pipeline produces a wrong result
**without raising an error**.

| Rule | Severity |
|---|---|
| `NZ101` label-interpolation | error |
| `NZ102` label-intensity | error (warning for min-max scaling) |
| `NZ103` geometry-split | error (warning when a key is never moved at all) |
| `NZ104` unknown-target | error |
| `NZ105` discrete-interpolation | error |
| `NZ106` saved-not-inverted | warning |
| `NZ107` label-guided-eval | error |
| `NZ108` random-in-eval | warning |
| `NZ109` train-eval-mismatch | warning |
| `NZ110` missing-channel-first | info |
| `NZ111` no-determinism | info |

**`lint_compose()`, for pipelines built in Python.** Reads the attributes of a
live `monai.transforms.Compose` (including `OneOf`, `SomeOf`, `RandomOrder` and
nested `Compose`) into the same model, so the same rules apply.

**What it knows about MONAI was read from MONAI 1.6.1**: the default `mode` of
every resampling transform, `Invertd`'s `nearest_interp=True`, the intensity
defaults compared between training and evaluation, and the 2446 names MONAI's
`ComponentLocator` resolves. The live test suite re-checks all of it against
the installed MONAI.

**Tested against MONAI's model zoo.** All 37 bundles, 157 config files and 174
pipelines: 6 errors and 7 warnings, every error read against the bundle source.
Seven false positives found during development became clean tests (listed in
TESTING.md). The six true errors, in four bundles, are in the README.

Zero runtime dependencies. PyYAML for YAML bundles and MONAI for
`lint_compose()` are optional extras.
