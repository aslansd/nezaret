# Changelog

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
