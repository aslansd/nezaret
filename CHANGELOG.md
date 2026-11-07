# Changelog

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
