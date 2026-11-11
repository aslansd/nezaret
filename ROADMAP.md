# Roadmap

*Last revised 11 November 2026, against nezaret 0.2.0.*

---

## Where this stands

**0.2.0**, Apache 2.0, **zero runtime dependencies**, **120 tests**,
**nineteen rules** (eleven for pipelines, eight for data), checked against
MONAI 1.6.1.

| | Status |
|---|---|
| `nezaret lint`: bundle configs, JSON and YAML, static `@`/`%`/`$@a + @b` resolution | shipped |
| `lint_compose()` on live MONAI pipelines | shipped |
| Run over all 37 model-zoo bundles, every error read by hand | done, see README |
| Upstream reports for the model-zoo findings | **not filed yet, see §1** |
| `nezaret data`: datalist + NIfTI/DICOM checks (0.2.0) | shipped, §2 |
| `nezaret data` on a real multi-patient cohort | **not done yet, see §2** |
| `nezaret result` | planned, §3 |
| **External users** | **none yet, the thing that actually matters** |

```
tests/test_rules.py        56
tests/test_cli.py          10
tests/test_datalist.py     13
tests/test_data.py         25   (skips without nibabel)
tests/test_examples.py      2
tests/test_live_monai.py   11   (skips without MONAI)
tests/test_model_zoo.py     3   (opt-in: NEZARET_MODEL_ZOO)
                          ---
                          120
```

---

## 1. The upstream findings: file them first

Running 0.1.0 over the model zoo found four bundles with faults that change
results without raising an error (README, "What it found"). Each is a small,
checkable report, and filing them is worth more than any feature below: a
merged fix is public evidence that the rules find real things, which is the
case for the package.

| Bundle | Report |
|---|---|
| `prostate_mri_anatomy` | `RandAffined`/`Rand3DElasticd`/`RandZoomd` need `mode` with `nearest` for the label; `Invertd` should come before `AsDiscreted` in inference |
| `renalStructures_CECT_segmentation` | `RandZoomd` label mode; training is RAS (`as_closest_canonical`), inference reorients to LPS |
| `wholeBrainSeg_Large_UNEST_segmentation` | move `NormalizeIntensityd` into `deterministic_transforms` so validation normalises |
| `mednist_reg` | `IgniteMetric` → `IgniteMetricHandler` |

One issue per bundle, each with the config lines and the one-line fix, and
**without** leading with nezaret. Mention it at the end as how the fault was
found. Before filing the orientation one, check by loading one case both ways
that the axes really differ; it is the only finding not visible from the
config alone.

---

## 2. 0.2.0: `nezaret data`: shipped, one thing left

Shipped as planned: geometry agreement (NZ202), label sanity (NZ203, NZ204),
class coverage (NZ205), patient leakage from DICOM `PatientID`, datalist
fields or path regexes, folds included (NZ206), exact and near-duplicate
voxels (NZ207), scanner and site balance (NZ208), plus missing files (NZ201).
nibabel and pydicom are the optional `[data]` extra; the core stays
dependency-free, and a bare install still finds missing files and shared
entries.

**What is still open: a real cohort.** Every data test runs on phantoms,
nibabel's single-subject sample scans and synthetic DICOM. Before telling
anyone 0.2.0 works on their data:

1. **Check that one open dataset downloads from Tehran.** The Medical
   Segmentation Decathlon's Task09_Spleen (41 training CTs, about 1.5 GB) is
   the natural first one, because its `dataset.json` is exactly the format
   nezaret reads. If it does not download, try a TCIA collection; if neither
   does, ask a collaborator abroad to run `nezaret data --format json` and
   send the report, which contains no images and no patient identifiers.
2. **Run `nezaret data` on it, with and without `--no-pixels`, and read every
   finding.** MSD has no leakage by construction, so the expected result is
   close to clean: anything NZ202 or NZ203 says is either a real header
   problem in a famous dataset (worth reporting) or a false positive (worth a
   clean test).
3. **Calibrate NZ207 on an atlas-registered cohort** (BraTS, or MSD Task01,
   which is skull-stripped and co-registered). Different subjects registered
   to one template are the most likely place for the 0.995 fingerprint
   threshold to fire wrongly. Record the highest between-subject correlation
   in the changelog, as 0.2.0 recorded the sample-scan calibration.

---

## 3. 0.3.0: `nezaret result`

Re-score saved predictions against labels under each defensible choice and
report whether the conclusion survives, as `ayar` does for spike sorting:

- empty ground-truth cases: excluded (MONAI's `ignore_empty`), scored 1, or
  scored 0;
- Hausdorff/surface-distance implementation variants;
- per-case mean vs pooled;
- bootstrap intervals, and whether the ranking of two methods holds across all
  of the above.

The reference for which choices matter is the Metrics Reloaded guidelines and
the distance-metric implementation study (arXiv 2410.02630).

## 4. 0.4.0: nnU-Net

nnU-Net is the other framework most of this audience uses. Read its dataset
folders, split files and results summaries and run `data` and `result` on them.
Check the v2 file layout against a current install before writing anything.

---

## Smaller things, in no order

- **Inline suppression.** MONAI passes unknown keys to the transform, so an
  `_nezaret_ignore_` key in a transform dict would break the bundle. A
  `nezaret.toml` next to `configs/` with per-location ignores is the likely
  shape.
- **SARIF output**, so GitHub shows findings on the pull request diff.
- **A daftar adapter**: record the lint verdict of the bundle a run used, so
  a result carries "pipeline checked by nezaret 0.1.0, 0 errors" in its
  provenance.
- **Python source.** Pipelines written in a training script, not a config or a
  live object. AST reading of `Compose([...])` literals would cover the
  common case.

---

## Not planned

- **Running pipelines.** Every check is on configs, objects or files. The day
  nezaret needs a GPU, it has become a different tool.
- **Fixing configs automatically.** The fix is usually one line and the author
  should read it.
