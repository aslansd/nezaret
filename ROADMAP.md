# Roadmap

*Last revised 14 November 2026, against nezaret 0.3.0.*

---

## Where this stands

**0.3.0**, Apache 2.0, **zero runtime dependencies**, **148 tests**,
**twenty-six rules** (eleven for pipelines, eight for data, seven for
results), checked against MONAI 1.6.1 and MedPy 0.5.2.

| | Status |
|---|---|
| `nezaret lint`: bundle configs, JSON and YAML, static `@`/`%`/`$@a + @b` resolution | shipped |
| `lint_compose()` on live MONAI pipelines | shipped |
| Run over all 37 model-zoo bundles, every error read by hand | done, see README |
| Upstream reports for the model-zoo findings | **not filed yet, see §1** |
| `nezaret data`: datalist + NIfTI/DICOM checks (0.2.0) | shipped, §2 |
| `nezaret data` on real cohorts: MSD Task09 and Task01 | done, [VALIDATION.md](VALIDATION.md) |
| `nezaret result`: re-scoring under every convention, bootstrap, ranking (0.3.0) | shipped, §3 |
| `nezaret result` on two real models' predictions | done, [VALIDATION.md](VALIDATION.md) |
| **External users** | **none yet, the thing that actually matters** |

```
tests/test_rules.py        56
tests/test_cli.py          10
tests/test_datalist.py     14
tests/test_data.py         27   (skips without nibabel)
tests/test_result.py       18   (skips without nibabel/scipy)
tests/test_live_metrics.py  7   (MONAI and MedPy parts skip without them)
tests/test_examples.py      2
tests/test_live_monai.py   11   (skips without MONAI)
tests/test_model_zoo.py     3   (opt-in: NEZARET_MODEL_ZOO)
                          ---
                          148
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

## 2. 0.2.0: `nezaret data`: shipped and validated on real data

Shipped as planned: geometry agreement (NZ202), label sanity (NZ203, NZ204),
class coverage (NZ205), patient leakage from DICOM `PatientID`, datalist
fields or path regexes, folds included (NZ206), exact and near-duplicate
voxels (NZ207), scanner and site balance (NZ208), plus missing files (NZ201).
nibabel and pydicom are the optional `[data]` extra; the core stays
dependency-free, and a bare install still finds missing files and shared
entries.

**The real-cohort check is done** ([VALIDATION.md](VALIDATION.md), 14
November 2026). MSD Task09 downloaded from Tehran and gave no findings. On
MSD Task01, the calibration found 178 pairs above 0.995, all of them either
the same scan stored twice (116) or within runs of consecutive cases that look
like repeat scans of one patient (62): the threshold stands. It also showed
that duplicates inside one split went unreported, which 0.3.0 now reports.

**Still open:**

1. **Check the remaining 49 Task01 training/test pairs at full resolution.**
   0.3.0 reports 52 test images that are a training scan re-saved; three
   (BRATS_243/720, 027/682, 112/695) are confirmed in VALIDATION.md, all four
   channels at r = 1.0000.
2. **Report the Task01 duplicates.** First check whether they are already
   documented (BraTS forums, the MSD paper's errata, papers that use Task01);
   if not, a short note with the group list from
   `nezaret data --format json` is useful to everyone who trains on Task01
   with a random split, and the training/test overlap matters to everyone who
   compares results on the MSD Task01 test set.

---

## 3. 0.3.0: `nezaret result`: shipped, first real run done

Shipped as planned: every prediction re-scored under three empty-ground-truth
conventions × two class/case orders, plus pooled Dice; sixteen HD95 and
sixteen ASSD variants (MONAI vs MedPy formula, mm vs voxels, 6- vs 26-neighbour
edges, empty masks excluded vs worst case); paired bootstrap intervals; the
ranking of every pair of methods under every variant. Seven rules, NZ301 to
NZ307. Each convention named after a library is checked against that
library: MONAI 1.6.1 and MedPy 0.5.2, per case and aggregated.

**Done:** the first real run ([VALIDATION.md](VALIDATION.md)): two model-zoo
bundles on 9 MSD spleen cases. The ranking was robust and nezaret said so;
HD95 for the same Swin UNETR predictions was 5.90 mm with MONAI's formula and
3.45 mm with MedPy's.

**Still open:**

1. **A real comparison that is close**, where NZ306 and NZ307 should fire:
   e.g. one bundle at two training checkpoints, or two runs with different
   seeds.
2. **Normalised surface distance (NSD) and boundary IoU.** Metrics Reloaded
   recommends NSD for many segmentation tasks, and the distance-metric
   implementation study (arXiv 2410.02630) found NSD implementations
   disagree too. Both reuse the edge distances nezaret already computes;
   each needs its tolerance as a parameter and a parity test against MONAI's
   `SurfaceDiceMetric`.

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
