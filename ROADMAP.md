# Roadmap

*Last revised 16 November 2026, against nezaret 0.3.1.*

---

## Where this stands

**0.3.1**, Apache 2.0, **zero runtime dependencies**, **148 tests**,
**twenty-six rules** (eleven for pipelines, eight for data, seven for
results), checked against MONAI 1.6.1 and MedPy 0.5.2.

| | Status |
|---|---|
| `nezaret lint`: bundle configs, JSON and YAML, static `@`/`%`/`$@a + @b` resolution | shipped |
| `lint_compose()` on live MONAI pipelines | shipped |
| Run over all 37 model-zoo bundles, every error read by hand | done, see README |
| Upstream reports for the model-zoo findings | **drafted in [upstream/](upstream/README.md), not filed yet, see §1** |
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

Running nezaret over the model zoo found four bundles with faults that change
results, mostly without an error. Filing them is worth more than any feature
below: a merged fix is public evidence that the rules find real things, which
is the case for the package.

**Drafts are ready** in [upstream/](upstream/README.md). There is one issue per
bundle, each with permalinked config lines, a reproduction that has been run,
the one-line fix, and nezaret mentioned only at the end. Everything was
re-checked on 16 November 2026 against model-zoo `feb2e8c` and MONAI 1.4.0
to 1.6.1. That check added two things the lint output alone did not show:

- `prostate_mri_anatomy`'s `RandZoomd` uses `min`/`max` instead of
  `min_zoom`/`max_zoom`. It is harmless up to MONAI 1.5.0, but from 1.5.1 every
  zoom below 1 raises `ValueError`, so training stops in the first epoch. This
  gives a fifth, optional, issue against MONAI itself: unknown keyword
  arguments should be rejected at construction.
- The renal orientation mismatch holds for **any** file: training always ends
  in RAS (`as_closest_canonical`), inference always in LPS. The bundle README
  says the data is stored in LPS, which does not change this.
  `tools/check_orientation.py` shows it on one case (`FLIPPED x,y`, or `SAME`
  once inference uses RAS).

| # | Bundle | Report | Before filing | Issue | Status |
|---|---|---|---|---|---|
| 1 | `prostate_mri_anatomy` | label interpolated in `RandAffined`/`Rand3DElasticd`/`RandZoomd`; `RandZoomd` `min`/`max` crash on ≥ 1.5.1; `Invertd` with `nearest_interp: false` after `AsDiscreted` | — | — | drafted |
| 2 | `renalStructures_CECT_segmentation` | training RAS, `inference.json` LPS; `RandZoomd` label mode; `download_data` 404 (Yandex link gone) | done 16 Nov: `FLIPPED x,y` on Task09 `spleen_10`; Yandex link dead in browser too | — | ready to file |
| 3 | `wholeBrainSeg_Large_UNEST_segmentation` | move `NormalizeIntensityd` into `deterministic_transforms` so validation normalises | read model-zoo #534 and discussion #557 | — | drafted |
| 4 | `mednist_reg` | `IgniteMetric` → `IgniteMetricHandler` (removed in 1.4; training does not start) | — | — | drafted |
| 5 | MONAI core | `Zoom`/`RandZoom` should reject unknown `**kwargs` at construction | file after #1, link it | — | optional |

Order: 4, 1, 3, 2, then 5. Search for duplicates first; the session that
drafted these could not query GitHub. Offer PRs rather than opening them
unasked; a model-zoo PR needs a `metadata.json` version bump and changelog
line.

**Announcing.** After the issues exist (about a week, or after a maintainer
replies):

1. a post in Project-MONAI/MONAI Discussions (draft in upstream/README.md);
2. a two-line pointer on each affected bundle's Hugging Face page, since that
   is where the bundles are downloaded from now;
3. a longer write-up on your own channels.

Fill in the Issue column as they are filed, and change Status to
`filed` / `acknowledged` / `fixed in <bundle version>`. A follow-up for the
linter: rule for unknown arguments to a `_target_` (NZ112), which would have
caught `min`/`max` directly (see "Smaller things").

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
- **NZ112: unknown arguments to a `_target_`.** Compare a transform dict's keys
  with the class signature. `RandZoomd(min=…, max=…)` in `prostate_mri_anatomy`
  passes MONAI's construction checks because the keys go into `**kwargs`, then
  crashes during training on MONAI ≥ 1.5.1. The rule needs the signatures
  shipped beside `monai_components.json`, and needs care where `**kwargs` is
  genuinely forwarded.
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
