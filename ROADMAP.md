# Roadmap

*Last revised 7 November 2026, against nezaret 0.1.0.*

---

## Where this stands

**0.1.0**, Apache 2.0, **zero runtime dependencies**, **82 tests**,
**eleven rules**, checked against MONAI 1.6.1.

| | Status |
|---|---|
| `nezaret lint`: bundle configs, JSON and YAML, static `@`/`%`/`$@a + @b` resolution | shipped |
| `lint_compose()` on live MONAI pipelines | shipped |
| Run over all 37 model-zoo bundles, every error read by hand | done, see README |
| Upstream reports for the model-zoo findings | **not filed yet, see §1** |
| `nezaret data`, `nezaret result` | planned, §2 and §3 |
| **External users** | **none yet, the thing that actually matters** |

```
tests/test_rules.py        56
tests/test_cli.py          10
tests/test_examples.py      2
tests/test_live_monai.py   11   (skips without MONAI)
tests/test_model_zoo.py     3   (opt-in: NEZARET_MODEL_ZOO)
                          ---
                           82
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

## 2. 0.2.0: `nezaret data`

The data list and the images, still without training anything.

- **Geometry agreement.** Image and label of each case: same shape, spacing,
  origin and direction. A label exported from a viewer with a flipped axis
  loads, trains, and is wrong.
- **Patient leakage across splits.** Same patient in training and test,
  found from DICOM `PatientID`/`StudyInstanceUID` and header fields, and from
  near-duplicate pixel hashes for when the IDs were scrubbed differently.
  `dicleak` checks a CSV manifest for overlap and says it does not look at
  DICOM metadata, pixels or hashes. That is the gap.
- **Site and scanner balance.** Scanner model, field strength and site per
  split, from the headers.
- **Label sanity.** Values that are not integers, classes absent from a split,
  empty masks counted.

Readers: nibabel and pydicom as an optional `[data]` extra, so the core stays
dependency-free.

**Check before starting:** that the open datasets used for testing (Medical
Segmentation Decathlon, a TCIA collection) download from Tehran. Some hosts
block sanctioned regions. If they do not, build the test fixtures from
synthetic phantoms written with nibabel, which is better for planted faults
anyway.

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
