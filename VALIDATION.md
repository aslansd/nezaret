# Validation on real data

*Run on 14 November 2026, nezaret 0.3.0 (before release), on a MacBook (Apple
silicon, no CUDA), Python 3.11, MONAI 1.6.1, from Tehran.*

The test suite runs on phantoms, nibabel's sample scans and synthetic DICOM.
This file records what nezaret did on real public data, how to repeat it, and
what each result changed in nezaret. Every number below was read from the
command output; nothing was estimated.

The helper scripts are in `tools/`.

---

## 1. `nezaret data` on MSD Task09 (spleen CT)

```bash
curl -L -C - -O https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar
md5 Task09_Spleen.tar              # 410d4a301da4e5b2f6f86ec3ddba524e
tar -xf Task09_Spleen.tar
nezaret data Task09_Spleen/dataset.json --no-pixels
nezaret data Task09_Spleen/dataset.json
```

The download (1535 MB, from the URL MONAI's `DecathlonDataset` uses) worked
from Tehran and matched MONAI's checksum.

**Result: 61 cases (41 training, 20 test), 102 files read, no findings**, with
and without `--no-pixels`. On a dataset with no planted fault, nezaret
reported nothing: no false positives on geometry, label values, empty labels
or duplicates.

---

## 2. Calibrating NZ207 on MSD Task01 (BraTS brain MRI)

Every BraTS case is registered to one template and skull-stripped, so
different patients look alike: the most likely place for the near-duplicate
fingerprint to fire wrongly.

```bash
curl -L -C - -O https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
md5 Task01_BrainTumour.tar         # 240a19d752f0d9e9101544901065d872
tar -xf Task01_BrainTumour.tar
python tools/fingerprint_calibration.py Task01_BrainTumour/dataset.json training
python tools/duplicate_groups.py Task01_BrainTumour/dataset.json training --csv brats_pairs.csv
```

(7255 MB; fingerprints for 484 cases in 95 s.)

### The distribution

| | fingerprint r |
|---|---|
| median over all 116,886 pairs | 0.9121 |
| 99th percentile | 0.9712 |
| 99.9th percentile | 0.9995 |
| pairs at or above NZ207's 0.995 | **178** |

### What the 178 pairs are

`duplicate_groups.py` reloaded both images of every pair at full resolution:

| fingerprint band | pairs | full-res r, lowest channel | label Dice, median |
|---|---|---|---|
| ≥ 0.9999 | 116 | 0.8257 – 1.0000 | 0.932 |
| 0.999 – 0.9999 | 6 | 0.9308 – 0.9909 | 0.831 |
| 0.995 – 0.999 | 56 | 0.7883 – 0.9898 | 0.742 |

274 of the 484 training images fall into 127 groups (120 pairs, 3 triples, and
groups of 5, 6, 7 and 7).

**The 116 pairs at ≥ 0.9999 are the same scans stored twice.** 114 of them
have full-resolution r = 1.0000 in every channel, yet 10–20 % of voxels differ
by more than 1 % of the intensity range (BRATS_277 vs BRATS_009: largest
difference 854 for intensities up to 1621). Correlation ignores a change of
scale, so these are the same voxels with different intensity normalisation.
In two pairs (BRATS_414/096, BRATS_114/423) one of the four MRI sequences
differs (lowest-channel r 0.87 and 0.83) and the other three match.

- **20 pairs, BRATS_275–294 against BRATS_001–020, have identical labels**
  (Dice 1.0 or 0.9999).
- **The other 96 have labels that differ** (Dice 0.65 – 0.96): the same scan,
  annotated twice.

**The 62 pairs below 0.9999 come from runs of consecutive case numbers**
(068–073, 054–056, 100–101, 121–122, 133–134, 149–152, 175–180, 183–189,
209–216, 241–242, 245–247): similar at full resolution (r 0.79 – 0.99), but
with tumours that differ (label Dice 0.17 – 0.92). They look like repeat scans
of the same patients at different times. That is an inference from the
numbers; the case numbers carry no patient identity.

**Why this is plausible.** The MSD paper describes Task01 as a subset of the
BraTS 2016 and 2017 data (Simpson et al., arXiv:1902.09063). BraTS 2017 reused
the BraTS 2012–13 images and annotations, and re-annotated the TCIA glioma
scans that BraTS 2014–16 had used with algorithmic labels and that mixed pre-
and post-operative scans (BraTS 2017 data description). That predicts exactly
the three patterns above: identical images and labels, identical images with
revised labels, and several scans of one patient. It has not been confirmed
case by case.

### Training against test, and inside the test set

```bash
nezaret data Task01_BrainTumour/dataset.json --format json -o task01.json
```

On all 750 cases (484 training, 266 test; 1234 files), the released 0.3.0
reports **52 errors and 2 warnings, from 201 duplicate groups**:

| | groups | images | same scan |
|---|---|---|---|
| across training and test | **52** | 104 (52 test images) | 52 |
| inside `training` | 127 | 274 | 116 |
| inside `test` | 22 | 49 | 5 |

- **52 of the 266 test images are a training image re-saved** (fingerprint
  r ≥ 0.9999; none byte-identical). The test images involved are
  BRATS_676–728, with partners across the training range, e.g. BRATS_243 /
  BRATS_720, BRATS_027 / BRATS_682, BRATS_112 / BRATS_695.
- **The 127 training groups are exactly those `tools/duplicate_groups.py`
  found** (274 images, 116 same-scan), so the reworked NZ207 agrees with the
  independent full-resolution analysis.
- **The test set has its own 22 groups** (49 images; the largest is
  BRATS_555–557).

**Confirmed at full resolution** for three of the 52 pairs (the test labels
are not public, so only the images can be compared):

```bash
python tools/compare_pairs.py Task01_BrainTumour BRATS_243 BRATS_720 BRATS_027 BRATS_682 BRATS_112 BRATS_695
```

| training | test | channel correlations | voxels that differ | headers |
|---|---|---|---|---|
| BRATS_243 | BRATS_720 | 1.0000, 1.0000, 1.0000, 1.0000 | 5,818,555 of 35,712,000 (16.3 %) | same |
| BRATS_027 | BRATS_682 | 1.0000, 1.0000, 1.0000, 1.0000 | 5,702,181 of 35,712,000 (16.0 %) | same |
| BRATS_112 | BRATS_695 | 1.0000, 1.0000, 1.0000, 1.0000 | 5,422,049 of 35,712,000 (15.2 %) | same |

The same pattern as the training duplicates: identical up to a change of
intensity scale. Task01's test set is the one the MSD challenge was scored
on, so about one test case in five is a training scan stored again.

### What this changed in nezaret

1. **The 0.995 threshold stands.** Every pair above it was a re-stored scan or
   one of a run of consecutive cases; none looked like two unrelated
   patients. The margin is smaller than on CT: unrelated registered brains
   reach 0.97 at the 99th percentile.
2. **NZ207 now reports duplicates inside one split.** 0.3.0 as first built
   compared images across splits only, so on this dataset, where every
   duplicate is inside `training`, it would have said nothing. But a user who
   holds out part of `training` for validation (as most bundles do) leaks
   these groups. It is now one warning per split, with the groups in the JSON
   report.
3. **NZ207 reports groups, not pairs**, and treats fingerprint r ≥ 0.9999 as
   the same scan (an error across splits), not a near-duplicate. All 116 BraTS
   pairs in that band were the same scan; only 2 of them had identical voxels.

---

## 3. `nezaret result` on two real models

Two MONAI model-zoo bundles scored on the 9 MSD spleen cases that
`spleen_ct_segmentation` holds out for validation (the last 9 in sorted order):

- **spleen_ct_segmentation**, a 3D UNet trained on MSD Task09;
- **swin_unetr_btcv_segmentation**, a Swin UNETR trained on BTCV (13 organs;
  class 1 is spleen), so this is also a cross-dataset test.

```bash
pip install fire pytorch-ignite itk einops huggingface-hub    # what bundle inference needs
python -m monai.bundle download --name spleen_ct_segmentation --bundle_dir bundles
python -m monai.bundle download --name swin_unetr_btcv_segmentation --bundle_dir bundles

for b in spleen_ct_segmentation swin_unetr_btcv_segmentation; do
  (cd bundles/$b && python -m monai.bundle run --config_file configs/inference.json --bundle_root . \
    --dataset_dir ~/nezaret-validation/Task09_Spleen \
    --datalist '$list(sorted(glob.glob(@dataset_dir + "/imagesTr/*.nii.gz")))[-9:]' \
    --output_dir ~/nezaret-validation/preds/$b \
    --checkpointloader#map_location cpu --inferer#sw_batch_size 1)
done

python tools/prepare_spleen_eval.py Task09_Spleen preds/spleen_ct_segmentation preds/swin_unetr_btcv_segmentation eval
nezaret result --labels eval/labels --pred spleen_unet=eval/spleen_unet --pred swin_unetr=eval/swin_unetr
```

Notes from running it:

- `monai.bundle download` reached Hugging Face from Tehran once
  `huggingface-hub` was installed. The weights were also available from
  NVIDIA's CDN through `python -m monai.bundle download_large_files`.
- Both bundles' weights were saved on a GPU and their configs set no
  `map_location`, so on a machine without CUDA `--checkpointloader#map_location
  cpu` is needed.
- On the Mac's CPU, the UNet took 2 min 39 s for all 9 cases. Swin UNETR with
  the bundle's `sw_batch_size` of 4 ran out of memory and swapped (67 minutes
  for the first case); with `--inferer#sw_batch_size 1`, which does not change
  the prediction, it took 3 h 9 min.

### Result

| | spleen_unet | swin_unetr |
|---|---|---|
| Dice, MONAI default | **0.9652** [0.9571, 0.9708] | 0.9557 [0.9505, 0.9604] |
| Dice range over the 7 conventions | 0.9652 – 0.9679 | 0.9557 – 0.9573 |
| HD95, MONAI formula, mm | **2.18** | 5.90 |
| HD95, MedPy formula, mm | 1.99 | 3.45 |
| HD95 in voxel units (spacing not passed) | 1.08 | 2.53 |
| ASSD, mm | 0.39 | 0.87 |

Paired difference spleen_unet − swin_unetr: Dice +0.0095 [+0.0045, +0.0139]
over 9 cases, P(spleen_unet better) = 1.00; spleen_unet wins under all 7 Dice
conventions and all 16 HD95 implementations.

**What this showed:**

- **No false alarm where the answer is clear.** The ranking is robust, so
  NZ306 and NZ307 were rightly silent.
- **On a real organ, HD95 depends heavily on the implementation.** For
  swin_unetr, MONAI's formula gives 5.90 mm and MedPy's 3.45 mm from the same
  predictions: MONAI's is 71 % higher. For the UNet the difference is 9 %.
- **Forgetting `spacing` roughly halves HD95 on these CTs** (thick slices), as
  the NZ305 warnings said: 2.18 mm becomes 1.08.

---

## Still to do

- Repeat section 3 with predictions that are close, where NZ306 and NZ307
  should fire on real data.
