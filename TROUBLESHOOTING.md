# Troubleshooting

First step, always:

```bash
nezaret doctor
```

It reports the nezaret version, the MONAI release its rules were checked
against, whether PyYAML is there (for YAML bundles), and whether MONAI imports
(for `lint_compose()`), or **`BROKEN`** with the exception if it is installed
and fails to import, which usually means a torch problem underneath.

---

## "nezaret: no config files checked"

Exit code 2. Either:

- **none found.** nezaret looks for `configs/` under the path you gave,
  or for `.json`/`.yaml` files containing `_target_` directly in it. Point it
  at the bundle folder, the `configs/` folder or a single file.
- **none could be read.** The notes above the summary say why. The usual case
  is a YAML bundle without PyYAML: `pip install "nezaret[yaml]"`.

## A finding I think is wrong

Read `nezaret explain <rule>` first; several rules fire on things that look
deliberate and are not. If it really is a false positive:

```bash
nezaret lint PATH --ignore NZ108
```

and please open an issue with the config lines. Every false positive found so
far became a test (see TESTING.md), and that is how the rules get better.

### NZ101/NZ102/NZ103 on a key that is not a label

Label keys are inferred from names (`label`, `seg`, `mask`, `gt`, …) and from
per-key `mode` lists. If your keys are named differently, say which ones hold
label maps, and only those are used:

```bash
nezaret lint PATH --label-keys organs,lesions
```

### NZ103 on a key that is meant to be left behind

nezaret already allows: a crop guide (`label_key` of a label-guided crop),
centred crops of different sizes, keys removed with `DeleteItemsd` or
`SelectItemsd`, and in evaluation a label kept in original space. If your
pipeline drops a key some other way, remove it with `DeleteItemsd` before the
spatial transform. That also makes the intent visible to the next reader.

### NZ104 on my own transform

A bare `_target_` is resolved by MONAI against its own modules only. Your own
classes need a qualified name (`scripts.transforms.MyTransform`), which
nezaret skips. If MONAI resolves the bare name for you, the class is in a
MONAI version newer than the list nezaret ships (see `nezaret doctor`), and
`tools/refresh_monai_components.py` updates it.

### NZ109 when training data was preprocessed offline

`Spacingd only in evaluation` is correct if the training images were resampled
to the same grid before training. The message says so; ignore the rule for
that bundle, and consider writing the offline step down in the bundle's README.

---

## Nothing is reported for a pipeline I know is wrong

- **Is it built by an expression?** `"transforms": "$build(...)"` cannot be
  read without running code, so the pipeline is skipped. `report.pipelines`
  (and the summary line) says how many pipelines were found.
- **Are its keys or modes expressions?** `"mode": "$@modes"` is unknown, so
  NZ101 skips that transform rather than guessing.
- **Is there a custom step before it?** After a `scripts.*` transform the
  key-tracking rules stop, because the step may have created or merged keys.
- **Was the role detected?** Evaluation-only rules (NZ107, NZ108) need the
  pipeline's path or file name to say validation, evaluation, inference or
  test. The JSON report shows the detected role of the pipeline behind every
  finding (`"pipeline": "train.json:validate#preprocessing (eval)"`).

## `lint_compose()` finds nothing in my post-processing pipeline

Pass `role="post"`, and for NZ106 pass the preprocessing it inverts:

```python
nezaret.lint_compose(post, role="post", preprocessing=val_transforms)
```

The preprocessing is used as context only. Lint it separately with
`role="eval"` if you want it checked too.

---

## `nezaret data`

### "N file(s) not checked: reading this needs nibabel"

Install the data extra: `pip install "nezaret[data]"`. Without it nezaret
still checks paths (NZ201) and entries shared between splits (NZ206), and says
which files it could not open. It never treats an unread file as clean.

### "pixel data not decodable"

The DICOM is compressed (JPEG, JPEG 2000, JPEG-LS) and pydicom has no decoder.
Headers are still read; for the voxel checks install one, e.g.
`pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg`.

### "N series in one folder"

Each datalist entry must point at one DICOM series: a file, or a folder of
slices sharing one `SeriesInstanceUID`. Split the folder by series.

### No NZ206 or NZ208 on a NIfTI dataset I know has repeat patients or several sites

NIfTI headers carry no patient or scanner information. Tell nezaret where it
is: a `patient_id` field in the datalist entries, `--patient-key` for another
field name, or a regex on the file path whose first group is the identifier:

```bash
nezaret data dataset.json --patient-id-regex "(sub-\d+)" --site-regex "/(site\d+)/"
```

### NZ207 warns about duplicates inside `training`

That is deliberate. If you hold out part of `training` for validation, which
most bundles do, members of one group end up on both sides. The JSON report
lists the groups (`nezaret data dataset.json --format json`, then
`stats.duplicate_groups`): keep each group on one side of the split. If you
never split that list, the warning is about the size of your effective
dataset, not leakage.

### NZ207 near-duplicates between different patients

The fingerprint is coarse on purpose (16×16×16 block means). It was
calibrated on BraTS (MSD Task01), where every subject is registered to one
template: unrelated subjects stayed below the 0.995 threshold (99th percentile
0.971), and every pair above it was the same scan or a likely repeat scan of
one patient (see VALIDATION.md). If it fires between subjects you know are
different, raise the threshold (`--near-duplicate 0.999`) and please report
the correlation, with `tools/duplicate_groups.py`'s output if you can.

### NZ202 on a dataset that trains fine

Check which header is right before re-exporting anything: open the image and
the label together in a viewer (ITK-SNAP, 3D Slicer). MONAI uses each file's
own header, so if the label's header is wrong the network trains on labels
that sit somewhere other than where they appear in the viewer when opened
alone.

### It is slow

Voxel checks read every file once, and decompressing `.nii.gz` dominates
(about 0.4 s per CT-sized case). `--no-pixels` checks paths, geometry, patient
identity and scanners from headers alone in a fraction of a second.

---

## `nezaret result`

### NZ301: every prediction is "missing"

Files are paired by case name: the file name without `.nii`/`.nii.gz` and
without a postfix such as `_seg`, `_pred` or `_trans`. If your names differ
in another way (`pred_case01.nii.gz` vs `case01.nii.gz`), give a regex whose
first group is the case name:

```bash
nezaret result --labels labels/ --pred m=preds/ --case-regex "(case\d+)"
```

### "N files map to case 'x'; none of them scored"

Two files in one folder have the same case name after postfixes are removed,
e.g. `x_seg.nii.gz` and `x_pred.nii.gz`. Keep one prediction per case per
folder, or use `--case-regex`.

### NZ302 on every case

The predictions were saved on the preprocessed grid. Run `nezaret lint` on
the bundle: `NZ106` usually explains it. Put `Invertd` before `SaveImaged`, or
set `resample=true` on `SaveImaged`.

### NZ305 as info on every run

That is expected. HD95 depends on the implementation for most real
predictions, by 10-20 % between MONAI's and MedPy's formula on the phantom
example. It is reported so the number in a paper can name its implementation.
It is a warning only when voxel units would give a different number from mm.

### NZ307: "difference unresolved" on a result I believe

The paired bootstrap uses the cases both methods were scored on, with an empty
ground truth excluded (MONAI's default). With few cases or a small difference,
the interval includes zero. More test cases, not more bootstrap resamples,
narrow it.

### It is slow

Surface distances dominate. `--no-distances` scores Dice only, about eight
times faster on CT-sized volumes. `--classes 1` scores one class.
