**Title:** `wholeBrainSeg_Large_UNEST_segmentation`: validation skips `NormalizeIntensityd`, so the validation Dice and best-checkpoint selection use un-normalised inputs

**Labels to suggest:** bug

---

**Describe the bug**

In `configs/train.json` (bundle 0.2.8), `NormalizeIntensityd(keys="image", nonzero=true, channel_wise=true)` sits at the end of `train#random_transforms` ([`#L129-L134`](https://github.com/Project-MONAI/model-zoo/blob/feb2e8c318e1964de5d3e1c1000474571e03ad50/models/wholeBrainSeg_Large_UNEST_segmentation/configs/train.json#L129-L134)), even though it is not random. The two pipelines use different transform lists:

```json
"train":    {"preprocessing": {"transforms": "$@train#deterministic_transforms + @train#random_transforms"}},
"validate": {"preprocessing": {"transforms": "%train#deterministic_transforms"}}
```

As a result:

- Training sees normalised crops.
- `configs/inference.json` normalises too ([`#L44-L49`](https://github.com/Project-MONAI/model-zoo/blob/feb2e8c318e1964de5d3e1c1000474571e03ad50/models/wholeBrainSeg_Large_UNEST_segmentation/configs/inference.json#L44-L49)).
- **Validation does not normalise.** It feeds the network raw T1 intensities it never saw in training.

The validation mean Dice is the key metric for `CheckpointSaver(save_key_metric=true)`. So the checkpoint kept as `model.pt` is chosen by a number computed on out-of-distribution input. Nothing raises an error; the validation curve just reads lower and noisier than the model really is.

A smaller second effect: because normalisation runs after `RandSpatialCropd`, training normalises each crop with that crop's own statistics, whereas inference normalises the whole volume.

**To Reproduce**

Read the two `preprocessing` entries above. Alternatively, instantiate `validate#preprocessing` and check that no `NormalizeIntensityd` is in it:

```python
from monai.bundle import ConfigParser
p = ConfigParser(); p.read_config("configs/train.json")
print([type(t).__name__ for t in p.get_parsed_content("validate#preprocessing").transforms])
```

**Expected behavior**

Validation preprocesses images the way training and inference do.

**Suggested fix** (one line)

Move the `NormalizeIntensityd` block from the end of `train#random_transforms` to the end of `train#deterministic_transforms`. Validation then normalises, and training normalises the whole volume before cropping, as inference does. A short fine-tune or re-selection of the checkpoint would show whether the released `model.pt` is the best one.

I'm happy to open a PR with this change and the version bump.

**Environment**

<!-- paste the output of: python -c 'import monai; monai.config.print_debug_info()' -->

**Additional context**

Possibly related to #534 (preprocessing question) and discussion #557 (noisy segmentations). Found by running the configuration linter [nezaret](https://github.com/aslansd/nezaret) (`nezaret lint models/wholeBrainSeg_Large_UNEST_segmentation`) over the model zoo at `feb2e8c`. It flags intensity transforms that run in training but not in validation.
