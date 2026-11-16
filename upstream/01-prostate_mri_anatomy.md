**Title:** `prostate_mri_anatomy`: label maps are interpolated in training augmentation and in inference inversion; `RandZoomd` crashes on MONAI ≥ 1.5.1

**Labels to suggest:** bug

---

**Describe the bug**

Three places in `prostate_mri_anatomy` (bundle 0.3.7) treat the label map as a continuous image. None of them raises an error on the declared MONAI 1.4.0, so each one changes training or the saved prediction without notice.

1. **Training augmentation interpolates the label.** `RandAffined` and `Rand3DElasticd` are applied to `["image", "label"]` without `mode`, so both keys use the default `bilinear`.
   - [`configs/train.json#L128-L138`](https://github.com/Project-MONAI/model-zoo/blob/feb2e8c318e1964de5d3e1c1000474571e03ad50/models/prostate_mri_anatomy/configs/train.json#L128-L138) (`RandAffined`)
   - [`configs/train.json#L149-L167`](https://github.com/Project-MONAI/model-zoo/blob/feb2e8c318e1964de5d3e1c1000474571e03ad50/models/prostate_mri_anatomy/configs/train.json#L149-L167) (`Rand3DElasticd`)

   The label has three classes (0, 1, 2). Bilinear interpolation produces fractional values at every class boundary, and where 0 meets 2 it creates a rim of 1. `DiceFocalLoss(to_onehot_y=True)` then truncates those values to integers, so the network is trained on boundaries that are not in the annotation. The deterministic `Spacingd` a few lines earlier already uses `"mode": ["bilinear", "nearest"]`; the random transforms do not.

2. **`RandZoomd` is configured with `min`/`max` instead of `min_zoom`/`max_zoom`, and has no `mode`.** See [`configs/train.json#L168-L177`](https://github.com/Project-MONAI/model-zoo/blob/feb2e8c318e1964de5d3e1c1000474571e03ad50/models/prostate_mri_anatomy/configs/train.json#L168-L177):

   ```json
   {"_target_": "RandZoomd", "keys": ["image", "label"], "prob": 0.15, "max": 1.1, "min": 0.9}
   ```

   `min` and `max` are not parameters of `RandZoomd`. They land in `**kwargs` and are passed on to the padding call. The defaults are 0.9/1.1 anyway, so the zoom range is unchanged, but:
   - On MONAI ≤ 1.5.0 the extra keys are silently dropped.
   - On MONAI ≥ 1.5.1, every zoom below 1 (which needs padding) raises `ValueError: unsupported keyword arguments for mode 'edge': {'min', 'max'}`. That happens on about half of the 15 % of samples where the transform fires, so training stops within the first epoch.
   - With no `mode`, the label is also zoomed with the default `area`.

3. **Inference inverts the discrete prediction with interpolation.** In `configs/inference.json`, postprocessing runs `AsDiscreted(argmax=True)` ([L104-L108](https://github.com/Project-MONAI/model-zoo/blob/feb2e8c318e1964de5d3e1c1000474571e03ad50/models/prostate_mri_anatomy/configs/inference.json#L104-L108)) and then `Invertd(..., "nearest_interp": false)` ([L117-L125](https://github.com/Project-MONAI/model-zoo/blob/feb2e8c318e1964de5d3e1c1000474571e03ad50/models/prostate_mri_anatomy/configs/inference.json#L117-L125)). The inversion undoes `Spacingd(mode="bilinear")`, which resamples the 0/1/2 label map bilinearly back to the original grid. The saved file therefore contains values such as 0.2 and 1.64 at boundaries, and 1 between 0 and 2.

**To Reproduce**

This uses only MONAI and synthetic data. It runs the three random spatial transforms exactly as configured, then as fixed.

```python
import numpy as np, torch, monai.transforms as T

lab = torch.zeros(1, 48, 48, 24); lab[0, 10:38, 10:38, 4:20] = 1; lab[0, 18:30, 18:30, 8:16] = 2
k = ["image", "label"]

def chain(fixed):
    m = {"mode": ["bilinear", "nearest"]} if fixed else {}
    z = dict(min_zoom=0.9, max_zoom=1.1, mode=["area", "nearest"]) if fixed else dict(min=0.9, max=1.1)
    return T.Compose([
        T.RandAffined(k, prob=1, rotate_range=5, shear_range=0.5, translate_range=25, **m),
        T.Rand3DElasticd(k, prob=1, magnitude_range=[0.5, 1.5], rotate_range=5, shear_range=0.5,
                         sigma_range=[0.5, 1.5], translate_range=25, **m),
        T.RandZoomd(k, prob=1, **z)])

for fixed in (False, True):
    crashes, values = 0, set()
    for seed in range(30):
        c = chain(fixed); c.set_random_state(seed)
        try:
            out = c({"image": torch.rand(1, 48, 48, 24), "label": lab.clone()})["label"]
            values |= set(np.round(out.numpy().ravel(), 2).tolist())
        except Exception:
            crashes += 1
    print("fixed" if fixed else "as shipped", "| crashes:", crashes, "/ 30 | distinct label values:", len(values))
```

On MONAI 1.6.1 this prints:

```
as shipped | crashes: 15 / 30 | distinct label values: 201
fixed | crashes: 0 / 30 | distinct label values: 3
```

For the inference part, `Invertd` with `nearest_interp=False` after `AsDiscreted` behaves the same way. In a synthetic 3-class check, about 1 % of output voxels were non-integer; with `nearest_interp=True` there were none.

**Expected behavior**

Label maps stay integer-valued throughout training augmentation and in the saved prediction.

**Suggested fix** (one line each)

- `RandAffined` and `Rand3DElasticd`: add `"mode": ["bilinear", "nearest"]`.
- `RandZoomd`: use `"min_zoom": 0.9, "max_zoom": 1.1, "mode": ["area", "nearest"]`.
- `inference.json` `Invertd`: set `"nearest_interp": true`. Alternatively, move `Invertd` before `AsDiscreted`, so the probabilities are inverted and the argmax is taken in the original space.

I'm happy to open a PR with these changes and the version bump, if that helps.

**Environment**

<!-- paste the output of: python -c 'import monai; monai.config.print_debug_info()' -->

The `RandZoomd` crash reproduces on MONAI 1.5.1, 1.5.2 and 1.6.1 but not on 1.4.0 or 1.5.0. The interpolation behaviour is the same on all of them.

**Additional context**

Found by running the configuration linter [nezaret](https://github.com/aslansd/nezaret) (`nezaret lint models/prostate_mri_anatomy`) over the model zoo at `feb2e8c`, then confirmed with the script above.
