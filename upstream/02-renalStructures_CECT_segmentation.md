**Title:** `renalStructures_CECT_segmentation`: training feeds RAS arrays but `inference.json` reorients to LPS; `RandZoomd` interpolates the label; `download_data` returns 404

**Labels to suggest:** bug

---

**Describe the bug**

1. **Training and inference give the network differently oriented arrays.**
   - Training loads every phase with `LoadImaged(..., "as_closest_canonical": true)` ([`configs/train.json#L65-L78`](https://github.com/Project-MONAI/model-zoo/blob/feb2e8c318e1964de5d3e1c1000474571e03ad50/models/renalStructures_CECT_segmentation/configs/train.json#L65-L78)) and applies no `Orientationd`. Validation reuses the same `deterministic_transforms`. So whatever orientation the files are stored in, the arrays are reordered to RAS before the network sees them, and `evaluate` reports Dice on RAS inputs.
   - `configs/inference.json` loads without `as_closest_canonical` and then applies `Orientationd(axcodes="LPS")` ([`configs/inference.json#L55-L63`](https://github.com/Project-MONAI/model-zoo/blob/feb2e8c318e1964de5d3e1c1000474571e03ad50/models/renalStructures_CECT_segmentation/configs/inference.json#L55-L63)). The README's note supports this ("underlying data is in LPS orientation … reorient it from RAS to LPS").

   RAS and LPS differ by a flip of the first two axes. As a result, `inference.json` runs the network on inputs mirrored left–right and front–back relative to everything it was trained and validated on. That holds for any file, because one path always ends in RAS and the other in LPS. The bundle's validation data can't currently be downloaded (part 3), so here is the check on a public CT (MSD Task09, `spleen_10.nii.gz`). The training-style array and the inference-style array are the same voxels mirrored along x and y:

   ```
   $ python tools/check_orientation.py Task09_Spleen/imagesTr/spleen_10.nii.gz

Task09_Spleen/imagesTr/spleen_10.nii.gz
  stored shape      (512, 512, 55)
  stored axcodes    RAS
  affine            [[   0.977    0.       0.    -499.023]
                     [   0.       0.977    0.    -499.023]
                     [   0.       0.       5.       0.   ]
                     [   0.       0.       0.       1.   ]]
  training array    (1, 512, 512, 55)
  inference array   (1, 512, 512, 55)
  verdict           FLIPPED x,y
   ```

   `RandAxisFlipd` and `RandRotate90d` (p = 0.3 each) mean the model has seen some mirrored crops, so the damage may be partial. It is still a mismatch between the pipeline that produced the reported Dice and the one users run on their own data. Left and right kidney structures are where it would show.

2. **`RandZoomd` zooms the label with the default `area` mode.** See [`configs/train.json#L169-L176`](https://github.com/Project-MONAI/model-zoo/blob/feb2e8c318e1964de5d3e1c1000474571e03ad50/models/renalStructures_CECT_segmentation/configs/train.json#L169-L176):

   ```json
   {"_target_": "RandZoomd", "keys": ["image", "label"], "prob": 0.3}
   ```

   Area interpolation averages neighbouring label values, so boundaries between classes get fractional labels. These are then truncated when `DiceLoss(to_onehot_y=True)` converts the label to one-hot.

3. **The validation data can no longer be downloaded.** `python -m monai.bundle run download_data --meta_file configs/metadata.json --config_file "['configs/train.json', 'configs/evaluate.json']"` fails with `urllib.error.HTTPError: HTTP Error 404: NOT FOUND`. The 404 comes from the Yandex API call in [`configs/evaluate.json#L46`](https://github.com/Project-MONAI/model-zoo/blob/feb2e8c318e1964de5d3e1c1000474571e03ad50/models/renalStructures_CECT_segmentation/configs/evaluate.json#L46), and the public link https://disk.yandex.ru/d/pWEKt6D3qi3-aw no longer opens in a browser either (checked 16 November 2026). Without it, the README's expected validation Dice (~0.784) can't be reproduced. Separately, `docs/data_license.txt` links the BraTS 2018 page as the source of this dataset, which looks like a copy-paste slip.

**To Reproduce**

For part 1: run `tools/check_orientation.py` from nezaret on any NIfTI. It loads the file as `train.json` does and as `inference.json` does, then compares the two arrays. The comparison is in roughly 40 lines of plain MONAI (`LoadImage(as_closest_canonical=True)` vs `LoadImage()` + `Orientation("LPS")`). It reports `FLIPPED x,y`, and `SAME` once the inference axcodes are set to RAS.

For part 2: `RandZoomd(["image", "label"], prob=1)` applied to an integer label returns non-integer values; with `mode=["area", "nearest"]` it does not.

**Expected behavior**

The network sees inputs in the same orientation at training, validation and inference, label maps stay integer-valued, and `download_data` fetches the validation set.

**Suggested fix** (one line each)

- `inference.json` `Orientationd`: `"axcodes": "RAS"`. Alternatively, add `"as_closest_canonical": true` to its `LoadImaged` and drop the `Orientationd`. Either way the README note about LPS should change too. With RAS, the check above reports `SAME`, and `Invertd` still writes the prediction back in the input's own orientation.
- `train.json` `RandZoomd`: add `"mode": ["area", "nearest"]`.
- Data: re-share the validation archive, or move it to the bundle's Hugging Face repo or another host, and fix the source link in `docs/data_license.txt`.

I'm happy to open a PR with these changes and the version bump.

**Environment**

<!-- paste the output of: python -c 'import monai; monai.config.print_debug_info()' -->

**Additional context**

Found by running the configuration linter [nezaret](https://github.com/aslansd/nezaret) (`nezaret lint models/renalStructures_CECT_segmentation`) over the model zoo at `feb2e8c`. It flags orientation that differs between a bundle's training and inference configs. The orientation was then confirmed by loading the data both ways, as above.
