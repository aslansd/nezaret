**File at:** Project-MONAI/**MONAI** (not model-zoo). This one is optional. File it after the prostate issue and link the two.

**Title:** `Zoom`/`RandZoom(d)`: unknown keyword arguments are passed on to padding; ignored up to 1.5.0, intermittent `ValueError` from 1.5.1

**Labels to suggest:** bug

---

**Describe the bug**

`RandZoomd(keys, prob, min=0.9, max=1.1)` is accepted at construction. `min` and `max` are not parameters (the intended names are `min_zoom` and `max_zoom`), so they go into `**kwargs`, which `Zoom` hands to the pad call when `keep_size=True` needs padding.

- MONAI 1.4.0 and 1.5.0: the keys are dropped and nothing happens.
- MONAI 1.5.1, 1.5.2 and 1.6.1: torch padding rejects them, MONAI falls back to `np.pad`, and numpy raises `ValueError: unsupported keyword arguments for mode 'edge': {'min', 'max'}`. This only happens when the sampled zoom is below 1, so the error is intermittent and appears mid-training rather than at construction.

This is how `prostate_mri_anatomy` in the model zoo stopped training on recent MONAI (Project-MONAI/model-zoo#\_\_\_).

**To Reproduce**

```python
import torch, monai.transforms as T
for seed in range(20):
    t = T.RandZoomd(["image"], prob=1.0, min=0.9, max=1.1); t.set_random_state(seed)
    t({"image": torch.rand(1, 32, 32, 16)})   # raises for seeds whose zoom < 1 on MONAI >= 1.5.1
```

**Expected behavior**

A misspelled or unsupported argument is rejected when the transform is constructed, with a message naming it.

**Suggestion**

Check `**kwargs` against the pad function's accepted arguments in `Zoom.__init__` (and in the other transforms that forward `**kwargs` to padding), and raise `TypeError` there.

**Environment**

<!-- paste the output of: python -c 'import monai; monai.config.print_debug_info()' -->

**Additional context**

Found while checking a model-zoo bundle flagged by [nezaret](https://github.com/aslansd/nezaret).
