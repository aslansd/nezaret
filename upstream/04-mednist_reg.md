**Title:** `mednist_reg`: `train.yaml` uses `IgniteMetric`, removed in MONAI 1.4, so training fails to instantiate

**Labels to suggest:** bug

---

**Describe the bug**

[`configs/train.yaml#L178-L185`](https://github.com/Project-MONAI/model-zoo/blob/feb2e8c318e1964de5d3e1c1000474571e03ad50/models/mednist_reg/configs/train.yaml#L178-L185) (bundle 0.0.8):

```yaml
loss_metric:
  metric_handler:
    _target_: IgniteMetric
```

`IgniteMetric` was deprecated in MONAI 1.2 and removed in 1.4 (`@deprecated(since="1.3", removed="1.4", msg_suffix="Use IgniteMetricHandler instead of IgniteMetric.")`). `loss_metric#metric_handler` is used as an evaluator metric (`additional_metrics`, L157). On any MONAI ≥ 1.4, `monai.bundle run training` therefore stops with:

```
ModuleNotFoundError: Cannot locate class or function path: 'IgniteMetric'.
```

The metadata declares `monai_version` 1.1.0, so CI on that version would not catch it.

**To Reproduce**

```python
from monai.bundle import ConfigParser
p = ConfigParser(); p.read_config("configs/train.yaml")
p.get_parsed_content("loss_metric#metric_handler")   # ModuleNotFoundError on MONAI >= 1.4
```

**Suggested fix** (one line)

`_target_: IgniteMetricHandler`. It takes the same arguments, and the same snippet then instantiates on 1.6.1. `IgniteMetricHandler` has existed since 1.3, so the bundle's `monai_version` would need raising to at least 1.3.

I'm happy to open a PR with this change and the version bump.

**Environment**

<!-- paste the output of: python -c 'import monai; monai.config.print_debug_info()' -->

**Additional context**

Found by running the configuration linter [nezaret](https://github.com/aslansd/nezaret) (`nezaret lint models/mednist_reg`) over the model zoo at `feb2e8c`. It checks every `_target_` against the components of the installed MONAI.
