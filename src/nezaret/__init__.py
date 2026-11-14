"""nezaret: find MONAI pipeline faults that do not raise errors.

    >>> import nezaret
    >>> report = nezaret.lint("my_bundle/")
    >>> print(report)

or, on a pipeline built in Python:

    >>> nezaret.lint_compose(train_transforms, role="train")

and on the data a bundle trains on:

    >>> nezaret.check_data("dataset.json", base_dir="/data/Task09_Spleen")

and on the predictions it produced:

    >>> nezaret.check_results("labelsTs/", {"unet": "preds/unet/", "swin": "preds/swin/"})
"""

from .__version__ import __version__
from .lint import Report, lint, lint_compose, lint_config
from .rules import RULES, Finding, Rule
from .dataset import check_data  # also registers the NZ2xx rules
from .result import check_results  # also registers the NZ3xx rules

__all__ = ["__version__", "lint", "lint_config", "lint_compose", "check_data", "check_results",
           "Report",
           "Finding", "Rule", "RULES"]
