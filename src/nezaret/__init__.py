"""nezaret: find MONAI pipeline faults that do not raise errors.

    >>> import nezaret
    >>> report = nezaret.lint("my_bundle/")
    >>> print(report)

or, on a pipeline built in Python:

    >>> nezaret.lint_compose(train_transforms, role="train")
"""

from .__version__ import __version__
from .lint import Report, lint, lint_compose, lint_config
from .rules import RULES, Finding, Rule

__all__ = ["__version__", "lint", "lint_config", "lint_compose", "Report",
           "Finding", "Rule", "RULES"]
