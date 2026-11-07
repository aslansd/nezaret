"""A transform pipeline as nezaret sees it: an ordered list of steps.

Two sources produce the same model, so the rules do not care where a pipeline
came from:

* :func:`pipelines_from_config` reads a MONAI Bundle config file.
* :func:`pipeline_from_compose` reads a live ``monai.transforms.Compose``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .config import ConfigFile, Unresolved
from .knowledge import CONTAINERS, base_name, is_monai_target

ROLES = ("train", "eval", "post", "unknown")

_EVAL_TOKENS = {"val", "valid", "validate", "validation", "validator", "eval",
                "evaluate", "evaluation", "evaluator", "infer", "inference",
                "inferer", "test", "testing", "predict", "prediction"}
_POST_TOKENS = {"postprocessing", "post", "postprocess", "post_transforms",
                "postprocessing_transforms"}
_TRAIN_TOKENS = {"train", "training", "trainer"}
_TOKEN_SPLIT = re.compile(r"[#_\-.\s/]+")


@dataclass
class Step:
    name: str                      # base name, e.g. "Spacingd"
    target: str                    # as written, e.g. "monai.transforms.Spacingd"
    args: dict[str, Any]
    location: str                  # "#"-path inside the file, or "[3]" for live
    file: str
    branch: str = ""               # "OneOf" etc. if inside a container
    monai: bool = True

    def arg(self, name, default=None):
        return self.args.get(name, default)

    @property
    def keys(self) -> list[str] | None:
        """Keys as a list, or None if they could not be resolved."""
        raw = self.args.get("keys")
        if raw is None:
            return None
        if isinstance(raw, Unresolved):
            return None
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, (list, tuple)):
            if any(isinstance(k, Unresolved) or not isinstance(k, str) for k in raw):
                return None
            return list(raw)
        return None

    @property
    def is_random(self) -> bool:
        if not self.name.startswith("Rand"):
            return False
        prob = self.args.get("prob")
        try:
            return prob is None or float(prob) > 0
        except (TypeError, ValueError):
            return True


@dataclass
class Pipeline:
    steps: list[Step]
    file: str
    location: str
    role: str = "unknown"
    label_keys: set[str] = field(default_factory=set)

    @property
    def loaded_keys(self) -> list[str]:
        out: list[str] = []
        for s in self.steps:
            if s.name == "LoadImaged" and s.keys:
                out.extend(k for k in s.keys if k not in out)
        return out

    @property
    def has_loader(self) -> bool:
        return any(s.name == "LoadImaged" for s in self.steps)

    def describe(self) -> str:
        return f"{self.file}:{self.location} ({self.role})"


# --------------------------------------------------------------------------
# role detection
# --------------------------------------------------------------------------

def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_SPLIT.split(text) if t}


def detect_role(location: str, file_stem: str, steps: list[Step]) -> str:
    toks = _tokens(location)
    names = {s.name for s in steps}
    if toks & _POST_TOKENS or "postprocessing" in location.lower():
        return "post"
    if "LoadImaged" not in names and names & {"Invertd", "AsDiscreted", "Activationsd", "SaveImaged"}:
        return "post"
    if toks & _EVAL_TOKENS:
        return "eval"
    if toks & _TRAIN_TOKENS:
        return "train"
    ftoks = _tokens(file_stem)
    if ftoks & _EVAL_TOKENS:
        return "eval"
    if ftoks & _TRAIN_TOKENS:
        return "train"
    return "unknown"


# --------------------------------------------------------------------------
# from a config file
# --------------------------------------------------------------------------

def _target_of(node) -> str | None:
    if isinstance(node, dict):
        t = node.get("_target_")
        if isinstance(t, str) and not t.startswith("$"):
            return t
    return None


def _steps_from_items(items, cf_name: str, branch: str, seen: set, depth=0) -> list[Step] | None:
    steps: list[Step] = []
    for item, loc, owner in items:
        node, loc2, owner2 = owner.resolve(item, loc)
        if isinstance(node, dict) and loc2 != loc:
            loc = loc2
            owner = owner2
        target = _target_of(node)
        if target is None:
            # an expression or unknown object inside the pipeline
            steps.append(Step(name="?", target=str(node)[:60], args={}, location=loc,
                              file=owner.name, branch=branch, monai=False))
            continue
        name = base_name(target)
        if name in CONTAINERS and "transforms" in node and depth < 10:
            inner = owner.resolve_list(node["transforms"], f"{loc}#transforms")
            if inner is None:
                steps.append(Step(name="?", target=target, args={}, location=loc,
                                  file=owner.name, branch=branch, monai=False))
                continue
            sub = _steps_from_items(inner, cf_name, name if name != "Compose" else branch,
                                    seen, depth + 1)
            steps.extend(sub or [])
            continue
        args = {k: owner.resolve_deep(v) for k, v in node.items() if k != "_target_"}
        steps.append(Step(name=name, target=target, args=args, location=loc,
                          file=owner.name, branch=branch, monai=is_monai_target(target)))
    return steps


def _walk(node, path=""):
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _walk(v, f"{path}#{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}#{i}" if path else str(i))


def all_targets(cf: ConfigFile):
    """Every ``(location, target)`` written in the file."""
    for path, node in _walk(cf.data):
        t = _target_of(node)
        if t is not None:
            yield path, t


def pipelines_from_config(cf: ConfigFile) -> list[Pipeline]:
    out: list[Pipeline] = []
    seen_locs: set[tuple] = set()
    for path, node in _walk(cf.data):
        target = _target_of(node)
        if target is None or base_name(target) != "Compose" or "transforms" not in node:
            continue
        items = cf.resolve_list(node["transforms"], f"{path}#transforms")
        if items is None:
            continue
        steps = _steps_from_items(items, cf.name, "", set())
        if not steps:
            continue
        sig = (path, tuple(s.location for s in steps))
        if sig in seen_locs:
            continue
        seen_locs.add(sig)
        p = Pipeline(steps=steps, file=cf.name, location=path)
        p.role = detect_role(path, cf.stem, steps)
        out.append(p)
    # drop pipelines that are wholly nested inside another one
    nested = set()
    for a in out:
        for b in out:
            if a is not b and b.location.startswith(a.location + "#"):
                nested.add(id(b))
    return [p for p in out if id(p) not in nested]


# --------------------------------------------------------------------------
# from a live Compose
# --------------------------------------------------------------------------

def _plain(v):
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(v, tuple):
        return [_plain(x) for x in v]
    if isinstance(v, list):
        return [_plain(x) for x in v]
    if hasattr(v, "value") and type(v).__module__.startswith("monai"):
        return v.value  # StrEnum members such as GridSampleMode.BILINEAR
    return v


_NESTED = {
    "Spacingd": ("spacing_transform", ["pixdim"]),
    "Orientationd": ("ornt_transform", ["axcodes"]),
    "ScaleIntensityRanged": ("scaler", ["a_min", "a_max", "b_min", "b_max", "clip"]),
    "ScaleIntensityRangePercentilesd": ("scaler", ["lower", "upper", "b_min", "b_max",
                                                   "clip", "relative", "channel_wise"]),
    "NormalizeIntensityd": ("normalizer", ["subtrahend", "divisor", "nonzero", "channel_wise"]),
    "ScaleIntensityd": ("scaler", ["minv", "maxv", "factor", "channel_wise"]),
    "Resized": ("resizer", ["spatial_size"]),
}

_DIRECT = ("keys", "mode", "prob", "source_key", "label_key", "w_key", "nearest_interp",
           "argmax", "threshold", "to_onehot", "rounding", "orig_keys", "names", "name")


def _args_of(t) -> dict:
    name = type(t).__name__
    args: dict[str, Any] = {}
    attrs = vars(t) if hasattr(t, "__dict__") else {}
    for a in _DIRECT:
        if a in attrs:
            args[a] = _plain(attrs[a])
    if name in _NESTED:
        holder, params = _NESTED[name]
        sub = getattr(t, holder, None)
        if sub is not None:
            for p in params:
                if hasattr(sub, p):
                    args[p] = _plain(getattr(sub, p))
    if name == "LoadImaged":
        loader = getattr(t, "_loader", None)
        if loader is not None and hasattr(loader, "ensure_channel_first"):
            args["ensure_channel_first"] = bool(loader.ensure_channel_first)
    if name == "SaveImaged":
        saver = getattr(t, "saver", None)
        mk = getattr(saver, "meta_kwargs", None) or {}
        if "resample" in mk:
            args["resample"] = bool(mk["resample"])
    if name == "Invertd" and "transform" in attrs:
        args["transform"] = attrs["transform"]
    return args


def _flatten_live(transforms, prefix: str, branch: str, steps: list[Step]):
    for i, t in enumerate(transforms):
        loc = f"{prefix}[{i}]"
        cls = type(t)
        name = cls.__name__
        if name in CONTAINERS and hasattr(t, "transforms"):
            _flatten_live(t.transforms, loc, name if name != "Compose" else branch, steps)
            continue
        module = cls.__module__
        steps.append(Step(name=name, target=f"{module}.{name}", args=_args_of(t),
                          location=loc, file="<compose>", branch=branch,
                          monai=module.startswith("monai.")))


def pipeline_from_compose(compose, role: str = "unknown", name: str = "<compose>") -> Pipeline:
    transforms = getattr(compose, "transforms", compose)
    steps: list[Step] = []
    _flatten_live(list(transforms), "", "", steps)
    for s in steps:
        s.file = name
    return Pipeline(steps=steps, file=name, location="<compose>", role=role)
