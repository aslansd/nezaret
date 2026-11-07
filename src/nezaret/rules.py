"""The checks.

Each rule names a way a MONAI pipeline can produce a wrong result *without
raising an error*. A rule fires only on what it can read; anything behind an
unevaluated ``$`` expression is treated as unknown and skipped, never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable

from .config import Unresolved
from .knowledge import (
    GEOMETRY, GRID_CHANGING, GUIDE_ARGUMENT, INTENSITY, REMOVED,
    RESAMPLING_DEFAULT_MODE, SIGNATURE_PARAMS, base_name, is_nearest,
    looks_like_label, monai_components,
)
from .pipeline import Pipeline, Step

SEVERITIES = ("error", "warning", "info")
_RANK = {"error": 0, "warning": 1, "info": 2}


@dataclass(frozen=True)
class Rule:
    id: str
    slug: str
    severity: str
    title: str
    why: str
    fix: str


@dataclass
class Finding:
    rule: str
    slug: str
    severity: str
    file: str
    location: str
    message: str
    pipeline: str = ""

    def to_dict(self):
        return asdict(self)


RULES: dict[str, Rule] = {}


def _rule(id, slug, severity, title, why, fix):
    RULES[id] = Rule(id, slug, severity, title, why, fix)


_rule("NZ101", "label-interpolation", "error",
      "A label map is resampled with a non-nearest interpolation",
      "Bilinear, trilinear or area interpolation averages neighbouring class "
      "indices. A boundary between class 1 and class 3 becomes a band of 2s, "
      "and a binary mask becomes fractional. Training still runs and the loss "
      "still falls. Several MONAI transforms default to a non-nearest mode "
      "(Spacingd: bilinear, Resized and Zoomd: area), so leaving out `mode` is "
      "enough to cause it.",
      "Give `mode` one entry per key, with \"nearest\" for every label key, "
      "e.g. keys [\"image\", \"label\"] with mode [\"bilinear\", \"nearest\"].")

_rule("NZ102", "label-intensity", "error",
      "An intensity transform is applied to a label map",
      "Intensity transforms rewrite voxel values. On a label map that turns "
      "class indices into other numbers (scaled, shifted, noised or "
      "normalised), and nothing downstream checks that labels are still "
      "integers in range.",
      "Restrict the transform's `keys` to the image keys.")

_rule("NZ103", "geometry-split", "error",
      "A spatial transform moves some loaded arrays but not the others",
      "Image and label are loaded together so they stay voxel-aligned. A "
      "transform that changes orientation, spacing, crop, pad, flip or "
      "rotation on one of them and not the other breaks that alignment. If the "
      "shapes still match, nothing raises: the network is trained or scored "
      "against a label that belongs somewhere else.",
      "Apply the transform to every key that must stay aligned, or remove the "
      "unused key with DeleteItemsd before this point if it really is unused.")

_rule("NZ104", "unknown-target", "error",
      "A `_target_` does not exist in the MONAI version nezaret checks against",
      "MONAI resolves a bare `_target_` by searching its own modules. A name "
      "that is not there was removed, renamed or misspelt. Removed transforms "
      "are the common case after an upgrade (AddChanneld went in MONAI 1.3).",
      "Use the replacement named in the message, or a fully qualified "
      "`_target_` if the class lives in your own code.")

_rule("NZ105", "discrete-interpolation", "error",
      "A discrete prediction is inverted with interpolation",
      "After AsDiscreted (argmax, threshold or one-hot) a prediction holds class "
      "indices. Invertd with nearest_interp=false resamples it back to the "
      "original grid with the forward transform's interpolation, which "
      "averages those indices into values that are not classes.",
      "Invert before AsDiscreted (MONAI's own bundles do this), or set "
      "nearest_interp to true for that key.")

_rule("NZ106", "saved-not-inverted", "warning",
      "Predictions are saved in the preprocessed space, not the original one",
      "The preprocessing changes spacing, orientation or extent. A prediction "
      "saved without Invertd (or SaveImaged resample=true) is written on the "
      "resampled grid. Opened next to the original scan it is misaligned, and "
      "metrics computed against original-space labels are computed on "
      "mismatched voxels.",
      "Add Invertd for the prediction key before SaveImaged, pointing "
      "`transform` at the preprocessing and `orig_keys` at the image key.")

_rule("NZ107", "label-guided-eval", "error",
      "An evaluation pipeline uses the ground truth to decide where to look",
      "CropForegroundd(source_key=label) and the label-guided random crops "
      "choose the region from the ground truth. In validation or inference "
      "that hands the model the answer's location: reported scores are higher "
      "than the model would get on data without labels.",
      "Derive the crop from the image (source_key=\"image\") or evaluate on the "
      "whole volume with a sliding-window inferer.")

_rule("NZ108", "random-in-eval", "warning",
      "An evaluation pipeline contains a random transform",
      "A random transform in validation or inference makes each evaluation "
      "different. Scores move between runs for reasons unrelated to the "
      "model, and a checkpoint chosen by best validation score is partly "
      "chosen by luck.",
      "Move the transform to the training pipeline. If it is deliberate "
      "test-time augmentation, set a seed and ignore this rule explicitly.")

_rule("NZ109", "train-eval-mismatch", "warning",
      "Training and evaluation preprocess images differently",
      "Orientation, spacing, size and intensity scaling are part of what the "
      "model learned. If evaluation or inference uses different values, the "
      "model sees inputs from a distribution it was never trained on. It still "
      "returns predictions, usually worse ones, and nothing reports why.",
      "Define the deterministic preprocessing once and reference it from both "
      "pipelines (`%train#deterministic_transforms` or `@...`).")

_rule("NZ110", "missing-channel-first", "info",
      "A spatial transform runs before the array has a channel dimension",
      "MONAI's spatial transforms assume channel-first arrays. A 3D volume "
      "loaded without ensure_channel_first is read with its first spatial axis "
      "as the channel axis, so a 3D resample becomes a 2D one with the wrong "
      "spacing. nezaret cannot see the data list, which may supply several "
      "files per key (stacked as channels), so this is reported as info.",
      "Set ensure_channel_first=true on LoadImaged or add EnsureChannelFirstd "
      "before the first spatial transform.")

_rule("NZ111", "no-determinism", "info",
      "Random training transforms with no seed anywhere in the config",
      "Without a seed two runs of the same config draw different "
      "augmentations, so a difference between runs cannot be told apart from "
      "a difference between configs.",
      "Call monai.utils.set_determinism(seed=...) in the bundle's `initialize` "
      "section.")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def per_key(value, keys: list[str], default=None):
    """Expand a MONAI per-key argument. None if it cannot be read."""
    if value is None:
        return {k: default for k in keys}
    if isinstance(value, Unresolved):
        return None
    if isinstance(value, (list, tuple)):
        if any(isinstance(v, Unresolved) for v in value):
            return None
        if len(value) == len(keys):
            return dict(zip(keys, value))
        if len(value) == 1:
            return {k: value[0] for k in keys}
        return None
    return {k: value for k in keys}


def infer_label_keys(p: Pipeline, user: Iterable[str] | None) -> set[str]:
    if user:
        return set(user)
    keys: set[str] = set()
    for s in p.steps:
        for k in s.keys or []:
            if looks_like_label(k):
                keys.add(k)
        if s.name in RESAMPLING_DEFAULT_MODE and s.keys and len(s.keys) > 1:
            # a per-key mode list that mixes nearest with non-nearest is the
            # author saying which keys are label maps; a single "nearest" for
            # every key says nothing (some bundles resample images that way)
            modes = per_key(s.arg("mode"), s.keys, RESAMPLING_DEFAULT_MODE[s.name])
            if modes and None not in modes.values():
                near = {k for k, m in modes.items() if is_nearest(m)}
                if near and len(near) < len(modes):
                    keys.update(near)
        g = GUIDE_ARGUMENT.get(s.name)
        if g and isinstance(s.arg(g), str) and not isinstance(s.arg(g), Unresolved):
            if s.name != "CropForegroundd" or looks_like_label(s.arg(g)):
                keys.add(s.arg(g))
    return keys


def _f(rule_id, step_or_p, message, p: Pipeline, severity: str | None = None) -> Finding:
    r = RULES[rule_id]
    if isinstance(step_or_p, Step):
        file, loc = step_or_p.file, step_or_p.location
    else:
        file, loc = step_or_p.file, step_or_p.location
    return Finding(r.id, r.slug, severity or r.severity, file, loc, message, p.describe())


def _fmt(v):
    if isinstance(v, str):
        return f"'{v}'"
    return str(v)


# ---------------------------------------------------------------------------
# per-pipeline rules
# ---------------------------------------------------------------------------

def nz101(p: Pipeline):
    if p.role == "post":
        return
    for s in p.steps:
        default = RESAMPLING_DEFAULT_MODE.get(s.name)
        if default is None or not s.keys:
            continue
        modes = per_key(s.arg("mode"), s.keys, default)
        if modes is None:
            continue
        for k in s.keys:
            if k in p.label_keys and not is_nearest(modes[k]):
                if s.location.startswith("["):
                    how = ""  # a live object always carries a mode; origin unknown
                else:
                    how = " (MONAI's default)" if s.arg("mode") is None else " (set explicitly)"
                yield _f("NZ101", s, f"{s.name} resamples label key '{k}' with mode "
                         f"{_fmt(modes[k])}{how}; label maps need 'nearest'.", p)


_MINMAX = {"ScaleIntensityd", "ScaleIntensityRanged"}


def nz102(p: Pipeline):
    if p.role == "post":
        return
    for s in p.steps:
        if s.name in INTENSITY and s.keys:
            hit = [k for k in s.keys if k in p.label_keys]
            if not hit:
                continue
            keys = ", ".join(repr(k) for k in hit)
            if s.name in _MINMAX:
                # min-max scaling a 0/255 binary mask to 0/1 is a known idiom;
                # on a multi-class map it destroys the class indices
                yield _f("NZ102", s, f"{s.name} rescales label key(s) {keys}. That is only "
                         f"safe for a binary 0/255 mask; on a multi-class label map it "
                         f"changes the class indices.", p, severity="warning")
            else:
                yield _f("NZ102", s, f"{s.name} changes voxel values of label key(s) {keys}.", p)


# Centred crops and pads keep arrays centre-aligned even at different sizes
# (HoVerNet crops its labels smaller than its input on purpose).
_CENTRED = {"CenterSpatialCropd", "CenterScaleCropd", "SpatialPadd", "BorderPadd",
            "DivisiblePadd", "ResizeWithPadOrCropd"}


def nz103(p: Pipeline):
    if p.role == "post":
        return
    live: list[str] = []
    stopped = False
    touched_ever: set[str] = set()
    centred_ever: set[str] = set()
    for s in p.steps:
        if s.name in GEOMETRY and s.keys:
            touched_ever.update(s.keys)
            if s.name in _CENTRED:
                centred_ever.update(s.keys)
    partial: list[tuple[Step, list[str], list[str]]] = []
    untouched_first: Step | None = None
    for s in p.steps:
        if stopped:
            break
        if s.name == "LoadImaged":
            if s.keys is None:
                stopped = True
            else:
                live.extend(k for k in s.keys if k not in live)
            continue
        if not s.monai or s.name == "?":
            # a custom step may create, merge or drop keys; stop tracking
            stopped = True
            continue
        if s.name == "DeleteItemsd" and s.keys:
            live = [k for k in live if k not in s.keys]
            continue
        if s.name == "SelectItemsd" and s.keys:
            live = [k for k in live if k in s.keys]
            continue
        if s.name in ("CopyItemsd", "ConcatItemsd"):
            names = s.arg("names") if s.name == "CopyItemsd" else s.arg("name")
            if isinstance(names, str) and not isinstance(names, Unresolved):
                names = [names]
            if isinstance(names, list) and all(isinstance(n, str) for n in names):
                live.extend(n for n in names if n not in live)
            else:
                stopped = True
            continue
        if s.name not in GEOMETRY or s.keys is None or s.arg("allow_missing_keys") is True:
            continue
        guide = s.arg(GUIDE_ARGUMENT.get(s.name, ""), None)
        missing = [k for k in live if k not in s.keys and k != guide]
        touched = [k for k in live if k in s.keys]
        if isinstance(guide, str) and guide in live and guide not in s.keys:
            live.remove(guide)  # a crop guide left behind on purpose
        if not touched or not missing:
            continue
        if s.name in _CENTRED:
            missing = [k for k in missing if k not in centred_ever]
        never = [k for k in missing if k not in touched_ever]
        moved = [k for k in missing if k in touched_ever]
        if moved:
            partial.append((s, touched, moved))
        if never and untouched_first is None:
            untouched_first = s
            untouched_keys, untouched_by = never, touched
    for s, touched, moved in partial:
        yield _f("NZ103", s, f"{s.name} transforms {', '.join(repr(k) for k in touched)} "
                 f"but not {', '.join(repr(k) for k in moved)}, which were loaded together "
                 f"and stop being aligned here.", p)
    if untouched_first is not None and p.role != "eval":
        # in evaluation, leaving the label in original space is a deliberate
        # pattern (score after Invertd); in training it usually is not
        yield _f("NZ103", untouched_first,
                 f"{', '.join(repr(k) for k in untouched_keys)} never goes through a spatial "
                 f"transform while {', '.join(repr(k) for k in untouched_by)} does, starting with "
                 f"{untouched_first.name} here. Fine only if they are not meant to share a grid.",
                 p, severity="warning")


def nz104(p: Pipeline):
    """Removed/unknown transforms inside a pipeline (file-wide check lives in lint)."""
    return iter(())


def nz105(p: Pipeline):
    discrete: set[str] = set()
    for s in p.steps:
        if s.name == "AsDiscreted" and s.keys:
            for argname in ("argmax", "threshold", "to_onehot", "rounding"):
                vals = per_key(s.arg(argname), s.keys, None)
                if vals is None:
                    discrete.update(s.keys)  # cannot read: assume it discretises
                    break
                for k, v in vals.items():
                    if v not in (None, False, 0) and not (isinstance(v, list) and not any(v)):
                        discrete.add(k)
        if s.name == "Invertd" and s.keys:
            ni = per_key(s.arg("nearest_interp"), s.keys, True)
            if ni is None:
                continue
            for k in s.keys:
                if k in discrete and ni[k] is False:
                    yield _f("NZ105", s, f"Invertd interpolates '{k}' (nearest_interp=false) "
                             f"after AsDiscreted already made it a class map.", p)


def nz106(p: Pipeline, context: "Context"):
    if p.role != "post":
        return
    loaded = set(p.loaded_keys)
    inverted: set[str] = set()
    for s in p.steps:
        if s.name == "Invertd" and s.keys:
            inverted.update(s.keys)
        if s.name != "SaveImaged" or not s.keys:
            continue
        if s.arg("resample") is True:
            continue
        for k in s.keys:
            if k in loaded or k in inverted:
                continue
            pre = context.grid_changing_preprocessing(p)
            if pre is None:
                continue
            moved = {key for st in pre.steps if st.name in GRID_CHANGING for key in (st.keys or [])}
            if k in pre.loaded_keys and k not in moved:
                continue  # loaded and never moved: already in original space
            yield _f("NZ106", s, f"SaveImaged writes '{k}' without Invertd, but the "
                     f"preprocessing at {pre.file}:{pre.location} changes the grid "
                     f"({context.grid_changers(pre)}).", p)


def nz107(p: Pipeline):
    if p.role != "eval":
        return
    for s in p.steps:
        arg = GUIDE_ARGUMENT.get(s.name)
        if not arg:
            continue
        guide = s.arg(arg)
        if isinstance(guide, str) and not isinstance(guide, Unresolved) and guide in p.label_keys:
            yield _f("NZ107", s, f"{s.name} in an evaluation pipeline takes its region from "
                     f"ground-truth key '{guide}' ({arg}).", p)


def nz108(p: Pipeline):
    if p.role != "eval":
        return
    for s in p.steps:
        if not s.is_random:
            continue
        arg = GUIDE_ARGUMENT.get(s.name)
        if arg and s.arg(arg) in p.label_keys:
            continue  # already NZ107
        yield _f("NZ108", s, f"{s.name} is random (prob={s.arg('prob', 'default')}) "
                 f"inside an evaluation pipeline.", p)


def nz110(p: Pipeline):
    if p.role == "post" or not p.has_loader:
        return
    channel_first: set[str] = set()
    trusted_all = False
    warned: set[str] = set()
    for s in p.steps:
        if s.name == "LoadImaged":
            if s.arg("ensure_channel_first") is True or isinstance(s.arg("ensure_channel_first"), Unresolved):
                channel_first.update(s.keys or [])
                if s.keys is None:
                    trusted_all = True
            continue
        if s.name in ("EnsureChannelFirstd", "AsChannelLastd", "SplitDimd", "Transposed") \
                or "channel" in s.name.lower() or not s.monai or s.name in ("Lambdad", "?"):
            if s.keys is None:
                trusted_all = True
            else:
                channel_first.update(s.keys)
            continue
        if trusted_all or s.name not in GEOMETRY or not s.keys:
            continue
        for k in s.keys:
            if k not in channel_first and k not in warned and k in p.loaded_keys:
                warned.add(k)
                yield _f("NZ110", s, f"{s.name} runs on '{k}' before any channel-first step. "
                         f"Fine if the data list gives several files per key (LoadImaged "
                         f"stacks them as channels) or the files are already channel-first.", p)


PIPELINE_RULES: list[Callable] = [nz101, nz102, nz103, nz105, nz107, nz108, nz110]


# ---------------------------------------------------------------------------
# cross-pipeline rules
# ---------------------------------------------------------------------------

@dataclass
class Context:
    pipelines: list[Pipeline] = field(default_factory=list)

    def eval_pre(self, file: str | None = None):
        return [q for q in self.pipelines if q.role == "eval" and q.has_loader
                and (file is None or q.file == file)]

    def grid_changers(self, p: Pipeline) -> str:
        names = []
        for s in p.steps:
            if s.name in GRID_CHANGING and s.name not in names:
                names.append(s.name)
        return ", ".join(names)

    def grid_changing_preprocessing(self, post: Pipeline) -> Pipeline | None:
        for q in self.eval_pre(post.file):
            if any(s.name in GRID_CHANGING for s in q.steps):
                return q
        return None


def _norm(v):
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    return v


def _num_equal(a, b) -> bool:
    a, b = _norm(a), _norm(b)
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)), abs(float(b)))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_num_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, str) and isinstance(b, str):
        return a.upper() == b.upper() if len(a) <= 4 else a == b
    return a == b


def _has_unresolved(v) -> bool:
    if isinstance(v, Unresolved):
        return True
    if isinstance(v, (list, tuple)):
        return any(_has_unresolved(x) for x in v)
    if isinstance(v, dict):
        return any(_has_unresolved(x) for x in v.values())
    return False


def signature(p: Pipeline) -> dict[str, dict | None]:
    """First deterministic occurrence of each compared transform on an image key.

    A value of None for a transform means "present but unreadable".
    """
    sig: dict[str, dict | None] = {}
    for s in p.steps:
        if s.name == "LoadImaged" and s.arg("as_closest_canonical") in (True, "true", "True") \
                and "Orientationd" not in sig:
            # nibabel's closest canonical orientation is RAS+
            sig["Orientationd"] = {"axcodes": "RAS (LoadImaged as_closest_canonical)"}
            continue
        if s.name not in SIGNATURE_PARAMS or s.name in sig or s.branch:
            continue
        keys = s.keys
        if keys is None:
            continue
        if not [k for k in keys if k not in p.label_keys]:
            continue
        params = {}
        for name, default in SIGNATURE_PARAMS[s.name].items():
            v = s.arg(name, default)
            if _has_unresolved(v):
                params = None
                break
            params[name] = v
        sig[s.name] = params
    return sig


def compare_signatures(train: dict, ev: dict) -> list[str]:
    diffs = []
    for name in sorted(set(train) | set(ev)):
        a, b = train.get(name, "absent"), ev.get(name, "absent")
        if a is None or b is None:
            continue
        if a == "absent" or b == "absent":
            where = "training" if b == "absent" else "evaluation"
            note = " (fine only if the training data was already resampled offline to the same grid)" \
                if name in ("Spacingd", "Resized") and where == "evaluation" else ""
            diffs.append(f"{name} only in {where}{note}")
            continue
        for param in a:
            if param == "axcodes" and isinstance(a[param], str) and isinstance(b[param], str) \
                    and a[param][:3].upper() == b[param][:3].upper():
                continue
            if not _num_equal(a[param], b[param]):
                diffs.append(f"{name}.{param}: train {a[param]!r} vs eval {b[param]!r}")
    return diffs


def nz109(ctx: Context):
    train = [q for q in ctx.pipelines if q.role == "train" and q.has_loader]
    evals = [q for q in ctx.pipelines if q.role == "eval" and q.has_loader]
    if not train or not evals:
        return
    tsigs = [(t, signature(t)) for t in train]
    for e in evals:
        esig = signature(e)
        best = None
        for t, ts in tsigs:
            d = compare_signatures(ts, esig)
            if best is None or len(d) < len(best[1]):
                best = (t, d)
        if best and best[1]:
            t, diffs = best
            first = e.steps[0]
            yield Finding("NZ109", RULES["NZ109"].slug, RULES["NZ109"].severity,
                          e.file, e.location,
                          f"compared with {t.file}:{t.location}: " + "; ".join(diffs),
                          e.describe())


def nz111(ctx: Context, files):
    for cf in files:
        rand = [q for q in ctx.pipelines if q.file == cf.name and q.role == "train"
                and any(s.is_random for s in q.steps)]
        if not rand:
            continue
        texts = [cf.text] + [fb.text for fb in cf.fallbacks]
        if any("set_determinism" in t or '"seed"' in t for t in texts):
            continue
        p = rand[0]
        yield Finding("NZ111", RULES["NZ111"].slug, RULES["NZ111"].severity,
                      cf.name, p.location,
                      "random transforms in training and no set_determinism or seed "
                      "in this file or the config it overrides.", p.describe())


def nz104_file(cf, targets):
    version, names = monai_components()
    for loc, target in targets:
        if "." in target:
            continue  # qualified: belongs to torch, ignite, scripts.*, or a full monai path
        if target in names:
            continue
        repl = REMOVED.get(target)
        hint = f" It was removed or renamed in MONAI; use {repl}." if repl else \
            " Check the spelling, or give the fully qualified name if it is your own class."
        yield Finding("NZ104", RULES["NZ104"].slug, RULES["NZ104"].severity, cf.name, loc,
                      f"'{target}' is not a MONAI {version} component.{hint}", "")


def rank(severity: str) -> int:
    return _RANK[severity]
