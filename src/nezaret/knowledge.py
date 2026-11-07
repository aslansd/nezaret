"""What nezaret knows about MONAI transforms.

Every default below was read from ``inspect.signature`` of MONAI 1.6.1, not
from documentation. If MONAI changes a default, this file is the one place to
update, and ``tests/test_live_monai.py`` checks these values against whatever
MONAI is installed.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources

# Transforms that resample voxel values onto a new grid, with the default
# interpolation mode MONAI 1.6.1 uses when ``mode`` is not given.
RESAMPLING_DEFAULT_MODE = {
    "Spacingd": "bilinear",
    "Resized": "area",
    "Zoomd": "area",
    "Rotated": "bilinear",
    "Affined": "bilinear",
    "SpatialResampled": "bilinear",
    "ResampleToMatchd": "bilinear",
    "GridDistortiond": "bilinear",
    "RandAffined": "bilinear",
    "Rand2DElasticd": "bilinear",
    "Rand3DElasticd": "bilinear",
    "RandRotated": "bilinear",
    "RandZoomd": "area",
    "RandGridDistortiond": "bilinear",
}

# Transforms that change the spatial grid (shape, affine or both). If two
# arrays are meant to stay voxel-aligned, every one of these must touch both.
GEOMETRY = set(RESAMPLING_DEFAULT_MODE) | {
    "Orientationd", "Rotate90d", "Flipd", "Transposed",
    "RandRotate90d", "RandFlipd", "RandAxisFlipd",
    "CropForegroundd", "SpatialCropd", "CenterSpatialCropd", "CenterScaleCropd",
    "RandSpatialCropd", "RandScaleCropd", "RandSpatialCropSamplesd",
    "RandCropByPosNegLabeld", "RandCropByLabelClassesd", "RandWeightedCropd",
    "SpatialPadd", "BorderPadd", "DivisiblePadd", "ResizeWithPadOrCropd",
}

# Geometry transforms that change voxel spacing, orientation or extent so that
# a saved prediction no longer lines up with the original image.
GRID_CHANGING = set(RESAMPLING_DEFAULT_MODE) | {
    "Orientationd", "CropForegroundd", "SpatialCropd", "CenterSpatialCropd",
    "CenterScaleCropd", "SpatialPadd", "BorderPadd", "DivisiblePadd",
    "ResizeWithPadOrCropd", "Transposed",
}

# Transforms that rewrite voxel *values*. On an image that is the point; on a
# label map it silently turns class indices into other numbers.
INTENSITY = {
    "ScaleIntensityd", "ScaleIntensityRanged", "ScaleIntensityRangePercentilesd",
    "NormalizeIntensityd", "ThresholdIntensityd", "ShiftIntensityd",
    "StdShiftIntensityd", "ClipIntensityPercentilesd", "HistogramNormalized",
    "GaussianSmoothd", "GaussianSharpend", "MedianSmoothd", "SavitzkyGolaySmoothd",
    "AdjustContrastd",
    "RandGaussianNoised", "RandRicianNoised", "RandShiftIntensityd",
    "RandStdShiftIntensityd", "RandScaleIntensityd", "RandScaleIntensityFixedMeand",
    "RandBiasFieldd", "RandAdjustContrastd", "RandGaussianSmoothd",
    "RandGaussianSharpend", "RandHistogramShiftd", "RandGibbsNoised",
    "RandKSpaceSpikeNoised", "RandCoarseDropoutd", "RandCoarseShuffled",
    "RandSimulateLowResolutiond",
}

# Transforms that use one key to decide *where* to look in the others, and the
# argument that names that key.
GUIDE_ARGUMENT = {
    "CropForegroundd": "source_key",
    "RandCropByPosNegLabeld": "label_key",
    "RandCropByLabelClassesd": "label_key",
    "RandWeightedCropd": "w_key",
}

CONTAINERS = {"Compose", "OneOf", "SomeOf", "RandomOrder"}

# Deterministic preprocessing whose parameters must agree between training and
# evaluation, with the MONAI 1.6.1 defaults for the parameters compared.
SIGNATURE_PARAMS = {
    "Orientationd": {"axcodes": None},
    "Spacingd": {"pixdim": None},
    "Resized": {"spatial_size": None},
    "ScaleIntensityRanged": {"a_min": None, "a_max": None, "b_min": None,
                             "b_max": None, "clip": False},
    "ScaleIntensityRangePercentilesd": {"lower": None, "upper": None,
                                        "b_min": None, "b_max": None,
                                        "clip": False, "relative": False,
                                        "channel_wise": False},
    "NormalizeIntensityd": {"subtrahend": None, "divisor": None,
                            "nonzero": False, "channel_wise": False},
    "ScaleIntensityd": {"minv": 0.0, "maxv": 1.0, "factor": None,
                        "channel_wise": False},
}

# Transforms that existed in older MONAI and are gone in 1.6.1, with the
# replacement that does the same job.
REMOVED = {
    "AddChanneld": 'EnsureChannelFirstd with channel_dim="no_channel"',
    "AddChannel": 'EnsureChannelFirst with channel_dim="no_channel"',
    "AsChannelFirstd": "EnsureChannelFirstd with channel_dim=-1",
    "AsChannelFirst": "EnsureChannelFirst with channel_dim=-1",
    "LoadNiftid": "LoadImaged",
    "LoadNifti": "LoadImage",
    "LoadNumpyd": "LoadImaged with reader=\"NumpyReader\"",
    "LoadPNGd": "LoadImaged",
    "RandDeformGridd": "RandGridDistortiond",
    "IgniteMetric": "IgniteMetricHandler",
}

NEAREST_MODES = {"nearest", "nearest-exact", "0"}

LABEL_KEY_RE = re.compile(
    r"(^|[_\-.])(label|labels|lbl|seg|segs|segmentation|mask|masks|gt|"
    r"groundtruth|ground_truth|annotation|annotations|annot|target|targets)"
    r"([_\-.\d]|$)",
    re.IGNORECASE,
)


def base_name(target: str) -> str:
    """``monai.transforms.Spacingd`` -> ``Spacingd``."""
    return target.rsplit(".", 1)[-1]


def is_monai_target(target: str) -> bool:
    return "." not in target or target.startswith("monai.")


def is_nearest(mode) -> bool:
    if isinstance(mode, bool):
        return False
    if isinstance(mode, (int, float)):
        return int(mode) == 0
    text = str(mode).strip().lower()
    # enum reprs such as "GridSampleMode.NEAREST" or "<InterpolateMode.NEAREST: 'nearest'>"
    text = text.rsplit(".", 1)[-1].strip("<>'\" ")
    if ":" in text:
        text = text.split(":", 1)[1].strip(" '\"")
    return text in NEAREST_MODES


def looks_like_label(key: str) -> bool:
    return bool(LABEL_KEY_RE.search(str(key)))


@lru_cache(maxsize=1)
def monai_components() -> tuple[str, frozenset]:
    """Names MONAI's bundle parser can resolve from a bare ``_target_``."""
    data = json.loads(
        resources.files("nezaret").joinpath("data/monai_components.json").read_text()
    )
    return data["monai_version"], frozenset(data["names"])


MONAI_REFERENCE_VERSION = monai_components()[0]
