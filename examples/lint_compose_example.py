"""Lint a pipeline built in Python rather than in a bundle config.

Needs MONAI (pip install nezaret[monai]). Nothing is loaded, trained or
written: nezaret only reads the transform objects.

    python examples/lint_compose_example.py
"""

import monai.transforms as T

import nezaret

keys = ["image", "label"]

train = T.Compose([
    T.LoadImaged(keys=keys, ensure_channel_first=True),
    T.Orientationd(keys=keys, axcodes="RAS"),
    T.Spacingd(keys=keys, pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    T.ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0, b_max=1, clip=True),
    T.RandCropByPosNegLabeld(keys=keys, label_key="label", spatial_size=(96, 96, 96)),
    T.RandZoomd(keys=keys, prob=0.2, min_zoom=0.9, max_zoom=1.1),   # label zoomed with "area"
])

val = T.Compose([
    T.LoadImaged(keys=keys, ensure_channel_first=True),
    T.Orientationd(keys=keys, axcodes="RAS"),
    T.Spacingd(keys=keys, pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    T.ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0, b_max=1, clip=True),
    T.CropForegroundd(keys=keys, source_key="label"),               # crop chosen by the ground truth
])

post = T.Compose([
    T.Activationsd(keys="pred", softmax=True),
    T.AsDiscreted(keys="pred", argmax=True),
    T.Invertd(keys="pred", transform=val, orig_keys="image", nearest_interp=False),  # interpolates classes
])

for name, pipeline, role in [("train", train, "train"), ("val", val, "eval"), ("post", post, "post")]:
    report = nezaret.lint_compose(pipeline, role=role, name=name,
                                  preprocessing=val if role == "post" else None)
    print(report, end="\n\n")
