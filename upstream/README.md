# Upstream reports

Issue drafts for the faults that `nezaret lint` found in the MONAI Model Zoo at commit `feb2e8c` (20 October 2026). Every claim in them was re-checked on 16 November 2026 against MONAI 1.4.0, 1.5.0, 1.5.1, 1.5.2 and 1.6.1, and each code snippet was run as written. See ROADMAP §1 for status.

| # | File | Where to file | Needs a check first |
|---|---|---|---|
| 1 | [01-prostate_mri_anatomy.md](01-prostate_mri_anatomy.md) | Project-MONAI/model-zoo | no |
| 2 | [02-renalStructures_CECT_segmentation.md](02-renalStructures_CECT_segmentation.md) | Project-MONAI/model-zoo | done: check run on Task09, link confirmed dead |
| 3 | [03-wholeBrainSeg_Large_UNEST_segmentation.md](03-wholeBrainSeg_Large_UNEST_segmentation.md) | Project-MONAI/model-zoo | read #534 and discussion #557 first |
| 4 | [04-mednist_reg.md](04-mednist_reg.md) | Project-MONAI/model-zoo | filed: [#802](https://github.com/Project-MONAI/model-zoo/issues/802) |
| 5 | [05-monai-core-zoom-kwargs.md](05-monai-core-zoom-kwargs.md) | Project-MONAI/MONAI | optional; file after #1 and fill in its number |

## How to file

1. **Search for duplicates.** For each bundle, open
   `https://github.com/Project-MONAI/model-zoo/issues?q=is:issue+<bundle name>`
   (open and closed), and search for `IgniteMetric` and `nearest`. If an issue
   already covers a point, comment there with the reproduction instead of
   opening a new one.
2. **Open a new issue** at
   https://github.com/Project-MONAI/model-zoo/issues/new/choose → *Bug report*.
   The drafts follow that template's headings.
   - Paste the **Title** line into the title field.
   - Paste everything below the `---` into the body.
   - Replace the `<!-- … -->` comment with your own
     `python -c 'import monai; monai.config.print_debug_info()'`.
3. **The renal issue is ready.** The orientation check was run on
   `Task09_Spleen/imagesTr/spleen_10.nii.gz` (`FLIPPED x,y`, pasted into the
   draft). The Yandex link was confirmed dead in a browser on 16 November 2026,
   so part 3 stays.
4. **One issue a day is plenty.** File #4 first (short, certain, easy to
   merge), then #1, #3 and #2. File #5 once #1 has a number.
5. **Offer PRs, don't open them unasked.** Each draft offers one. A model-zoo PR
   must also bump `version` in `configs/metadata.json` and add a `changelog`
   line, or CI rejects it.
6. Record each issue URL in ROADMAP §1.

## Where to announce, and when

Announce only after the issues exist, so the post can link to them. Wait about a week or for a maintainer reply, whichever comes first. 0.3.0, which adds `nezaret result`, is already on PyPI, so `pip install nezaret` gives what the post describes.

| Venue | What to post | Why |
|---|---|---|
| **Project-MONAI/MONAI → Discussions** (*Show and tell* if listed, otherwise *General*) | The announcement below | Where MONAI users and maintainers read. Results first, tool second |
| **Hugging Face model pages** (`huggingface.co/MONAI/<bundle>` → *Community* → new discussion), one per affected bundle | Two lines with a link to the GitHub issue | Bundles are now downloaded from Hugging Face, so users of a bundle may never see the GitHub issue |
| Your own blog or Substack, LinkedIn | A longer write-up: what was checked, what was found, how | The record you can point to later, e.g. in applications |

Don't cross-post the full text to every venue, and don't announce before the issues are filed: a maintainer should read about a fault in their bundle in an issue, not in a post.

### Announcement draft (MONAI Discussions)

> **Title:** Four Model Zoo bundles with configuration faults that change results silently
>
> I ran a static check over every bundle config in the Model Zoo (commit `feb2e8c`). Four bundles have faults that alter training or saved predictions, mostly without raising an error. Each is filed as an issue with the config lines, a reproduction and a one-line fix:
>
> - `prostate_mri_anatomy`: the label is interpolated bilinearly in random augmentation and in `Invertd`. `RandZoomd` is given `min`/`max`, which crashes training on MONAI ≥ 1.5.1 (#…)
> - `renalStructures_CECT_segmentation`: training feeds RAS arrays and `inference.json` feeds LPS, and `RandZoomd` uses area interpolation on the label (#…)
> - `wholeBrainSeg_Large_UNEST_segmentation`: validation skips `NormalizeIntensityd`, so best-checkpoint selection uses un-normalised inputs (#…)
> - `mednist_reg`: `IgniteMetric` was removed in MONAI 1.4, so training does not start (#…)
>
> The pattern worth checking in your own configs: any spatial transform applied to `["image", "label"]` without `mode=[…, "nearest"]` interpolates the label. That includes `RandAffined`, `Rand3DElasticd` and `RandZoomd`, whose defaults are bilinear or area.
>
> The check is [nezaret](https://github.com/aslansd/nezaret) (`pip install nezaret`, `nezaret lint <bundle>`). It needs no GPU and doesn't run the bundle. Its `data` and `result` commands check datasets (leakage, duplicates) and reported metrics. Feedback on false positives is very welcome.

The Task01 duplicates (ROADMAP §2) are a separate report to a different audience: the MSD organisers and MONAI's `DecathlonDataset` users. They go out on their own once the remaining 49 pairs are checked.
