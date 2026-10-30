# Short-segment exploratory follow-up — 31 October 2026

The 20-second Kwon-style segmentation comparison remains **unevaluable** under
its original rules: 0/84 MRF windows were eligible. This follow-up asks a
smaller question using the same extracted traces: on portions of the existing
recordings that are long enough to evaluate, does dynamic MRF segmentation
change breathing-rate error? The [protocol](short_segment_protocol.md) was
written before calculating these errors.

## Why these recordings interrupt the method

Pixel-identical frame runs recur every roughly 15-19 seconds. Across the four
usable recordings there are 31 runs, each around 1.0-1.1 seconds. Their cause
cannot be determined from the supplied 8-bit images. We treat them as missing,
retain original timestamps, and allow the four-second BAF window to regain
support. As a result, uninterrupted valid segmented runs are generally around
10-14 seconds long. No 20-second MRF window passes the unchanged support rules.

## Paired 12-second results

The follow-up uses 12-second windows every two seconds, the same four usable
people, unchanged marker-1 references and the same missing-data and rate
estimators. It is **exploratory**; a 12-second rate estimate has much less
temporal evidence than the planned 20-second estimate.

| Condition and estimator | Paired windows | KLT ROI MAE (BPM) | KLT + MRF MAE (BPM) | Difference, MRF minus ROI |
|---|---:|---:|---:|---:|
| Still, spectral (primary follow-up) | 14 | 4.874 | 2.564 | -2.310 |
| Still, peak count | 14 | 2.856 | 2.111 | -0.745 |
| Overall, spectral | 29 | 10.583 | 8.613 | -1.970 |
| Overall, peak count | 27 | 5.296 | 4.963 | -0.333 |
| Moving, spectral | 12 | 12.543 | 12.591 | +0.049 |
| Moving, peak count | 10 | 6.752 | 7.585 | +0.833 |

Lower error is better. For still spectral windows, each of the four people has
lower mean error with MRF, but their contributions are unequal: Claudio has six
windows, Jaime five, Joao one and Manuel two. The per-person still spectral
errors change from 4.215 to 2.444, 6.914 to 1.009, 1.314 to 0.064, and 7.052
to 6.739 BPM respectively. Participant-macro MAE averages those four people
equally. The moving condition does **not** show a benefit.

There were 222 candidate 12-second windows per variant. KLT-only had 126
eligible; KLT+MRF had 29. In the MRF arm, 164 candidates exceeded the unchanged
20% missing-support limit and 16 lacked endpoint support. Reference checks
rejected 83 candidates, often overlapping signal exclusions. The 29 eligible
windows are heavily overlapping and come from only four independent people.

## What this supports

The still condition offers a measurable sign that the Kwon-inspired mask may
help rate estimation on short portions of these recordings. It does not prove
the mask selects the correct anatomical region. A support-based review sheet,
with one frame from the middle of each person's longest valid MRF run, shows
substantial non-nasal selections: face edges in Joao and Manuel, an eye and
face edge in Claudio, and nasal/cheek areas in Jaime. The Joao crop itself is
off the nose. This raises a serious tracking-and-mask validity concern even
where the short-window rate error improves. These images are review prompts,
not independent ground-truth labels. We cannot calculate segmentation overlap,
the paper's waveform correlation, or its cycle-accuracy metric from these data.
The published 20-second result remains unavailable, and the short-window
numbers should not replace it.

The next gate is to obtain/verify continuous source recordings and independent
nasal landmark and breathing-region annotations. A four-frame [review guide](mask_annotation_guide.md)
and blank local annotation CSV are ready. Then rerun a prespecified
20-second paired evaluation, reporting both accuracy and coverage. No MRF,
tracking, event or exclusion thresholds were changed for this follow-up.

Code: `thermal_face/evaluate_short_segments.py`. Saved window-level predictions,
exclusions, per-person metrics and trace hashes:
`artifacts/short_segment_experiment/evaluation.json`. The executed chapter is
also appended to `docs/thermal_respiration_walkthrough.ipynb`.
