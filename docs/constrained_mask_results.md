# Fixed perinasal mask constraint: exploratory result — 31 October 2026

The [locked protocol](constrained_mask_protocol.md) tests one simple response
to the prior visual review: keep the original MRF pixels only inside a fixed
ellipse around the central nose and upper-lip area of each tracked crop. The
same rule is used for all four people. Original frames, KLT transforms, BAF,
BAP, MRF masks, event labels, missing-data rules, and rate estimators are
unchanged. The frame pixels were verified against the original extraction's
hashes. This is an engineering ablation, not a claim of anatomical labeling.

## Paired breathing-rate results

| Window / condition / estimator | Paired windows | Original MRF MAE | Constrained MAE | Change, constrained minus original |
|---|---:|---:|---:|---:|
| 12 s / still / spectral (main) | 14 | 2.564 | 3.350 | **+0.786** |
| 12 s / still / peak count | 14 | 2.111 | 3.117 | +1.005 |
| 12 s / overall / spectral | 23 | 7.261 | 7.705 | +0.444 |
| 12 s / moving / spectral | 6 | 10.260 | 12.343 | +2.083 |
| 20 s / all conditions / either estimator | 0 | undefined | undefined | undefined |

Lower error is better. The main short-window result **worsens** with the
constraint. Original MRF has 29/222 eligible 12-second windows, while the
constrained version has 23/222. Moving coverage falls from 12 to 6 eligible
windows; still coverage remains 14. The unchanged 20-second gate remains at
0/84 for both variants. On the 14 paired still spectral windows, the
participant errors are:

| Person | Paired windows | Original MAE | Constrained MAE |
|---|---:|---:|---:|
| Joao | 1 | 0.064 | 0.064 |
| Claudio | 6 | 2.444 | 2.236 |
| Manuel | 2 | 6.739 | 9.864 |
| Jaime | 5 | 1.009 | 1.237 |

There are four independent people, not 14 independent observations; the
windows overlap. Spectral and peak-count results agree in direction on the
still comparison. On the original MRF-valid frames, the new minimum-pixel rule
rejects another 72 Joao, 52 Claudio, 178 Manuel, and 2 Jaime frames. No source
ROS bag files were present in the supplied `Manuel_clean` folder, so the frozen
frame runs cannot be recovered from it.

## What the visual review means

The local review image at
`artifacts/constrained_mask_experiment/constrained_mask_review.png`
uses exactly the four frames selected before this experiment's rate errors were
calculated. The ellipse removes some obvious eye/edge mask pixels. It also
throws away potentially useful temperature changes, especially for Manuel.
The Joao turned-head crop is still misplaced: restricting a mask **inside** an
incorrect crop cannot move it back to the nose. These observations explain
why a tidier-looking shape need not give a better breathing signal. They do
not establish the true breathing-associated region.

**Decision:** do not promote the constrained mask to the main pipeline. Keep
the original MRF result and this negative ablation visible. Independent
reviewers still need to annotate nasal visibility, localization error and mask
coverage on still and moving frames. Pixel-level Dice/IoU and paper-style
waveform metrics remain unavailable. Longer continuous, synchronized source
recordings are required to assess the planned 20-second detector.

A 12-frame [annotation guide](mask_annotation_guide.md) and local blank CSV now
support independent review across both assumed still and moving periods. No
human annotations have been collected yet.

Reproduce with `python -m thermal_face.constrained_mask`, then
`python -m thermal_face.evaluate_constrained_mask`, then
`python docs/build_constrained_mask_report.py`. Full window predictions,
exclusions, and trace hashes are saved locally in
`artifacts/constrained_mask_experiment/evaluation.json`; the executed chapter
is appended to `docs/thermal_respiration_walkthrough.ipynb`.
