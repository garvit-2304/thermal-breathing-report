# Corrected Kwon experiment — 18 October 2026

The project aims to detect breathing rate and segment breathing-associated
facial regions from thermal video. Kwon et al. (2023), DOI
10.1109/JTEHM.2023.3295775, guides the segmentation branch. The corrected
implementation and its first controlled experiment have been run, but
successful reproduction of the paper's published result is not established.

Implemented correct inverse KLT stabilization, four-second sample-standard-deviation
BAF, maximum-normalized BAP, dynamic two-class MRF/ICM, and integrated masked frame
differences. The earlier claim that the paper required variance without a square
root was wrong. Parameters unspecified by the paper are documented assumptions in
[the protocol](kwon_reproduction_protocol.md).

## Coverage and primary result

| Variant | Eligible / candidate 20-second windows |
|---|---:|
| Fixed ROI | 73 / 84 |
| Correct KLT, no MRF | 65 / 84 |
| Correct KLT + MRF | 0 / 84 |
| Wrong warp + MRF | 0 / 84 |

The MRF branches lose support around frozen frames and tracking failures, with
additional four-second warm-up/recovery. Seven windows lack endpoint support and
77 exceed the unchanged 20% missing-support limit. The primary still comparison
has zero eligible windows out of 36 candidates: MAE is undefined. This is a
coverage failure under the declared protocol, not a measured MRF accuracy failure.
The rules were not relaxed after observing the result.

An explicitly [exploratory 12-second follow-up](short_segment_results.md) later
produced paired estimates on supported portions of the same recordings. It is
reported separately and does not change the original 20-second outcome.

## Auxiliary alignment-only comparison

On 65 common windows, spectral macro MAE changes from 10.119 to 10.540 BPM;
peak-count MAE changes from 6.261 to 5.042 BPM. On 32 common still windows,
spectral MAE changes from 7.436 to 7.292 and peak-count MAE from 4.498 to
3.396 BPM. These estimator-dependent results concern alignment, not segmentation.

Fixed-time masks include cheeks and face edges as well as nasal regions.
No annotated pixel masks exist to measure Dice/IoU. Four usable participants,
unverified event semantics and condition boundaries, and different camera data
prevent a claim of matching the paper's 15-participant/BIOPAC results.

## Validation and next experiment

The suite ran 26 tests: 25 passed, one CUDA test skipped. New checks cover warp
direction and offsets, sample-SD equivalence, MRF localization on a synthetic
patch, constant-image mask changes, KLT translation, paired evaluation, and
synthetic peak rate. Synthetic checks establish mechanics, not clinical accuracy.

Next: audit freezes against original acquisition files, recover continuous
recordings, and predeclare a still-segment comparison with four seconds of warm-up
plus 20 seconds of valid synchronized observations. Independently annotate nasal
landmarks and segmentation masks. Confirm unspecified settings from author code
before claiming exact reproduction. Keep neural tuning secondary to this work.

Executed code, outputs, coverage plots, masks, limitations and next steps are in
[the original notebook](thermal_respiration_walkthrough.ipynb) and its
[HTML presentation](thermal_respiration_walkthrough.html). Full machine-readable
results are in `artifacts/kwon_experiment/evaluation.json`; source/input hashes and
frame exclusions are in `audit.json` and per-variant NPZ files.
