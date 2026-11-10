# Next experiment: short supported segments

Written 31 October 2026 **before calculating rate errors**. This is an
exploratory follow-up to the 20-second experiment, whose segmentation arm had
zero eligible windows. The primary project aim remains a thermal breathing-rate
detector with breathing-region segmentation.

## Reason for the new window

The existing `klt_mrf` traces have uninterrupted valid runs mostly around
10-14 seconds, while no 20-second window passes the unchanged support rules.
We inspected only support durations and reference eligibility to choose a
12-second window; we did not inspect rate predictions or errors for this choice.
At a two-second stride, the unchanged marker-1 reference rule (at least four
events, plausible intervals) and resampling rule (at most 20% missing support)
give 29 eligible segmented windows, 14 in the assumed still period. These
windows overlap substantially and are not independent participants.

## Locked comparison

- Input: the four existing original-frame trace pairs in
  `artifacts/kwon_experiment/klt_roi` and `klt_mrf`. `anestis` remains excluded
  because frame and reference row counts do not match. No feature re-extraction,
  model retraining, threshold adjustment, or label-based region selection.
- Candidate windows: 12 seconds long, two-second stride, from original elapsed
  time. The same `load_window_records` event and support checks apply.
- Main contrast: corrected KLT ROI **without vs with dynamic MRF** on the same
  eligible still windows. Secondary: overall, moving and transition windows.
- Main estimator: the existing spectral estimator, using the same 6-45 BPM
  range. Secondary estimator: the existing Butterworth-filtered peak count.
- Measure: participant-macro mean absolute breathing-rate error in BPM, paired
  by subject and window start. Also report each participant, paired count,
  coverage, and all exclusions. No reference-based parameter changes after
  evaluation.
- Interpretation: a nonzero paired score is a rate-estimation result on this
  small dataset, not pixel-level segmentation accuracy, waveform agreement, or
  reproduction of the paper's published results. Twelve-second rate estimation
  has coarse temporal support and may be sensitive to individual events.

## Independent next gate

Prepare fixed-time nasal crops and masks for human annotation. Landmark and
breathing-region labels must be made without looking at rate errors. Until
those labels exist, do not report Dice/IoU or claim anatomical correctness.
