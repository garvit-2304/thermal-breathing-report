# Comparison: timestamped reconstruction and CNN/GRU-MRF attempt

This note compares the local timestamped thermal-respiration reconstruction with the four-page progress report supplied as `Report.pdf`. The supplied report describes experiments completed by 18 October 2026. The local reconstruction report was generated on 31 October 2026. These are separate experiments and their headline errors are not a controlled head-to-head comparison.

## Results in the supplied report

The supplied report evaluates four participants and 77 eligible, overlapping 20-second segments for its fixed-versus-tracked model comparison. Its seed-7 macro MAEs are:

| Method | Fixed region (BPM) | Tracked region (BPM) |
| --- | ---: | ---: |
| Training-set typical-rate baseline | 2.501 | 2.501 |
| Frequency estimator | 10.618 | 10.809 |
| CNN | 3.577 | 3.268 |
| CNN + GRU | 2.986 | 2.969 |

The CNN + GRU tracking change is only 0.017 BPM in this run. Across three initializations, the report gives CNN MAE of 3.682 fixed versus 3.247 tracked, and CNN + GRU MAE of 6.803 versus 6.795; one initialization reaches about 14.87 BPM for both CNN + GRU variants. The training-rate baseline beats both learned models in the seed-7 comparison and does not use the video signal.

The report's Kwon-style MRF segmentation branch has zero eligible 20-second windows in its latest experiment (0/84 with correct alignment and 0/84 with deliberately incorrect alignment). Its segmentation accuracy therefore remains unmeasured in that experiment. The alignment-only comparison covers 65 shared windows and changes peak-counting MAE from 6.26 to 5.04 BPM, while frequency-estimator MAE changes from 10.12 to 10.54 BPM.

The GitHub repository also contains later MRF follow-ups beyond the supplied report. Its 31 October short-segment experiment reports spectral macro MAE changing from 4.874 to 2.564 BPM on 14 paired still windows; the moving condition does not improve. A subsequent fixed perinasal-mask ablation raises the still-window error to 3.350 BPM and reduces eligible short windows from 29 to 23. These are distinct, shorter-window experiments: they add evidence beyond the supplied report's 0/84 result but do not establish a general segmentation benefit or form a matched comparison with the local reconstruction.

## Results in the local reconstruction

The later reconstruction recovers all five recordings and 14,154 thermal frames from the ROS bags, retaining image-header timestamps and raw `mono16` counts. Its exploratory leave-one-subject-out nose-rate MAE is 4.72 BPM for raw signals, 4.62 for PWS graph filtering, and 4.91 for GTV. The small PWS change does not establish a gain.

For sparse inhalation-event detection, the TCN reaches macro F1 0.141 and event-span rate MAE 13.42 BPM. A training-fold-selected bandpass peak/valley comparator reaches macro F1 0.318 and event-span rate MAE 1.53 BPM. The comparator is stronger on this evaluation, although its event timing is still imperfect and its rate score should not replace event-level metrics.

An exploratory five-person region ablation gives 2.51 BPM MAE for raw mouth signals versus 4.72 for raw nose signals. Because regions were ranked on the same five people, this result is not an independently held-out ROI selection. Manual reference review covers 20 sampled frames: mean IoU is 0.56 for face boxes and 0.22 across visible facial-region boxes. This sparse, single-reviewer sample does not establish full-recording tracking accuracy.

## What the comparison supports

- The attached report explores CNN/GRU prediction, CSRT tracking, and dynamic pixel-level MRF masks. The local reconstruction instead emphasizes timestamp recovery, conventional signal/event baselines, graph-filter ablations, and sparse box-localization review.
- The local work resolves the fifth recording's image/reference count mismatch using timestamps, but the numeric `0/1/2` annotation mapping remains owner-confirmed rather than author-verified.
- The local work does not reproduce the MRF mask pipeline, and the supplied report does not evaluate event-level F1. Neither report supplies a common set of held-out windows with a shared protocol, so the 2.969, 4.72, and 1.53 BPM figures must not be ranked against one another as if they measured the same task.
- Both attempts point to localization and motion handling as important failure sources. The attached report notes tracker drift; the local sample shows much lower overlap for small facial regions than for full-face boxes.

## Recommended next comparison

Use one timestamp-matched set of recordings and eligible windows to compare the training-rate baseline, CNN + GRU, bandpass event detector, and any MRF/region method. Keep subject-level holdout, report coverage and exclusion reasons, and report rate error separately from event precision/recall/F1. The current five-person dataset supports an exploratory study, not a reliable claim of generalization.

This page contains aggregate results only. Raw recordings, participant-level result tables, and face-preview images are not included.
