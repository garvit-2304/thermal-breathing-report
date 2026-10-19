## CSRT tracking experiment — measured results

Executed on the restored original recordings, 17 October 2026. This is a
thermal-only adaptation of JoulesEye's tracking idea: the existing first-frame
boxes initialize CSRT; there is no RGB detector or detector-based recovery.
The complete [executed experiment notebook](tracking_experiment.ipynb) and
[HTML presentation](tracking_experiment.html) include code, per-frame diagnostics,
sample overlays, paired evaluation and limitations.

**Primary result (CNN + GRU, seed 7):** fixed crops **2.986 BPM** macro MAE;
tracked crops **2.969 BPM**. Tracking reduced
error by **0.017 BPM** (0.6%).

| Method | Fixed MAE | Tracked MAE | Tracked − fixed |
|---|---:|---:|---:|
| train_median | 2.501 | 2.501 | +0.000 |
| spectral | 10.618 | 10.809 | +0.190 |
| cnn | 3.577 | 3.268 | -0.309 |
| cnn_gru | 2.986 | 2.969 | -0.017 |

### Predetermined seed sensitivity

| Seed | Fixed CNN + GRU | Tracked CNN + GRU | Tracked − fixed |
|---|---:|---:|---:|
| 7 | 2.986 | 2.969 | -0.017 |
| 17 | 14.867 | 14.865 | -0.002 |
| 27 | 2.557 | 2.551 | -0.006 |


Across the three predetermined seeds (7, 17, 27), CNN + GRU mean MAE is
**6.803 BPM fixed** versus
**6.795 BPM tracked**.
Tracking improves 3 of 3 seeds.
These seeds measure optimization sensitivity, not additional independent people.

The primary CNN + GRU gain is very small. Seed 17 produces much larger errors
for both crop variants than seeds 7 and 27; this indicates substantial training
sensitivity under the unchanged 40-epoch recipe. The three-seed mean must not be
presented without that context. Tracking does not solve this optimization issue.

For the CNN-only comparison, three-seed mean MAE changes from
3.682 to
3.247 BPM, with improvement in
3/3 seeds. This is a larger
numerical benefit, but the tracked CNN still does not beat the median baseline
on the three-seed mean. The spectral comparator becomes worse with tracking.

### Coverage and fairness

- Fixed eligible windows: 77/84; tracked: 77/84.
- Both training and testing use the same 77 paired eligible windows,
  with leave-one-person-out splits and unchanged architectures, loss, optimizer,
  epoch count, normalization and reference labels.
- All four recomputed fixed traces matched the prior trace arrays exactly.
- `anestis` remains excluded because its frame/reference alignment is unresolved.
- Explicit CSRT failures: 0 frames.
  Successful updates are not a measurement of anatomical tracking accuracy.

### Visual finding: undetected drift

Inspection of predetermined 90 s and 115 s samples shows Joao's tracked box
moving onto the lower face/neck rather than remaining on the nose. Claudio's
90 s and 115 s samples shift toward the upper face/eye area. CSRT still returns
successful updates on these frames. Thus a lower respiration error, if present,
does not by itself validate nasal tracking or prove that the learned signal is
physiological breathing. The notebook includes these overlays for review.

### Interpretation and limitations

This experiment supports a reduction
in the primary model's seed-7 error on this pilot. It is not evidence that tracking
will generalize to new recordings. Only four people, one recording each, were
evaluated. Fixed initial boxes, possible undetected drift, uncertain marker
semantics, and the assumed motion boundary remain limitations. CSRT changes both
position and scale, so this experiment evaluates that combined change.

The training-median comparator remains essential: its paired seed-7 MAE is
2.501 BPM;
the tracked CNN + GRU does not beat it.
No tracker or model settings were revised after inspecting these errors.

Next, annotate nose locations through motion to quantify drift, test localization
and explicit recovery, verify reference events/alignment, and collect additional
people and sessions. Do not replace a baseline based solely on a favorable seed.

### Reproduction

Install `requirements-tracking.txt` in a clean environment (use the contrib
OpenCV distribution rather than combining multiple OpenCV wheels), then run:

```powershell
python -m thermal_face.tracking --root Manuel_clean --boxes config/nose_boxes.json --out artifacts/tracking_experiment --fixed-reference artifacts/traces
python -m thermal_face.compare_tracking --traces artifacts/tracking_experiment --epochs 40 --seeds 7 17 27 --device cpu
python docs/build_tracking_report.py
```

Machine-readable reports: `artifacts/tracking_experiment/tracking_audit.json`
and `artifacts/tracking_experiment/comparison.json`. The protocol is documented
in [tracking_experiment_protocol.md](tracking_experiment_protocol.md).
