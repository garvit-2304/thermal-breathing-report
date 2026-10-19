# Fixed versus tracked nasal crops: experiment protocol

Specified before inspecting respiration errors for the tracked pipeline, 17 October 2026.

## Change being tested

Borrow JoulesEye's use of CSRT to follow the nasal region. Initialize from the
existing manually specified first-frame box in `config/nose_boxes.json` because
the supplied recordings contain no synchronized RGB. Use OpenCV 4.12.0 CSRT
defaults, including scale adaptation. Do not optimize tracker settings, crop
sizes, missing-data thresholds, or neural hyperparameters using test errors.
This is a thermal-only adaptation, not an exact JoulesEye reproduction.

Read original 8-bit frames and original CSV time. At each frame, compute the same
nine patch means from both the fixed and tracked box. Preserve every timestamp.
Skip CSRT updates on identical consecutive frames; retain the previous tracking
status and apply the existing >=5-frame frozen-run exclusion afterward. Mark
failed, nonfinite, undersized, or out-of-image boxes missing. Never substitute
the fixed crop after failure. Continue CSRT updates to allow recovery, without
claiming detector-based reinitialization. A successful tracker update is not
proof that the box still covers the nose.

Recompute fixed traces and require exact numerical equality to the earlier
`artifacts/traces` arrays. Write new outputs separately under
`artifacts/tracking_experiment/{fixed,tracked}`. Record per-frame boxes and
status, trace hashes, raw pixel hashes, source hashes, failures and runtime.

## Evaluation

- Keep the four previously eligible people; exclude `anestis` for unresolved alignment.
- Keep 20-second windows, five-second stride, 10 Hz resampling, marker-1 labels,
  normalization, missing threshold and 60-second condition boundary unchanged.
- Use the intersection of fixed/tracked eligible windows for **both training
  and testing**, separately for each person. Report each variant's original
  candidate coverage and exclusions; do not silently omit a person if all their
  paired windows are unavailable.
- Train both CNN and CNN + GRU from scratch for 40 epochs per held-out-person
  fold. Use the existing optimizer/loss/batch settings unchanged.
- Primary comparison: CNN + GRU, seed 7, macro MAE across held-out people.
- Predetermined sensitivity runs: seeds 17 and 27, for both architectures and crops.
- Include the training-median and spectral comparators. Score each method on
  matching windows with predictions from both crops, so spectral abstention
  cannot change the neural methods' scoring subset.
- Report overall, per-person, and still/moving/transition errors; per-seed paired
  changes; mean and population standard deviation over seeds. Negative
  tracked-minus-fixed MAE means improvement. Repeated seeds are not independent
  participants and do not provide population-level confidence intervals.
- Do not compare a coverage-reduced result directly with the old all-window
  2.986 BPM figure. Compare the retrained paired fixed baseline instead.

## References

- [JoulesEye project and method](https://sustainability-lab.github.io/jouleseye/)
- [OpenCV CSRT API](https://docs.opencv.org/4.6.0/d2/da2/classcv_1_1TrackerCSRT.html)

The main research question is whether this concrete adaptation improves this
pilot's measured error. It cannot establish reliable generalization from four
people or validate the anatomical accuracy of tracking without frame labels.
