# Exploratory spatial constraint for the Kwon-style mask

Recorded 31 October 2026 before evaluating the new rate predictions. The
previous mask review showed frequent eye and face-edge selections. This test
asks whether restricting the **already computed** MRF mask to a fixed
perinasal area improves rate estimation. It does not supply ground-truth
nasal or breathing-region labels.

## Fixed rule

- Use the four already eligible source recordings, their original 8-bit frames,
  saved KLT transforms, and original MRF masks. Do not change tracking, BAF,
  BAP, MRF, event labels, rate estimators, or frozen-frame handling.
- Inside every tracked crop, use an ellipse centred at `(0.50 width,
  0.61 height)` with radii `(0.27 width, 0.23 height)`. These normalized
  coordinates are identical for all people and frames. They cover a central
  nose and upper-lip region in the frontal review crops while excluding the
  upper eye band and outer face edges. A side-view nose can lie outside it.
- Intersect this ellipse with the original binary MRF mask and the valid
  current/previous warp support. Require at least 12 selected pixels for a
  frame; otherwise mark it missing. The threshold prevents a few isolated
  pixels from defining the mean temperature increment.
- Integrate the mean current-minus-previous pixel intensity under that mask,
  with the same held-value convention as the original MRF trace. Retain all
  timestamps and reference markers.
- Compare the constrained trace with the **original MRF trace** on identical
  eligible windows. The 12-second, two-second-stride still spectral comparison
  is the main exploratory contrast; peak count and moving/overall outcomes are
  secondary. Also rerun the original 20-second, five-second-stride gate.
- Report candidate coverage, paired count, per-person MAE and full exclusions.
  An apparent reduction in error accompanied by lost coverage is not a clear
  detector improvement.

This is an engineering ablation chosen from visual mask failures, not a
reproduction of a spatial rule in Kwon et al. Human-drawn masks and independently
checked nasal localization are still needed to establish segmentation quality.
