# Kwon BAFR reproduction: corrected experiment protocol

Specified 18 October 2026 before computing corrected respiration errors.

The primary project objective is reproducing Kwon et al., *Breathing-Associated
Facial Region Segmentation for Thermal Camera-Based Indirect Breathing
Monitoring*, DOI 10.1109/JTEHM.2023.3295775. CNN/GRU and CSRT work is exploratory
background, not a substitute for this reproduction.

## Verified from the supplied paper

Pages 507-509 (PDF pages 3-5), equations 1-2 and Figure 4 specify:

1. KLT tracking of nasal ROIs.
2. A four-second moving window. BAF is **sample standard deviation**
   `sqrt(sum((BS - mean(BS))**2)/(N-1))`, not variance alone. Earlier project
   commentary incorrectly criticized the square root; that criticism is withdrawn.
3. BAP = BAF divided by the maximum BAF value for that time point. The equation
   prints `arg max`, but its prose clearly describes the maximum value, which
   is the interpretation implemented here.
4. Two-class MRF segmentation, initialized from the middle probability, with
   up to 30 ICM iterations **per frame**, rather than a whole-recording mean mask.
5. Breathing signals derived by integrating frame differences, comparing the
   same ROI with and without the mask.

The paper's published experiment has 15 participants, specified camera views,
640x480/25 Hz thermal recordings and 1000 Hz BIOPAC waveform references in a
stable setting. Our four aligned 382x288, approximately 27 Hz, 8-bit recordings
with annotated inhalation events are different data. Stillness is the primary
condition here; head movement is an additional stress test. We cannot reproduce
the paper's waveform correlation or published breathing-cycle accuracy from
these data and must not relabel rate MAE as those metrics.

## Implementation choices where the paper is incomplete

No complete likelihood formula, neighborhood, regularization weight, KLT
parameters, mask-normalization convention, or detailed peak detector is supplied.
This implementation makes those choices explicit, not claims of author code:

- Existing manually inspected nose boxes initialize KLT, restricted to a local
  face context (box expanded by one box width/height on each side).
- Pyramidal KLT, forward/backward error <=1.5 px, RANSAC partial affine with
  3 px threshold, >=6 inliers and >=60% inlier fraction. Per-step scale 0.8-1.25
  and reference-to-current total scale 0.5-2.0. Failed tracking is missing;
  reference-frame KLT is attempted for recovery. No stale fixed-box fallback.
- Use the inverse image warp correctly: the reference-to-current matrix is
  passed with `WARP_INVERSE_MAP` when sampling the current image in reference
  coordinates. Reject crops with <98% source-image support. Synthetic tests
  verify transform direction and crop-coordinate offsets.
- Retain original time; frozen runs >=5 frames remain missing. Four-second
  feature windows use actual timestamps and >=80% valid samples, with a minimum
  3.5 s valid span. No image-value conversion to degrees Celsius.
- BAF equals the sample SD of integrated per-pixel frame differences; subtracting
  the first image in the window is algebraically equivalent for its SD. Float
  maps are retained internally, without PNG quantization or renormalizing a
  whole-recording mean.
- Initialize at BAP >0.5 (a declared interpretation of 'middle probability').
  Use Gaussian class likelihoods, a four-neighbor Potts penalty of 0.6 (retained
  from the existing implementation, not specified by the paper), and red-black
  ICM for up to 30 sweeps. Class parameters are updated between sweeps. Label the
  higher-mean class breathing-associated. No forced minimum anatomical mask and
  no reference-label-based parameter selection. Empty masks abstain.
- For a time-varying mask, integrate the **masked mean frame difference**, rather
  than differencing means from two different masks. This avoids artificial
  jumps caused solely by changing mask membership. The spatial mean normalizes
  mask area; this normalization is an explicit implementation choice.

## Controlled comparisons and evaluation

Four prespecified variants use the same initial nasal ROI and timestamps:

1. `fixed_roi`: no alignment or segmentation.
2. `klt_roi`: corrected KLT alignment, no segmentation.
3. `klt_mrf`: corrected alignment plus four-second per-frame BAF/BAP/MRF.
4. `wrong_warp_mrf`: deliberately forward-warped alignment with the same dynamic
   MRF, isolating the stabilization-direction bug. This is an ablation, not an
   exact rerun of the old broad-ROI/static-mask pipeline.

Primary contrast: `klt_mrf` vs `klt_roi` on identical eligible **still** windows.
Secondary: overall/moving/transition contrasts and the wrong-warp ablation.
Twenty-second windows, five-second stride; existing marker-1 median-interval
labels and validity checks. Exclude `anestis` until synchronization is verified.

Estimate rate with a fixed, common spectral estimator (6-45 BPM, detrend, Hann
taper, 8x zero padding). Also report a prespecified 0.1-0.75 Hz fourth-order
Butterworth / peak-count estimator (minimum peak spacing 0.8 s, prominence 0.4
times filtered SD). These are **evaluation choices**, not exact paper detector
replications. No supervised models are trained.

Report all-method coverage, pairwise common-window MAE, per-person and condition
metrics, counts of tracking/mask failures, masks and waveforms. Preserve all
candidate windows and reasons for exclusion. Do not compare against the invalid
legacy 15.40 BPM score or infer segmentation Dice/IoU without annotated masks.
No parameter changes will be made in response to held-out reference errors in
this experiment. A negative result is a valid outcome; next changes require a
separate experiment.
