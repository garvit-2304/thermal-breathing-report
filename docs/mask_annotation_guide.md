# Independent review of the breathing-region masks

This is the next validation gate for the thermal breathing-rate detector. The
contact sheet is saved locally at
`artifacts/short_segment_experiment/mask_review.png`. It contains an aligned
crop, MRF mask, and overlay for each of the four usable people. Frames were
selected from the middle of each person's longest valid segmented run using
signal validity only; no breathing-rate errors were used for selection. The
frame index and timestamp are in `mask_review_selection.json`.

Reviewers should inspect the **crop first**, without rate predictions. Record:

1. Is the nose/nasal outlet visible in the aligned crop? (`yes`, `partly`, `no`)
2. Does the mask cover the nasal outlet or immediately adjacent upper-lip area?
   (`mostly`, `partly`, `no`)
3. Does the mask also cover clearly unrelated regions such as an eye, cheek,
   face edge, or background? (`yes`, `no`)
4. One short note identifying any tracking drift or obvious artifact.

Use the matching blank CSV at
`artifacts/short_segment_experiment/mask_review_annotations.csv`. Two reviewers
should fill separate copies without discussing labels first; disagreements can
then be adjudicated. These four examples are a pilot review, not a pixel-mask
ground-truth dataset. For Dice/IoU, annotate pixel regions on a larger,
preselected set of original frames, including still and moving periods and
failed tracking cases. Keep those labels separate from rate-error tuning.
