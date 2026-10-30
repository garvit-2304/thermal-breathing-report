# Independent review of the breathing-region masks

This is the next validation gate for the thermal breathing-rate detector. The
contact sheet is saved locally at
`artifacts/short_segment_experiment/mask_review.png`. It contains an aligned
crop, MRF mask, and overlay for each of the four usable people. Frames were
selected from the middle of each person's longest valid segmented run using
signal validity only; no breathing-rate errors were used for selection. The
frame index and timestamp are in `mask_review_selection.json`.

An expanded, rate-blind 12-frame packet is now also available locally:
`artifacts/constrained_mask_experiment/annotation_full_frames.png` shows the
original frame and tracked box; `annotation_mrf_overlays.png` shows the aligned
crop and original MRF mask. For each of the four people, the script selects
the nearest unfrozen frame to 15, 45 and 85 seconds without inspecting rate
predictions. The 15/45-second samples represent the assumed still period and
85 seconds the assumed moving period. Some masks are explicitly invalid at
these instants; review the crop/box anyway. Selection and validity are saved
in `annotation_packet_selection.json`. Run
`python docs/build_annotation_packet.py` to rebuild the packet.

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

For the 12-frame packet, use
`artifacts/constrained_mask_experiment/annotation_packet_blank.csv` instead.
Each reviewer should make a separate copy and fill the `reviewer` field plus
the five categorical judgments and note. Suggested values are `yes`, `partly`,
`no`, or `unclear`; use `not_applicable` for mask coverage when the MRF frame is
invalid. Inspect the full frame first, then the aligned crop, then the mask.
Do not inspect rate predictions while labeling. The packet remains local
because it contains participant images; only the code and blank protocol are
intended for the repository.
