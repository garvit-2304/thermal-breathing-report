"""Execute the short-segment follow-up and append it to the original notebook."""
import build_walkthrough as book

book.cells = []
book.md('''# Short supported segments: exploratory next experiment
**Executed 31 October 2026.** The 20-second Kwon-style MRF evaluation remains at zero eligible windows. This chapter asks whether the already extracted segmented signal helps rate estimation on 12-second portions with adequate support. It does **not** replace the original 20-second result.

The window, stride, estimators, inclusion rules, and interpretation were recorded before calculating errors in `docs/short_segment_protocol.md`. We inspected valid-run durations, not rate errors, to choose 12 seconds.''')
book.code('''from pathlib import Path
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image
root = Path.cwd()
if not (root / 'thermal_face').exists(): root = root.parent
import sys
if str(root) not in sys.path: sys.path.insert(0, str(root))
folder = root / 'artifacts/kwon_experiment'
display(Markdown((root / 'docs/short_segment_protocol.md').read_text(encoding='utf-8')))''')
book.md('''## 1. Diagnose the available source support
The source recordings repeatedly contain pixel-identical frame runs. We keep their timestamps and mark those frames missing. The code below counts those runs and the longest uninterrupted period where the segmented trace is valid. The recordings alone do not establish why the freezes occur.''')
book.code('''people = ['Joao', 'Claudio', 'Manuel', 'Jaime']
rows = ['| Person | Frozen runs | Frozen frames | Longest valid MRF span (s) | KLT failure frames |', '|---|---:|---:|---:|---:|']
for person in people:
    with np.load(folder / 'diagnostics' / f'{person}.npz') as d:
        t, frozen, track = d['seconds'], d['frozen'], d['track_ok']
    with np.load(folder / 'klt_mrf' / f'{person}.npz') as z:
        valid = z['valid']
    starts = np.where(valid & ~np.r_[False, valid[:-1]])[0]
    ends = np.where(valid & ~np.r_[valid[1:], False])[0]
    longest = max((t[e]-t[s] for s, e in zip(starts, ends)), default=0)
    frozen_runs = int(np.count_nonzero(frozen & ~np.r_[False, frozen[:-1]]))
    rows.append(f'| {person} | {frozen_runs} | {int(frozen.sum())} | {longest:.2f} | {int((~track).sum())} |')
display(Markdown('\\n'.join(rows)))''')
book.md('''## 2. Run the locked 12-second paired evaluation
We use the existing corrected KLT-only and KLT+MRF traces. Windows start every two seconds. Both methods must have an eligible signal and a valid marker-1 reference for the same person and start time. The code saves all candidates, predictions, exclusions, and hashes.''')
book.code('''from thermal_face.evaluate_short_segments import run
report = run(folder, root / 'artifacts/short_segment_experiment/evaluation.json')
def show(value): return 'undefined' if value is None else f'{value:.3f}'
rows = ['| Condition | Estimator | Paired windows | KLT only MAE | KLT+MRF MAE | MRF minus KLT |', '|---|---|---:|---:|---:|---:|']
for condition in ['still', 'overall', 'moving', 'transition']:
    for estimator in ['spectral', 'peak_count']:
        r = report['comparisons'][estimator][condition]
        rows.append(f"| {condition} | {estimator} | {r['common_windows']} | {show(r['first']['macro_mae_bpm'])} | {show(r['second']['macro_mae_bpm'])} | {show(r['second_minus_first_bpm'])} |")
display(Markdown('\\n'.join(rows)))
for variant, details in report['variants'].items():
    print(variant, 'eligible', details['eligible_windows'], '/', details['candidate_windows'],
          'signal exclusions', details['signal_exclusions'],
          'reference exclusions', details['reference_exclusions'])''')
book.md('''## 3. Inspect each participant, not just the overall average
Still-window spectral error falls for all four participants, but the number of overlapping windows varies. The plot gives each person one pair of bars. Four participants, rather than 14 windows, is the independent sample size.''')
book.code('''r = report['comparisons']['spectral']['still']
names = people
a = [r['first']['per_subject_mae_bpm'][p] for p in names]
b = [r['second']['per_subject_mae_bpm'][p] for p in names]
counts = {p: sum(k[0] == p for k in r['window_keys']) for p in names}
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.bar(x-.18, a, .36, label='KLT ROI only', color='#758b9a')
ax.bar(x+.18, b, .36, label='KLT + MRF', color='#148084')
ax.set_xticks(x, [f'{p} (n={counts[p]})' for p in names])
ax.set_ylabel('Still-window spectral MAE (breaths/min)')
ax.set_title('Exploratory 12-second paired comparison')
ax.grid(axis='y', alpha=.2); ax.legend(); fig.tight_layout()
plot_path = root / 'artifacts/short_segment_experiment/per_person_still.png'
fig.savefig(plot_path, dpi=160); plt.close(fig)
display(Image(filename=str(plot_path)))''')
book.md('''## 4. Prepare masks for independent review
The images below are chosen from the middle of each person's **longest valid MRF run**, using validity only. The middle column shows the dynamic mask and the last overlays it on the aligned crop. These are review prompts, not ground-truth annotations. A high-variation region may also reflect movement or image noise; only independent labels can establish whether the selected pixels are truly breathing-associated.''')
book.code('''import cv2
from thermal_face.kwon import crop_warp
boxes = json.loads((root / 'config/nose_boxes.json').read_text())
fig, axes = plt.subplots(4, 3, figsize=(9.5, 11))
review = []
for row, person in enumerate(people):
    paths = sorted((root / 'Manuel_clean' / person).glob('frame_*.png'))
    with np.load(folder / 'diagnostics' / f'{person}.npz') as d, np.load(folder / 'klt_mrf' / f'{person}.npz') as z:
        valid, t = z['valid'], z['seconds']
        starts = np.where(valid & ~np.r_[False, valid[:-1]])[0]
        ends = np.where(valid & ~np.r_[valid[1:], False])[0]
        s, e = max(zip(starts, ends), key=lambda se: t[se[1]]-t[se[0]])
        idx = int((s+e)//2)
        frame = cv2.imread(str(paths[idx]), cv2.IMREAD_GRAYSCALE)
        crop, support = crop_warp(frame, d['matrices'][idx], boxes[person])
        mask = np.unpackbits(d['klt_mrf_packed_masks'][idx], count=crop.size).reshape(crop.shape).astype(bool)
        assert valid[idx] and mask.any() and mask.shape == crop.shape
        base = np.repeat((crop/255.0)[..., None], 3, axis=2)
        overlay = base.copy(); overlay[mask] = .55*base[mask] + .45*np.array([0.0, 1.0, .85])
        for col, (image, title, cmap) in enumerate([(crop, 'Aligned crop', 'gray'), (mask, 'MRF mask', 'gray'), (overlay, 'Overlay', None)]):
            axes[row, col].imshow(image, cmap=cmap, vmin=0 if col < 2 else None, vmax=255 if col == 0 else (1 if col == 1 else None))
            axes[row, col].set_title(f'{person}, {t[idx]:.1f}s: {title}', fontsize=10)
            axes[row, col].axis('off')
        review.append({'person': person, 'frame_index_zero_based': idx, 'seconds': float(t[idx]),
                       'mask_fraction': float(mask.mean())})
fig.tight_layout()
review_path = root / 'artifacts/short_segment_experiment/mask_review.png'
fig.savefig(review_path, dpi=150); plt.close(fig)
(root / 'artifacts/short_segment_experiment/mask_review_selection.json').write_text(json.dumps(review, indent=2))
display(Image(filename=str(review_path)))
print('Selection:', review)''')
book.md('''## 5. Interpretation, limitations, and next step
On the prespecified follow-up endpoint, still spectral participant-macro MAE changes from **4.874 to 2.564 breaths/min** (14 paired windows, four people). Still peak-count MAE changes from **2.856 to 2.111**. The moving condition does not improve: spectral changes from 12.543 to 12.591, and peak count from 6.752 to 7.585. This is a small exploratory result on 12-second, overlapping windows. It is not evidence that the mask is anatomically correct or that the original 20-second experiment succeeded. The support-based review images show substantial non-nasal selections: face edges for Joao and Manuel, an eye and edge for Claudio, and nasal/cheek areas for Jaime. Joao's aligned crop itself misses the nose. These images make independent localization and mask annotation a necessary next gate even though still-window rate error improved.

**Limitations:** the four recordings differ from Kwon's data and lack a synchronized waveform and pixel-level mask labels. Original camera/acquisition logs are unavailable, so the cause of repeated frozen frames is unknown. The movement boundary and marker interpretation still need verification. MRF settings partly reflect documented implementation assumptions. Short windows provide a coarse breathing-rate estimate. Many candidates are excluded: 29/222 MRF windows pass both signal and reference checks, compared with 126/222 KLT-only windows. A score without its coverage would be misleading.

**Next:** verify and recover continuous source recordings if possible; have a reviewer annotate nasal landmarks and breathing-associated regions on the saved mask examples without looking at rate errors; then rerun a prespecified paired **20-second** comparison on longer synchronized recordings. Keep the original 0/84 result and this short-window follow-up separate. No thresholds were tuned against these errors.''')

book.build('short_segment_experiment')
from integrate_short_segments import main
main()
