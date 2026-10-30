"""Execute the spatial-constraint ablation and append it to the original notebook."""
import build_walkthrough as book

book.cells = []
book.md('''# Constraining the MRF mask to a perinasal area
**Exploratory experiment, executed 31 October 2026.** The earlier review found that the dynamic MRF mask often selected eyes or face edges. Here we intersect the existing mask with one fixed crop-relative ellipse. The [protocol](constrained_mask_protocol.md) was recorded before evaluating the new breathing-rate predictions. The ellipse is a rough anatomical prior, **not** a manually verified breathing-region annotation.''')
book.code('''from pathlib import Path
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image
root = Path.cwd()
if not (root / 'thermal_face').exists(): root = root.parent
import sys
if str(root) not in sys.path: sys.path.insert(0, str(root))
display(Markdown((root / 'docs/constrained_mask_protocol.md').read_text(encoding='utf-8')))
print('Original acquisition bag files in supplied dataset:',
      len(list((root / 'Manuel_clean').glob('*.bag'))))''')
book.md('''## 1. Apply the same spatial rule to all four usable recordings
The source frame pixels are hashed and checked against the original experiment. We use its stored KLT transforms and MRF masks. Each new trace integrates current-minus-previous intensity only where the original mask intersects the ellipse and both aligned frames have image support. Too few selected pixels become an explicit missing frame.''')
book.code('''from thermal_face.constrained_mask import extract
out = root / 'artifacts/constrained_mask_experiment'
audit = extract(root / 'Manuel_clean', root / 'artifacts/kwon_experiment',
                out / 'traces', root / 'config/nose_boxes.json')
rows = ['| Person | Original MRF-valid frames | Constrained-valid frames | Additional frames without enough pixels |', '|---|---:|---:|---:|']
for person, item in audit['subjects'].items():
    rows.append(f"| {person} | {item['original_valid_frames']} | {item['constrained_valid_frames']} | {item['too_few_prior_mask_pixels']} |")
display(Markdown('\\n'.join(rows)))''')
book.md('''## 2. Evaluate accuracy **and** coverage on paired windows
The same marker-1 labels, signal-support rules, spectral and peak-count estimators, and still/moving boundary are reused. The main comparison is 12-second still spectral error. The original 20-second gate is repeated unchanged. A paired error is calculated only where both masks produce an eligible prediction for the same person and time.''')
book.code('''from thermal_face.evaluate_constrained_mask import run
report = run(root / 'artifacts/kwon_experiment/klt_mrf', out / 'traces', out / 'evaluation.json')
def fmt(value): return 'undefined' if value is None else f'{value:.3f}'
for experiment, evaluation in report['evaluations'].items():
    print(experiment, 'eligible windows:', {k: f"{v['eligible_windows']}/{v['candidate_windows']}"
                                       for k, v in evaluation['variants'].items()})
    rows = ['| Estimator / condition | Paired | Original MRF MAE | Constrained MAE | Change |', '|---|---:|---:|---:|---:|']
    for estimator in ['spectral', 'peak_count']:
        for condition in ['still', 'overall', 'moving']:
            r = evaluation['comparisons'][estimator][condition]
            rows.append(f"| {estimator} / {condition} | {r['common_windows']} | {fmt(r['first']['macro_mae_bpm'])} | {fmt(r['second']['macro_mae_bpm'])} | {fmt(r['second_minus_first_bpm'])} |")
    display(Markdown('\\n'.join(rows)))
still = report['evaluations']['short_12s']['comparisons']['spectral']['still']
rows = ['| Person | Paired windows | Original MAE | Constrained MAE |', '|---|---:|---:|---:|']
for person in ['Joao','Claudio','Manuel','Jaime']:
    n = sum(k[0] == person for k in still['window_keys'])
    rows.append(f"| {person} | {n} | {fmt(still['first']['per_subject_mae_bpm'].get(person))} | {fmt(still['second']['per_subject_mae_bpm'].get(person))} |")
display(Markdown('Still spectral by person:\\n\\n' + '\\n'.join(rows)))''')
book.md('''## 3. Inspect the actual mask change
These are the same four frames chosen before rate-error inspection in the previous chapter. The last column shows the constrained pixels in cyan. Removing eye or edge pixels does not prove that the remaining pixels are breathing-associated. Joao's turned-head crop still misses the nose, so a within-crop rule cannot repair that tracking failure.''')
book.code('''import cv2
from thermal_face.kwon import crop_warp
from thermal_face.constrained_mask import perinasal_prior
selection = json.loads((root / 'artifacts/short_segment_experiment/mask_review_selection.json').read_text())
boxes = json.loads((root / 'config/nose_boxes.json').read_text())
fig, axes = plt.subplots(4, 3, figsize=(9.4, 11))
for row, chosen in enumerate(selection):
    person, index = chosen['person'], chosen['frame_index_zero_based']
    path = sorted((root / 'Manuel_clean' / person).glob('frame_*.png'))[index]
    frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    with np.load(root / 'artifacts/kwon_experiment/diagnostics' / f'{person}.npz') as d:
        crop, support = crop_warp(frame, d['matrices'][index], boxes[person])
        mask = np.unpackbits(d['klt_mrf_packed_masks'][index], count=crop.size).reshape(crop.shape).astype(bool)
    constrained = mask & perinasal_prior(crop.shape)
    gray = np.repeat((crop/255.0)[...,None],3,axis=2)
    overlay = gray.copy(); overlay[constrained] = .55*gray[constrained] + .45*np.array([0,1,.85])
    for col,(image,title,cmap) in enumerate([(crop,'Aligned crop','gray'),(mask,'Original MRF mask','gray'),(overlay,'Constrained overlay',None)]):
        axes[row,col].imshow(image,cmap=cmap,vmin=0 if col<2 else None,vmax=255 if col==0 else (1 if col==1 else None))
        axes[row,col].set_title(f"{person}, {chosen['seconds']:.1f}s: {title}",fontsize=9)
        axes[row,col].axis('off')
fig.tight_layout()
review_path = out / 'constrained_mask_review.png'
fig.savefig(review_path,dpi=150);plt.close(fig)
display(Image(filename=str(review_path)))''')
book.md('''## 4. Prepare an independent, rate-blind review packet
The four examples above were selected from long valid MRF runs. To avoid evaluating only successful frames, the next packet selects the nearest unfrozen frame to 15, 45, and 85 seconds for each person, regardless of whether the MRF mask is valid. It shows the original full frame and tracked box alongside the aligned crop and original mask. Two independent reviewers can fill separate copies of the blank CSV without seeing rate predictions. The script creates no human labels.''')
book.code('''import importlib.util
spec = importlib.util.spec_from_file_location('annotation_packet', root / 'docs/build_annotation_packet.py')
packet = importlib.util.module_from_spec(spec); spec.loader.exec_module(packet)
selection = packet.build(root)
print('Selected frames:', len(selection), 'valid MRF frames:', sum(x['mrf_frame_valid'] for x in selection))
display(Image(filename=str(out / 'annotation_mrf_overlays.png')))
print('Blank review form:', out / 'annotation_packet_blank.csv')''')
book.md('''## 5. Interpretation, limitations, and next step
The fixed perinasal constraint **worsens** the main exploratory result: still-window spectral macro MAE rises from **2.564 to 3.350 breaths/min** on the same 14 windows. Peak-count MAE rises from **2.111 to 3.117**. Overall eligible 12-second windows fall from **29 to 23**; moving eligible windows fall from **12 to 6**. Both variants still have **0 eligible 20-second windows**. These comparisons do not support adopting the constraint as the detector's default. In the still spectral comparison, Claudio improves slightly, Joao is unchanged, while Jaime and particularly Manuel worsen. A spatially tidy-looking mask can discard useful temperature signal, and a crop that misses the nose remains wrong even if its mask is restricted.

**Limitations:** This is one hand-chosen geometric rule, based on the visual review, and the same four people with overlapping short windows. The rule may exclude legitimate breathing-associated regions in profiles. Rate error is not pixel-mask accuracy. The supplied dataset has no original ROS bag files, independent nasal/breathing-region annotations, or reference waveform; the repeated frozen runs remain. We have not repaired the acquisition gaps or validated tracker geometry.

**Next:** independent reviewers should annotate nasal visibility, mask coverage and unrelated regions on the prepared four-frame sheet, then expand to a preselected still/moving sample including tracking failures. Use those labels to measure localization and mask overlap **without rate-label tuning**. Source recordings with continuous timestamps and preferably radiometric frames are needed before the 20-second result can be assessed. Keep this negative ablation alongside the original and short-segment results.''')

book.build('constrained_mask_experiment')
from integrate_constrained_mask import main
main()
