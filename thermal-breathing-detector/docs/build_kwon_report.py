"""Execute the Kwon report and integrate its saved outputs into the original notebook."""
import json
from pathlib import Path
import build_walkthrough as book

ROOT = book.ROOT
book.cells = []
book.md('''# Kwon BAFR reproduction: corrected implementation and coverage experiment
**Executed 18 October 2026.** This is the Kwon-inspired segmentation branch of the broader breathing-rate detector project. Earlier CNN/GRU and CSRT studies test other parts of the same project.

Source: Kwon et al., *Breathing-Associated Facial Region Segmentation for Thermal Camera-Based Indirect Breathing Monitoring*, [DOI 10.1109/JTEHM.2023.3295775](https://doi.org/10.1109/JTEHM.2023.3295775), supplied PDF pages 3–5, equations 1–2 and Figure 4.

The paper specifies KLT nasal tracking → four-second BAF → maximum-normalized BAP → two-class MRF with 30 ICM iterations per frame → integrated frame differences. **BAF is sample standard deviation, not variance alone.** Our earlier criticism of the square root was incorrect and is withdrawn.

This chapter records a method reimplementation on different data, not a reproduction of the paper's published accuracy. The complete assumptions were recorded before evaluating errors in `docs/kwon_reproduction_protocol.md`.''')
book.code('''from pathlib import Path
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image
ROOT = Path.cwd()
if not (ROOT / 'thermal_face').exists():
    ROOT = ROOT.parent
folder = ROOT / 'artifacts/kwon_experiment'
evaluation = json.loads((folder / 'evaluation.json').read_text())
audit = json.loads((folder / 'audit.json').read_text())
print('Recorded status:', evaluation['status'])
print('Primary contrast:', evaluation['primary'])
display(Markdown((ROOT / 'docs/kwon_reproduction_protocol.md').read_text()))''')
book.md('''## Step 1. Correct the implementation before judging segmentation
The legacy code estimated reference-to-current motion but applied it in the opposite sampling direction. The new code uses `WARP_INVERSE_MAP` with the correct ROI-coordinate offset. It computes a fresh four-second sample-SD map and MRF mask at each valid frame, instead of one whole-recording mask. Original timestamps and exclusions remain intact.

The signal integrates the mean frame difference under the current mask. Changing mask membership alone cannot create an intensity jump. Gaussian likelihoods, four-neighbor Potts weight 0.6, red-black ICM, and KLT thresholds are explicitly declared choices because the paper does not fully specify them. These are not claimed to be the authors' original implementation.

The following cell exposes the actual extraction and evaluation source for presentation and review.''')
book.code('''display(Markdown('```python\\n' + (ROOT / 'thermal_face/kwon.py').read_text() + '\\n```'))
display(Markdown('```python\\n' + (ROOT / 'thermal_face/evaluate_kwon.py').read_text() + '\\n```'))''')
book.md('''## Step 2. Run four controlled variants
The experiment used original frames for Joao, Claudio, Manuel and Jaime. `anestis` remains excluded because frame and CSV counts disagree.

1. Fixed nasal ROI, no segmentation.
2. Corrected KLT alignment, no segmentation.
3. Corrected KLT plus dynamic BAF/BAP/MRF.
4. Deliberately wrong warp plus the same dynamic MRF, isolating warp direction.

Commands actually run:
```powershell
python -m thermal_face.kwon --root Manuel_clean --out artifacts/kwon_experiment --boxes config/nose_boxes.json
python -m thermal_face.evaluate_kwon --traces artifacts/kwon_experiment --out artifacts/kwon_experiment/evaluation.json
python -m unittest discover -s tests -v
```
The report cells read saved experiment outputs; they do not silently reprocess data. Source and input hashes are recorded in the audit and evaluation files.''')
book.code('''print((ROOT / 'artifacts/kwon_tests.txt').read_text().split('----------------------------------------------------------------------')[-1])
for person, info in audit['subjects'].items():
    print(person, json.dumps(info, indent=2))''')
book.md('''## Step 3. Check coverage before interpreting rate errors
The unchanged evaluator uses 20-second windows every five seconds. It rejects windows with more than 20% missing interpolated support, and does not extrapolate beyond valid endpoints. The four-second BAF support rule additionally rejects windows with fewer than 80% valid source samples or less than 3.5 seconds of valid span.

Frozen frames, tracking failures and BAF warm-up/recovery compound. This is a pipeline coverage failure under the declared rules; it is not evidence that the MRF cannot identify breathing pixels.''')
book.code('''rows = ['| Variant | Eligible / candidate windows | Signal exclusions |', '|---|---:|---|']
for name, result in evaluation['variants'].items():
    rows.append(f"| {name} | {result['eligible']} / {result['candidates']} | {result['signal_exclusions']} |")
display(Markdown('\\n'.join(rows)))
fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
for ax, person in zip(axes, ['Joao', 'Claudio', 'Manuel', 'Jaime']):
    for level, variant in enumerate(evaluation['variants']):
        z = np.load(folder / variant / f'{person}.npz')
        ax.scatter(z['seconds'][z['valid']], np.full(z['valid'].sum(), level), s=1)
    ax.set_yticks(range(4), list(evaluation['variants']))
    ax.set_title(person); ax.grid(alpha=.2)
axes[-1].set_xlabel('Original elapsed seconds; dots indicate valid extracted samples')
fig.tight_layout(); fig.savefig(folder / 'coverage.png', dpi=150); display(Image(filename=str(folder / 'coverage.png'))); plt.close(fig)''')
book.md('''## Step 4. Inspect fixed-time segmentation examples
These snapshots were selected at fixed times before error evaluation. Black masks can represent unavailable BAF support, not necessarily a failed MRF optimization. Masks label high thermal variability; without pixel annotations we cannot establish anatomical accuracy or Dice/IoU. Visual inspection shows cheek and facial-edge regions in several masks, alongside nasal regions. Spatial smoothness alone does not establish breathing specificity.''')
book.code('''fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for row, person in enumerate(['Joao', 'Claudio', 'Manuel', 'Jaime']):
    z = np.load(folder / 'diagnostics' / f'{person}.npz')
    for col, (key, label) in enumerate([('raw_samples','Raw crop'), ('aligned_samples','Correct KLT'), ('klt_mrf_bap','BAP'), ('klt_mrf_masks','MRF')]):
        axes[row, col].imshow(z[key][0], cmap='gray' if col < 2 else 'magma')
        axes[row, col].set_title(f"{person}, {z['sample_seconds'][0]:.1f}s: {label}", fontsize=9)
        axes[row, col].axis('off')
fig.tight_layout(); fig.savefig(folder / 'masks.png', dpi=150); display(Image(filename=str(folder / 'masks.png'))); plt.close(fig)''')
book.md('''## Step 5. Report the measured result without substituting another endpoint
**The primary MRF comparison is unevaluable: 0 common eligible windows, including 0/36 still candidates. Its MAE is undefined, not zero.** No improvement claim is supported. Both correct- and wrong-warp MRF branches have zero eligible windows, so real-data rate errors cannot quantify the benefit of correcting warp direction.

The auxiliary alignment-only comparison is evaluable on identical windows. Peak-count and spectral outcomes differ; neither is a substitute for the primary segmentation result.''')
book.code('''rows = ['| Contrast / estimator / condition | Paired windows | First MAE | Second MAE | Change (BPM) |', '|---|---:|---:|---:|---:|']
fmt = lambda value: 'undefined' if value is None else f'{value:.3f}'
for name in ['segmentation', 'stabilization', 'warp_correction']:
    for method in ['spectral', 'peak_count']:
        for condition in ['still', 'overall']:
            r = evaluation['comparisons'][name][method][condition]
            rows.append(f"| {name} / {method} / {condition} | {r['common_windows']} | {fmt(r['first']['macro_mae_bpm'])} | {fmt(r['second']['macro_mae_bpm'])} | {fmt(r['second_minus_first_bpm'])} |")
display(Markdown('\\n'.join(rows)))''')
book.md('''## Limitations and inference
- The earlier legacy 15.40 BPM result mixes timing, ROI, reference and processing errors; it cannot isolate MRF failure.
- Corrected processing is implemented and tested, but this experiment does **not** establish successful breathing segmentation or reproduce published performance.
- Four people with overlapping windows provide very limited independent evidence. Still/moving labels assume a 60-second boundary.
- The paper uses 15 participants, specific camera views and a BIOPAC waveform. Our 8-bit recordings and event labels cannot yield its waveform correlation or equivalent cycle accuracy.
- Gaussian/Potts and tracking choices remain reproduction assumptions. High temporal SD can also reflect motion, camera changes or noise; MRF spatial smoothness does not identify respiration by itself.
- Missing source frames cannot be reconstructed by changing timestamps or duplicating frames. Relaxing validity thresholds after observing this result would define a new exploratory experiment.

## What to do next, in order
1. Audit freeze timing against original acquisition files and reference timestamps. Recover uninterrupted source recordings if available; preserve the current run as the first corrected experiment.
2. Predeclare a clean-segment experiment: obtain continuous still recordings with at least four seconds of warm-up followed by 20 seconds of valid synchronized data. Use the same KLT-only versus KLT+MRF comparison and the same rate estimators. Report all exclusions.
3. Annotate nasal landmarks and breathing-associated pixels independently of rate errors. Quantify alignment error and segmentation overlap, separating tracking, segmentation and downstream rate estimation failures.
4. Confirm underspecified MRF settings from author code or correspondence; assess alternatives only in a separate development set. Do not tune on held-out rate labels.
5. Acquire paper-compatible views, radiometric recordings, repeated subjects/sessions and a synchronized respiratory waveform for a full replication.

**Presentation conclusion:** We corrected the pipeline toward Kwon's method, including dynamic BAF/BAP/MRF and inverse stabilization. The next controlled experiment revealed insufficient valid temporal coverage, so MRF accuracy remains unmeasured. The immediate research priority is valid continuous data and independent segmentation validation, not additional neural-model tuning.''')
book.build('kwon_reproduction')
from integrate_kwon import main
main()
