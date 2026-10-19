"""Present actual saved CSRT experiment outputs without retraining during export."""
from pathlib import Path
import json

import build_walkthrough as book

ROOT = book.ROOT
folder = ROOT / 'artifacts' / 'tracking_experiment'
report = json.loads((folder / 'comparison.json').read_text())
audit = json.loads((folder / 'tracking_audit.json').read_text())
primary = report['runs'][0]
metric = primary['comparison']['cnn_gru']
fixed = metric['fixed']['macro_mae_bpm']
tracked = metric['tracked']['macro_mae_bpm']
change = tracked-fixed
percent = 100*(fixed-tracked)/fixed
aggregate = report['aggregate']['cnn_gru']
coverage = report['coverage']
candidate = sum(v['candidates'] for v in coverage['fixed'].values())
counts = {variant: sum(v['eligible'] for v in values.values()) for variant, values in coverage.items()}
paired_count = sum(len(v) for v in report['paired_starts'].values())

summary = f"""## CSRT tracking experiment — measured results

Executed on the restored original recordings, 17 October 2026. This is a
thermal-only adaptation of JoulesEye's tracking idea: the existing first-frame
boxes initialize CSRT; there is no RGB detector or detector-based recovery.
The complete [executed experiment notebook](tracking_experiment.ipynb) and
[HTML presentation](tracking_experiment.html) include code, per-frame diagnostics,
sample overlays, paired evaluation and limitations.

**Primary result (CNN + GRU, seed 7):** fixed crops **{fixed:.3f} BPM** macro MAE;
tracked crops **{tracked:.3f} BPM**. Tracking {'reduced' if change < 0 else 'increased'}
error by **{abs(change):.3f} BPM** ({abs(percent):.1f}%).

| Method | Fixed MAE | Tracked MAE | Tracked − fixed |
|---|---:|---:|---:|
"""
for method, metrics in primary['comparison'].items():
    a, b = (metrics[v]['macro_mae_bpm'] for v in ['fixed', 'tracked'])
    summary += f'| {method} | {a:.3f} | {b:.3f} | {b-a:+.3f} |\n'
summary += '\n### Predetermined seed sensitivity\n\n| Seed | Fixed CNN + GRU | Tracked CNN + GRU | Tracked − fixed |\n|---|---:|---:|---:|\n'
for run in report['runs']:
    m = run['comparison']['cnn_gru']
    a, b = (m[v]['macro_mae_bpm'] for v in ['fixed', 'tracked'])
    summary += f"| {run['seed']} | {a:.3f} | {b:.3f} | {b-a:+.3f} |\n"
summary += f"""

Across the three predetermined seeds (7, 17, 27), CNN + GRU mean MAE is
**{aggregate['fixed_mean_mae_bpm']:.3f} BPM fixed** versus
**{aggregate['tracked_mean_mae_bpm']:.3f} BPM tracked**.
Tracking improves {aggregate['seeds_with_improvement']} of {len(report['runs'])} seeds.
These seeds measure optimization sensitivity, not additional independent people.

The primary CNN + GRU gain is very small. Seed 17 produces much larger errors
for both crop variants than seeds 7 and 27; this indicates substantial training
sensitivity under the unchanged 40-epoch recipe. The three-seed mean must not be
presented without that context. Tracking does not solve this optimization issue.

For the CNN-only comparison, three-seed mean MAE changes from
{report['aggregate']['cnn']['fixed_mean_mae_bpm']:.3f} to
{report['aggregate']['cnn']['tracked_mean_mae_bpm']:.3f} BPM, with improvement in
{report['aggregate']['cnn']['seeds_with_improvement']}/3 seeds. This is a larger
numerical benefit, but the tracked CNN still does not beat the median baseline
on the three-seed mean. The spectral comparator becomes worse with tracking.

### Coverage and fairness

- Fixed eligible windows: {counts['fixed']}/{candidate}; tracked: {counts['tracked']}/{candidate}.
- Both training and testing use the same {paired_count} paired eligible windows,
  with leave-one-person-out splits and unchanged architectures, loss, optimizer,
  epoch count, normalization and reference labels.
- All four recomputed fixed traces matched the prior trace arrays exactly.
- `anestis` remains excluded because its frame/reference alignment is unresolved.
- Explicit CSRT failures: {sum(s['tracking_failed_frames'] for s in audit['subjects'].values())} frames.
  Successful updates are not a measurement of anatomical tracking accuracy.

### Visual finding: undetected drift

Inspection of predetermined 90 s and 115 s samples shows Joao's tracked box
moving onto the lower face/neck rather than remaining on the nose. Claudio's
90 s and 115 s samples shift toward the upper face/eye area. CSRT still returns
successful updates on these frames. Thus a lower respiration error, if present,
does not by itself validate nasal tracking or prove that the learned signal is
physiological breathing. The notebook includes these overlays for review.

### Interpretation and limitations

This experiment {'supports a reduction' if change < 0 else 'does not support a reduction'}
in the primary model's seed-7 error on this pilot. It is not evidence that tracking
will generalize to new recordings. Only four people, one recording each, were
evaluated. Fixed initial boxes, possible undetected drift, uncertain marker
semantics, and the assumed motion boundary remain limitations. CSRT changes both
position and scale, so this experiment evaluates that combined change.

The training-median comparator remains essential: its paired seed-7 MAE is
{primary['comparison']['train_median']['fixed']['macro_mae_bpm']:.3f} BPM;
the tracked CNN + GRU {'beats' if tracked < primary['comparison']['train_median']['fixed']['macro_mae_bpm'] else 'does not beat'} it.
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
"""
(ROOT / 'docs' / 'tracking_experiment_results.md').write_text(summary, encoding='utf-8')

book.cells = []
book.md("""
# Does CSRT nose tracking improve our respiration model?
**Executed experiment · 17 October 2026**

We borrow [JoulesEye's](https://sustainability-lab.github.io/jouleseye/) tracking idea,
using [OpenCV CSRT](https://docs.opencv.org/4.6.0/d2/da2/classcv_1_1TrackerCSRT.html)
with the existing manually inspected first-frame boxes. RGB initialization is
unavailable in this dataset. This notebook presents actual saved experiment
outputs; reading those reports is executed below. Synthetic test outputs are
kept separate from real-data results. The source listings and opt-in rerun cells
make the extraction/training steps reviewable and reproducible.

The original fixed-crop walkthrough is in `thermal_respiration_walkthrough.ipynb`.
This local report embeds participant thermal frames.
""")
book.md((ROOT / 'docs' / 'tracking_experiment_protocol.md').read_text())
book.code("""
from pathlib import Path
import json, sys, hashlib
import numpy as np
from IPython.display import display, Markdown, Image
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'thermal_face').is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FOLDER = ROOT / 'artifacts' / 'tracking_experiment'
report = json.loads((FOLDER / 'comparison.json').read_text())
audit = json.loads((FOLDER / 'tracking_audit.json').read_text())
print('Recorded training environment:', json.dumps(report['environment'], indent=2))
print('Tracker:', audit['tracker'], '| OpenCV:', audit['opencv'])
print('Seeds:', report['protocol']['seeds'], '| epochs:', report['protocol']['epochs'])
print('Training/evaluation wall seconds:', round(report['elapsed_seconds'], 2))
print('Extraction wall seconds:', round(sum(s['elapsed_seconds'] for s in audit['subjects'].values()), 2))
# Verify that the reports refer to the trace archives currently present.
for variant, hashes in report['trace_sha256'].items():
    for person, expected in hashes.items():
        assert hashlib.sha256((FOLDER / variant / f'{person}.npz').read_bytes()).hexdigest() == expected
print('All trace hashes verified.')
""")
book.md("""
## Reproduce extraction and training (opt-in)
The saved results shown below were generated using these module entry points.
`RERUN=False` keeps a presentation from launching several minutes of tracking.
For this workspace only, OpenCV contrib was installed in
`artifacts/tracking_dependencies`; a clean environment can instead install
`requirements-tracking.txt`. Restart the notebook kernel after changing OpenCV
packages if an older cv2 module has already been imported.
""")
book.code("""
RERUN = False
if RERUN:
    local_cv = ROOT / 'artifacts' / 'tracking_dependencies'
    if local_cv.is_dir():
        sys.path.insert(0, str(local_cv))
    from thermal_face.tracking import extract
    from thermal_face.compare_tracking import run
    audit = extract(ROOT / 'Manuel_clean', FOLDER, ROOT / 'config' / 'nose_boxes.json',
                    ROOT / 'artifacts' / 'traces')
    report = run(FOLDER, FOLDER / 'comparison.json', seeds=[7, 17, 27], epochs=40, device='cpu')
else:
    print('Displaying the completed extraction and training reports; no retraining during this export.')
""")
book.md("""
## Tracking behavior and missing-data accounting
Frozen runs keep their original time coordinates. Failed boxes would become
missing samples; they are never replaced by a fixed crop. Duplicate frames do
not update the tracker. The nine patch features, timestamp resampling and
window rules remain the same. CSRT includes scale adaptation, so patch areas
can change with the box dimensions. A zero failure count does not exclude drift.
""")
book.code("""
table = '| Person | Frames | Frozen | Failed tracking | Valid tracked | Fixed matches prior |\\n|---|---:|---:|---:|---:|---|\\n'
for person, s in audit['subjects'].items():
    table += f"| {person} | {s['frames']} | {s['frozen_frames']} | {s['tracking_failed_frames']} | {s['valid_tracked_frames']} | {s['fixed_matches_previous_trace_exactly']} |\\n"
display(Markdown(table))
for variant, people in report['coverage'].items():
    print(variant, {p: {k: s[k] for k in ['candidates', 'eligible', 'paired']} for p, s in people.items()})
""")
book.md("""
## Visual inspection: fixed box versus tracked box
Red = fixed starting box; green = CSRT box. Samples are selected at predetermined
elapsed times (0, 30, 60, 90, 115 seconds), not selected based on prediction error.
This qualitative inspection cannot replace annotated landmark accuracy.
Joao's 90 s and 115 s boxes visibly drift to the lower face/neck, while
Claudio's move toward the upper face/eye area. These updates are not flagged as
failures by CSRT. No settings were adjusted in response to these observations.
""")
book.code("""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image as PILImage
import io
def show_plot(fig):
    stream = io.BytesIO()
    fig.savefig(stream, format='png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    display(Image(data=stream.getvalue()))
boxes = json.loads((ROOT / 'config' / 'nose_boxes.json').read_text())
people = sorted(audit['subjects'])
fig, axes = plt.subplots(len(people), 5, figsize=(14, 10))
for row, person in enumerate(people):
    paths = sorted((ROOT / 'Manuel_clean' / person).glob('frame_*.png'))
    with np.load(FOLDER / 'tracked' / f'{person}.npz') as z:
        seconds, tracked_boxes, valid = z['seconds'], z['boxes'], z['track_ok']
    for col, time_s in enumerate([0, 30, 60, 90, 115]):
        i = int(np.argmin(abs(seconds-time_s)))
        with PILImage.open(paths[i]) as picture:
            axes[row, col].imshow(np.array(picture), cmap='gray')
        for box, color in [(boxes[person], 'red'), (tracked_boxes[i], 'lime')]:
            if box[0] >= 0:
                x, y, w, h = box
                axes[row, col].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.3))
        axes[row, col].set_title(f'{person} {seconds[i]:.1f}s', fontsize=9)
        axes[row, col].axis('off')
fig.suptitle('Fixed (red) and tracked (green) nasal crops; no label-based selection')
fig.tight_layout()
show_plot(fig)
""")
book.md(summary)
book.md("""
## Per-person and condition results, primary seed 7
All errors are breaths/minute. Each person contributes equally to macro MAE.
The four people are the independent evaluation units; overlapping windows and
repeated seeds do not enlarge that number.
""")
book.code("""
primary = report['runs'][0]
table = '| Person | Fixed CNN + GRU | Tracked CNN + GRU | Change |\\n|---|---:|---:|---:|\\n'
for person in sorted(primary['comparison']['cnn_gru']['fixed']['per_subject_mae_bpm']):
    a, b = [primary['comparison']['cnn_gru'][v]['per_subject_mae_bpm'][person] for v in ['fixed', 'tracked']]
    table += f'| {person} | {a:.3f} | {b:.3f} | {b-a:+.3f} |\\n'
display(Markdown(table))
table = '| Condition | Fixed CNN + GRU | Tracked CNN + GRU |\\n|---|---:|---:|\\n'
for condition in ['still', 'moving', 'transition']:
    vals = [primary['variants'][v]['by_condition'][condition]['methods']['cnn_gru']['macro_mae_bpm'] for v in ['fixed', 'tracked']]
    table += '| ' + condition + ' | ' + ' | '.join(f'{v:.3f}' if v is not None else 'N/A' for v in vals) + ' |\\n'
display(Markdown(table))
fig, ax = plt.subplots(figsize=(9, 4))
methods = list(primary['comparison'])
positions = np.arange(len(methods))
for offset, variant, color in [(-0.18, 'fixed', '#637c91'), (0.18, 'tracked', '#188c7b')]:
    vals = [primary['comparison'][m][variant]['macro_mae_bpm'] for m in methods]
    bars = ax.bar(positions+offset, vals, 0.36, label=variant, color=color)
    ax.bar_label(bars, fmt='%.2f', padding=3)
ax.set(xticks=positions, xticklabels=methods, ylabel='Macro MAE (BPM)', title='Measured paired comparison: seed 7')
ax.legend()
show_plot(fig)
""")
book.md("""
## Verify implementation checks
This output is the saved test log from the implementation run. It includes
synthetic translated-target tracking, failure/recovery handling, frozen-run
timing, matched train/test windows, and existing evaluation/model tests.
These checks do not establish anatomical accuracy or clinical validity.
""")
book.code("""
text = (ROOT / 'artifacts' / 'tracking_tests.txt').read_text(encoding='utf-8')
lines = text.splitlines()
print('\\n'.join(line for line in lines if line.startswith('test_')))
print('\\n'.join(lines[-5:]))
""")
book.md('## Implementation source used for this experiment\nReference listings; rerun cells above call these modules.')
for name in ['tracking.py', 'compare_tracking.py']:
    book.md(f"### `thermal_face/{name}`\n\n```python\n{(ROOT / 'thermal_face' / name).read_text()}\n```")
book.md("""
## Limitations and next steps
- This is an adaptation of CSRT tracking, not a full JoulesEye reproduction.
  No synchronized RGB, automatic initialization, or detector-based recovery is available.
- Fixed manual initial boxes, possible undetected drift and box-scale changes
  need annotation-based evaluation. The sampled overlays are qualitative only.
- Four participants with one session each cannot establish generalization.
  Repeated seeds assess training variability, not population uncertainty.
- Marker semantics, row synchronization and the assumed 60-second motion boundary
  require verification. `anestis` is still excluded.
- Compare error together with coverage; any abstentions must remain visible.
- Next: annotate nose locations and drift during motion, verify references,
  test recovery without label-based tuning, and expand the independent test set.
  Keep the median and spectral baselines and retain both crop variants for review.
""")
book.build('tracking_experiment')

# Integrate the complete experiment into the original presentation without retraining.
if (ROOT / 'docs' / 'thermal_respiration_walkthrough.ipynb').exists():
    from integrate_tracking import main as integrate_original
    integrate_original()
