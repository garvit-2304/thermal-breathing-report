"""Build and execute the presentation notebook without requiring Jupyter tooling."""
from pathlib import Path
import html
import json
import os
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
cells = []

def md(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=text.strip()))

def code(text):
    cells.append(dict(cell_type="code", metadata={}, source=text.strip(), execution_count=None, outputs=[]))

md("""
# Thermal face respiration research: a step-by-step walkthrough
**Presentation edition · prepared 17 October 2026**

**Research question:** Can changes in thermal intensity around the nose estimate breathing rate for a person unseen during training?

This notebook tells the development story in logical sequence: initial signal processing → data corrections → neural pilots → fair comparison → limitations and next work. The repository does not provide a complete dated experiment log, so stage numbers indicate development order, **not invented dates or elapsed training times**.

**Evidence labels used throughout**
- **Executed now:** outputs produced by running the visible cell while building this notebook.
- **Historical/documented:** values transcribed from `README.md`, `docs/research_strategy.md`, or legacy reports; not freshly reproduced measurements.
- **Synthetic demonstration:** invented signals used only to explain and verify mechanics, never participant results.
- **Conditional reproduction:** real-data work runs only when inputs exist and the explicit rerun switch is enabled.

The delivered notebook contains executed outputs and can be presented without rerunning it. A companion HTML edition is suitable for browser presentation. This local edition embeds participant thermal frames and supplied processing maps; review dataset sharing terms before distributing it.
""")
md("""
## 1. What this project does
Breathing produces temporal changes around the nose in thermal imagery. The current pipeline summarizes nine nasal patches, preserves timestamps, builds windows, and predicts breaths per minute (BPM).

`Thermal frames + CSV timestamps → fixed nose crop → 9 intensity traces → 10 Hz resampling + missing-data channel → 20 s windows → CNN / CNN + GRU → breathing rate`

The broader ambition includes recognizing identity, but the implemented pilot is respiration regression. It is not a validated face recognition or clinical monitoring system. The available images are 8-bit grayscale intensities, not calibrated degrees Celsius.

### Research foundations
| Paper | Role in this project |
|---|---|
| [Coşar et al., 2018, Thermal Camera Based Physiological Monitoring with an Assistive Robot](https://doi.org/10.1109/EMBC.2018.8513201) | Dataset source according to project notes; facial thermal signals and Fourier analysis |
| [Kwon et al., 2023, Breathing-Associated Facial Region Segmentation for Thermal Camera-Based Indirect Breathing Monitoring](https://doi.org/10.1109/JTEHM.2023.3295775) | BAFR segmentation inspiration; the PDF included in the repository |
| [Adhikary et al., 2023, JoulesEye](https://doi.org/10.1145/3631422) | Supporting ideas for tracking and handling motion/occlusion |

These are different studies and datasets. Kwon's BIOPAC reference is not the reference used by this project's dataset. JoulesEye's energy-expenditure error is not respiration-rate error. The current CNN–GRU is this repository's exploratory implementation, not an exact reproduction of these papers.
""")
md("""
## 2. Environment, provenance, and input availability
Run from the repository root or its `docs` directory. Standard processing needs NumPy and Pillow; neural experiments need PyTorch. Notebook display uses IPython and Matplotlib. Install the project's `requirements.txt` in your chosen environment; install notebook/display packages separately if needed.

This edition uses the restored data in `Manuel_clean/`. `RERUN_EXPERIMENTS=True` executes extraction, MLP, and unified evaluation. Set it to `False` before using Run All during a presentation to avoid retraining; saved outputs remain viewable without execution. New reports use notebook-specific filenames. A GPU is optional; `auto` selects CUDA if available.
""")
code("""
from pathlib import Path
import sys, json, hashlib, platform, importlib.util
import numpy as np
from IPython.display import display, Markdown, Image

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / 'thermal_face' / 'data.py').exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_ROOT = ROOT / 'Manuel_clean'
RERUN_EXPERIMENTS = True
DEVICE, EPOCHS, SEED = 'auto', 40, 7
TRACE_DIR = ROOT / 'artifacts' / 'traces'
print('Python:', platform.python_version())
print('NumPy:', np.__version__)
print('Retraining enabled:', RERUN_EXPERIMENTS)
from thermal_face.data import SUBJECTS
availability = []
for person in SUBJECTS:
    frames = len(list((DATA_ROOT / person).glob('frame_*.png')))
    csvs = len(list(DATA_ROOT.glob(f'*{person}_respiration.csv')))
    availability.append((person, frames, csvs))
    print(f'{person:8} | frames present: {frames:5} | respiration CSV files: {csvs}')
RAW_AVAILABLE = all(n > 0 and c == 1 for _, n, c in availability)
print('Complete original dataset available:', RAW_AVAILABLE)
print('Saved trace directory available:', TRACE_DIR.is_dir())
print('PyTorch installed:', importlib.util.find_spec('torch') is not None)
""")
code("""
# Source fingerprints identify the implementation inspected by this presentation.
for relative in ['README.md', 'docs/research_strategy.md',
                 'thermal_face/features.py', 'thermal_face/windows.py',
                 'thermal_face/models.py', 'thermal_face/train_rate.py',
                 'thermal_face/evaluate.py']:
    digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    print(relative, digest)
""")
md("""
## 3. First approach: the legacy image-processing pipeline
The archived scripts preserve an earlier approach inspired by breathing-associated facial regions:

| Order | Operation | Source |
|---|---|---|
| 1 | Load previously extracted 8-bit PNG frames | `legacy/step2_step3.py` |
| 2–3 | Detect frozen runs; derive a perinasal crop from a warm face region | `legacy/step2_step3.py`, `legacy/step3_roi.py` |
| 4 | KLT optical-flow stabilization | `legacy/step4_5_6.py` |
| 5–6 | Rolling pixel standard deviation; normalize to a breathing association map | `legacy/step4_5_6.py` |
| 7 | MRF/ICM segmentation of the averaged map | `legacy/step7_8_9_10.py` |
| 8–10 | Masked waveform, bandpass filtering, peak counting, summaries | `legacy/step7_8_9_10.py` |
| Evaluation | Compare peak/FFT rates with reference events | `legacy/gt_comparison.py` |

Kwon equation 1 specifies sample standard deviation, so the square root is appropriate. The substantive legacy departures are reversed stabilization, whole-recording static masking, and inconsistent timing/signal extraction. These scripts contain machine-specific paths and are shown as historical evidence, not executed here.
""")
code("""
# Executed now: read the archived output. Its claims are corrected immediately below.
legacy_text = (ROOT / 'legacy' / 'gt_comparison_report.txt').read_text(encoding='utf-8')
print('HISTORICAL OUTPUT — known labeling/timing problems; not a valid benchmark')
print(legacy_text)
""")
md("""
## 4. Why the first result was not a trustworthy benchmark
The historical peak-count MAE was **15.40 BPM**, but it cannot be compared directly with the corrected neural evaluation:

- Actual CSV spacing is about 27 Hz; old processing assumed 25 FPS and removed/renumbered frozen frames, compressing time.
- A broad face crop could include chest pixels or poorly localize the nose.
- Counting both markers 1 and 2 approximately doubled Joao's reference rate. The current code counts marker 1 only; the semantics still need independent confirmation.
- The legacy report calls the reference BIOPAC/chest belt. Project research notes identify manually annotated inhalation times instead.
- A 256-sample spectral segment at 25 Hz has approximately 5.86 BPM bin spacing, limiting precision.
- `anestis` has unequal frame and reference-row counts.

**Presenter note:** We corrected the measurement protocol before interpreting model accuracy. Do not claim an improvement “from 15.40 to 2.986” because these are different, non-comparable evaluations.
""")
code("""
from datetime import datetime, timedelta
from thermal_face.data import breath_events, recording_rate
# Synthetic marker example: three inhalation events and three other-phase events.
times = [datetime(2020, 1, 1) + timedelta(seconds=s) for s in range(10)]
markers = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
print('Synthetic example; not participant data')
print('Correct marker-1 event count:', len(breath_events(times, markers)))
print('Full-duration marker-1 rate:', recording_rate(times, markers), 'BPM')
print('Incorrect all-nonzero rate:', 60 * sum(m != 0 for m in markers) / 9, 'BPM')
print('Legacy FFT bin spacing:', 60 * 25 / 256, 'BPM')
""")
md("""
## 5. Audit recordings before building training examples
The table below is a **historical transcription** of the research notes. The following code independently audits the restored files for comparison. Matching row counts are necessary evidence, not proof of timestamp synchronization.

| Person | Frames | CSV rows | Marker-1 events | Full-CSV rate (BPM) | Current use |
|---|---:|---:|---:|---:|---|
| Joao | 3036 | 3036 | 37 | 18.22 | provisional alignment |
| anestis | 2863 | 3526 | 44 | 21.72 | excluded until aligned |
| Claudio | 2903 | 2903 | 38 | 18.62 | provisional alignment |
| Manuel | 2700 | 2700 | 38 | 18.68 | provisional alignment |
| Jaime | 2652 | 2652 | 43 | 21.24 | provisional alignment |

Full-recording rate uses `60 × event count / recording duration`. Later window labels use `60 / median inter-inhalation interval`; these are related but different statistics.
""")
code("""
from thermal_face.data import audit
if RAW_AVAILABLE:
    audit_rows = audit(DATA_ROOT)
    print(json.dumps(audit_rows, indent=2))
else:
    print('NOT RECOMPUTED: original frames and respiration CSVs are absent/incomplete.')
    print('Reproduction command: python -m thermal_face.data --out artifacts/data_audit.json')
""")
md("""
## 6. Corrected feature extraction: retain time, isolate the nose
`config/nose_boxes.json` contains provisional, person-specific manually inspected crops. Divide each crop into a **3 × 3 grid** and average intensity within each patch, producing nine traces per recording. Flag identical-frame runs of at least five frames as missing, while retaining original CSV elapsed time. Skip mismatched recordings.

These fixed crops are not an automatic detector or tracker. Head motion can move the nose outside the box. The current neural path uses these traces; it does not consume legacy MRF masks or full-frame spatial CNN features.
""")
code("""
boxes = json.loads((ROOT / 'config' / 'nose_boxes.json').read_text())
print('Configured crops [x, y, width, height]:')
print(json.dumps(boxes, indent=2))
from thermal_face.features import extract
if RERUN_EXPERIMENTS and RAW_AVAILABLE:
    extract(DATA_ROOT, TRACE_DIR, ROOT / 'config' / 'nose_boxes.json')
else:
    print('Extraction not run: restore inputs and enable RERUN_EXPERIMENTS.')
    print('Reproduction command: python -m thermal_face.features --out artifacts/traces')
""")
md("""
## 7. Resample, normalize, and build supervised windows
- Resample to **10 Hz**, using valid timestamps as interpolation anchors.
- Mark interpolation gaps over **0.25 s** with a tenth, missing-data channel; reject windows with more than 20% missing samples.
- Normalize each nasal channel by its window median and standard deviation (with a scale floor of 1).
- Build **20-second windows every 5 seconds**, each shaped `(10, 200)`.
- Label each window with `60 / median(diff(marker-1 event times))`. Require at least four events, intervals from 0.8–10 s, and rates from 6–45 BPM.
- Exclude `anestis` until alignment is resolved. Treat the first 60 s as still, later windows as moving, and boundary-crossing windows as transition; this is a protocol assumption.

The following cell runs the **actual preprocessing functions on a synthetic sinusoid** to illustrate their behavior. It is not evidence of real-data accuracy.
""")
code("""
from thermal_face.windows import resample_signal, reference_rate, condition_for, load_window_records
from thermal_face.baselines import spectral_rate
t = np.arange(0, 21, 1 / 27)
traces = np.stack([100 + (4 + j / 4) * np.sin(2 * np.pi * 0.3 * t)
                   for j in range(9)], axis=1)
valid = np.ones(len(t), dtype=bool)
target = np.arange(0, 20, 0.1)
x_demo, missing, reason = resample_signal(t, traces, valid, target)
events = np.arange(0.5, 20, 60 / 18)
label, label_reason = reference_rate(events, 0, 20)
estimated, spectral_reason = spectral_rate(x_demo)
print('SYNTHETIC 18 BPM signal')
print('Input shape:', x_demo.shape, '| missing fraction:', missing)
print('Reference:', label, 'BPM | spectral estimate:', estimated, 'BPM')
print('Rejection reasons:', reason, label_reason, spectral_reason)
print('Conditions:', [condition_for(a, a + 20) for a in [0, 50, 60]])
assert x_demo.shape == (10, 200) and abs(estimated - 18) < 0.75
""")
code("""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
def show_plot(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    display(Image(data=buffer.getvalue()))
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(target, x_demo[0], color='#187b8d', label='Normalized nasal patch (synthetic)')
for event in events:
    ax.axvline(event, color='#cc7a00', alpha=0.4)
ax.set(xlabel='Elapsed time (seconds)', ylabel='Normalized intensity',
       title='Synthetic demonstration: thermal oscillation and illustrative event times')
ax.legend(loc='upper right')
show_plot(fig)
""")
code("""
if TRACE_DIR.is_dir() and list(TRACE_DIR.glob('*.npz')):
    records = load_window_records(TRACE_DIR)
    for person, windows in records.items():
        print(person, 'candidate windows:', len(windows),
              'eligible:', sum(w.eligible for w in windows))
else:
    print('Real window generation not recomputed: saved traces are absent.')
""")
md("""
### Restored dataset: original frames and supplied BAP/BAFR outputs
The following montage reads the supplied files; it does not recompute the legacy segmentation. BAF denotes the rolling variability feature, BAP its normalized association map, and BAFR the segmented region. A high association value can also reflect motion and is not a calibrated probability of breathing.

The original frame and legacy maps use different coordinates and image sizes. They are displayed separately, not overlaid as though aligned. The red box on the original frame is the current fixed nose crop. Supplied masks and overlays remain historical processing outputs; they are not ground-truth labels. A reported zero tracking-fallback count is not proof of accurate tracking.
""")
code("""
from PIL import Image as PILImage
from matplotlib.patches import Rectangle
inventory = '| Person | Original | Clean | Stable | BAF | BAP |\\n|---|---:|---:|---:|---:|---:|\\n'
for person in SUBJECTS:
    counts = [len(list((DATA_ROOT / (person + suffix)).glob('*')))
              for suffix in ['', '_clean', '_stable', '_baf', '_bap']]
    inventory += '| ' + person + ' | ' + ' | '.join(map(str, counts)) + ' |\\n'
display(Markdown(inventory))
print('Original frames actually present:', sum(n for _, n, _ in availability))
print('The supplied README_pipeline.md headline says 15,154; the actual folder counts total 14,154.')
fig, axes = plt.subplots(5, 4, figsize=(13, 13))
for row, person in enumerate(SUBJECTS):
    paths = [next(iter(sorted((DATA_ROOT / person).glob('frame_*.png')))),
             DATA_ROOT / f'{person}_bap_mean.png',
             DATA_ROOT / f'{person}_mask.png',
             DATA_ROOT / f'{person}_mask_overlay.png']
    for col, (path, title) in enumerate(zip(paths, ['Original + current crop', 'Supplied mean BAP', 'Supplied BAFR mask', 'Supplied mask overlay'])):
        with PILImage.open(path) as picture:
            axes[row, col].imshow(np.array(picture), cmap='gray')
        axes[row, col].set_title(f'{person}: {title}', fontsize=9)
        axes[row, col].axis('off')
    bx, by, bw, bh = boxes[person]
    axes[row, 0].add_patch(Rectangle((bx, by), bw, bh, fill=False, edgecolor='red', linewidth=1.4))
fig.suptitle('Restored local data — supplied segmentation outputs, not recomputed masks')
fig.tight_layout()
show_plot(fig)
""")
md("""
### Actual extracted nasal traces and annotations
These plots use freshly extracted traces from original frames and CSV timestamps. Orange vertical lines mark marker-1 events. Shaded points indicate frozen-run samples, which are retained on the original time axis but excluded from interpolation anchors. Intensity is not calibrated temperature.
""")
code("""
fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
for ax, person in zip(axes, ['Claudio', 'Jaime', 'Joao', 'Manuel']):
    with np.load(TRACE_DIR / f'{person}.npz') as trace_file:
        seconds, signal, valid, marker = [trace_file[k] for k in ['seconds', 'trace', 'valid', 'marker']]
    ax.plot(seconds, signal.mean(axis=1), linewidth=0.8, color='#187b8d')
    ax.scatter(seconds[~valid], signal.mean(axis=1)[~valid], s=3, color='red', label='Frozen-run samples')
    for event in seconds[marker == 1]:
        ax.axvline(event, color='#cc7a00', alpha=0.25)
    ax.set(title=person, ylabel='Mean intensity')
axes[-1].set_xlabel('Elapsed seconds from CSV timestamps')
axes[0].legend()
fig.suptitle('Actual data: nasal intensity and annotated inhalation events')
fig.tight_layout()
show_plot(fig)
""")
md("""
## 8. First learned model: spectral MLP
The MLP takes normalized FFT magnitudes from the nine traces in the 0.1–0.75 Hz band, plus missing fraction. Training-only feature standardization prevents using the held-out person's distribution to fit the model. A 16-unit tanh hidden layer predicts rate; the NumPy implementation trains for 800 updates with seed 7.

**Historical finding:** macro MAE **4.80 BPM**, versus **2.50 BPM** for the training-median comparator. The MLP did not establish added value. Its result motivates investigating temporal features and localization; it does not prove that model size is the problem.
""")
code("""
if RERUN_EXPERIMENTS and TRACE_DIR.is_dir():
    from thermal_face.train_mlp import run as run_mlp
    mlp_report = run_mlp(TRACE_DIR, ROOT / 'artifacts' / 'notebook_mlp.json')
else:
    print('MLP training not rerun. Historical MAE: 4.80 BPM; median baseline: 2.50 BPM.')
    print('Reproduction command: python -m thermal_face.train_mlp --traces artifacts/traces --out artifacts/notebook_mlp.json')
""")
md("""
## 9. Temporal CNN and CNN + GRU
The CNN learns local patterns along time. The GRU integrates the resulting sequence. Both receive **nine traces plus a missing-data channel**, not images.

| CNN + GRU stage | Output dimensions (excluding batch) |
|---|---|
| Input | 10 channels × 200 samples |
| Conv1D 10→24, kernel 9, ReLU, average pool /2 | 24 × 100 |
| Conv1D 24→32, kernel 9, ReLU, average pool /2 | 32 × 50 |
| Transpose, GRU with hidden size 32 | 50 steps × 32 features → final hidden state |
| Linear 32→16, ReLU, Linear 16→1 | one rate in BPM |

The comparison CNN adds a third convolution and averages across time before its regression head. Training uses **AdamW**, learning rate 0.001, weight decay 0.01, batch size 16, Smooth L1 loss (beta 3), 40 epochs, and seed 7. The complete window is required; GRU state resets between windows. The missing channel is a feature, not a recurrent-state mask.
""")
code("""
# Executed architecture inspection; random weights below are not trained results.
if importlib.util.find_spec('torch'):
    import torch
    from thermal_face.models import CNNRateNet, CNNGRURateNet
    print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
    for cls in [CNNRateNet, CNNGRURateNet]:
        model = cls().eval()
        with torch.no_grad():
            output = model(torch.from_numpy(x_demo[None]))
        print(cls.__name__, '| parameters:', sum(p.numel() for p in model.parameters()),
              '| output shape:', tuple(output.shape))
    print(CNNGRURateNet())
else:
    print('Architecture inspection skipped: PyTorch is not installed.')
""")
md("""
## 10. Evaluation that keeps people separate
Use **leave-one-person-out** evaluation: train on three people, test on the fourth, and repeat. Every overlapping window from a person remains in that person's fold. Never randomly split overlapping windows across train/test.

The training-median baseline predicts the median training-window label for every held-out window. It is an essential check because the subjects' rates occupy a narrow range. The spectral baseline detrends, applies a Hann taper, and averages normalized nasal-patch power in 6–45 BPM. Eightfold zero padding gives denser frequency samples, not better underlying resolution (about 3 BPM for 20 s).

**MAE** is mean absolute predicted-minus-reference error. **Macro MAE** first computes each person's MAE, then averages across people so each person has equal weight. Pooled window MAE instead weights people by window count.
""")
code("""
people = ['Claudio', 'Jaime', 'Joao', 'Manuel']
for held_out in people:
    training = [p for p in people if p != held_out]
    assert held_out not in training
    print('Train:', ', '.join(training), '| test:', held_out)
print('Four independent held-out people; 77 overlapping windows are not 77 independent subjects.')
""")
md("""
### Reproduce the full evaluation
This cell trains both neural architectures and compares them with spectral and median baselines. It writes fold membership, per-window predictions, source-trace hashes, environment information, exclusions, coverage, and condition summaries. `DEVICE='cuda'` explicitly requires CUDA; `auto` allows CPU.

The commands in the repository were documented as prior runs; the cell below is the actual runnable equivalent used by this notebook when enabled. No original terminal-session history is available.
""")
code("""
report = None
if RERUN_EXPERIMENTS and TRACE_DIR.is_dir() and importlib.util.find_spec('torch'):
    from time import perf_counter
    from thermal_face.evaluate import run as evaluate
    evaluation_started = perf_counter()
    report = evaluate(TRACE_DIR, ROOT / 'artifacts' / 'notebook_evaluation.json',
                      epochs=EPOCHS, seed=SEED, device=DEVICE, verbose=True)
    print('Fresh evaluation wall time (seconds):', round(perf_counter() - evaluation_started, 2))
    print(json.dumps(report['overall'], indent=2))
else:
    print('Full training/evaluation NOT rerun in this presentation build.')
    print('python -m thermal_face.evaluate --device auto --epochs 40 --seed 7 --out artifacts/notebook_evaluation.json')
""")
md("""
### Fresh evaluation on the restored data
The following outputs are newly computed by the notebook, using the actual dataset. These are distinct from the historical comparison in section 11. Training time below is the wall time of this run, not an inferred historical duration.
""")
code("""
if report is not None:
    print('Environment:', json.dumps(report['environment'], indent=2))
    print('Protocol:', json.dumps(report['protocol'], indent=2))
    summary = report['overall']
    print('Coverage:', summary['eligible_windows'], '/', summary['candidate_windows'])
    print('Reference rejections:', summary['reference_rejections'])
    table = '| Method | Overall macro MAE | Still | Moving | Transition |\\n|---|---:|---:|---:|---:|\\n'
    for method, metrics in summary['common_comparison']['methods'].items():
        vals = [metrics['macro_mae_bpm']] + [report['by_condition'][c]['common_comparison']['methods'][method]['macro_mae_bpm'] for c in ['still', 'moving', 'transition']]
        table += '| ' + method + ' | ' + ' | '.join(f'{v:.3f}' if v is not None else 'N/A' for v in vals) + ' |\\n'
    display(Markdown('**Fresh run: MAE in BPM**\\n\\n' + table))
    median_error = summary['common_comparison']['methods']['train_median']['macro_mae_bpm']
    gru_error = summary['common_comparison']['methods']['cnn_gru']['macro_mae_bpm']
    print('Fresh-run interpretation: CNN + GRU', 'beats' if gru_error < median_error else 'does not beat', 'the median baseline overall.')
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for ax, person in zip(axes, ['Claudio', 'Jaime', 'Joao', 'Manuel']):
        rows = [r for r in report['windows'] if r['subject'] == person and r['eligible']]
        centers = [(r['start_s'] + r['end_s']) / 2 for r in rows]
        ax.plot(centers, [r['reference_bpm'] for r in rows], 'o-', label='Reference', color='black')
        for method, label in [('cnn_gru', 'CNN + GRU'), ('train_median', 'Training median')]:
            ax.plot(centers, [r['predictions_bpm'][method] for r in rows], '.--', label=label)
        ax.axvline(60, color='gray', linestyle=':')
        ax.set(title=person, ylabel='BPM')
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel('Window center (seconds); each point summarizes a 20-second window')
    fig.suptitle('Fresh held-out-person predictions on actual data')
    fig.tight_layout()
    show_plot(fig)
else:
    print('No fresh report in this execution; enable RERUN_EXPERIMENTS to reproduce it.')
""")
md("""
## 11. Recorded results and interpretation
The following tables and plots use **historical values transcribed from `docs/research_strategy.md`**, for 40 epochs and seed 7. They are deliberately independent of the fresh run above. Original historical neural JSON artifacts were not supplied; new per-window predictions are saved in `artifacts/notebook_evaluation.json`.

The documented CPU and GPU runs agree at three decimal places. Using a GPU changes runtime, not evidence of model accuracy. No training-duration claim is made because complete logs are unavailable.
""")
code("""
historical = {
    'Training median': [2.501, 1.944, 2.711, 3.527],
    'Spectral': [10.618, 7.192, 14.046, 12.771],
    'CNN': [3.577, 2.787, 3.832, 5.452],
    'CNN + GRU': [2.986, 2.286, 3.573, 3.885],
}
table = '| Method | Overall | Still | Moving | Transition |\\n|---|---:|---:|---:|---:|\\n'
for method, values in historical.items():
    table += '| ' + method + ' | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |\\n'
display(Markdown('**Historical macro MAE, BPM (lower is better)**\\n\\n' + table))
fig, ax = plt.subplots(figsize=(9, 3.6))
names = list(historical)
values = [historical[n][0] for n in names]
bars = ax.barh(names, values, color=['#267764', '#9299a3', '#4482ad', '#b07935'])
ax.bar_label(bars, fmt='%.3f', padding=4)
ax.set(xlim=(0, 12), xlabel='Macro MAE (breaths/minute)',
       title='Historical documented results — not rerun measurements')
ax.invert_yaxis()
show_plot(fig)
""")
md("""
### Per-person errors from the recorded pilot
| Held-out person | CNN + GRU | CNN | Median baseline |
|---|---:|---:|---:|
| Claudio | 2.565 | 4.276 | 1.956 |
| Jaime | 4.241 | 3.830 | 2.626 |
| Joao | 1.653 | 3.104 | 2.146 |
| Manuel | 3.484 | 3.097 | 3.277 |

All values are MAE in BPM. CNN + GRU improves aggregate error relative to this CNN run, but **does not beat the training-median baseline overall**. It beats that baseline only for Joao. These single-seed results do not establish reliable generalization or clinical suitability.

### Coverage and exclusions
The historical evaluation used **77/84 candidate windows (91.7%)**: still 36/36, moving 29/36, transition 12/12. Six windows had implausible reference-event intervals; one had fewer than four events. No candidate window failed signal screening in that run.

Coverage includes reference-label screening, so it is **not live system availability**. Missing labels especially affect moving-period results. Spectral error alone does not prove a particular failure cause; inspect crops and labels before drawing that conclusion.
""")
md("""
## 12. Verification performed for this notebook
Run the repository tests and report their real output. Tests verify implementation behavior, not research validity. The source notes describe historical test passes; the output below records only what this notebook actually executes in its current environment. CUDA-dependent tests may skip when CUDA is unavailable.
""")
code("""
import subprocess, os
# Keep temporary test artifacts inside this writable checkout.
test_temp = ROOT / 'artifacts' / 'notebook_test_tmp'
test_temp.mkdir(parents=True, exist_ok=True)
test_env = dict(os.environ, TMP=str(test_temp), TEMP=str(test_temp))
test_run = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                          cwd=ROOT, env=test_env, capture_output=True, text=True)
print(test_run.stdout)
print(test_run.stderr)
print('Test process exit code:', test_run.returncode)
if test_run.returncode:
    print('Some checks failed or dependencies are unavailable; inspect the output before claiming verification.')
""")
md("""
## 13. Source-code appendix: implementation used by the walkthrough
The following source snapshots preserve the exact current Python implementation for explanation without opening other files. They are **reference listings**, not extra executed training cells. The executable cells above import these same modules from the repository. Legacy code remains in `legacy/`; it is not the corrected benchmark implementation.
""")
for name in ['data', 'features', 'windows', 'baselines', 'train_mlp', 'models', 'train_rate', 'evaluate']:
    path = ROOT / 'thermal_face' / f'{name}.py'
    md(f"### `thermal_face/{name}.py`\n\n```python\n{path.read_text(encoding='utf-8')}\n```")
tracking_summary = ROOT / 'docs' / 'tracking_experiment_results.md'
if tracking_summary.exists():
    md(tracking_summary.read_text(encoding='utf-8'))
md("""
## 14. Limitations
1. **Very small dataset:** five recordings, four usable people, one session each. Overlapping windows inflate sample count without adding independent people.
2. **Synchronization is provisional:** equal frame/CSV row counts are not independent proof of alignment. `anestis` remains excluded.
3. **Reference uncertainty:** marker semantics and seven rejected windows need original annotation guidance and review.
4. **Fixed crops:** person-specific nose boxes do not adapt to motion, new people, pose, or occlusion. No automatic detector/tracker has been validated.
5. **Limited thermal information:** 8-bit grayscale values cannot recover calibrated 16-bit temperatures.
6. **No demonstrated baseline improvement:** CNN + GRU remains worse than the training-median predictor overall; a single seed is insufficient for robust conclusions.
7. **Assumed conditions:** the 60-second still/moving boundary requires verification. Coverage includes label screening and is not deployment availability.
8. **Historical provenance is incomplete:** the restored folder provides original frames, reference CSVs and legacy outputs. Original historical neural JSON reports and complete execution logs were not supplied. This edition regenerates traces and neural reports, while keeping historical figures explicitly labeled.
9. **Deployment is unfinished:** training saves reports but not model checkpoints; streaming inference, quality confidence, recovery, and prospective validation are not established.
10. **Identity recognition is future work:** repeated sessions and a separate identity evaluation are needed. Existing results concern respiration only.

## 15. What to do next, in order
| Priority | Action | Evidence of completion |
|---|---|---|
| 1 | Preserve the restored inputs and newly generated reports; recover original timestamp/annotation documentation | Dataset inventory and hashes retained; this notebook now reproduces real outputs |
| 2 | Verify marker meaning and frame synchronization against annotation instructions / ROS timestamps | Reviewed alignment record per subject; resolve `anestis` before inclusion |
| 3 | Generate a visual diagnostics report | Nose-box overlays, traces, event markers, largest errors, and all seven rejected windows reviewed |
| 4 | Improve nose localization and tracking with explicit loss flags | Fixed-vs-tracked comparison on the same held-out people, reporting error and coverage |
| 5 | Save fold checkpoints and preprocessing metadata | Reloaded inference reproduces predictions without retraining |
| 6 | Repeat predetermined seeds and collect more people/sessions | Stable comparison against median and spectral baselines; uncertainty measured over people |
| 7 | Expand the model only after data/localization validation | Event or waveform supervision where available; prospective independent test set |
| 8 | Develop identity recognition separately | Consented repeated sessions, held-out-session testing, and unknown-person rejection |

### Closing statement for the presentation
“We built and audited a contactless thermal respiration pipeline and evaluated temporal neural models on held-out people. The recurrent model improved over the CNN in the documented run, but did not beat a simple median predictor. Our next priority is verified labels, synchronization, tracking, and more independent recordings before claiming reliable generalization.”
""")

def build(out_stem='thermal_respiration_walkthrough'):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from traitlets.config import Config
    configuration = Config()
    configuration.HistoryManager.hist_file = ':memory:'
    shell = InteractiveShell.instance(config=configuration)
    count = 0
    for cell in cells:
        if cell['cell_type'] != 'code':
            continue
        count += 1
        started = time.perf_counter()
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell['source'], store_history=False)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f"Cell {count} failed: {captured.stdout} {captured.stderr}")
        cell['execution_count'] = count
        cell['metadata']['execution_seconds'] = round(time.perf_counter() - started, 3)
        for key in ['stdout', 'stderr']:
            value = getattr(captured, key)
            if value:
                cell['outputs'].append(dict(output_type='stream', name=key, text=value))
        for output in captured.outputs:
            cell['outputs'].append(dict(output_type='display_data', data=output.data, metadata=output.metadata))
        print(f'Executed cell {count}', flush=True)
    for i, cell in enumerate(cells):
        cell['id'] = f'walkthrough-{i:03d}'
    notebook = dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version=__import__('platform').python_version()),
        presentation_provenance='Historical metrics are explicitly labeled; all code-cell outputs executed during build.'
    ), nbformat=4, nbformat_minor=5)
    out = ROOT / 'docs' / f'{out_stem}.ipynb'
    if out_stem == 'thermal_respiration_walkthrough' and (out.parent / 'tracking_experiment.ipynb').exists():
        from integrate_tracking import integrate
        notebook = integrate(notebook, json.loads((out.parent / 'tracking_experiment.ipynb').read_text(encoding='utf-8')))
    if out_stem == 'thermal_respiration_walkthrough' and (out.parent / 'kwon_reproduction.ipynb').exists():
        from integrate_kwon import integrate as integrate_kwon
        notebook = integrate_kwon(notebook, json.loads((out.parent / 'kwon_reproduction.ipynb').read_text(encoding='utf-8')))
    render(notebook, out)


def render(notebook, out):
    """Export saved notebook outputs without rerunning scientific experiments."""
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
    import markdown
    rendered = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            rendered.append(markdown.markdown(cell['source'], extensions=['tables', 'fenced_code']))
        else:
            rendered.append(f"<div class='cell'><small>Executed cell {cell['execution_count']}</small><pre>{html.escape(cell['source'])}</pre>")
            for output in cell['outputs']:
                if output['output_type'] == 'stream':
                    rendered.append('<pre class="output">' + html.escape(output['text']) + '</pre>')
                else:
                    data = output['data']
                    if 'image/png' in data:
                        rendered.append('<img src="data:image/png;base64,' + data['image/png'] + '">')
                    elif 'text/markdown' in data:
                        rendered.append(markdown.markdown(data['text/markdown'], extensions=['tables']))
                    elif 'text/html' in data:
                        rendered.append(data['text/html'])
                    elif 'text/plain' in data:
                        rendered.append('<pre>' + html.escape(data['text/plain']) + '</pre>')
            rendered.append('</div>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Thermal respiration walkthrough</title><style>
body{font:17px/1.65 system-ui,sans-serif;max-width:1100px;margin:48px auto;padding:0 28px;color:#253444;background:#fafcfe}h1,h2,h3{line-height:1.25;color:#123f51}h2{margin-top:56px;border-top:2px solid #dce7ed;padding-top:24px}table{border-collapse:collapse;width:100%;font-size:15px}td,th{border:1px solid #cddbe3;padding:9px;text-align:left}th{background:#e6f0f4}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf2f6;padding:16px;font-size:13px;line-height:1.5}code{font-size:.88em}.cell{border-left:3px solid #328698;padding-left:16px;margin:24px 0}.output{background:#fff;border:1px solid #dce7ed}img{max-width:100%}a{color:#096c88}small{color:#667887}@media print{body{font-size:11pt;margin:0}pre{font-size:8pt}h2{break-after:avoid}tr,img{break-inside:avoid}}
</style><body>''' + '\n'.join(rendered) + '</body></html>'
    out.with_suffix('.html').write_text(page, encoding='utf-8')
    print(f"Wrote {out}; {len(notebook['cells'])} cells, "
          f"{sum(c['cell_type'] == 'code' for c in notebook['cells'])} code cells with saved outputs")

if __name__ == '__main__':
    build()
