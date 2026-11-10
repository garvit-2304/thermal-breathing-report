## LCAS thermal respiration reconstruction

The second project is included in [`lcas-thermal-respiration-reconstruction/`](./lcas-thermal-respiration-reconstruction/). Its scripts, workflow documentation, data setup notes, requirements, and notebooks are kept together in that folder. The original reconstruction report PDF remains available in the [source repository](https://github.com/garvit-2304/lcas-thermal-respiration-reconstruction/blob/main/reports/thermal_respiration_project_report.pdf).

---

# Thermal face respiration research

The project aims to build a breathing-rate detector with facial breathing-region segmentation from thermal video. Kwon et al. guides the segmentation work; JoulesEye informs nasal tracking during movement. Data processing, signal-based estimation, CNN/GRU comparisons, tracking and segmentation are all part of this project. The complete detector is not yet validated.

The corrected KLT → four-second sample-SD BAF → BAP → per-frame MRF implementation is in `thermal_face/kwon.py`. See [protocol](docs/kwon_reproduction_protocol.md) and [executed reproduction notebook](docs/kwon_reproduction.ipynb), also integrated into the original walkthrough. The first corrected run has **0/84 eligible MRF evaluation windows** under the declared temporal-support rules, so segmentation improvement is **unmeasured**, not disproved. Recovering valid continuous recordings and validating masks are the next priorities. This dataset differs from the paper's; published performance has not been reproduced.

The repository contains **five people, one recording each**, with four usable frame/reference pairs. It remains a research pilot.

The [31 October exploratory follow-up](docs/short_segment_results.md) evaluates 12-second supported portions of the same recordings. On 14 paired still windows from four people, spectral macro MAE changes from 4.874 to 2.564 breaths/min with dynamic MRF; the moving condition does not improve. This short-window result does not replace the original unevaluable 20-second comparison. See the [locked protocol](docs/short_segment_protocol.md) and [executed notebook](docs/short_segment_experiment.ipynb).

A further [fixed perinasal-mask experiment](docs/constrained_mask_results.md) tested whether excluding obvious eye and face-edge pixels improves the original MRF trace. It did not: still spectral error rose from 2.564 to 3.350 breaths/min on the same 14 short windows, and overall eligible short windows fell from 29 to 23. The original 20-second test remains unevaluable. The constraint is retained as a negative ablation, not the default pipeline. A local 12-frame [annotation packet guide](docs/mask_annotation_guide.md) prepares independent localization and mask review across still and moving frames; no human labels have been collected yet.

## What was wrong with the previous result

The saved legacy result reports 15.40 breaths/min MAE from peak counting. It is not a trustworthy model comparison:

- The original dataset is recorded at about **27 Hz** with per-frame CSV timestamps. Legacy code assumes 25 fps and renumbers frames after deleting frozen runs, compressing elapsed time.
- The legacy face box includes much of the chest and sometimes starts at the nose. A mean over this box is not a clean nasal temperature trace.
- Joao has marker `1` and marker `2` events. The old evaluator counts both as breaths, giving 36.4 breaths/min. Counting only marker `1` gives **18.22 breaths/min** for the full CSV.
- The old FFT uses a 256-sample Welch segment: its frequency bins are about 5.86 breaths/min apart at 25 Hz. All five reported FFT estimates round to 11.7 breaths/min.
- This dataset's respiration reference is **manually annotated inhalation times**, according to Cosar et al. It is not a BIOPAC chest belt recording. The BIOPAC setup belongs to the separate Kwon et al. paper.
- `anestis` has 2,863 thermal frames but 3,526 reference rows. Frame-to-reference alignment cannot be assumed for this recording.

Run `python -m thermal_face.data --out artifacts/data_audit.json` to regenerate the audit. The old 15.40 figure is kept in `legacy/gt_comparison_report.txt` for provenance and must not be compared to a new model as a corrected benchmark.

## Layout

The presentation walkthrough is available in
[`docs/thermal_respiration_walkthrough.ipynb`](docs/thermal_respiration_walkthrough.ipynb)
and its browser-readable [HTML edition](docs/thermal_respiration_walkthrough.html).
The restored dataset is nested under `Manuel_clean/` (which contains all five
people, not just Manuel), together with supplied legacy BAF/BAP maps and BAFR
masks. The notebook keeps these supplied maps separate from freshly computed
timestamp-preserving traces and held-out-person evaluation results.

To process this nested dataset without moving or changing its source files:

```powershell
python -m thermal_face.data --root Manuel_clean --out artifacts/data_audit.json
python -m thermal_face.features --root Manuel_clean --boxes config/nose_boxes.json --out artifacts/traces
```

- `Joao/`, `anestis/`, `Claudio/`, `Manuel/`, `Jaime/`: original frames, retained locally.
- `*_respiration.csv`, `*_heartbeat.csv`: original reference files.
- `thermal_face/`: current data audit, feature extraction, and temporal neural network pilot.
- `config/nose_boxes.json`: provisional nose crops inspected on each existing subject.
- `docs/research_strategy.md`: paper synthesis, evaluation protocol, and roadmap.
- `legacy/`: old scripts and reports, retained as a record of the prior pipeline.
- `artifacts/`: generated reports and compressed traces; ignored by Git.

The provisional boxes in `config/nose_boxes.json` are dataset-specific annotations. They must be replaced by a detector and tracker before testing new people or camera positions.

The measured spectral MLP pilot achieved **4.80 breaths/min macro MAE** across four held-out people, while a training-median predictor achieved **2.50**. See `artifacts/rate_mlp_pilot.json`. This does not establish a useful learned model; it identifies the data and localization work needed before expanding model size.

The initial CNN + GRU run (40 epochs, seed 7) achieved **2.986 breaths/min
macro MAE**, compared with **3.577** for the CNN and **2.501** for the median
baseline on the same four people and 77 windows. The recurrent model does not
yet beat the simple baseline. See `docs/research_strategy.md` for per-person
results, architecture details, limitations, and the next experiments.

## Run the pilot

Use Python 3.10-3.12 in a virtual environment. Install NumPy and Pillow for extraction and the spectral MLP. PyTorch is needed for the temporal CNN and CNN + GRU experiments. `requirements.txt` lists all three.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m thermal_face.data --out artifacts/data_audit.json
python -m thermal_face.features --out artifacts/traces
python -m thermal_face.train_mlp --traces artifacts/traces --out artifacts/rate_mlp_pilot.json
python -m thermal_face.train_rate --architecture cnn-gru --epochs 40 --seed 7
python -m thermal_face.train_rate --architecture cnn --epochs 40 --seed 7
python -m thermal_face.evaluate --epochs 40 --seed 7
python -m unittest discover -s tests -v
```

The feature extractor reads the original frame folders and retains CSV time. It flags long identical-frame runs as missing, resamples the remaining samples, and records a missing-data channel. It refuses to use `anestis` until its reference alignment is established. The neural pilots use nine nasal patch traces in 20-second windows and report **leave-one-person-out** MAE alongside a training-median baseline. `train_mlp` is a small spectral MLP; `train_rate` defaults to a temporal CNN + GRU; `--architecture cnn` selects the original CNN comparison. Each architecture writes a separate report with its seed, epochs, parameter count, and per-window predictions. The recurrent model uses two convolution/pooling blocks, a 32-unit GRU, and a small rate-regression head. Overlapping windows are correlated; the four held-out people, not the number of windows, determine how much evidence the result provides. Training may underperform the baseline and should be reported as such.

## Evaluate all methods

`python -m thermal_face.evaluate` writes `artifacts/evaluation.json`, comparing
CNN + GRU, CNN, a timestamp-aware spectral estimate and the training-median
baseline. It reports still/moving/transition errors, per-subject metrics,
per-window predictions, coverage and exclusion reasons. The 60-second movement
boundary is assumed from the protocol and configurable with `--still-until-s`.

At 40 epochs and seed 7, all methods cover 77/84 candidate windows (91.7%).
Macro MAE is 2.501 BPM for the median baseline, 2.986 for CNN + GRU, 3.577 for
CNN and 10.618 for the spectral estimate. Seven windows fail reference-event
checks; coverage includes label screening and does not describe live availability.
See `docs/research_strategy.md` for the condition breakdown and remaining gates.

## Train on the NVIDIA GPU

The trainer and evaluator accept `--device auto|cpu|cuda`. The default `auto`
uses CUDA when available. Explicit `cuda` fails if CUDA is unavailable rather
than switching to CPU. Reports record the actual device, GPU name, PyTorch
version and CUDA runtime.

For this machine's RTX 4060, install the CUDA build in the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade "torch==2.14.0+cu130" --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m thermal_face.evaluate --device cuda --epochs 40 --seed 7 --out artifacts/evaluation_gpu.json
```

To run only CNN + GRU:

```powershell
.\.venv\Scripts\python.exe -m thermal_face.train_rate --architecture cnn-gru --device cuda --epochs 40 --seed 7 --out artifacts/rate_cnn_gru_gpu.json
```

The CUDA package comes from the [official PyTorch wheel index](https://download.pytorch.org/whl/cu130/torch/).
CPU remains selectable with `--device cpu`. The models and all neural training
and inference tensors use the selected device; trace loading and the spectral
baseline use NumPy on CPU. CPU and CUDA numerical results need not be identical.
Training currently saves evaluation reports, not model checkpoints.
Both commands print flushed startup, device, fold and epoch progress (first, every
10th, and final epoch), with training loss and elapsed time. Ctrl+C stops the run
with a concise interruption message; rerunning starts training again.

## Optional CSRT tracking experiment

`thermal_face.tracking` adapts JoulesEye's nasal-region tracking idea using the
existing first-frame nose boxes. It requires `requirements-tracking.txt` and
does not require RGB; consequently it does not reproduce JoulesEye's RGB
initialization/recovery. Failed boxes become missing data, never fixed-crop
fallbacks. CSRT can drift while reporting success, so inspect the overlays.

```powershell
python -m thermal_face.tracking --root Manuel_clean --boxes config/nose_boxes.json --out artifacts/tracking_experiment --fixed-reference artifacts/traces
python -m thermal_face.compare_tracking --traces artifacts/tracking_experiment --epochs 40 --seeds 7 17 27 --device cpu
python docs/build_tracking_report.py
```

The comparison trains and tests both crop variants on identical eligible
windows. The [protocol](docs/tracking_experiment_protocol.md) specifies the
comparison before error inspection. The [results](docs/tracking_experiment_results.md)
and [executed notebook](docs/tracking_experiment.ipynb) report errors, coverage,
seed sensitivity, and sampled overlays. Fixed crops remain the default pipeline.

## Data handling

Original frames and CSVs are local data and excluded by `.gitignore`. Do not publish face recordings, identity labels, or trained recognition embeddings without confirming participant consent and dataset terms. The image files are 8-bit RGB containers with identical grayscale channels; they are not recoverable 16-bit temperature measurements. Do not convert these intensities into degrees Celsius.
