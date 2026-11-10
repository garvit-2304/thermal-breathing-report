# Thermal Breathing Rate Detection and Facial Region Segmentation

This repository brings together two related research efforts on contactless breathing analysis from thermal video. The shared goal is to build a breathing-rate detector and study how facial regions associated with breathing can be located and segmented.

## Project overview

The work combines two complementary pipelines:

1. **Thermal breathing-rate detection.** This pipeline audits the L-CAS recordings, preserves image timestamps, extracts nasal and facial-region signals, and compares classical rate estimators with CNN and CNN+GRU models. It also explores movement tracking, graph filtering, and inhalation-event detection.
2. **BAFR segmentation reconstruction.** This pipeline studies the breathing-associated facial region (BAFR) using perinasal regions of interest, temporal breathing-associated features, breathing-associated probabilities, and Markov Random Field (MRF) segmentation. It follows ideas from Kwon et al. and tests how these steps behave on the available L-CAS data.

The two pipelines use related recordings but have different processing choices and evaluation setups. Their scores should be read within the experiment that produced them, rather than treated as one directly comparable benchmark.

## Research references

- Kwon et al., [“Breathing-Associated Facial Region Segmentation for Thermal Camera-Based Indirect Breathing Monitoring”](https://doi.org/10.1109/JTEHM.2023.3295775), informs the BAFR segmentation work.
- Coşar et al., [“Thermal Camera Based Physiological Monitoring with an Assistive Robot”](https://doi.org/10.1109/EMBC.2018.8513276), describes the L-CAS thermal respiration data and its annotated breathing references.
- The JoulesEye paper informs the nasal-region tracking experiments, particularly tracking through head movement.

## What the repository contains

- `thermal_face/`: thermal data loading, timestamp-aware signal processing, breathing-rate models, tracking, and Kwon-inspired segmentation.
- `lcas-thermal-respiration-reconstruction/`: the companion reconstruction project, including scripts, setup notes, workflow documentation, and its report.
- `config/`: provisional facial and nasal region definitions used in experiments.
- `docs/`: experiment protocols, dataset audits, results, limitations, and notebooks.
- `legacy/`: earlier scripts and outputs retained for reference.
- `reports/`: project reports and presentation materials.

## Data and privacy

The original L-CAS recordings and participant-level files are not included in the repository. Obtain and use the dataset under its applicable terms, then place the source files in the local paths described in the relevant project documentation.

The original thermal frames, ROS bags, participant CSV files, and generated face previews are excluded from Git. Do not upload them without confirming that sharing is permitted. Thermal recordings can still contain identifiable facial information.

## Current status

The repository contains working experiments, not a clinically validated breathing monitor. The recording set is small, region labels are limited, and the available reference signals do not support every comparison reported in the papers. Segmentation coverage and breathing-rate error depend on the chosen window length, signal estimator, and quality checks.

The results show mixed effects: MRF segmentation can increase the number of usable windows and improve spectral estimates in one comparison, while peak-count estimates may worsen. Other model and filtering experiments also have limitations. Reports and protocols document the evaluation conditions so that results are not presented as stronger than the evidence supports.

## Getting started

Use Python 3.10–3.12 and create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The root requirements support the main thermal pipeline. Some optional tracking and segmentation experiments use additional packages listed in `requirements-tracking.txt` and `requirements-kwon.txt`.

See the [main workflow notebook](docs/thermal_respiration_walkthrough.ipynb), the [research strategy](docs/research_strategy.md), and the companion project's [workflow guide](lcas-thermal-respiration-reconstruction/docs/workflow.md) for data setup and experiment commands. Individual scripts also provide command-line help.

## Next steps

- Collect more recordings and validate on people who were not used to set up the current pipeline.
- Add independent pixel-level labels for breathing-associated facial regions.
- Compare all methods on the same timestamp-aligned windows and report both error and coverage.
- Improve tracking recovery and assess performance during head movement.
- Validate the reference annotations and synchronization before drawing broader conclusions.
- Keep the dataset files local unless its terms explicitly allow redistribution.
