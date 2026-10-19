"""Evaluate the Kwon-style mask/no-mask experiment without training a neural model."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import signal

from .baselines import spectral_rate
from .kwon import VARIANTS
from .windows import load_window_records


def peak_rate(x):
    wave = signal.detrend(x[0].astype(float))
    if np.std(wave) < 1e-8:
        return None, 'flat_signal'
    sos = signal.butter(4, [0.1, 0.75], btype='bandpass', fs=10, output='sos')
    filtered = signal.sosfiltfilt(sos, wave)
    if np.std(filtered) < 1e-8:
        return None, 'flat_filtered_signal'
    peaks, _ = signal.find_peaks(filtered, distance=8, prominence=0.4*np.std(filtered))
    if len(peaks) < 2:
        return None, 'fewer_than_two_peaks'
    return float(len(peaks)*60/(len(wave)/10)), None


def metrics(rows, estimator):
    errors = {}
    for row in rows:
        prediction = row['predictions_bpm'][estimator]
        if row['eligible'] and prediction is not None:
            errors.setdefault(row['subject'], []).append(abs(prediction-row['reference_bpm']))
    per = {p: float(np.mean(e)) for p, e in errors.items()}
    return {'macro_mae_bpm': float(np.mean(list(per.values()))) if per else None,
            'pooled_mae_bpm': float(np.mean([e for values in errors.values() for e in values])) if per else None,
            'per_subject_mae_bpm': per, 'people': len(per), 'windows': sum(map(len, errors.values()))}


def pairwise(a, b, estimator, condition=None):
    def index(rows):
        return {(r['subject'], r['start_s']): r for r in rows
                if condition is None or r['condition'] == condition}
    ia, ib = index(a), index(b)
    if ia.keys() != ib.keys():
        raise ValueError('Candidate windows differ')
    for key in ia:
        if (ia[key]['reference_bpm'], ia[key]['end_s']) != (ib[key]['reference_bpm'], ib[key]['end_s']):
            raise ValueError('Reference labels differ')
    keys = [key for key in ia if all(index[key]['eligible'] and index[key]['predictions_bpm'][estimator] is not None
                                   for index in [ia, ib])]
    first, second = metrics([ia[k] for k in keys], estimator), metrics([ib[k] for k in keys], estimator)
    a_mae, b_mae = first['macro_mae_bpm'], second['macro_mae_bpm']
    return {'first': first, 'second': second, 'common_windows': len(keys), 'candidates': len(ia),
            'second_minus_first_bpm': b_mae-a_mae if a_mae is not None and b_mae is not None else None,
            'window_keys': [[p, t] for p, t in keys]}


def run(folder: Path, out: Path):
    variants = {}
    for variant in VARIANTS:
        records = load_window_records(folder / variant)
        rows = []
        for person, windows in records.items():
            for window in windows:
                predictions, reasons = {}, {}
                for estimator, function in [('spectral', spectral_rate), ('peak_count', peak_rate)]:
                    predictions[estimator], reasons[estimator] = function(window.x) if window.eligible else (None, 'ineligible_window')
                rows.append(window.metadata() | {'predictions_bpm': predictions, 'estimator_reasons': reasons})
        variants[variant] = {
            'windows': rows, 'candidates': len(rows), 'eligible': sum(r['eligible'] for r in rows),
            'signal_exclusions': dict(Counter(r['signal_rejection'] for r in rows if r['signal_rejection'])),
            'reference_exclusions': dict(Counter(r['reference_rejection'] for r in rows if r['reference_rejection'])),
            'own_coverage_metrics': {method: metrics(rows, method) for method in ['spectral', 'peak_count']},
            'trace_sha256': {person: hashlib.sha256((folder / variant / f'{person}.npz').read_bytes()).hexdigest() for person in records},
        }
    comparisons = {}
    for name, a, b in [('segmentation', 'klt_roi', 'klt_mrf'),
                       ('stabilization', 'fixed_roi', 'klt_roi'),
                       ('warp_correction', 'wrong_warp_mrf', 'klt_mrf')]:
        comparisons[name] = {'first': a, 'second': b}
        for method in ['spectral', 'peak_count']:
            comparisons[name][method] = {condition or 'overall': pairwise(variants[a]['windows'], variants[b]['windows'], method, condition)
                                         for condition in [None, 'still', 'moving', 'transition']}
    common = {}
    for method in ['spectral', 'peak_count']:
        indices = {v: {(r['subject'], r['start_s']): r for r in data['windows']} for v, data in variants.items()}
        keys = [key for key in indices[VARIANTS[0]] if all(index[key]['eligible'] and index[key]['predictions_bpm'][method] is not None for index in indices.values())]
        common[method] = {v: metrics([index[k] for k in keys], method) for v, index in indices.items()}
    report = {'status': 'method_reimplementation_on_different_dataset_not_published_result_replication',
              'protocol': 'docs/kwon_reproduction_protocol.md',
              'primary': 'segmentation/spectral/still; paired corrected KLT without versus with MRF',
              'paper_waveform_correlation': None, 'paper_cycle_accuracy': None, 'mask_dice_iou': None,
              'unavailable_metrics_reason': 'No BIOPAC waveform, paper-view protocol, or annotated pixel masks supplied',
              'variants': variants, 'comparisons': comparisons, 'all_variant_common_metrics': common,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({name: data['spectral'] for name, data in comparisons.items()}, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--traces', type=Path, default=Path('artifacts/kwon_experiment'))
    p.add_argument('--out', type=Path, default=Path('artifacts/kwon_experiment/evaluation.json'))
    args = p.parse_args()
    run(args.traces, args.out)
