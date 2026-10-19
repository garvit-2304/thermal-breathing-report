"""Paired fixed-versus-CSRT evaluation on identical training and test windows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .baselines import spectral_rate
from .evaluate import METHODS, summarize
from .train_rate import execution_environment, fit_predict
from .windows import load_window_records


def paired_windows(fixed, tracked):
    """Use the eligibility intersection for BOTH training and testing."""
    if set(fixed) != set(tracked) or len(fixed) < 3:
        raise ValueError('Need the same three or more subjects in both variants')
    paired = {}
    for subject in sorted(fixed):
        a = {w.start_s: w for w in fixed[subject]}
        b = {w.start_s: w for w in tracked[subject]}
        if a.keys() != b.keys():
            raise ValueError(f'Candidate timestamps differ for {subject}')
        for start in a:
            wa, wb = a[start], b[start]
            if (wa.end_s, wa.condition, wa.reference_bpm, wa.reference_rejection) != (
                    wb.end_s, wb.condition, wb.reference_bpm, wb.reference_rejection):
                raise ValueError(f'Reference labels or conditions differ for {subject} at {start}')
        paired[subject] = [start for start in sorted(a) if a[start].eligible and b[start].eligible]
        if not paired[subject]:
            raise ValueError(f'No paired eligible windows for {subject}; cannot silently drop a person')
    return paired


def run(folder: Path, out: Path, seeds=(7, 17, 27), epochs=40, device='cpu'):
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or epochs < 1:
        raise ValueError('Specify distinct seeds and positive epochs')
    started = perf_counter()
    records = {variant: load_window_records(folder / variant) for variant in ['fixed', 'tracked']}
    paired = paired_windows(records['fixed'], records['tracked'])
    people = sorted(paired)
    coverage = {}
    for variant in records:
        coverage[variant] = {}
        for person, windows in records[variant].items():
            coverage[variant][person] = {
                'candidates': len(windows), 'eligible': sum(w.eligible for w in windows),
                'paired': len(paired[person]),
                'windows': [w.metadata() for w in windows],
            }
    report = {
        'protocol': {
            'primary_seed': seeds[0], 'seeds': list(seeds), 'epochs': epochs,
            'split': 'leave-one-person-out',
            'paired_training_and_test': True,
            'paired_rule': 'intersection of fixed/tracked eligibility by subject and exact window timestamp',
            'window_s': 20, 'step_s': 5, 'sample_hz': 10,
            'model_selection': 'no selection or tuning using held-out results',
            'interpretation': 'seed variation is not independent-person uncertainty; four-person exploratory pilot',
            'excluded_subjects': {'anestis': 'unresolved frame/reference alignment'},
        },
        'environment': execution_environment(device), 'coverage': coverage,
        'paired_starts': paired, 'runs': [],
        'trace_sha256': {variant: {person: hashlib.sha256((folder / variant / f'{person}.npz').read_bytes()).hexdigest()
                                    for person in people} for variant in records},
        'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ['compare_tracking.py', 'train_rate.py', 'models.py', 'windows.py', 'baselines.py']},
    }
    for seed in seeds:
        seed_result = {'seed': seed, 'variants': {}}
        for variant in records:
            examples = {person: [w for w in records[variant][person] if w.start_s in paired[person]] for person in people}
            rows, folds = [], []
            for held_out in people:
                training_people = [p for p in people if p != held_out]
                train = [(w.x, np.float32(w.reference_bpm)) for p in training_people for w in examples[p]]
                test = examples[held_out]
                test_x = np.stack([w.x for w in test])
                median = float(np.median(np.array([y for _, y in train], dtype=np.float32)))
                print(f'seed={seed} {variant} held_out={held_out}: train={len(train)}, test={len(test)}', flush=True)
                predicted = {}
                parameters = {}
                for architecture in ['cnn', 'cnn-gru']:
                    values, count = fit_predict(train, test_x, epochs, seed, architecture, device)
                    if np.shape(values) != (len(test),) or not np.isfinite(values).all():
                        raise ValueError('Nonfinite or malformed neural predictions; do not silently exclude them')
                    predicted[architecture.replace('-', '_')] = values
                    parameters[architecture] = count
                for i, window in enumerate(test):
                    spectrum, reason = spectral_rate(window.x)
                    row = window.metadata() | {
                        'predictions_bpm': {'train_median': median, 'spectral': spectrum,
                                            'cnn': float(predicted['cnn'][i]),
                                            'cnn_gru': float(predicted['cnn_gru'][i])},
                        'abstention_reasons': dict.fromkeys(METHODS),
                    }
                    row['abstention_reasons']['spectral'] = reason
                    rows.append(row)
                folds.append({'held_out': held_out, 'train_subjects': training_people,
                              'train_windows': len(train), 'test_windows': len(test),
                              'train_median_bpm': median, 'parameters': parameters})
            seed_result['variants'][variant] = {
                'folds': folds, 'windows': rows, 'overall': summarize(rows),
                'by_condition': {c: summarize([r for r in rows if r['condition'] == c]) for c in ['still', 'moving', 'transition']},
                'by_subject': {p: summarize([r for r in rows if r['subject'] == p]) for p in people},
            }
        # Pair predictions separately per method, so spectral abstention cannot affect CNN scoring.
        comparisons = {}
        variants = seed_result['variants']
        index = {v: {(r['subject'], r['start_s']): r for r in variants[v]['windows']} for v in variants}
        for method in METHODS:
            keys = [k for k in index['fixed'] if all(index[v][k]['predictions_bpm'][method] is not None for v in index)]
            from .evaluate import error_metrics
            metrics = {v: error_metrics([index[v][k] for k in keys], method) for v in index}
            a, b = [metrics[v]['macro_mae_bpm'] for v in ['fixed', 'tracked']]
            comparisons[method] = {
                **metrics, 'paired_prediction_windows': len(keys),
                'tracked_minus_fixed_mae_bpm': b-a if a is not None and b is not None else None,
            }
        seed_result['comparison'] = comparisons
        report['runs'].append(seed_result)
        print('Seed comparison:', json.dumps(comparisons, indent=2), flush=True)
    report['aggregate'] = {}
    for method in METHODS:
        values = np.array([[r['comparison'][method][v]['macro_mae_bpm'] for v in ['fixed', 'tracked']]
                           for r in report['runs']], dtype=float)
        if not np.isfinite(values).all():
            report['aggregate'][method] = {'status': 'no paired predictions in at least one seed'}
            continue
        differences = values[:, 1] - values[:, 0]
        report['aggregate'][method] = {
            'fixed_mean_mae_bpm': float(values[:, 0].mean()),
            'tracked_mean_mae_bpm': float(values[:, 1].mean()),
            'fixed_seed_sd_bpm': float(values[:, 0].std()),
            'tracked_seed_sd_bpm': float(values[:, 1].std()),
            'mean_tracked_minus_fixed_bpm': float(differences.mean()),
            'seeds_with_improvement': int((differences < 0).sum()),
        }
    report['elapsed_seconds'] = perf_counter()-started
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces', type=Path, default=Path('artifacts/tracking_experiment'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/tracking_experiment/comparison.json'))
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 17, 27])
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    args = parser.parse_args()
    report = run(args.traces, args.out, args.seeds, args.epochs, args.device)
    print(json.dumps(report['aggregate'], indent=2))


if __name__ == '__main__':
    main()
