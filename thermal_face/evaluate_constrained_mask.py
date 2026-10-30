"""Paired evaluation of original and spatially constrained Kwon-style masks."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .baselines import spectral_rate
from .evaluate_kwon import pairwise, peak_rate
from .windows import load_window_records


ESTIMATORS = {'spectral': spectral_rate, 'peak_count': peak_rate}
CONDITIONS = ('still', 'overall', 'moving', 'transition')


def _windows(folder: Path, seconds: float, stride: float) -> dict:
    records = load_window_records(folder, window_s=seconds, step_s=stride)
    rows = []
    for windows in records.values():
        for window in windows:
            predictions, reasons = {}, {}
            for name, estimator in ESTIMATORS.items():
                predictions[name], reasons[name] = (
                    estimator(window.x) if window.eligible else (None, 'ineligible_window')
                )
            rows.append(window.metadata() | {
                'predictions_bpm': predictions, 'estimator_reasons': reasons,
            })
    return {
        'windows': rows,
        'candidate_windows': len(rows),
        'eligible_windows': sum(row['eligible'] for row in rows),
        'eligible_by_condition': {
            condition: sum(row['eligible'] and row['condition'] == condition for row in rows)
            for condition in ('still', 'moving', 'transition')
        },
        'signal_exclusions': dict(Counter(
            row['signal_rejection'] for row in rows if row['signal_rejection']
        )),
        'reference_exclusions': dict(Counter(
            row['reference_rejection'] for row in rows if row['reference_rejection']
        )),
        'trace_sha256': {
            person: hashlib.sha256((folder / f'{person}.npz').read_bytes()).hexdigest()
            for person in records
        },
    }


def run(original: Path, constrained: Path, out: Path) -> dict:
    report = {
        'status': 'exploratory_fixed_spatial_constraint',
        'protocol': 'docs/constrained_mask_protocol.md',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'pixel_mask_accuracy': None,
        'evaluations': {},
    }
    for name, seconds, stride in [('short_12s', 12.0, 2.0),
                                  ('original_20s', 20.0, 5.0)]:
        variants = {
            'original_mrf': _windows(original, seconds, stride),
            'constrained_mrf': _windows(constrained, seconds, stride),
        }
        comparisons = {
            estimator: {
                condition: pairwise(
                    variants['original_mrf']['windows'], variants['constrained_mrf']['windows'],
                    estimator, None if condition == 'overall' else condition,
                ) for condition in CONDITIONS
            } for estimator in ESTIMATORS
        }
        report['evaluations'][name] = {
            'window_seconds': seconds, 'stride_seconds': stride,
            'variants': variants, 'comparisons': comparisons,
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, default=Path('artifacts/kwon_experiment/klt_mrf'))
    parser.add_argument('--constrained', type=Path,
                        default=Path('artifacts/constrained_mask_experiment/traces'))
    parser.add_argument('--out', type=Path,
                        default=Path('artifacts/constrained_mask_experiment/evaluation.json'))
    args = parser.parse_args()
    result = run(args.original, args.constrained, args.out)
    for name, evaluation in result['evaluations'].items():
        print(name, {variant: details['eligible_windows']
                     for variant, details in evaluation['variants'].items()})
        print(json.dumps({estimator: {
            condition: {'paired': value['common_windows'],
                        'original_mae_bpm': value['first']['macro_mae_bpm'],
                        'constrained_mae_bpm': value['second']['macro_mae_bpm'],
                        'change_bpm': value['second_minus_first_bpm']}
            for condition, value in conditions.items()}
                          for estimator, conditions in evaluation['comparisons'].items()}, indent=2))
