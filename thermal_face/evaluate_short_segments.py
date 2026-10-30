"""Exploratory 12-second paired Kwon-style segmentation comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .baselines import spectral_rate
from .evaluate_kwon import metrics, pairwise, peak_rate
from .windows import load_window_records


VARIANTS = ('klt_roi', 'klt_mrf')
ESTIMATORS = {'spectral': spectral_rate, 'peak_count': peak_rate}


def run(folder: Path, out: Path):
    variants = {}
    for variant in VARIANTS:
        records = load_window_records(folder / variant, window_s=12.0, step_s=2.0)
        rows = []
        for windows in records.values():
            for window in windows:
                predictions = {}
                reasons = {}
                for name, estimator in ESTIMATORS.items():
                    predictions[name], reasons[name] = (
                        estimator(window.x) if window.eligible
                        else (None, 'ineligible_window')
                    )
                rows.append(window.metadata() | {
                    'predictions_bpm': predictions, 'estimator_reasons': reasons,
                })
        variants[variant] = {
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
                person: hashlib.sha256((folder / variant / f'{person}.npz').read_bytes()).hexdigest()
                for person in records
            },
        }
    comparisons = {
        estimator: {
            condition: pairwise(variants['klt_roi']['windows'], variants['klt_mrf']['windows'],
                                estimator, None if condition == 'overall' else condition)
            for condition in ('still', 'overall', 'moving', 'transition')
        }
        for estimator in ESTIMATORS
    }
    report = {
        'status': 'exploratory_short_segment_comparison',
        'protocol': 'docs/short_segment_protocol.md',
        'window_seconds': 12.0, 'stride_seconds': 2.0,
        'primary': 'spectral/still, KLT ROI without vs with MRF on paired windows',
        'variants': variants, 'comparisons': comparisons,
        'pixel_mask_accuracy': None,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces', type=Path, default=Path('artifacts/kwon_experiment'))
    parser.add_argument('--out', type=Path,
                        default=Path('artifacts/short_segment_experiment/evaluation.json'))
    args = parser.parse_args()
    result = run(args.traces, args.out)
    print(json.dumps({name: {condition: {'windows': value['common_windows'],
                                     'first_mae_bpm': value['first']['macro_mae_bpm'],
                                     'second_mae_bpm': value['second']['macro_mae_bpm'],
                                     'change_bpm': value['second_minus_first_bpm']}
                         for condition, value in conditions.items()}
                      for name, conditions in result['comparisons'].items()}, indent=2))
