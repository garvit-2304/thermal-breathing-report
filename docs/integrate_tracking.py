"""Integrate the completed tracking study into the original presentation in place."""
import copy
import json
import re
from pathlib import Path


ENDING = """## 15. Limitations after the tracking experiment

1. **Small dataset:** four usable people, one recording each. The 77 overlapping windows and three training seeds are not additional independent participants.
2. **Unverified references:** exact frame/CSV synchronization and marker semantics need original annotation instructions. `anestis` remains excluded; seven candidate windows fail reference checks.
3. **Tracking drift:** CSRT reports successful updates even when sampled boxes move toward the neck or eyes. It has no anatomical detector or RGB recovery. It changes crop size as well as position.
4. **Very small main-model gain:** at seed 7, CNN + GRU improves from 2.986 to 2.969 BPM (0.017 BPM, about 0.6%), still worse than the 2.501 BPM median baseline. This does not establish a practically reliable improvement.
5. **Training sensitivity:** seed 17 gives about 14.87 BPM CNN + GRU error for both crop variants. Report all seeds; do not present the three-seed average without explaining this variation.
6. **Mixed effects across methods:** CNN improves across all three tested seeds (mean 3.682 to 3.247 BPM), while the spectral baseline worsens (10.618 to 10.809 BPM). Tracking is not a universal accuracy improvement.
7. **Coverage and conditions:** both variants retain 77/84 windows, but coverage includes reference screening. The still/moving boundary at 60 s is assumed and not independently verified.
8. **Thermal precision and deployment:** inputs are 8-bit intensities, not calibrated temperatures. Automatic initialization, reliable recovery, deployment checkpoints and prospective evaluation remain unfinished.
9. **Historical provenance:** the new comparison is supported by saved predictions and hashes; older historical reports and full original execution logs remain incomplete.
10. **Identity recognition:** this work estimates respiration. Identity recognition needs repeated sessions and its own evaluation.

## 16. What to do next, in order

| Priority | Next action | Evidence required |
|---|---|---|
| 1 | Annotate nose positions during motion and quantify tracker drift | Localization error, failure detection and recovery assessed independently of respiration labels |
| 2 | Verify reference markers and frame synchronization | Reviewed annotation semantics and timestamps; resolve `anestis` before inclusion |
| 3 | Investigate CNN + GRU training sensitivity using training-only diagnostics | Loss curves and predeclared training changes evaluated without tuning on held-out subjects |
| 4 | Test improved localization and recovery | Paired fixed/tracked comparison on the same people and windows, preserving coverage reporting |
| 5 | Collect more participants and repeated sessions | Independent prospective test set; uncertainty over people rather than windows or seeds |
| 6 | Save checkpoints with preprocessing and fold metadata | Reloaded inference reproduces saved predictions |
| 7 | Develop identity recognition as a separate task | Held-out-session evaluation and unknown-person rejection |

**Completed in this notebook:** the first CSRT adaptation, paired fixed-versus-tracked evaluation, three predetermined seeds, sampled overlays, and implementation checks (18 tests passed; one CUDA test skipped). Fixed crops remain the default; tracking is available as an optional experiment.

### Closing statement for the presentation
“We implemented and tested CSRT nose-region tracking using identical data splits and windows. It improved CNN-only results, but the CNN + GRU gain was very small and neither approach beat the simple median baseline overall. Visible tracking drift and training instability show why our next priorities are verified localization, reliable labels, and more independent recordings.”
"""


def integrate(original, experiment):
    notebook = copy.deepcopy(original)
    kept = []
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        cell['source'] = source
        if (cell.get('metadata', {}).get('tracking_integrated') or
                cell.get('metadata', {}).get('kwon_integrated') or
                cell.get('metadata', {}).get('short_segment_integrated')):
            continue
        if cell['cell_type'] == 'markdown' and (
            source.startswith('## CSRT tracking experiment') or
            source.startswith('## 14. Limitations') or
            source.startswith('## 15. Limitations after')):
            continue
        kept.append(cell)
    revision = '\n\n**Updated 18 October 2026:** Section 14 integrates the complete CSRT experiment, followed by revised limitations and next steps. Original and tracking-run outputs are preserved from their actual executions; this integration does not retrain the models. Cell numbers indicate presentation order. The earlier sections describe the fixed-crop baseline.\n'
    if '**Updated 18 October 2026:**' not in kept[0]['source']:
        kept[0]['source'] += revision
    for i, source_cell in enumerate(experiment['cells'][:-1]):
        cell = copy.deepcopy(source_cell)
        source = ''.join(cell['source'])
        cell['metadata']['tracking_integrated'] = True
        if i == 0:
            source = '## 14. Borrowing JoulesEye’s tracking idea: implementation and measured results\n\n' + source.split('\n', 1)[1]
            source = source.replace('The original fixed-crop walkthrough is in `thermal_respiration_walkthrough.ipynb`.',
                                    'The preceding sections document the fixed-crop baseline used in this comparison.')
        elif cell['cell_type'] == 'markdown':
            # Shift headings only outside fenced source listings.
            fenced = False
            lines = []
            for line in source.splitlines():
                if line.startswith('```'):
                    fenced = not fenced
                if not fenced and line.startswith('#'):
                    line = re.sub(r'^(#{1,6}) ', lambda m: '#' * min(6, len(m[1])+1 if len(m[1]) > 1 else 3) + ' ', line)
                lines.append(line)
            source = '\n'.join(lines)
        cell['source'] = source
        kept.append(cell)
    kept.append({'cell_type': 'markdown', 'source': ENDING, 'metadata': {'tracking_integrated': True}})
    count = 0
    for i, cell in enumerate(kept):
        cell['id'] = f'integrated-{i:03d}'
        if cell['cell_type'] == 'code':
            count += 1
            cell['metadata'].setdefault('original_execution_count', cell['execution_count'])
            cell['execution_count'] = count
            for output in cell['outputs']:
                if output['output_type'] == 'execute_result':
                    output['execution_count'] = count
    notebook['cells'] = kept
    notebook['metadata']['presentation_provenance'] = (
        'Integrated 2026-10-18 from the original walkthrough and executed tracking experiment. '
        'Saved outputs preserved; counters reordered for presentation, original counters in cell metadata. '
        'Historical values remain labeled separately from measured experiment outputs.')
    return notebook


def main():
    from build_walkthrough import render
    folder = Path(__file__).resolve().parent
    original_path = folder / 'thermal_respiration_walkthrough.ipynb'
    original = json.loads(original_path.read_text(encoding='utf-8'))
    experiment = json.loads((folder / 'tracking_experiment.ipynb').read_text(encoding='utf-8'))
    merged = integrate(original, experiment)
    assert len([c for c in merged['cells'] if c['cell_type'] == 'code']) == 24
    assert not any(o['output_type'] == 'error' for c in merged['cells'] if c['cell_type'] == 'code' for o in c['outputs'])
    render(merged, original_path)
    if (folder / 'kwon_reproduction.ipynb').exists():
        from integrate_kwon import main as integrate_kwon
        integrate_kwon()


if __name__ == '__main__':
    main()
