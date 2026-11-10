"""Append the executed spatial-constraint experiment to the original notebook."""
import copy
import json
from pathlib import Path


def integrate(original, experiment):
    notebook = copy.deepcopy(original)
    kept = [cell for cell in notebook['cells']
            if not cell.get('metadata', {}).get('constrained_mask_integrated')]
    notice = ('\n\n**Further experiment on 31 October 2026:** Section 17 tests a fixed '
              'perinasal constraint on the existing MRF masks, with paired rate '
              'errors and coverage. It remains exploratory.\n')
    if '**Further experiment on 31 October 2026:**' not in kept[0]['source']:
        kept[0]['source'] += notice
    for i, source_cell in enumerate(experiment['cells']):
        cell = copy.deepcopy(source_cell)
        cell.setdefault('metadata', {})['constrained_mask_integrated'] = True
        if i == 0:
            cell['source'] = cell['source'].replace(
                '# Constraining', '## 17. Constraining', 1)
        kept.append(cell)
    count = 0
    for i, cell in enumerate(kept):
        cell['id'] = f'constraint-integrated-{i:03d}'
        if cell['cell_type'] == 'code':
            count += 1
            cell['metadata'].setdefault('original_execution_count', cell['execution_count'])
            cell['execution_count'] = count
            for output in cell['outputs']:
                if output['output_type'] == 'execute_result':
                    output['execution_count'] = count
    notebook['cells'] = kept
    return notebook


def main():
    from build_walkthrough import render
    folder = Path(__file__).resolve().parent
    path = folder / 'thermal_respiration_walkthrough.ipynb'
    result = integrate(
        json.loads(path.read_text(encoding='utf-8')),
        json.loads((folder / 'constrained_mask_experiment.ipynb').read_text(encoding='utf-8')),
    )
    render(result, path)


if __name__ == '__main__':
    main()
