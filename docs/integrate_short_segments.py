"""Append the executed short-segment follow-up to the original presentation."""
import copy
import json
from pathlib import Path


def integrate(original, experiment):
    notebook = copy.deepcopy(original)
    kept = [cell for cell in notebook['cells'] if not cell.get('metadata', {}).get('short_segment_integrated')]
    notice = ('\n\n**Follow-up on 31 October 2026:** Section 16 evaluates 12-second supported '
              'segments after the original 20-second MRF comparison yielded no eligible windows. '
              'The follow-up is exploratory and does not replace the original result.\n')
    if '**Follow-up on 31 October 2026:**' not in kept[0]['source']:
        kept[0]['source'] += notice
    for i, source_cell in enumerate(experiment['cells']):
        cell = copy.deepcopy(source_cell)
        cell.setdefault('metadata', {})['short_segment_integrated'] = True
        if i == 0:
            cell['source'] = cell['source'].replace('# Short supported', '## 16. Short supported', 1)
        kept.append(cell)
    count = 0
    for i, cell in enumerate(kept):
        cell['id'] = f'short-integrated-{i:03d}'
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
    render(integrate(json.loads(path.read_text(encoding='utf-8')),
                     json.loads((folder / 'short_segment_experiment.ipynb').read_text(encoding='utf-8'))), path)


if __name__ == '__main__':
    main()
