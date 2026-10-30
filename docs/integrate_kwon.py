"""Preserve historical outputs while appending the primary Kwon experiment."""
import copy
import json
from pathlib import Path

CORRECTION = "BAF in Kwon equation 1 is sample standard deviation. The legacy square root is therefore appropriate; the substantive departures are reversed stabilization, whole-recording static masking, and inconsistent timing/signal extraction. Historical scripts contain machine-specific paths and are not executed here."

def integrate(original, experiment):
    result = copy.deepcopy(original)
    kept = []
    for cell in result['cells']:
        source = ''.join(cell['source'])
        if (cell.get('metadata', {}).get('kwon_integrated') or
                cell.get('metadata', {}).get('short_segment_integrated') or
                source.startswith('## 15. Limitations after')):
            continue
        source = source.replace("The standard-deviation map in these scripts should not be called an exact reproduction of the paper's variance-based method. These scripts also contain machine-specific paths and are shown as historical evidence, not executed here.", CORRECTION)
        cell['source'] = source
        kept.append(cell)
    old_notice = '**Primary objective: reproduce Kwon et al. BAFR segmentation.** Section 15 contains the corrected implementation and latest experiment. Earlier neural and CSRT results are historical exploratory work, not completion of the reproduction. The new experiment has zero eligible MRF windows; segmentation accuracy remains unmeasured.'
    kept[0]['source'] = kept[0]['source'].replace(old_notice, '')
    notice = '\n\n**Project aim:** build a thermal-video breathing-rate detector with breathing-region segmentation. Kwon et al. guides the segmentation branch, and JoulesEye guides tracking. Section 15 records the corrected Kwon-style implementation and original 20-second experiment; Section 16 is an exploratory short-segment follow-up.\n'
    if '**Project aim:** build a thermal-video' not in kept[0]['source']:
        kept[0]['source'] += notice
    for i, source_cell in enumerate(experiment['cells']):
        cell = copy.deepcopy(source_cell)
        cell.setdefault('metadata', {})['kwon_integrated'] = True
        if i == 0:
            cell['source'] = cell['source'].replace('# Kwon BAFR', '## 15. Kwon BAFR', 1)
        kept.append(cell)
    count = 0
    for i, cell in enumerate(kept):
        cell['id'] = f'kwon-integrated-{i:03d}'
        if cell['cell_type'] == 'code':
            count += 1
            cell['metadata'].setdefault('original_execution_count', cell['execution_count'])
            cell['execution_count'] = count
            for output in cell['outputs']:
                if output['output_type'] == 'execute_result':
                    output['execution_count'] = count
    result['cells'] = kept
    return result

def main():
    from build_walkthrough import render
    folder = Path(__file__).resolve().parent
    path = folder / 'thermal_respiration_walkthrough.ipynb'
    render(integrate(json.loads(path.read_text(encoding='utf-8')), json.loads((folder / 'kwon_reproduction.ipynb').read_text(encoding='utf-8'))), path)
    if (folder / 'short_segment_experiment.ipynb').exists():
        from integrate_short_segments import main as integrate_short_segments
        integrate_short_segments()

if __name__ == '__main__':
    main()
