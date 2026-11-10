"""Create a local, rate-blind packet for independent nasal/mask review."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from thermal_face.kwon import crop_warp


PEOPLE = ('Joao', 'Claudio', 'Manuel', 'Jaime')
TARGET_SECONDS = (15.0, 45.0, 85.0)


def build(root: Path = Path('.')) -> list[dict]:
    boxes = json.loads((root / 'config/nose_boxes.json').read_text(encoding='utf-8'))
    source = root / 'artifacts/kwon_experiment'
    out = root / 'artifacts/constrained_mask_experiment'
    out.mkdir(parents=True, exist_ok=True)
    full_fig, full_axes = plt.subplots(4, 3, figsize=(11, 11))
    crop_fig, crop_axes = plt.subplots(4, 3, figsize=(9, 10))
    selection = []
    for row, person in enumerate(PEOPLE):
        paths = sorted((root / 'Manuel_clean' / person).glob('frame_*.png'))
        with np.load(source / 'diagnostics' / f'{person}.npz') as d, \
                np.load(source / 'klt_mrf' / f'{person}.npz') as z:
            times, matrices, frozen = d['seconds'], d['matrices'], d['frozen']
            packed, valid = d['klt_mrf_packed_masks'], z['valid']
            available = np.flatnonzero(~frozen)
            for col, target in enumerate(TARGET_SECONDS):
                index = int(available[np.argmin(abs(times[available] - target))])
                frame = cv2.imread(str(paths[index]), cv2.IMREAD_GRAYSCALE)
                if frame is None:
                    raise ValueError(f'Unreadable {paths[index]}')
                full = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                matrix = matrices[index]
                tracked = bool(np.isfinite(matrix).all())
                crop = np.zeros((boxes[person][3], boxes[person][2]), np.uint8)
                mask = np.zeros(crop.shape, bool)
                if tracked:
                    x, y, w, h = boxes[person]
                    corners = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],np.float32)
                    projected = corners @ matrix[:,:2].T + matrix[:,2]
                    cv2.polylines(full, [np.rint(projected).astype(np.int32)], True, (0,255,255), 2)
                    crop, _ = crop_warp(frame, matrix, boxes[person])
                    mask = np.unpackbits(packed[index], count=crop.size).reshape(crop.shape).astype(bool)
                overlay = np.repeat((crop/255.0)[...,None],3,axis=2)
                overlay[mask] = .55*overlay[mask] + .45*np.array([0,1,.85])
                title = f'{person} {times[index]:.1f}s; MRF {"valid" if valid[index] else "invalid"}'
                full_axes[row,col].imshow(full); full_axes[row,col].set_title(title,fontsize=9)
                crop_axes[row,col].imshow(overlay); crop_axes[row,col].set_title(title,fontsize=9)
                full_axes[row,col].axis('off'); crop_axes[row,col].axis('off')
                selection.append({
                    'person': person, 'target_second': target, 'actual_second': float(times[index]),
                    'frame_index_zero_based': index, 'condition_assumed': 'still' if target<60 else 'moving',
                    'track_transform_available': tracked, 'mrf_frame_valid': bool(valid[index]),
                    'original_mask_pixels': int(mask.sum()),
                })
    for fig, name in [(full_fig,'annotation_full_frames.png'),
                      (crop_fig,'annotation_mrf_overlays.png')]:
        fig.tight_layout(); fig.savefig(out / name,dpi=150); plt.close(fig)
    (out / 'annotation_packet_selection.json').write_text(
        json.dumps(selection,indent=2),encoding='utf-8')
    with (out / 'annotation_packet_blank.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=[
            'reviewer','person','frame_index_zero_based','actual_second',
            'nose_visible_in_full_frame','tracked_box_contains_nose',
            'nose_visible_in_aligned_crop','original_mrf_covers_breathing_region',
            'original_mrf_selects_unrelated_region','note',
        ])
        writer.writeheader()
        for item in selection:
            writer.writerow({k:item[k] for k in ('person','frame_index_zero_based','actual_second')})
    return selection


if __name__ == '__main__':
    result=build(Path(__file__).resolve().parents[1])
    print(f'Prepared {len(result)} local review frames without rate predictions')
