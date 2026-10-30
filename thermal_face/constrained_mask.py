"""Intersect saved Kwon-style masks with a fixed crop-relative perinasal ellipse.

This is an exploratory ablation, not an anatomical ground-truth segmenter.
The rule and evaluation were fixed in docs/constrained_mask_protocol.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .kwon import crop_warp, differential_increment


PEOPLE = ('Joao', 'Claudio', 'Manuel', 'Jaime')
MIN_PIXELS = 12


def perinasal_prior(shape: tuple[int, int]) -> np.ndarray:
    """The same normalized ellipse for every person and every frame."""
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    return (((xx + 0.5) / width - 0.50) / 0.27) ** 2 + (
        ((yy + 0.5) / height - 0.61) / 0.23
    ) ** 2 <= 1


def intersect_mask(packed: np.ndarray, shape: tuple[int, int],
                   current_support: np.ndarray, previous_support: np.ndarray) -> np.ndarray:
    original = np.unpackbits(packed, count=shape[0] * shape[1]).reshape(shape).astype(bool)
    return original & perinasal_prior(shape) & current_support & previous_support


def extract(root: Path, original: Path, out: Path, boxes_path: Path) -> dict:
    cv2.setNumThreads(2)
    boxes = json.loads(boxes_path.read_text(encoding='utf-8'))
    original_audit = json.loads((original / 'audit.json').read_text(encoding='utf-8'))
    out.mkdir(parents=True, exist_ok=True)
    audit = {
        'protocol': 'docs/constrained_mask_protocol.md',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'boxes_sha256': hashlib.sha256(boxes_path.read_bytes()).hexdigest(),
        'min_pixels': MIN_PIXELS,
        'prior': {'center_x': 0.50, 'center_y': 0.61,
                  'radius_x': 0.27, 'radius_y': 0.23},
        'subjects': {},
    }
    for person in PEOPLE:
        paths = sorted((root / person).glob('frame_*.png'))
        with np.load(original / 'diagnostics' / f'{person}.npz') as d, \
                np.load(original / 'klt_mrf' / f'{person}.npz') as source:
            seconds = source['seconds']
            marker = source['marker']
            original_valid = source['valid']
            matrix = d['matrices']
            packed = d['klt_mrf_packed_masks']
            if not (len(paths) == len(seconds) == len(matrix) == len(packed)):
                raise ValueError(f'{person}: frame/diagnostic length mismatch')
            x, y, width, height = boxes[person]
            if tuple(d['klt_mrf_mask_shape']) != (height, width):
                raise ValueError(f'{person}: mask/box shape mismatch')
            trace = np.zeros(len(paths), np.float32)
            valid = np.zeros(len(paths), bool)
            mask_pixels = np.zeros(len(paths), np.int32)
            reasons = np.full(len(paths), 'original_mrf_invalid', dtype='U32')
            pixel_hash = hashlib.sha256()
            previous_crop = previous_support = None
            last = 0.0
            for i, path in enumerate(paths):
                frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if frame is None or frame.dtype != np.uint8:
                    raise ValueError(f'Unreadable original image {path}')
                pixel_hash.update(frame.tobytes())
                crop = support = None
                if np.isfinite(matrix[i]).all():
                    crop, support = crop_warp(frame, matrix[i], boxes[person])
                if original_valid[i]:
                    if crop is None or previous_crop is None:
                        raise ValueError(f'{person}: original valid frame lacks aligned pair at {i}')
                    mask = intersect_mask(packed[i], (height, width), support, previous_support)
                    mask_pixels[i] = int(mask.sum())
                    if mask_pixels[i] >= MIN_PIXELS:
                        last += differential_increment(previous_crop, crop, mask)
                        valid[i] = True
                        reasons[i] = ''
                    else:
                        reasons[i] = 'too_few_prior_mask_pixels'
                trace[i] = last
                previous_crop, previous_support = crop, support
            actual_hash = pixel_hash.hexdigest()
            if actual_hash != original_audit['subjects'][person]['source_pixel_sha256']:
                raise ValueError(f'{person}: source pixels differ from original experiment')
            np.savez_compressed(out / f'{person}.npz', seconds=seconds, marker=marker,
                                trace=np.repeat(trace[:, None], 9, axis=1),
                                valid=valid, frame_reason=reasons,
                                mask_fraction=mask_pixels / (width * height),
                                mask_pixels=mask_pixels)
            audit['subjects'][person] = {
                'frames': len(paths), 'original_valid_frames': int(original_valid.sum()),
                'constrained_valid_frames': int(valid.sum()),
                'too_few_prior_mask_pixels': int((reasons == 'too_few_prior_mask_pixels').sum()),
                'source_pixel_sha256': actual_hash,
                'original_trace_sha256': hashlib.sha256(
                    (original / 'klt_mrf' / f'{person}.npz').read_bytes()).hexdigest(),
                'diagnostics_sha256': hashlib.sha256(
                    (original / 'diagnostics' / f'{person}.npz').read_bytes()).hexdigest(),
            }
            print(person, audit['subjects'][person]['constrained_valid_frames'],
                  '/', audit['subjects'][person]['original_valid_frames'], flush=True)
    (out / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('Manuel_clean'))
    parser.add_argument('--original', type=Path, default=Path('artifacts/kwon_experiment'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/constrained_mask_experiment/traces'))
    parser.add_argument('--boxes', type=Path, default=Path('config/nose_boxes.json'))
    args = parser.parse_args()
    extract(args.root, args.original, args.out, args.boxes)
