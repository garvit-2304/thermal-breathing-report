"""Kwon-style BAF/BAP/MRF processing with explicit reproduction assumptions.

See docs/kwon_reproduction_protocol.md. This is not the authors' original code.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from scipy import ndimage

from .data import SUBJECTS, read_respiration
from .tracking import unfrozen_mask


def crop_warp(frame, reference_to_current, box, correct=True):
    """Sample current image into reference-box coordinates; exclude padded pixels."""
    x, y, w, h = box
    matrix = reference_to_current if correct else cv2.invertAffineTransform(reference_to_current)
    local = matrix.copy()
    local[:, 2] += matrix[:, :2] @ np.array([x, y])
    options = dict(flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    crop = cv2.warpAffine(frame, local, (w, h), **options)
    support = cv2.warpAffine(np.ones(frame.shape, np.float32), local, (w, h), **options)
    return crop, support >= 0.999


def local_features(frame, box):
    x, y, w, h = (int(v) for v in box)
    mask = np.zeros(frame.shape, np.uint8)
    mask[max(0, y-h):min(frame.shape[0], y+2*h), max(0, x-w):min(frame.shape[1], x+2*w)] = 255
    return cv2.goodFeaturesToTrack(frame, maxCorners=150, qualityLevel=0.01,
                                  minDistance=4, blockSize=7, mask=mask)


def klt_step(previous, current, points):
    if points is None or len(points) < 6:
        return None, None
    options = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    forward, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **options)
    if forward is None:
        return None, None
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, **options)
    if backward is None:
        return None, None
    keep = (status.ravel() > 0) & (back_status.ravel() > 0)
    keep &= np.linalg.norm(backward.reshape(-1, 2)-points.reshape(-1, 2), axis=1) <= 1.5
    keep &= np.isfinite(forward.reshape(-1, 2)).all(axis=1)
    if keep.sum() < 6:
        return None, None
    matrix, inliers = cv2.estimateAffinePartial2D(points[keep], forward[keep], method=cv2.RANSAC,
                                                ransacReprojThreshold=3.0)
    if matrix is None or inliers.sum() < 6 or inliers.mean() < 0.6:
        return None, None
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    if not 0.8 <= scale <= 1.25 or not np.isfinite(matrix).all():
        return None, None
    return matrix, forward[keep][inliers.ravel() > 0]


class LocalKLT:
    def __init__(self, frame, box):
        self.reference = frame.copy()
        self.previous = frame.copy()
        self.box = box
        self.matrix = np.eye(3)
        self.points = local_features(frame, box)
        self.ok = True
        self.steps = 0

    def update(self, frame, repeated=False):
        if repeated:
            return self.matrix[:2].copy() if self.ok else None
        self.steps += 1
        if self.ok:
            increment, points = klt_step(self.previous, frame, self.points)
            candidate = np.vstack([increment, [0, 0, 1]]) @ self.matrix if increment is not None else None
        else:
            candidate, points = None, None
        if candidate is None:
            recovery, points = klt_step(self.reference, frame, local_features(self.reference, self.box))
            candidate = np.vstack([recovery, [0, 0, 1]]) if recovery is not None else None
        if candidate is not None and not 0.5 <= np.hypot(candidate[0, 0], candidate[1, 0]) <= 2.0:
            candidate = None
        self.ok = candidate is not None
        if self.ok:
            self.matrix = candidate
            self.points = points
            if self.steps % 100 == 0 or self.points is None or len(self.points) < 20:
                x, y, w, h = self.box
                center = self.matrix @ np.array([x+w/2, y+h/2, 1])
                self.points = local_features(frame, (center[0]-w/2, center[1]-h/2, w, h))
        self.previous = frame.copy()
        return self.matrix[:2].copy() if self.ok else None


def baf_bap(frames, valid_pixels=None):
    """Equation 1: sample SD; integrated pixel differences equal I(t)-I(t0)."""
    frames = np.asarray(frames, dtype=np.float64)
    if len(frames) < 2:
        raise ValueError('At least two samples required')
    baf = frames.std(axis=0, ddof=1)
    support = np.ones(baf.shape, bool) if valid_pixels is None else valid_pixels
    baf[~support] = 0
    maximum = float(baf.max())
    return baf.astype(np.float32), (baf/maximum).astype(np.float32) if maximum > 1e-8 else np.zeros_like(baf, np.float32)


NEIGHBORS = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)


def mrf_icm(bap, support=None, weight=0.6, sweeps=30):
    """Gaussian/Potts implementation assumption; red-black ICM, not parallel flips."""
    bap = np.asarray(bap, dtype=np.float64)
    if bap.ndim != 2 or not np.isfinite(bap).all() or np.any((bap < 0) | (bap > 1)):
        raise ValueError('BAP must be a finite two-dimensional [0,1] map')
    support = np.ones(bap.shape, bool) if support is None else support.astype(bool)
    if support.sum() < 2 or np.ptp(bap[support]) < 1e-8:
        return np.zeros(bap.shape, bool), 0
    labels = (bap > 0.5) & support
    degree = ndimage.convolve(support.astype(float), NEIGHBORS, mode='constant')
    parity = np.indices(bap.shape).sum(axis=0) % 2
    for iteration in range(sweeps):
        old = labels.copy()
        parameters = []
        for cls in [False, True]:
            values = bap[support & (labels == cls)]
            if len(values) < 2:
                return np.zeros(bap.shape, bool), iteration
            parameters.append((values.mean(), max(values.std(), 1e-3)))
        data = [np.log(sigma) + 0.5*((bap-mean)/sigma)**2 for mean, sigma in parameters]
        for color in [0, 1]:
            count = ndimage.convolve((labels & support).astype(float), NEIGHBORS, mode='constant')
            select = support & (parity == color)
            energy0 = data[0] + weight*count
            energy1 = data[1] + weight*(degree-count)
            # Keep current label on ties to avoid gratuitous oscillation.
            labels[select & (energy1 < energy0)] = True
            labels[select & (energy0 < energy1)] = False
        if np.array_equal(labels, old):
            break
    if labels[support].all() or not labels.any():
        return np.zeros(bap.shape, bool), iteration+1
    if bap[labels].mean() < bap[support & ~labels].mean():
        labels = support & ~labels
    return labels, iteration+1


def differential_increment(previous, current, mask):
    if not np.any(mask):
        return None
    return float((current.astype(float)-previous.astype(float))[mask].mean())


VARIANTS = ('fixed_roi', 'klt_roi', 'klt_mrf', 'wrong_warp_mrf')


def extract(root: Path, out: Path, boxes_path: Path):
    cv2.setNumThreads(2)
    cv2.setRNGSeed(7)
    boxes = json.loads(boxes_path.read_text())
    for variant in (*VARIANTS, 'diagnostics'):
        (out / variant).mkdir(parents=True, exist_ok=True)
    audit = {'opencv': cv2.__version__, 'protocol': 'docs/kwon_reproduction_protocol.md', 'subjects': {}, 'excluded': {},
             'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'boxes_sha256': hashlib.sha256(boxes_path.read_bytes()).hexdigest()}
    for person in SUBJECTS:
        paths = sorted((root / person).glob('frame_*.png'))
        csv = next(root.glob(f'*{person}_respiration.csv'))
        times, markers = read_respiration(csv)
        if len(paths) != len(times):
            audit['excluded'][person] = 'frame/reference row mismatch'
            continue
        started = perf_counter()
        seconds = np.array([(t-times[0]).total_seconds() for t in times])
        box = boxes[person]
        x, y, w, h = box
        n = len(paths)
        raw = np.empty((n, h, w), np.uint8)
        aligned = np.zeros_like(raw)
        wrong = np.zeros_like(raw)
        supports = np.zeros((n, h, w), bool)
        wrong_supports = np.zeros_like(supports)
        matrices = np.full((n, 2, 3), np.nan)
        track_ok = np.zeros(n, bool)
        repeated = np.zeros(n, bool)
        previous = None
        pixels_hash = hashlib.sha256()
        print(f'{person}: reading/tracking {n} original frames', flush=True)
        for i, path in enumerate(paths):
            frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if frame is None or frame.dtype != np.uint8:
                raise ValueError(f'Unreadable original image {path}')
            pixels_hash.update(frame.tobytes())
            raw[i] = frame[y:y+h, x:x+w]
            if i == 0:
                tracker = LocalKLT(frame, box)
                matrix = np.eye(2, 3)
            else:
                repeated[i] = np.array_equal(frame, previous)
                matrix = tracker.update(frame, bool(repeated[i]))
            previous = frame
            if matrix is not None:
                matrices[i] = matrix
                aligned[i], supports[i] = crop_warp(frame, matrix, box)
                wrong[i], wrong_supports[i] = crop_warp(frame, matrix, box, correct=False)
                track_ok[i] = supports[i].mean() >= 0.98
            if (i+1) % 1000 == 0:
                print(f'  read {i+1}/{n}; tracking failures {(~track_ok[:i+1]).sum()}', flush=True)
        unfrozen = unfrozen_mask(repeated)
        frame_valid = {'fixed_roi': unfrozen, 'klt_roi': unfrozen & track_ok,
                       'klt_mrf': unfrozen & track_ok,
                       'wrong_warp_mrf': unfrozen & track_ok & (wrong_supports.mean(axis=(1, 2)) >= 0.98)}
        crops = {'fixed_roi': raw, 'klt_roi': aligned, 'klt_mrf': aligned, 'wrong_warp_mrf': wrong}
        samples = sorted(set(int(np.argmin(abs(seconds-t))) for t in [10, 30, 50, 70, 90, 110]))
        diagnostics = dict(seconds=seconds, matrices=matrices, track_ok=track_ok, frozen=~unfrozen,
                           sample_indices=np.array(samples), sample_seconds=seconds[samples],
                           raw_samples=raw[samples], aligned_samples=aligned[samples], wrong_samples=wrong[samples])
        statistics = {}
        for variant in VARIANTS:
            frames = crops[variant]
            good = frame_valid[variant]
            valid = np.zeros(n, bool)
            signal = np.zeros(n, np.float32)
            area = np.zeros(n, np.float32)
            reasons = np.full(n, 'invalid_frame', dtype='U32')
            sample_masks, sample_baps = [], []
            packed = np.zeros((n, (h*w+7)//8), np.uint8)
            last_value = 0.0
            indices = deque()
            # Running moments avoid recomputing SD across 100 images at every frame.
            total = np.zeros((h, w), float)
            squares = np.zeros_like(total)
            support_count = np.zeros((h, w), np.int32)
            masks_source = supports if variant != 'wrong_warp_mrf' else wrong_supports
            left = 0
            for i, t in enumerate(seconds):
                while left < i and seconds[left] < t-4:
                    left += 1
                if good[i]:
                    value = frames[i].astype(float)
                    indices.append(i)
                    total += value
                    squares += value**2
                    support_count += masks_source[i] if variant != 'fixed_roi' else 1
                while indices and seconds[indices[0]] < t-4:
                    old = indices.popleft()
                    value = frames[old].astype(float)
                    total -= value
                    squares -= value**2
                    support_count -= masks_source[old] if variant != 'fixed_roi' else 1
                bap = np.zeros((h, w), np.float32)
                mask = np.zeros((h, w), bool)
                if not good[i]:
                    reason = 'invalid_frame'
                elif i == 0 or not good[i-1]:
                    reason = 'no_valid_previous_frame'
                elif variant.endswith('mrf'):
                    if t < 4 or len(indices) < max(2, 0.8*(i-left+1)) or t-seconds[indices[0]] < 3.5:
                        reason = 'insufficient_four_second_support'
                    else:
                        count = len(indices)
                        variance = np.maximum((squares-total**2/count)/(count-1), 0)
                        baf = np.sqrt(variance)
                        support = (support_count == count) & masks_source[i-1]
                        baf[~support] = 0
                        if baf.max() <= 1e-8:
                            reason = 'flat_baf'
                        else:
                            bap = (baf/baf.max()).astype(np.float32)
                            mask, _ = mrf_icm(bap, support)
                            reason = '' if mask.any() else 'empty_mrf_mask'
                else:
                    mask = np.ones((h, w), bool) if variant == 'fixed_roi' else supports[i] & supports[i-1]
                    reason = '' if mask.any() else 'empty_roi'
                if not reason:
                    last_value += differential_increment(frames[i-1], frames[i], mask)
                    valid[i] = True
                    area[i] = mask.mean()
                signal[i] = last_value
                reasons[i] = reason
                packed[i] = np.packbits(mask.ravel())
                if i in samples:
                    sample_masks.append(mask)
                    sample_baps.append(bap)
                if variant.endswith('mrf') and (i+1) % 1000 == 0:
                    print(f'  {person} {variant}: {i+1}/{n} masks', flush=True)
            np.savez_compressed(out / variant / f'{person}.npz', seconds=seconds,
                                marker=np.asarray(markers, np.int8), trace=np.repeat(signal[:, None], 9, axis=1),
                                valid=valid, frame_reason=reasons, mask_fraction=area)
            if variant.endswith('mrf'):
                diagnostics[variant+'_masks'] = np.array(sample_masks)
                diagnostics[variant+'_bap'] = np.array(sample_baps)
                diagnostics[variant+'_packed_masks'] = packed
                diagnostics[variant+'_mask_shape'] = np.array([h, w])
            unique, counts = np.unique(reasons[~valid], return_counts=True)
            statistics[variant] = {'valid_frames': int(valid.sum()), 'excluded_reasons': dict(zip(unique.tolist(), counts.tolist())),
                                    'mean_mask_fraction_on_valid': float(area[valid].mean()) if valid.any() else None}
        np.savez_compressed(out / 'diagnostics' / f'{person}.npz', **diagnostics)
        audit['subjects'][person] = {'frames': n, 'frozen_frames': int((~unfrozen).sum()),
                                    'tracking_failed_frames': int((~track_ok).sum()),
                                    'variants': statistics, 'elapsed_seconds': perf_counter()-started,
                                    'source_pixel_sha256': pixels_hash.hexdigest(),
                                    'csv_sha256': hashlib.sha256(csv.read_bytes()).hexdigest()}
        (out / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        print(f'{person} complete in {perf_counter()-started:.1f}s', flush=True)
    (out / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('Manuel_clean'))
    p.add_argument('--out', type=Path, default=Path('artifacts/kwon_experiment'))
    p.add_argument('--boxes', type=Path, default=Path('config/nose_boxes.json'))
    args = p.parse_args()
    extract(args.root, args.out, args.boxes)


if __name__ == '__main__':
    main()
