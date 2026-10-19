"""CSRT nasal tracking experiment, initialized from the existing first-frame box.

No RGB initialization, detector, label-based tuning, or manual recovery is used.
Unsuccessful/out-of-frame boxes are missing measurements, never fixed-box fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .data import SUBJECTS, read_respiration


def create_tracker():
    import cv2
    factory = getattr(cv2, 'TrackerCSRT_create', None)
    if factory is None:
        factory = getattr(getattr(cv2, 'legacy', None), 'TrackerCSRT_create', None)
    if factory is None:
        raise RuntimeError('CSRT requires opencv-contrib-python; the base opencv-python package is insufficient')
    return factory()  # Published OpenCV defaults; no tuning against respiration labels.


def integer_box(box, shape):
    """Reject invalid boxes rather than silently changing the region by clipping."""
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        return None
    x, y, w, h = (int(round(v)) for v in values)
    if min(w, h) < 3 or min(x, y) < 0 or x + w > shape[1] or y + h > shape[0]:
        return None
    return x, y, w, h


def patch_means(frame, box):
    x, y, w, h = box
    crop = frame[y:y+h, x:x+w].astype(np.float32)
    return np.asarray([crop[r*h//3:(r+1)*h//3, c*w//3:(c+1)*w//3].mean()
                       for r in range(3) for c in range(3)], dtype=np.float32)


def unfrozen_mask(repeated):
    """Match the fixed extractor's >=5 identical-frame exclusion exactly."""
    valid = np.ones(len(repeated), dtype=bool)
    start = None
    for i in range(1, len(repeated) + 1):
        same = i < len(repeated) and repeated[i]
        if same and start is None:
            start = i - 1
        elif not same and start is not None:
            if i - start >= 5:
                valid[start:i] = False
            start = None
    return valid


class NoseTracker:
    """Track only images; no access to subject labels or reference events."""
    def __init__(self, frame, box, factory=create_tracker):
        self.tracker = factory()
        self.box = integer_box(box, frame.shape)
        if self.box is None:
            raise ValueError('Initial box must fit inside the image and contain nine nonempty patches')
        result = self.tracker.init(frame, self.box)
        if result is False:
            raise RuntimeError('CSRT initialization failed')
        self.ok = True

    def update(self, frame, repeated=False):
        if repeated:
            return self.box if self.ok else None, 'duplicate' if self.ok else 'lost'
        was_ok = self.ok
        success, candidate = self.tracker.update(frame)
        box = integer_box(candidate, frame.shape) if success else None
        self.ok = box is not None
        if self.ok:
            self.box = box
        # Continue asking CSRT to update after loss; no synthetic/reinitialized boxes.
        return box, ('tracked' if was_ok else 'recovered') if self.ok else 'lost'


def extract(root: Path, out: Path, boxes_path: Path, fixed_reference: Path | None = None,
            subjects=SUBJECTS):
    import cv2
    cv2.setNumThreads(2)
    cv2.setRNGSeed(7)
    boxes = json.loads(boxes_path.read_text(encoding='utf-8'))
    out.mkdir(parents=True, exist_ok=True)
    for variant in ['fixed', 'tracked']:
        (out / variant).mkdir(exist_ok=True)
    report = {
        'tracker': 'OpenCV CSRT, default parameters', 'opencv': cv2.__version__,
        'initialization': 'existing hand-inspected first-frame box; no RGB',
        'failure_policy': 'missing; no fixed-box fallback; subsequent CSRT updates may recover',
        'freeze_policy': 'skip tracker update on identical frames; flag runs >=5 as missing',
        'box_policy': 'CSRT position and scale; rounded integer boxes; reject outside image',
        'seed': 7, 'opencv_threads': 2, 'subjects': {}, 'excluded': {},
        'boxes_sha256': hashlib.sha256(boxes_path.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for subject in subjects:
        paths = sorted((root / subject).glob('frame_*.png'))
        times, markers = read_respiration(next(root.glob(f'*{subject}_respiration.csv')))
        if len(paths) != len(times):
            report['excluded'][subject] = f'{len(paths)} frames versus {len(times)} reference rows'
            print(f'Skipping {subject}: unverified alignment', flush=True)
            continue
        if not paths:
            raise ValueError(f'No frames for {subject}')
        started = perf_counter()
        fixed = np.zeros((len(paths), 9), dtype=np.float32)
        tracked = np.zeros_like(fixed)
        repeated = np.zeros(len(paths), dtype=bool)
        track_ok = np.zeros(len(paths), dtype=bool)
        track_boxes = np.full((len(paths), 4), -1, dtype=np.int32)
        states = np.empty(len(paths), dtype='U10')
        previous = None
        pixels_hash = hashlib.sha256()
        print(f'{subject}: extracting and tracking {len(paths)} original frames', flush=True)
        for i, path in enumerate(paths):
            frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if frame is None:
                raise ValueError(f'Unreadable image: {path}')
            if frame.ndim == 3 and frame.shape[2] == 3:
                if not (np.array_equal(frame[:, :, 0], frame[:, :, 1]) and
                        np.array_equal(frame[:, :, 0], frame[:, :, 2])):
                    raise ValueError(f'Expected identical grayscale channels: {path}')
                gray = frame[:, :, 0]
            elif frame.ndim == 2:
                gray = frame
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                raise ValueError(f'Unsupported frame: {path}')
            if gray.dtype != np.uint8:
                raise ValueError('This experiment expects the original 8-bit images')
            pixels_hash.update(gray.tobytes())
            fixed_box = integer_box(boxes[subject], gray.shape)
            if fixed_box is None:
                raise ValueError(f'Invalid configured box for {subject}')
            fixed[i] = patch_means(gray, fixed_box)
            if i == 0:
                tracker = NoseTracker(frame, fixed_box)
                box, state = fixed_box, 'initial'
            else:
                repeated[i] = np.array_equal(gray, previous)
                box, state = tracker.update(frame, bool(repeated[i]))
            previous = gray.copy()
            states[i] = state
            if box is not None:
                track_ok[i] = True
                track_boxes[i] = box
                tracked[i] = patch_means(gray, box)
            if (i + 1) % 500 == 0:
                print(f'  {subject}: {i+1}/{len(paths)} frames; {np.count_nonzero(~track_ok[:i+1])} failed boxes; '
                      f'{perf_counter()-started:.1f}s', flush=True)
        valid = unfrozen_mask(repeated)
        seconds = np.array([(t-times[0]).total_seconds() for t in times])
        base = dict(seconds=seconds, marker=np.asarray(markers, dtype=np.int8))
        comparison = None
        if fixed_reference is not None:
            reference = fixed_reference / f'{subject}.npz'
            with np.load(reference, allow_pickle=False) as saved:
                comparison = all(np.array_equal(saved[k], v) for k, v in
                                 (base | {'trace': fixed, 'valid': valid}).items())
            if not comparison:
                raise AssertionError(f'Fixed extraction differs from the existing baseline: {subject}')
        np.savez_compressed(out / 'fixed' / f'{subject}.npz', **base, trace=fixed, valid=valid)
        np.savez_compressed(out / 'tracked' / f'{subject}.npz', **base,
                            trace=tracked, valid=valid & track_ok,
                            track_ok=track_ok, boxes=track_boxes, state=states, frozen=~valid)
        info = {
            'frames': len(paths), 'frozen_frames': int((~valid).sum()),
            'tracking_failed_frames': int((~track_ok).sum()),
            'tracking_success_fraction': float(track_ok.mean()),
            'valid_tracked_frames': int((valid & track_ok).sum()),
            'recoveries': int((states == 'recovered').sum()),
            'source_pixel_sha256': pixels_hash.hexdigest(),
            'reference_csv_sha256': hashlib.sha256(next(root.glob(f'*{subject}_respiration.csv')).read_bytes()).hexdigest(),
            'fixed_matches_previous_trace_exactly': comparison,
            'elapsed_seconds': perf_counter()-started,
        }
        report['subjects'][subject] = info
        (out / 'tracking_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'{subject} completed: {info}', flush=True)
    (out / 'tracking_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('artifacts/tracking_experiment'))
    parser.add_argument('--boxes', type=Path, default=Path('config/nose_boxes.json'))
    parser.add_argument('--fixed-reference', type=Path)
    args = parser.parse_args()
    extract(args.root, args.out, args.boxes, args.fixed_reference)


if __name__ == '__main__':
    main()
