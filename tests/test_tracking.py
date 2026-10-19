"""Tracking loss, timing preservation, and paired-comparison invariants."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from thermal_face.tracking import NoseTracker, integer_box, patch_means, unfrozen_mask
from thermal_face.compare_tracking import paired_windows, run
from thermal_face.windows import Window


class TrackingTests(unittest.TestCase):
    def test_loss_duplicate_and_recovery_never_return_stale_box(self):
        class FakeTracker:
            def init(self, frame, box):
                self.calls = 0

            def update(self, frame):
                self.calls += 1
                return [(True, (8, 8, 9, 9)), (False, (8, 8, 9, 9)),
                        (True, (-5, 8, 9, 9)), (True, (10, 8, 9, 9))][self.calls-1]

        frame = np.zeros((30, 40, 3), dtype=np.uint8)
        tracker = NoseTracker(frame, (7, 7, 9, 9), FakeTracker)
        self.assertEqual(tracker.update(frame), ((8, 8, 9, 9), 'tracked'))
        self.assertEqual(tracker.update(frame), (None, 'lost'))
        self.assertEqual(tracker.update(frame, repeated=True), (None, 'lost'))
        self.assertEqual(tracker.tracker.calls, 2)
        self.assertEqual(tracker.update(frame), (None, 'lost'))
        self.assertEqual(tracker.update(frame), ((10, 8, 9, 9), 'recovered'))

    def test_freeze_runs_preserve_length_including_final_run(self):
        repeat = np.array([False, True, True, True, True, False, True, False, True, True, True, True])
        mask = unfrozen_mask(repeat)
        self.assertEqual(len(mask), len(repeat))
        np.testing.assert_array_equal(mask, [False]*5 + [True]*2 + [False]*5)
        self.assertIsNone(integer_box((0, 0, 2, 9), (30, 40)))
        self.assertIsNone(integer_box((39, 0, 9, 9), (30, 40)))
        image = np.arange(81, dtype=np.uint8).reshape(9, 9)
        np.testing.assert_array_equal(patch_means(image, (0, 0, 9, 9)),
                                      [10, 13, 16, 37, 40, 43, 64, 67, 70])

    def test_actual_csrt_follows_translated_texture(self):
        import cv2
        if not hasattr(cv2, 'TrackerCSRT_create'):
            self.skipTest('OpenCV contrib CSRT not installed')
        cv2.setNumThreads(2)
        rng = np.random.default_rng(1)
        texture = rng.integers(40, 240, (30, 30, 3), dtype=np.uint8)
        def frame(dx):
            image = np.zeros((120, 160, 3), dtype=np.uint8)
            image[40:70, 40+dx:70+dx] = texture
            return image
        tracker = NoseTracker(frame(0), (40, 40, 30, 30))
        for dx in range(1, 11):
            box, state = tracker.update(frame(dx))
            self.assertIsNotNone(box)
        self.assertAlmostEqual(box[0] + box[2]/2, 65, delta=5)

    def test_pairing_excludes_windows_from_training_and_testing(self):
        def window(person, start, valid=True):
            return Window(person, start, start+20, 'still', np.zeros((10, 200), np.float32),
                          18.0, 0.0, None if valid else 'missing_fraction_over_20_percent', None)
        fixed = {p: [window(p, 0), window(p, 5)] for p in ['a', 'b', 'c']}
        tracked = {p: [window(p, 0), window(p, 5, p != 'a')] for p in fixed}
        self.assertEqual(paired_windows(fixed, tracked)['a'], [0])
        tracked['b'][0].reference_bpm = 20
        with self.assertRaisesRegex(ValueError, 'Reference labels'):
            paired_windows(fixed, tracked)

    def test_comparison_folds_are_identical_and_training_labels_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for variant in ['fixed', 'tracked']:
                (folder / variant).mkdir()
                for person, period in [('a', 3), ('b', 4), ('c', 5)]:
                    t = np.arange(0, 30.01, 0.1)
                    trace = np.stack([np.sin(t * (2*np.pi/period) + j/10) for j in range(9)], axis=1)
                    marker = np.zeros(len(t), np.int8)
                    marker[::period*10] = 1
                    valid = np.ones(len(t), dtype=bool)
                    if variant == 'tracked' and person == 'a':
                        valid[t > 23] = False
                    np.savez(folder / variant / f'{person}.npz', seconds=t, trace=trace, valid=valid, marker=marker)
            calls = []
            def predictor(train, test_x, *args):
                calls.append((len(train), len(test_x)))
                return np.full(len(test_x), np.median([y for _, y in train])), 0
            with patch('thermal_face.compare_tracking.fit_predict', side_effect=predictor):
                report = run(folder, folder / 'result.json', seeds=[7], epochs=1)
            variants = report['runs'][0]['variants']
            self.assertEqual(variants['fixed']['folds'], variants['tracked']['folds'])
            self.assertEqual(calls[:6], calls[6:])
            self.assertEqual(report['runs'][0]['comparison']['cnn_gru']['tracked_minus_fixed_mae_bpm'], 0)
            for fold in variants['fixed']['folds']:
                self.assertNotIn(fold['held_out'], fold['train_subjects'])
                labels = [w['reference_bpm'] for w in variants['fixed']['windows'] if w['subject'] in fold['train_subjects']]
                self.assertEqual(fold['train_median_bpm'], float(np.median(labels)))
            self.assertLess(report['coverage']['tracked']['a']['eligible'], report['coverage']['fixed']['a']['eligible'])


if __name__ == '__main__':
    unittest.main()
