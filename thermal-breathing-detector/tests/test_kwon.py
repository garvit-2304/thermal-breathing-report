"""Tests of paper equations, stabilization direction and temporal-mask signal extraction."""
import unittest
import cv2
import numpy as np

from thermal_face.kwon import baf_bap, crop_warp, differential_increment, klt_step, mrf_icm
from thermal_face.evaluate_kwon import pairwise, peak_rate


class KwonTests(unittest.TestCase):
    def test_inverse_warp_and_roi_offset_restore_translation(self):
        rng = np.random.default_rng(4)
        reference = rng.integers(0, 255, (80, 90), dtype=np.uint8)
        matrix = np.float64([[1, 0, 5], [0, 1, 3]])
        current = cv2.warpAffine(reference, matrix, (90, 80))
        correct, support = crop_warp(current, matrix, (20, 20, 30, 30))
        wrong, _ = crop_warp(current, matrix, (20, 20, 30, 30), correct=False)
        np.testing.assert_array_equal(correct, reference[20:50, 20:50])
        self.assertTrue(support.all())
        self.assertGreater(np.abs(wrong.astype(float)-correct).mean(), 10)
        _, outside = crop_warp(current, matrix, (70, 60, 30, 30))
        self.assertLess(outside.mean(), 0.98)

    def test_baf_is_sample_sd_of_integrated_differences(self):
        frames = np.array([[[1., 5.]], [[3., 5.]], [[5., 5.]]])
        baf, bap = baf_bap(frames)
        np.testing.assert_allclose(baf, [[2, 0]])
        np.testing.assert_allclose(bap, [[1, 0]])
        integrated = np.concatenate([np.zeros_like(frames[:1]), np.cumsum(np.diff(frames, axis=0), axis=0)])
        np.testing.assert_allclose(baf_bap(integrated)[0], baf)

    def test_mrf_localizes_high_bap_without_forcing_half_the_image(self):
        rng = np.random.default_rng(3)
        bap = np.clip(0.1 + rng.normal(0, 0.01, (30, 30)), 0, 1)
        bap[10:18, 12:20] = 0.9 + rng.normal(0, 0.01, (8, 8))
        mask, iterations = mrf_icm(bap)
        self.assertTrue(mask[10:18, 12:20].all())
        self.assertLess(mask.mean(), 0.1)
        self.assertLessEqual(iterations, 30)
        self.assertFalse(mrf_icm(np.zeros((10, 10)))[0].any())
        support = np.ones((30, 30), bool)
        support[10:18, 12:20] = False
        self.assertFalse(mrf_icm(bap, support)[0][~support].any())

    def test_changing_mask_does_not_create_signal_on_static_image(self):
        image = np.arange(100).reshape(10, 10)
        first = image < 20
        second = image > 80
        self.assertNotEqual(image[first].mean(), image[second].mean())
        self.assertEqual(differential_increment(image, image, second), 0)
        self.assertEqual(differential_increment(image, image+3, first), 3)
        self.assertIsNone(differential_increment(image, image, np.zeros_like(image, bool)))

    def test_klt_estimates_known_small_motion(self):
        cv2.setNumThreads(2)
        rng = np.random.default_rng(4)
        image = rng.integers(0, 255, (100, 100), dtype=np.uint8)
        points = cv2.goodFeaturesToTrack(image, 70, 0.03, 5)
        shifted = cv2.warpAffine(image, np.float32([[1, 0, 2], [0, 1, 1]]), (100, 100))
        matrix, _ = klt_step(image, shifted, points)
        self.assertIsNotNone(matrix)
        np.testing.assert_allclose(matrix[:, 2], [2, 1], atol=0.3)

    def test_pairwise_scoring_does_not_reward_dropped_hard_windows(self):
        def row(person, prediction):
            return dict(subject=person, start_s=5., end_s=25., condition='still',
                        reference_bpm=18., eligible=True, predictions_bpm={'spectral': prediction})
        first = [row('a', 18), row('b', 30)]
        second = [row('a', 20), row('b', None)]
        result = pairwise(first, second, 'spectral')
        self.assertEqual(result['common_windows'], 1)
        self.assertEqual(result['second_minus_first_bpm'], 2)
        second[0]['reference_bpm'] = 19
        with self.assertRaises(ValueError):
            pairwise(first, second, 'spectral')

    def test_peak_counter_on_known_synthetic_rate(self):
        t = np.arange(200)/10
        x = np.repeat(np.sin(2*np.pi*0.3*t)[None], 10, axis=0)
        predicted, reason = peak_rate(x)
        self.assertIsNone(reason)
        self.assertAlmostEqual(predicted, 18, delta=3)


if __name__ == '__main__':
    unittest.main()
