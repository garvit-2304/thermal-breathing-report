"""Checks that the exploratory spatial rule rejects the observed edge failure."""
import unittest

import numpy as np

from thermal_face.constrained_mask import intersect_mask, perinasal_prior


class ConstrainedMaskTests(unittest.TestCase):
    def test_center_nose_kept_while_eye_and_face_edge_excluded(self):
        shape = (56, 62)
        original = np.zeros(shape, bool)
        original[27:35, 27:35] = True  # central nose/upper-lip area
        original[5:13, 27:35] = True   # eye band
        original[24:35, 56:62] = True  # face edge
        packed = np.packbits(original.ravel())
        support = np.ones(shape, bool)
        mask = intersect_mask(packed, shape, support, support)
        self.assertGreater(mask[27:35, 27:35].sum(), 40)
        self.assertFalse(mask[5:13].any())
        self.assertFalse(mask[:, 56:].any())
        support[30, 30] = False
        self.assertFalse(intersect_mask(packed, shape, support, support)[30, 30])

    def test_prior_is_crop_relative_and_never_selects_corners(self):
        for shape in [(48, 55), (52, 63), (56, 62)]:
            prior = perinasal_prior(shape)
            self.assertTrue(prior[round(shape[0]*0.61), round(shape[1]*0.5)])
            self.assertFalse(prior[0].any())
            self.assertFalse(prior[:, 0].any())


if __name__ == '__main__':
    unittest.main()
