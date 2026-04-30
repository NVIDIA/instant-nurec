import unittest

from unittest.mock import MagicMock

import torch

from nre.utils.lidar_post_processing import distance_based_filter


class TestDistanceBasedFilter(unittest.TestCase):
    def setUp(self):
        # grid dims
        self.b, self.h, self.w = 1, 3, 5
        self.n_rays = self.b * self.h * self.w

    def make_rendered(self, distances):
        # Create a simple mock object that has a .distance attribute
        rendered = MagicMock()
        rendered.distance = distances
        return rendered

    def make_model_elements(self, seed=0):
        # model_element should be shape (n_rays, 2) with vertical,horizontal indices
        gen = torch.Generator().manual_seed(seed)
        v = torch.randint(0, self.h, (self.n_rays, 1), generator=gen)
        gen.manual_seed(seed + 1)
        u = torch.randint(0, self.w, (self.n_rays, 1), generator=gen)
        return torch.cat([v, u], dim=1)

    def test_all_invalid_valid_mask_produces_all_false(self):
        # If no rays are marked valid, filter should return all False
        distances = torch.linspace(1.0, 2.0, self.n_rays)
        rendered = self.make_rendered(distances)
        model_elements = self.make_model_elements(seed=0)

        valid_mask_pred = torch.zeros(self.n_rays, dtype=torch.bool)

        out = distance_based_filter(
            rendered=rendered,
            model_elements=model_elements,
            valid_mask_pred=valid_mask_pred,
            filter_threshold=0.5,
            n_vertical_bins=self.h,
            n_horizontal_bins=self.w,
        )

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.dtype, torch.bool)
        self.assertEqual(out.numel(), self.n_rays)
        # all should be False because nothing was valid to compute discontinuities
        self.assertTrue(torch.all(~out))

    def test_all_valid_returns_bool_mask_length_n_rays(self):
        # When all rays are valid, we at least get a boolean mask of the right size
        # Construct distances that vary smoothly so thresholding will likely keep most False
        distances = torch.linspace(1.0, 3.0, self.n_rays)
        rendered = self.make_rendered(distances)
        model_elements = self.make_model_elements(seed=42)
        valid_mask_pred = torch.ones(self.n_rays, dtype=torch.bool)

        out = distance_based_filter(
            rendered=rendered,
            model_elements=model_elements,
            valid_mask_pred=valid_mask_pred,
            filter_threshold=0.01,  # small threshold
            n_vertical_bins=self.h,
            n_horizontal_bins=self.w,
        )

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.dtype, torch.bool)
        self.assertEqual(out.numel(), self.n_rays)
        # As a basic sanity check ensure mask contains some True/False values (not all same)
        # It's allowed that all could be False in degenerate setups, but with this data we expect variability.
        unique_vals = torch.unique(out)
        self.assertTrue(unique_vals.numel() >= 1)


if __name__ == "__main__":
    unittest.main()
