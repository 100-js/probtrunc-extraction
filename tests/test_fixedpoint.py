import unittest

from probtrunc_extraction.fixedpoint import (
    INT64_MAX,
    INT64_MIN,
    check_arithmetic,
    fits_int64,
    round_nearest_even,
    scaled_quotient,
    signed_bit_width,
)


class FixedPointTests(unittest.TestCase):
    def test_nearest_even_and_signed_bit_width(self):
        for numerator, scale, expected in (
            (5, 2, 2),
            (7, 2, 4),
            (-5, 2, -2),
            (-7, 2, -4),
            (-4, 3, -1),
            (-5, 3, -2),
            ((1 << 62) + 3, 2, (1 << 61) + 2),
        ):
            with self.subTest(numerator=numerator, scale=scale):
                self.assertEqual(round_nearest_even(numerator, scale), expected)

        self.assertEqual(signed_bit_width(0), 1)
        self.assertEqual(signed_bit_width(7), 4)
        self.assertEqual(signed_bit_width(8), 5)
        self.assertEqual(signed_bit_width(45_128_270_587_055_141), 57)
        with self.assertRaises(ValueError):
            scaled_quotient(1, 0)

    def test_signed_bounds(self):
        self.assertTrue(fits_int64(INT64_MIN))
        self.assertTrue(fits_int64(INT64_MAX))
        self.assertFalse(fits_int64(INT64_MIN - 1))
        self.assertFalse(fits_int64(INT64_MAX + 1))

    def test_signed_scalar_distribution(self):
        for seed in (0, 42, 2026):
            with self.subTest(seed=seed):
                records = check_arithmetic(seed)["scalar_truncation_distribution"]
                self.assertEqual([record["lower"] for record in records], [0, -1, -1])
                self.assertEqual(
                    [record["upper_count_over_all_offsets"] for record in records], [5, 3, 0]
                )
                for record in records:
                    observed = record["sampled_upper_count"] / record["sample_count"]
                    self.assertAlmostEqual(observed, record["upper_probability"], delta=0.03)


if __name__ == "__main__":
    unittest.main()
