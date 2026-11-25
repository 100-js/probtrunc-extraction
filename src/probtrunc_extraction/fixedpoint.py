"""Exact integer reference utilities and sanity checks for signed fixed-point arithmetic."""

from operator import index
from random import Random

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


def fits_int64(value: int) -> bool:
    return INT64_MIN <= index(value) <= INT64_MAX


def signed_bit_width(magnitude: int) -> int:
    """Return signed two's-complement bit width 1 + ceil(log2(magnitude + 1))."""
    mag = abs(index(magnitude))
    return mag.bit_length() + 1


def scaled_quotient(numerator: int, scale: int) -> tuple[int, int]:
    """Return the Euclidean floor quotient and nonnegative remainder in [0, scale)."""
    scale = index(scale)
    if scale <= 0:
        raise ValueError("Fixed-point scale must be positive")
    return divmod(index(numerator), scale)


def round_nearest_even(numerator: int, scale: int) -> int:
    """Round numerator / scale to the nearest integer with ties-to-even semantics."""
    lower, remainder = scaled_quotient(numerator, scale)
    return lower + int(2 * remainder > scale or (2 * remainder == scale and lower % 2 == 1))


def check_arithmetic(seed: int = 42) -> dict:
    """Verify signed int64 bounds, nearest-even rounding, and PT carry probabilities."""
    ties = {
        "5/2": round_nearest_even(5, 2),
        "7/2": round_nearest_even(7, 2),
        "-5/2": round_nearest_even(-5, 2),
        "-7/2": round_nearest_even(-7, 2),
        "(2^62+3)/2": round_nearest_even((1 << 62) + 3, 2),
    }
    bounds = {
        "minimum_fits": fits_int64(INT64_MIN),
        "maximum_fits": fits_int64(INT64_MAX),
        "below_minimum_fits": fits_int64(INT64_MIN - 1),
        "above_maximum_fits": fits_int64(INT64_MAX + 1),
    }
    if list(ties.values()) != [2, 4, -2, -4, (1 << 61) + 2]:
        raise ArithmeticError("Nearest-even rounding failed an integer fixture")
    if list(bounds.values()) != [True, True, False, False]:
        raise ArithmeticError("Signed 64-bit bounds failed an integer fixture")

    scale = 8
    samples = 4096
    rng = Random(seed)
    distribution = []
    for numerator in (5, -5, -8):
        lower, remainder = scaled_quotient(numerator, scale)
        exhaustive_upper_count = sum(
            (numerator + draw) // scale == lower + 1 for draw in range(scale)
        )
        observed_upper_count = sum(
            (numerator + rng.randrange(scale)) // scale == lower + 1 for _ in range(samples)
        )
        if exhaustive_upper_count != remainder or lower * scale + remainder != numerator:
            raise ArithmeticError("Signed stochastic truncation failed an integer fixture")
        distribution.append({
            "numerator": numerator,
            "scale": scale,
            "lower": lower,
            "upper": lower + 1,
            "upper_probability": remainder / scale,
            "upper_count_over_all_offsets": exhaustive_upper_count,
            "sampled_upper_count": observed_upper_count,
            "sample_count": samples,
        })

    return {
        "seed": seed,
        "nearest_even": ties,
        "signed_int64": bounds,
        "scalar_truncation_distribution": distribution,
    }
