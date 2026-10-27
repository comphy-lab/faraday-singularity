"""Bessel function ``J0`` as an even polynomial, usable inside pyoomph expressions.

pyoomph has no Bessel functions, but the free surface only spans ``0 <= r <= R_c`` with
``R_c = j_{1,n}`` of order a few, where the power series ``J0(x) = sum_k (-1)^k (x^2/4)^k/(k!)^2``
converges fast. Truncated after ``K`` terms its error is below the first omitted term
``(x/2)^(2K+2)/((K+1)!)^2``, which is under 1e-18 for ``x <= 4`` with the default ``K = 16`` and
under 1e-15 for ``x <= 7.1`` (the second mode) with ``K = 24``.
"""

from __future__ import annotations

import math


def j0_series_coefficients(terms: int) -> list[float]:
    """Coefficients ``c_k`` of ``J0(x) = sum_k c_k (x^2)^k``, ``k = 0..terms-1``."""
    return [(-1) ** k / (4.0**k * math.factorial(k) ** 2) for k in range(terms)]


def terms_for_radius(x_max: float, tol: float = 1e-15) -> int:
    """Smallest series length whose first omitted term is below ``tol`` on ``0 <= x <= x_max``."""
    k = 0
    while (x_max / 2.0) ** (2 * k + 2) / math.factorial(k + 1) ** 2 >= tol:
        k += 1
    return k + 1


def j0_polynomial(x, terms: int):
    """Horner evaluation of the truncated series; ``x`` may be a float, array or pyoomph expression."""
    coeffs = j0_series_coefficients(terms)
    x2 = x * x
    p = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        p = p * x2 + c
    return p
