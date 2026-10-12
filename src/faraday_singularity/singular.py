"""Singular time and exponent of a length that vanishes as a power law, from its latest samples.

If ``x = A |t - t0|^alpha``, the ratio of ``x`` to its rate of change is linear in time on both
sides of the singular time::

    q = x/(dx/dt) = (t - t0)/alpha,

whatever the prefactor ``A``. A least-squares line ``q = a + b t`` through the latest samples
therefore gives ``alpha = 1/b`` and ``t0 = -a/b`` without a search over ``t0``. Before ``t0`` the
length shrinks (``q < 0``) and ``t0`` lies after the samples; after ``t0`` it grows (``q > 0``) and
``t0`` lies before them. ``dx/dt`` is the second-order three-point difference on the non-uniform
time grid, evaluated at the interior samples of the window.

The estimate is NaN when it is ill-conditioned: too few samples, non-increasing times, a length
that is not positive or not monotonic over the window, a slope ``b <= 0``, an exponent outside
``alpha_range``, ``t0`` on the wrong side of the samples, ``t0`` farther from the nearest sample
than ``max_extrapolation`` times the duration of the window, or a line that does not describe ``q``
(root-mean-square residual above ``max_residual`` times the range of ``q``).

The extrapolation rule rejects windows too short for the distance to ``t0``. For a power law it
requires ``|t - t0|`` to change over the window by at least the factor ``1 + 1/max_extrapolation``
(1.5 by default), whatever the exponent: a window that covers only a small change of a smooth
length, such as the apex radius of a trough that does not collapse, otherwise fits a line to ``q``
and extrapolates it to a finite but meaningless ``t0``.
"""

from __future__ import annotations

import math

import numpy as np


def singular_time(t, x, *, min_samples: int = 6, alpha_range: tuple[float, float] = (0.1, 5.0),
                  max_residual: float = 0.05,
                  max_extrapolation: float = 2.0) -> tuple[float, float]:
    """``(t0, alpha)`` of ``x = A |t - t0|^alpha`` from the samples ``(t, x)``, or NaNs."""
    nan = (math.nan, math.nan)
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.shape != x.shape or t.ndim != 1 or t.size < max(min_samples, 5):
        return nan
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(x)) and np.all(x > 0)):
        return nan
    h = np.diff(t)
    dx = np.diff(x)
    if np.any(h <= 0):
        return nan
    shrinking = bool(np.all(dx < 0))
    if not (shrinking or np.all(dx > 0)):
        return nan
    h1, h2 = h[:-1], h[1:]
    dxdt = (-h2 / (h1 * (h1 + h2)) * x[:-2] + (h2 - h1) / (h1 * h2) * x[1:-1]
            + h1 / (h2 * (h1 + h2)) * x[2:])
    if np.any(np.sign(dxdt) != np.sign(dx[0])):
        return nan
    q = x[1:-1] / dxdt
    tm = t[1:-1]
    tc = tm.mean()  # centred for conditioning: q = a_c + b (t - tc)
    b, a_c = np.polyfit(tm - tc, q, 1)
    if not (b > 0):
        return nan
    alpha = 1.0 / b
    t0 = tc - a_c / b
    if not (alpha_range[0] <= alpha <= alpha_range[1]):
        return nan
    if (shrinking and t0 <= t[-1]) or (not shrinking and t0 >= t[0]):
        return nan
    if (t0 - t[-1] if shrinking else t[0] - t0) > max_extrapolation * (t[-1] - t[0]):
        return nan
    span = float(q.max() - q.min())
    residual = float(np.sqrt(np.mean((q - (a_c + b * (tm - tc))) ** 2)))
    if span <= 0 or residual > max_residual * span:
        return nan
    return float(t0), float(alpha)
