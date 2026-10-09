"""Fits of a mode-amplitude history ``a(t)`` for the linear verification cases.

``fit_damped_cosine`` returns the complex rate ``s`` of ``a ~ Re[C exp(s t)]`` for a freely decaying
mode. ``floquet_exponent`` returns the real Floquet exponent ``mu`` of a subharmonic response, for
which ``a(t + T) = -exp(mu T) a(t)`` once one Floquet mode dominates (``T`` the forcing period).
Both expect a single dominant mode in the fitting window; their residuals report how well that holds.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def _window(t: np.ndarray, a: np.ndarray, t_start: float, t_end: float | None):
    t = np.asarray(t, dtype=float)
    a = np.asarray(a, dtype=float)
    sel = t >= t_start
    if t_end is not None:
        sel &= t <= t_end
    if sel.sum() < 8:
        raise ValueError(f"too few samples in the fitting window [{t_start}, {t_end}]")
    return t[sel], a[sel]


def fit_damped_cosine(t, a, *, t_start: float, t_end: float | None = None,
                      omega_guess: float, rate_guess: float = 0.0) -> dict:
    """Least-squares fit of ``a = A exp(sr t) cos(si t + phi) + c`` on ``[t_start, t_end]``.

    ``c`` absorbs any slow mean offset of the projected amplitude. Returns ``s_real``, ``s_imag``,
    the relative RMS residual and the window actually used.
    """
    tw, aw = _window(t, a, t_start, t_end)
    t0 = tw[0]
    x = tw - t0
    scale = np.max(np.abs(aw))

    def model(p):
        amp, sr, si, phi, c = p
        return amp * np.exp(sr * x) * np.cos(si * x + phi) + c

    def resid(p):
        return (model(p) - aw) / scale

    i0 = int(np.argmax(np.abs(aw[: max(8, len(aw) // 10)])))
    phi0 = 0.0 if aw[i0] > 0 else np.pi
    p0 = [scale, rate_guess, omega_guess, phi0 - omega_guess * x[i0], 0.0]
    sol = least_squares(resid, p0, x_scale="jac", method="lm", xtol=1e-14, ftol=1e-14)
    _, sr, si, _, c = sol.x
    rms = float(np.sqrt(np.mean(sol.fun**2)))
    return {"s_real": float(sr), "s_imag": float(abs(si)), "offset": float(c),
            "relative_rms_residual": rms, "t_start": float(tw[0]), "t_end": float(tw[-1]),
            "samples": int(tw.size)}


def floquet_exponent(t, a, *, period: float, t_start: float, t_end: float | None = None,
                     samples_per_period: int = 64) -> dict:
    """Real Floquet exponent of a subharmonic response from ``a(t + T) = -exp(mu T) a(t)``.

    ``a`` is interpolated onto a uniform grid of ``samples_per_period`` points per forcing period
    over the window, and the multiplier is the least-squares ratio
    ``-sum a(t) a(t+T) / sum a(t)**2``. A positive multiplier confirms the subharmonic response; a
    negative one (a harmonic response) is reported rather than hidden.
    """
    tw, aw = _window(t, a, t_start, t_end)
    n_periods = int(np.floor((tw[-1] - tw[0]) / period))
    if n_periods < 2:
        raise ValueError("need at least two forcing periods in the fitting window")
    grid = tw[0] + np.arange(n_periods * samples_per_period + 1) * (period / samples_per_period)
    ag = np.interp(grid, tw, aw)
    lead, lag = ag[:-samples_per_period], ag[samples_per_period:]
    multiplier = -float(np.dot(lead, lag) / np.dot(lead, lead))
    resid = lag + multiplier * lead
    rel = float(np.sqrt(np.mean(resid**2) / np.mean(lag**2)))
    out = {"multiplier": multiplier, "relative_rms_residual": rel, "periods": n_periods,
           "t_start": float(grid[0]), "t_end": float(grid[-1])}
    out["mu"] = float(np.log(multiplier) / period) if multiplier > 0 else float("nan")
    return out
