"""The duct of the gas, read from the free surface.

The open chain of interface nodes runs from the cavity's bottom on the axis up the pocket wall,
through the neck and up the cavity wall to the container wall, with the liquid velocity at each
node (``FaradayProblem.interface_arrays``: ``r, z, u_r, u_z``). Between the heights ``z_lo`` and
``z_hi`` of the gas model the chain must enter once, be strictly monotone in ``z`` and never come
back below ``z_hi`` beyond it (:func:`single_valued`); the cross-section of the gas is then
``S(z) = pi r(z)^2``, with ``r(z)`` piecewise linear through every node, the midside nodes of the
quadratic elements included.

:class:`ChainGeom` is the per-frame geometry of the one-way neck gas check (its ``FrameGeom``) on a
chain given as arrays: the gas volume below ``z``, ``Vcum(z)``, the exact sum of frusta from the
axis, and its kinematic rate ``dVcum/dt = int 2 pi r (u_r dz - u_z dr)`` along the chain below
``z`` (Reynolds transport, the normal velocity only), both exact for a piecewise-linear chain with
piecewise-linear velocities, and ``dS/dt`` at fixed ``z``. Its arithmetic is that of the check, so
that the coupler reproduces the check's gas solve on the same chains. Units are the solver's.
"""

from __future__ import annotations

import math

import numpy as np

from ..collapse import Profile, necks


class GasGeometryError(ValueError):
    """The chain is not a duct the gas model can read between ``z_lo`` and ``z_hi``; ``info``
    says why (``reason``) and where."""

    def __init__(self, message: str, info: dict):
        super().__init__(message)
        self.info = dict(info)


def _seg_int(a0, da, b0, db, w):
    """int_0^w (a0 + s da)(b0 + s db) ds."""
    return a0 * b0 * w + 0.5 * (a0 * db + da * b0) * w ** 2 + da * db * w ** 3 / 3.0


def single_valued(r, z, z_lo: float, z_hi: float) -> dict | None:
    """Why the chain ``(r, z)`` (axis first) is no duct between ``z_lo`` and ``z_hi``, or None: it
    must start below ``z_lo`` (the bubble zone has a volume), enter ``[z_lo, z_hi]`` once, be
    strictly increasing in ``z`` there, and lie below ``z_lo`` before and above ``z_hi`` after
    (it never returns below ``z_hi``)."""
    r, z = np.asarray(r, dtype=float), np.asarray(z, dtype=float)
    inside = np.where((z >= z_lo) & (z <= z_hi))[0]
    base = {"z_lo": float(z_lo), "z_hi": float(z_hi), "z_axis": float(z[0])}
    if not z[0] < z_lo:
        return dict(base, reason="bottom-above-z_lo")
    if inside.size == 0:
        return dict(base, reason="no-entry")
    runs = np.split(inside, np.where(np.diff(inside) > 1)[0] + 1)
    if len(runs) != 1:
        return dict(base, reason="re-entry", runs=len(runs),
                    z=[float(z[x[0]]) for x in runs])
    run = runs[0]
    if run.size > 1 and not np.all(np.diff(z[run]) > 0):
        k = int(run[np.argmax(np.diff(z[run]) <= 0)])
        return dict(base, reason="not-monotone", r=float(r[k]), z=float(z[k]))
    before, after = z[:run[0]], z[run[-1] + 1:]
    if before.size and not np.all(before < z_lo):
        return dict(base, reason="above-before-entry", z=float(before.max()))
    if after.size and not np.all(after > z_hi):
        return dict(base, reason="returns-below-z_hi", z=float(after.min()))
    return None


def window_minimum(r, z, lo: float, hi: float) -> tuple[float, float, int]:
    """The smallest radius of the nodes with ``lo < z < hi`` and its height and index (NaN, -1
    without a node there)."""
    r, z = np.asarray(r, dtype=float), np.asarray(z, dtype=float)
    win = (z > lo) & (z < hi)
    if not win.any():
        return math.nan, math.nan, -1
    j = int(np.where(win)[0][np.argmin(r[win])])
    return float(r[j]), float(z[j]), j


def chain_neck(r, z, lo: float, hi: float, window: tuple[float, float]) -> tuple[float, float, int,
                                                                             str]:
    """The neck the gas model follows: the narrowest interior local minimum of ``r`` along the
    chain (``collapse.necks``, as the collapse driver measures it) with ``lo < z < hi``, else the
    smallest radius in ``window`` (before a neck has formed). Returns ``(r_n, z_n, index,
    source)``, ``source`` ``neck`` or ``window``."""
    for r_n, z_n, i in necks(Profile(np.asarray(r, dtype=float), np.asarray(z, dtype=float))):
        if lo < z_n < hi:
            return r_n, z_n, i, "neck"
    r_n, z_n, i = window_minimum(r, z, *window)
    return r_n, z_n, i, "window"


class ChainGeom:
    """The chain restricted to ``[z_lo, z_hi]`` with its cumulative volume and kinematic rate."""

    def __init__(self, r, z, ur, uz, z_lo: float, z_hi: float):
        r, z, ur, uz = (np.asarray(x, dtype=float) for x in (r, z, ur, uz))
        dz, dr = np.diff(z), np.diff(r)
        dur, duz = np.diff(ur), np.diff(uz)
        vol = math.pi * dz * (r[:-1] ** 2 + r[:-1] * r[1:] + r[1:] ** 2) / 3.0
        rate = 2 * math.pi * (dz * _seg_int(r[:-1], dr, ur[:-1], dur, 1.0)
                              - dr * _seg_int(r[:-1], dr, uz[:-1], duz, 1.0))
        Vc = np.r_[0.0, np.cumsum(vol)]
        Rc = np.r_[0.0, np.cumsum(rate)]
        # monotone portion covering [z_lo, z_hi]
        i0 = int(np.where(z < z_lo)[0].max()) if (z < z_lo).any() else 0
        above = np.where(z > z_hi)[0]
        i1 = int(above[above > i0].min()) if above.size else len(z) - 1
        sl = slice(i0, i1 + 1)
        zz = z[sl]
        if not np.all(np.diff(zz) > 0):
            raise GasGeometryError(f"z not monotone in [{z_lo}, {z_hi}]",
                                   {"reason": "not-monotone", "z_lo": float(z_lo),
                                    "z_hi": float(z_hi)})
        self.z, self.r, self.ur, self.uz = zz, r[sl], ur[sl], uz[sl]
        self.Vc, self.Rc = Vc[sl], Rc[sl]
        self.z_lo, self.z_hi = float(z_lo), float(z_hi)

    def at(self, zq):
        """S, Vcum, dVcum/dt and dS/dt (fixed z) at heights zq."""
        z, r, ur, uz = self.z, self.r, self.ur, self.uz
        i = np.clip(np.searchsorted(z, zq) - 1, 0, len(z) - 2)
        dz = z[i + 1] - z[i]
        w = (zq - z[i]) / dz
        dr = r[i + 1] - r[i]
        rq = r[i] + w * dr
        urq = ur[i] + w * (ur[i + 1] - ur[i])
        uzq = uz[i] + w * (uz[i + 1] - uz[i])
        S = math.pi * rq ** 2
        Vc = self.Vc[i] + math.pi * (zq - z[i]) * (r[i] ** 2 + r[i] * rq + rq ** 2) / 3.0
        Rc = self.Rc[i] + 2 * math.pi * (dz * _seg_int(r[i], dr, ur[i], ur[i + 1] - ur[i], w)
                                         - dr * _seg_int(r[i], dr, uz[i], uz[i + 1] - uz[i], w))
        dSdt = 2 * math.pi * rq * (urq - uzq * dr / dz)
        return S, Vc, Rc, dSdt

    def radius(self, zq):
        """``r(z)`` at heights ``zq`` (NaN outside the portion)."""
        zq = np.asarray(zq, dtype=float)
        out = np.interp(zq, self.z, self.r)
        return np.where((zq >= self.z[0]) & (zq <= self.z[-1]), out, np.nan)
