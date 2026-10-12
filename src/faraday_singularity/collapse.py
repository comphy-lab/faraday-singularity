"""Geometry of a collapsing cavity and of the jet it launches, from the ordered free surface.

The free surface is passed as arrays ordered along the curve from its axis end (``r = 0``) to the
wall. Overturned profiles are fine: nothing assumes that ``z`` is a function of ``r``.

Definitions (lengths in units of ``1/k``):

* **Apex**: the free surface on the axis, height ``z_axis``. Before the jet forms it is the cavity
  bottom; afterwards it is the jet tip. ``axis_curvature`` is ``d2z/dr2`` there, from a polynomial
  fit in ``r**2`` over the nodes nearest the axis.
* **Jet base**: the lowest point of the free surface between the axis and ``r_search``.
  ``r_j`` is its radius and ``z_b`` its height. While the cavity bottom is the lowest point,
  ``r_j = 0``; once a jet rises on the axis the lowest point moves off it, and the jet is born
  (``jet_born``) when the axis stands above the base by more than ``rise_tol``.
* **Jet-base flux** ``Q_j = 2 pi int_0^{r_j} u_z(r, z_b) r dr`` through the horizontal disc at the
  base, as in the Basilisk bursting-bubble analysis of Gordillo, Rodriguez-Rodriguez & Sanjay.
  The mean jet speed is ``V_j = Q_j/(pi r_j^2)``, the flux per unit length ``q_j = V_j r_j`` and the
  local Weber number ``We_j = V_j^2 r_j`` (density and surface tension are 1).
* **Neck**: an interior local minimum of ``r`` along the curve. It marks an approaching pinch-off:
  a bubble closing above the cavity bottom, or a drop forming on the jet.
* **Cone angle**: the semi-angle ``beta`` between the axis and the free-surface tangent,
  ``atan2(dr, dz)`` along the curve, sampled outside the base at given radii, and fitted as a
  straight cone ``r = (z - z0) tan(beta)`` over a window of radii.
* **Wall kinematics**: over a window of radii on the same branch, the median ``|cos|`` of the
  angle between the surface velocity and the surface normal, and the slope of ``ln|u|`` against
  ``ln r``. They tell the two inertial far fields of a collapsing cone apart (Gordillo,
  Rodriguez-Rodriguez & Sanjay, arXiv:2607.08972, eqs. (2)-(3)): where the velocity potential
  vanishes on a cone of semi-angle ``beta``, the flow meets the wall at right angles
  (``|cos| = 1``) and ``|u| ~ r^(nu - 1)``; a line sink along the axis drives a horizontal inflow,
  ``|cos| = cos(beta)`` on a straight wall, with ``|u| ~ 1/r``. The velocities are those of the
  container frame, so a wall that moves as a whole biases both where the focusing flow is weak.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Profile:
    r: np.ndarray
    z: np.ndarray

    @classmethod
    def ordered(cls, r, z) -> "Profile":
        """Orient a single polyline so that it starts at its end nearest the axis."""
        r = np.asarray(r, dtype=float)
        z = np.asarray(z, dtype=float)
        if r[0] > r[-1]:
            r, z = r[::-1], z[::-1]
        return cls(r.copy(), z.copy())

    @property
    def arclength(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(self.r), np.diff(self.z)))])


def axis_curvature(p: Profile, nodes: int = 6) -> float:
    """``d2z/dr2`` at the axis from ``z = c0 + c1 r^2 + c2 r^4`` through the nearest nodes.

    Only the nodes before the first turn of ``r`` along the curve are used, so an overturned
    surface near the axis does not corrupt the fit. Returns NaN with fewer than three nodes.
    """
    dr = np.diff(p.r)
    stop = int(np.argmax(dr <= 0)) + 1 if np.any(dr <= 0) else len(p.r)
    m = min(nodes, stop)
    if m < 3:
        return float("nan")
    r2 = p.r[:m] ** 2
    cols = [np.ones(m), r2] + ([r2**2] if m >= 4 else [])
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), p.z[:m], rcond=None)
    return float(2.0 * coef[1])


def lowest_point(p: Profile, r_search: float) -> int:
    """Index of the lowest node among those with ``r <= r_search``."""
    idx = np.nonzero(p.r <= r_search)[0]
    return int(idx[np.argmin(p.z[idx])])


def refine_minimum(p: Profile, i: int) -> tuple[float, float]:
    """``(r, z)`` at the minimum of ``z`` near node ``i``, by a parabola through three nodes in s."""
    if i <= 0 or i >= len(p.r) - 1:
        return float(p.r[i]), float(p.z[i])
    s = p.arclength[i - 1:i + 2]
    cz = np.polyfit(s - s[1], p.z[i - 1:i + 2], 2)
    if cz[0] <= 0:
        return float(p.r[i]), float(p.z[i])
    ds = float(np.clip(-cz[1] / (2 * cz[0]), s[0] - s[1], s[2] - s[1]))
    cr = np.polyfit(s - s[1], p.r[i - 1:i + 2], 2)
    return float(np.polyval(cr, ds)), float(np.polyval(cz, ds))


def necks(p: Profile, skip: int = 2) -> list[tuple[float, float, int]]:
    """Interior local minima of ``r`` along the curve: ``(r, z, index)``, narrowest first.

    The first ``skip`` nodes are ignored so the axis end itself is not reported.
    """
    out = []
    for i in range(max(1, skip), len(p.r) - 1):
        if p.r[i] < p.r[i - 1] and p.r[i] <= p.r[i + 1]:
            out.append((float(p.r[i]), float(p.z[i]), i))
    return sorted(out)


def _branch_length(p: Profile, start: int) -> int:
    """Number of nodes from ``start`` along which ``r`` keeps increasing."""
    dr = np.diff(p.r[start:])
    return int(np.argmax(dr <= 0)) + 1 if np.any(dr <= 0) else len(p.r) - start


def tangent_angles(p: Profile) -> np.ndarray:
    """Angle between the local tangent and the axis, ``atan2(dr, dz)``, at the nodes."""
    dr = np.gradient(p.r, p.arclength)
    dz = np.gradient(p.z, p.arclength)
    return np.arctan2(dr, dz)


def cone_angles(p: Profile, start: int, radii) -> list[float]:
    """Tangent semi-angle at each radius in ``radii`` on the branch after node ``start``.

    The branch is followed while ``r`` increases; radii beyond it give NaN.
    """
    stop = _branch_length(p, start)
    r = p.r[start:start + stop]
    ang = tangent_angles(p)[start:start + stop]
    out = []
    for rk in radii:
        out.append(float(np.interp(rk, r, ang)) if len(r) >= 2 and r[0] <= rk <= r[-1]
                   else float("nan"))
    return out


def cone_fit(p: Profile, start: int, r_lo: float, r_hi: float) -> tuple[float, float, int]:
    """Least-squares cone ``r = (z - z0) tan(beta)`` over ``r_lo <= r <= r_hi`` after ``start``.

    Returns ``(beta, z0, points)``; ``beta`` is NaN with fewer than four points.
    """
    stop = _branch_length(p, start)
    r, z = p.r[start:start + stop], p.z[start:start + stop]
    sel = (r >= r_lo) & (r <= r_hi)
    if sel.sum() < 4:
        return float("nan"), float("nan"), int(sel.sum())
    slope, intercept = np.polyfit(z[sel], r[sel], 1)  # r = slope*z + intercept
    beta = math.atan(slope) if slope > 0 else float("nan")
    z0 = -intercept / slope if slope != 0 else float("nan")
    return float(beta), float(z0), int(sel.sum())


@dataclass
class WallKinematics:
    cos_normal: float  # median |cos| of the angle between the surface velocity and the normal
    speed_slope: float  # least-squares slope of ln|u| against ln r
    points: int


def wall_kinematics(p: Profile, ur, uz, start: int, r_lo: float, r_hi: float) -> WallKinematics:
    """Direction and decay of the surface velocity over ``r_lo <= r <= r_hi`` after ``start``.

    ``ur`` and ``uz`` are the velocity components at the nodes of ``p``. The branch is followed
    while ``r`` increases, as in :func:`cone_fit`; nodes at rest are skipped. With the tangent
    ``(sin a, cos a)`` in ``(r, z)``, ``a`` the tangent angle, the normal is ``(cos a, -sin a)``.
    Both values are NaN with fewer than four nodes.
    """
    stop = _branch_length(p, start)
    r = p.r[start:start + stop]
    ang = tangent_angles(p)[start:start + stop]
    u_r = np.asarray(ur, dtype=float)[start:start + stop]
    u_z = np.asarray(uz, dtype=float)[start:start + stop]
    speed = np.hypot(u_r, u_z)
    sel = (r >= r_lo) & (r <= r_hi) & np.isfinite(speed) & (speed > 0)
    if sel.sum() < 4:
        return WallKinematics(float("nan"), float("nan"), int(sel.sum()))
    cos_n = np.abs(u_r[sel] * np.cos(ang[sel]) - u_z[sel] * np.sin(ang[sel])) / speed[sel]
    slope = np.polyfit(np.log(r[sel]), np.log(speed[sel]), 1)[0]
    return WallKinematics(float(np.median(cos_n)), float(slope), int(sel.sum()))


def gauss_disc_points(r_j: float, n: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Radii and weights for ``2 pi int_0^{r_j} f(r) r dr`` by Gauss-Legendre in ``r``."""
    x, w = np.polynomial.legendre.leggauss(n)
    rr = 0.5 * r_j * (x + 1.0)
    return rr, 2.0 * math.pi * 0.5 * r_j * w * rr


@dataclass
class JetBase:
    r_j: float
    z_b: float
    index: int
    jet_born: bool


def jet_base(p: Profile, r_search: float, rise_tol: float) -> JetBase:
    i = lowest_point(p, r_search)
    r_b, z_b = refine_minimum(p, i)
    born = i > 0 and (p.z[0] - z_b) > rise_tol
    if not born:
        r_b = 0.0 if i == 0 else r_b
    return JetBase(r_j=float(r_b), z_b=float(z_b), index=i, jet_born=bool(born))
