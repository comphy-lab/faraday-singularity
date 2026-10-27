"""Similarity laws for the birth of a Worthington jet: inertial cone, line sink, inertio-capillary.

References: [G26] J. M. Gordillo, J. Rodriguez-Rodriguez and V. Sanjay, Self-similar Worthington
jets, arXiv:2607.08972 (2026); [GB23] J. M. Gordillo and F. J. Blanco-Rodriguez, Phys. Rev. Fluids
8, 073606 (2023).

Each law is a power ``r_j ~ tau^alpha`` of the jet-base radius in the time ``tau`` from the jet
inception, with ``v_j ~ tau^(alpha - 1)`` and ``We_j = r_j v_j^2 ~ tau^(3 alpha - 2)``. In terms of
the jet radius ([G26] p. 3), with ``q_j = v_j r_j`` and ``Q_j = pi r_j q_j``::

    q_j ~ r_j^((2 alpha - 1)/alpha),   Q_j ~ r_j^((3 alpha - 1)/alpha),
    We_j ~ r_j^((3 alpha - 2)/alpha).

The far-field condition on the velocity potential selects the law; the cone angle alone does not:

* **Inertial cone** ([G26] eqs. (3)-(4), p. 2): the potential vanishes on a cone of semi-angle
  ``beta`` in an intermediate region ``r_j << r << 1`` at inception, where streamlines meet the
  cone at right angles. The far field is ``r^nu P_nu(-cos theta)`` with ``P_nu(-cos beta) = 0``,
  and ``alpha = 1/(2 - nu)``. ``nu`` is taken on the branch continuous with Taylor's cone ([G26]
  App. A, p. 6); for ``0 < beta < 90 deg`` it is the root of ``P_nu(-cos beta)`` in ``(0, 1)``.
  ``beta`` is the semi-angle of the gas cavity measured from the axis on the jet side, the liquid
  filling ``beta < theta <= pi``. Then ``We_j ~ r_j^(2 nu - 1)``: it diverges as ``r_j -> 0`` only
  for cones sharper than Taylor's, ``beta < BETA_TAYLOR`` (49.29 deg), where ``nu < 1/2``. Above
  that angle ``We_j -> 0``, so capillarity, which the theory neglects, grows relative to inertia;
  [G26] calls ``We_j >> 1`` a necessary condition (p. 2), so the law is not self-consistent there.
* **Line sink** ([G26] eq. (2), p. 2; [GB23] eq. (2) and App. A): a time-independent far-field
  flux per unit length, ``r_p dphi/dr_p -> -q_inf``, drives the jet from a truncated cone. Then
  ``nu = 0`` in effect, ``alpha = 1/2``, ``q_j`` stays constant and ``We_j ~ r_j^-1``, whatever
  ``beta``; the inflow near the cavity wall is horizontal.
* **Inertio-capillary**: ``alpha = 2/3`` and ``We_j`` constant, for every ``beta``.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.special import lpmv


def legendre_p(nu: float, x: float) -> float:
    """Legendre function of the first kind of real degree ``nu`` (order 0) on ``-1 < x < 1``."""
    return float(lpmv(0, nu, x))


def nu_of_beta(beta: float) -> float:
    """Degree ``nu`` with ``P_nu(-cos beta) = 0`` on the branch through Taylor's cone, ``beta`` in rad."""
    if not 0.0 < beta < 0.5 * math.pi:
        raise ValueError("need 0 < beta < pi/2")
    x = -math.cos(beta)
    return brentq(lambda nu: legendre_p(nu, x), 1e-12, 1.0 - 1e-12, xtol=1e-14, rtol=1e-14)


def alpha_of_beta(beta: float) -> float:
    """Inertial jet-radius exponent ``1/(2 - nu(beta))`` ([G26] p. 3)."""
    return 1.0 / (2.0 - nu_of_beta(beta))


# Taylor's cone, P_{1/2}(-cos beta) = 0: nu = 1/2 and alpha = 2/3 ([G26] p. 3 gives 49.3 deg).
# 0.8602743 rad = 49.29009 deg.
BETA_TAYLOR = brentq(lambda b: legendre_p(0.5, -math.cos(b)), 0.5, 1.2, xtol=1e-15, rtol=1e-15)


def exponents_of_alpha(alpha: float) -> dict[str, float]:
    """Exponents in time and in jet radius of a law ``r_j ~ tau^alpha``."""
    a = alpha
    return {"alpha": a, "v_j_tau": a - 1.0, "We_j_tau": 3 * a - 2,
            "q_j_rj": (2 * a - 1) / a, "Q_j_rj": (3 * a - 1) / a, "We_j_rj": (3 * a - 2) / a}


# truncated cone driven by a constant far-field flux ([G26] eq. (2), p. 2): delta = 0, alpha = 1/2
LINE_SINK = {"nu": 0.0, **exponents_of_alpha(0.5), "We_j_diverges": True}
INERTIO_CAPILLARY = {**exponents_of_alpha(2.0 / 3.0), "We_j_diverges": False}


def exponents(beta: float) -> dict[str, float | bool]:
    """Predicted exponents of the inertial cone of semi-angle ``beta``.

    ``We_j_diverges`` is ``beta < BETA_TAYLOR``: necessary for the cone law to hold as
    ``r_j -> 0``, not sufficient, since the law also needs the potential to vanish on the cone.
    """
    nu = nu_of_beta(beta)
    return {"nu": nu, **exponents_of_alpha(1.0 / (2.0 - nu)), "We_j_diverges": beta < BETA_TAYLOR}
