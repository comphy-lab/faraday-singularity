"""Contract tests for the collapse geometry on synthetic free surfaces with known answers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faraday_singularity.collapse import (Profile, axis_curvature, cone_angles,  # noqa: E402
                                          cone_fit, gauss_disc_points, jet_base, necks,
                                          wall_kinematics)


def _polyline(*pieces):
    r = np.concatenate([p[0] for p in pieces])
    z = np.concatenate([p[1] for p in pieces])
    keep = np.concatenate([[True], np.hypot(np.diff(r), np.diff(z)) > 1e-14])
    return Profile(r[keep], z[keep])


def _rounded_cone(beta, rho, r_max=1.0, n=4000):
    """Cavity: a sphere cap of radius rho on the axis, tangent to a cone of semi-angle beta."""
    t = np.linspace(0.0, math.pi / 2 - beta, 200)
    cap = (rho * np.sin(t), rho * (1 - np.cos(t)))
    r1, z1 = cap[0][-1], cap[1][-1]
    r = np.linspace(r1, r_max, n)
    return _polyline(cap, (r, z1 + (r - r1) / math.tan(beta)))


def test_axis_curvature_of_paraboloid():
    r = np.linspace(0, 0.2, 41)
    p = Profile(r, 1.0 + 3.0 * r**2 - 0.5 * r**4)
    assert axis_curvature(p) == pytest.approx(6.0, rel=1e-10)


def test_axis_curvature_of_sphere_cap():
    p = _rounded_cone(math.radians(40), rho=0.01)
    assert axis_curvature(p) == pytest.approx(1 / 0.01, rel=1e-3)


@pytest.mark.parametrize("beta_deg", [35.0, 46.0, 60.0])
def test_cone_angle_and_fit(beta_deg):
    beta = math.radians(beta_deg)
    p = _rounded_cone(beta, rho=0.002)
    assert all(a == pytest.approx(beta, abs=1e-6) for a in cone_angles(p, 0, [0.05, 0.2, 0.5]))
    fitted, _, npts = cone_fit(p, 0, 0.05, 0.5)
    assert npts > 100
    assert fitted == pytest.approx(beta, abs=1e-9)


def test_jet_base_of_jet_in_cone():
    beta, rj, z_b = math.radians(45), 0.02, -0.3
    t = np.linspace(0, math.pi / 2, 50)  # rounded jet tip of radius rj at z = 0
    tip = (rj * np.sin(t), -rj * (1 - np.cos(t)))
    side = (np.full(200, rj), np.linspace(-rj, z_b + rj, 200))
    th = np.linspace(math.pi, 1.5 * math.pi, 50)  # quarter fillet: jet wall to a flat base
    fillet = (2 * rj + rj * np.cos(th), z_b + rj + rj * np.sin(th))
    r_out = np.linspace(2 * rj, 1.0, 400)
    cone = (r_out, z_b + (r_out - 2 * rj) / math.tan(beta))
    p = _polyline(tip, side, fillet, cone)
    jb = jet_base(p, r_search=1.0, rise_tol=1e-6)
    assert jb.jet_born
    # the shape has a slope kink at its lowest node, where the three-node parabola refinement
    # overshoots slightly; smooth surfaces do not have this
    assert jb.z_b == pytest.approx(z_b, abs=1e-3 * rj)
    assert jb.r_j == pytest.approx(2 * rj, rel=1e-2)  # the lowest point is the fillet bottom
    assert not necks(p)


def test_cavity_bottom_is_not_a_jet():
    p = _rounded_cone(math.radians(45), rho=0.01)
    jb = jet_base(p, r_search=1.0, rise_tol=1e-6)
    assert not jb.jet_born and jb.r_j == 0.0 and jb.index == 0


def test_neck_detection():
    # a bubble of radius 0.05 under a neck of radius 0.01, then a cone
    t = np.linspace(0, 0.8 * math.pi, 300)
    bubble = (0.05 * np.sin(t), -0.05 * np.cos(t))
    r0, z0 = bubble[0][-1], bubble[1][-1]
    neck_z = z0 + 0.02
    s = np.linspace(0, 1, 100)
    neck = (r0 + (0.01 - r0) * np.sin(0.5 * math.pi * s), z0 + (neck_z - z0) * s)
    r_out = np.linspace(0.01, 0.5, 200)
    cone = (r_out, neck_z + (r_out - 0.01))
    p = _polyline(bubble, neck, cone)
    (r_n, z_n, _), *_ = necks(p)
    assert r_n == pytest.approx(0.01, abs=1e-4)
    assert z_n == pytest.approx(neck_z, abs=1e-3)


def _cone_wall_velocity(p, beta, field):
    """Velocity at the nodes of a rounded cone: zero on the cap, ``field`` on the straight wall.

    ``field`` is ``"cone"`` for the gradient of ``r_p^nu P_nu(-cos theta)`` about the virtual apex
    of the wall, with ``P_nu(-cos beta) = 0``, or ``"sink"`` for a line sink along the axis,
    ``u = (-1/r, 0)``.
    """
    sys.path.insert(0, str(ROOT / "modelComparisonCases"))
    from inertial_similarity import legendre_p, nu_of_beta

    wall = np.abs(np.arctan2(np.gradient(p.r), np.gradient(p.z)) - beta) < 1e-9
    i0 = int(np.argmax(wall))
    z_apex = p.z[i0] - p.r[i0] / math.tan(beta)
    nu = nu_of_beta(beta)

    def phi(r, z):
        rho = math.hypot(r, z - z_apex)
        return rho**nu * legendre_p(nu, -(z - z_apex) / rho)

    ur, uz = np.zeros_like(p.r), np.zeros_like(p.r)
    for i in np.nonzero(wall)[0]:
        r, z = p.r[i], p.z[i]
        if field == "sink":
            ur[i] = -1.0 / r
            continue
        h = 1e-6 * math.hypot(r, z - z_apex)
        ur[i] = (phi(r + h, z) - phi(r - h, z)) / (2 * h)
        uz[i] = (phi(r, z + h) - phi(r, z - h)) / (2 * h)
    return ur, uz, nu


@pytest.mark.parametrize("beta_deg", [38.4, 60.0])
def test_wall_kinematics_cone_and_line_sink(beta_deg):
    beta = math.radians(beta_deg)
    p = _rounded_cone(beta, rho=0.002)
    # phi = 0 on the cone: the flow meets the wall at right angles and |u| ~ r^(nu - 1)
    ur, uz, nu = _cone_wall_velocity(p, beta, "cone")
    k = wall_kinematics(p, ur, uz, 0, 0.05, 0.5)
    assert k.points > 100
    assert k.cos_normal == pytest.approx(1.0, abs=1e-6)
    assert k.speed_slope == pytest.approx(nu - 1, abs=1e-6)
    # line sink along the axis: horizontal inflow, |cos| = cos(beta) and |u| ~ 1/r
    ur, uz, _ = _cone_wall_velocity(p, beta, "sink")
    k = wall_kinematics(p, ur, uz, 0, 0.05, 0.5)
    assert k.cos_normal == pytest.approx(math.cos(beta), abs=1e-9)
    assert k.speed_slope == pytest.approx(-1.0, abs=1e-9)


def test_wall_kinematics_needs_four_nodes():
    p = _rounded_cone(math.radians(45), rho=0.002, n=50)
    k = wall_kinematics(p, np.ones_like(p.r), np.zeros_like(p.r), 0, 0.5, 0.52)
    assert k.points < 4 and math.isnan(k.cos_normal) and math.isnan(k.speed_slope)


def test_disc_quadrature():
    rj, W = 0.03, 2.0
    rr, ww = gauss_disc_points(rj, 16)
    assert np.sum(ww) == pytest.approx(math.pi * rj**2, rel=1e-13)
    q = np.dot(ww, W * (1 - (rr / rj) ** 2))  # Poiseuille-like profile
    assert q == pytest.approx(0.5 * math.pi * rj**2 * W, rel=1e-13)



def test_inertial_similarity_exponents():
    sys.path.insert(0, str(ROOT / "modelComparisonCases"))
    from inertial_similarity import BETA_TAYLOR, alpha_of_beta, legendre_p, nu_of_beta
    from scipy.optimize import brentq

    # Legendre polynomial at integer degree
    assert legendre_p(2.0, 0.3) == pytest.approx(0.5 * (3 * 0.09 - 1), rel=1e-12)
    # Taylor's cone: P_{1/2}(-cos beta) = 0 at beta = 49.29 deg, where nu = 1/2 and alpha = 2/3
    x0 = brentq(lambda x: legendre_p(0.5, x), -0.99, 0.0)
    beta_taylor = math.acos(-x0)
    assert math.degrees(beta_taylor) == pytest.approx(49.29, abs=0.01)
    assert beta_taylor == pytest.approx(BETA_TAYLOR, abs=1e-12)
    assert nu_of_beta(beta_taylor) == pytest.approx(0.5, abs=1e-10)
    assert alpha_of_beta(beta_taylor) == pytest.approx(2 / 3, abs=1e-10)
    # value quoted for bubble bursting at beta = 38.4 deg (alpha = 0.629)
    assert alpha_of_beta(math.radians(38.4)) == pytest.approx(0.629, abs=5e-4)
    # slender cones: nu -> 0 logarithmically, nu ~ 1/(2 ln(2/beta)); nearly flat ones: nu -> 1
    for deg in (0.5, 1.0, 2.0):
        b = math.radians(deg)
        assert nu_of_beta(b) == pytest.approx(1 / (2 * math.log(2 / b)), rel=0.02)
    assert nu_of_beta(math.radians(89.0)) > 0.95


def test_similarity_regimes():
    sys.path.insert(0, str(ROOT / "modelComparisonCases"))
    from inertial_similarity import BETA_TAYLOR, INERTIO_CAPILLARY, LINE_SINK, exponents, legendre_p

    # Taylor's angle, P_{1/2}(-cos beta) = 0: 0.8602743 rad = 49.29009 deg
    assert BETA_TAYLOR == pytest.approx(0.8602743, abs=1e-7)
    assert math.degrees(BETA_TAYLOR) == pytest.approx(49.29009, abs=1e-5)
    assert legendre_p(0.5, -math.cos(BETA_TAYLOR)) == pytest.approx(0.0, abs=1e-13)
    taylor = exponents(BETA_TAYLOR)
    assert taylor["alpha"] == pytest.approx(2 / 3, abs=1e-12)
    assert taylor["We_j_rj"] == pytest.approx(0.0, abs=1e-12)
    # cone: q_j ~ r_j^nu, Q_j ~ r_j^(1 + nu), We_j ~ r_j^(2 nu - 1), diverging only below Taylor
    for deg in (20.0, 38.4, 49.0, 49.6, 56.0, 64.0):
        e = exponents(math.radians(deg))
        assert e["q_j_rj"] == pytest.approx(e["nu"], abs=1e-12)
        assert e["Q_j_rj"] == pytest.approx(1 + e["nu"], abs=1e-12)
        assert e["We_j_rj"] == pytest.approx(2 * e["nu"] - 1, abs=1e-12)
        assert e["We_j_diverges"] == (deg < 49.29)
    # angles above Taylor's: alpha above 2/3 and We_j vanishing as r_j -> 0
    assert exponents(math.radians(56.0))["alpha"] == pytest.approx(0.6953, abs=1e-4)
    assert exponents(math.radians(56.0))["We_j_rj"] == pytest.approx(0.124, abs=1e-3)
    assert exponents(math.radians(64.0))["alpha"] == pytest.approx(0.7372, abs=1e-4)
    assert exponents(math.radians(64.0))["We_j_rj"] == pytest.approx(0.287, abs=1e-3)
    # line sink: delta = 0, alpha = 1/2, q_j constant, We_j ~ r_j^-1
    assert LINE_SINK == {"nu": 0.0, "alpha": 0.5, "v_j_tau": -0.5, "We_j_tau": -0.5,
                         "q_j_rj": 0.0, "Q_j_rj": 1.0, "We_j_rj": -1.0, "We_j_diverges": True}
    expected = {"alpha": 2 / 3, "v_j_tau": -1 / 3, "We_j_tau": 0.0, "q_j_rj": 0.5, "Q_j_rj": 1.5,
                "We_j_rj": 0.0}
    assert set(INERTIO_CAPILLARY) == set(expected) | {"We_j_diverges"}
    assert not INERTIO_CAPILLARY["We_j_diverges"]
    for key, value in expected.items():
        assert INERTIO_CAPILLARY[key] == pytest.approx(value, abs=1e-12)
