"""Contract tests of the reconnection geometry: settings, chains, liquid axis spans, the planner on
gas necks and the replacement of its caps. No finite-element problem is built.

The planner tests on stored profiles of a real neck run only when ``FARADAY_RECONNECT_PROFILES``
names a directory with ``profile_001028.txt``, ``profile_001055.txt`` and ``profile_001076.txt``
(columns ``r z u_r u_z``, a JSON header with the neck radius as ``ell``).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faraday_singularity import reconnect as rc  # noqa: E402

R_C = 3.8317059702075125
PROFILES = os.environ.get("FARADAY_RECONNECT_PROFILES")


# --- settings -----------------------------------------------------------------------------------


def test_settings_parse_and_round_trip():
    s = rc.ReconnectSettings.from_dict({"delta": 3e-4, "fill": "zero", "tau_end": None})
    assert s.delta == 3e-4 and s.fill == "zero" and math.isinf(s.tau_end)
    assert s.reservoir_depth == 3.0 and s.bubble_model == "constant-pressure"
    assert rc.ReconnectSettings.from_dict(s.to_dict()) == s
    json.dumps(s.to_dict())  # finite and JSON-serialisable


@pytest.mark.parametrize("bad", [{}, {"delta": 0.0}, {"delta": 1e-4, "fill": "nearest"},
                                 {"delta": 1e-4, "bubble_model": "fixed-volume"},
                                 {"delta": 1e-4, "restart": "bdf2"}, {"delta": 1e-4, "max_events": 2},
                                 {"delta": 1e-4, "min_landing_ratio": 1.0},
                                 {"delta": 1e-4, "volume_correction": True},
                                 {"delta": 1e-4, "projection": 1}])
def test_settings_reject_invalid(bad):
    with pytest.raises(ValueError):
        rc.ReconnectSettings.from_dict(bad)


def test_collapse_block_split_leaves_the_driver_block_unchanged():
    block = {"neck_stop": 3e-4, "h_floor": 2e-5}
    rest, recon, keep = rc.split_collapse_block(block)
    assert rest == block and recon is None and keep is False
    rest, recon, keep = rc.split_collapse_block(dict(block, reconnect={"delta": 3e-4},
                                                     keep_stop_state=True))
    assert rest == block and recon == {"delta": 3e-4} and keep is True
    with pytest.raises(ValueError):
        rc.split_collapse_block({"keep_stop_state": "yes"})


# --- chains and the axis -------------------------------------------------------------------------


def test_liquid_axis_spans():
    assert rc.liquid_axis_spans([4.6]) == [(0.0, 4.6)]  # one chain, cavity or crest on the axis
    assert rc.liquid_axis_spans([4.688, 4.6, 4.687]) == [(0.0, 4.6), (4.687, 4.688)]
    # a drop above the cavity: its own span between its two tips
    assert rc.liquid_axis_spans([4.0, 5.0, 5.5]) == [(0.0, 4.0), (5.0, 5.5)]
    with pytest.raises(RuntimeError):
        rc.liquid_axis_spans([4.6, 4.7])
    with pytest.raises(RuntimeError):
        rc.liquid_axis_spans([4.6, 4.7, 4.7])


def test_orient_chain_snaps_and_orders_ends():
    tol = 1e-9
    wall_first = [(R_C - 1e-12, 6.0), (1.0, 5.0), (1e-12, 4.0)]
    c = rc.orient_chain(wall_first, R_C, tol)
    assert c.end_types == ("axis", "fixed") and c.points[0, 0] == 0.0 and c.points[-1, 0] == R_C
    assert c.points[0, 1] == 4.0
    top_first = [(0.0, 4.7), (0.01, 4.65), (0.0, 4.6)]
    b = rc.orient_chain(top_first, R_C, tol)
    assert b.end_types == ("axis", "axis") and b.points[0, 1] == 4.6 and not b.is_open
    with pytest.raises(RuntimeError):
        rc.orient_chain([(0.5, 1.0), (1.0, 2.0)], R_C, tol)


def test_chain_contact_is_refused():
    a = rc.Chain(np.array([[0.0, 1.0], [0.1, 1.1], [R_C, 2.0]]), ("axis", "fixed"))
    b = rc.Chain(np.array([[0.0, 0.5], [0.2, 1.2], [0.0, 1.5]]), ("axis", "axis"))  # crosses a
    with pytest.raises(RuntimeError):
        rc.check_chains([a, b])
    c = rc.Chain(np.array([[0.0, 0.5], [0.05, 0.7], [0.0, 0.9]]), ("axis", "axis"))
    assert rc.check_chains([a, c]) > 0.0


def test_tangent_arc_meets_the_axis_at_a_right_angle():
    for tz in (1.0, -1.0):
        beta = math.radians(5.0)
        t = np.array([-math.sin(beta), tz * math.cos(beta)])  # narrowing towards the tip
        P_s = np.array([1.3, 2.0])
        arc, R, z_tip = rc.tangent_arc(P_s, t, 40)
        assert R == pytest.approx(1.3 / math.cos(beta))
        assert arc[-1, 0] == 0.0 and arc[-1, 1] == z_tip
        # centre on the axis, every point at distance R from it
        zc = z_tip - tz * R
        assert np.allclose(np.hypot(arc[:, 0], arc[:, 1] - zc), R)
        # tangent at the splice: the first chord turns by half an arc step from t
        d = arc[0] - P_s
        assert math.degrees(math.acos(np.dot(d, t) / np.hypot(*d))) < 2.0
        # perpendicular to the axis at the tip
        e = arc[-1] - arc[-2]
        assert abs(e[1]) / np.hypot(*e) < 0.05
        assert rc.cap_backtrack(arc[::-1]) == 0.0


# --- a synthetic gas neck ------------------------------------------------------------------------


def synthetic_neck(r_n: float, beta_deg: float = 10.0, z_n: float = 4.7, a: float = 0.05,
                   per_radius: float = 10.0):
    """A gas boundary from the bottom of a bubble on the axis, through a hyperbolic neck of radius
    ``r_n`` and asymptotic half-angle ``beta``, to the wall: ``r(z)`` ordered from the axis.

    Within about 20 r_n of the neck the shape and the node spacing (``r_n/per_radius``) scale with
    ``r_n``, so a cut at a fixed ``delta/r_n`` is geometrically similar for every ``r_n``."""
    tb = math.tan(math.radians(beta_deg))
    z0 = z_n - 0.3  # bottom of the bubble
    zc = z0 + a  # centre of its spherical bottom

    def r_of(z):
        cone = np.sqrt(r_n**2 + ((z - z_n) * tb) ** 2) + 4.5 * np.maximum(0.0, z - z_n - 0.1) ** 2
        bulb = np.sqrt(np.maximum(a**2 - np.minimum(z - zc, 0.0) ** 2, 0.0))
        return np.where(z < z_n, np.minimum(cone, np.where(z < zc, bulb, a)), cone)

    z_w = z_n + 0.1 + math.sqrt((R_C - 0.2) / 4.5)
    # fine parametric sampling by arc length, then resampling with a local spacing
    t = np.linspace(0.0, 1.0, 400001)
    zz = z0 + (z_w - z0) * t
    rr = r_of(zz)
    # the spherical bottom: sample by angle so that the curve leaves the axis smoothly
    th = np.linspace(0.0, 0.5 * math.pi, 2001)
    bottom = np.column_stack([a * np.sin(th), zc - a * np.cos(th)])
    body = np.column_stack([rr, zz])
    body = body[body[:, 1] > zc]
    P = np.vstack([bottom, body])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    neck = np.array([r_n, z_n])

    def h(p):
        d = math.hypot(*(p - neck))
        return min(0.02, max(r_n / per_radius, 0.1 * d))

    out = [P[0]]
    si = 0.0
    while True:
        si += h(out[-1])
        if si >= s[-1]:
            break
        out.append(np.array([np.interp(si, s, P[:, 0]), np.interp(si, s, P[:, 1])]))
    out.append(np.array([R_C, z_w]))
    Q = np.array(out)
    Q[0, 0] = 0.0
    Q[-1, 0] = R_C
    return Q[:, 0], Q[:, 1]


def test_synthetic_neck_measured_as_the_driver_does():
    r, z = synthetic_neck(5e-3)
    r_n, z_n, _ = rc.neck_radius(r, z)
    assert r_n == pytest.approx(5e-3, rel=1e-3) and z_n == pytest.approx(4.7, abs=1e-3)


@pytest.fixture(scope="module")
def synthetic_scan():
    out = {}
    for r_n in (1e-2, 5e-3, 2.5e-3):
        r, z = synthetic_neck(r_n)
        rn, _, _ = rc.neck_radius(r, z)
        out[r_n] = (r, z, rn)
    return out


def test_planner_classes_on_a_synthetic_neck(synthetic_scan):
    s = rc.ReconnectSettings(delta=1.0)
    for r_n, (r, z, rn) in synthetic_scan.items():
        assert rc.plan_on_profile(r, z, 0.97 * rn, s)[0] == "none"
        assert rc.plan_on_profile(r, z, 1.0 * rn, s)[0] == "none"
        for f in (1.005, 1.01, 1.02):
            assert rc.plan_on_profile(r, z, f * rn, s)[0] in ("none", "deferral", "unplannable",
                                                              "refused", "plan")
        outcome, plan = rc.plan_on_profile(r, z, 1.05 * rn, s)
        assert outcome == "plan"
        ends = sorted(tuple(c.end_types) for c in plan.new_chains)
        assert ends == [("axis", "axis"), ("axis", "fixed")]
        assert [e.kind for e in plan.events] == ["pinch"]


def test_replaced_caps_on_a_synthetic_neck_scale_with_the_cut(synthetic_scan):
    removed, bridges = [], []
    for r_n, (r, z, rn) in synthetic_scan.items():
        delta = 1.05 * rn
        s = rc.ReconnectSettings(delta=delta)
        outcome, plan = rc.plan_on_profile(r, z, delta, s)
        assert outcome == "plan"
        edits = rc.replace_caps(plan.new_chains, delta, s)
        assert len(edits) == 2
        for e in edits:
            assert e.min_angle >= 30.0 and e.splice_angle > 150.0
            assert e.r_s >= delta and 1.0 < e.radius / delta < 3.0
        for nc in plan.new_chains:
            assert np.all(np.diff(nc.zeta) > 0)
            assert len(nc.points) == len(nc.zeta) == len(nc.sizes) == len(nc.origin)
        cg = rc.cut_geometry(rc.plan_chains(plan.new_chains), s.reservoir_depth,
                             sum(plan.fragment_volumes_before))
        assert cg.bridge > 0 and cg.chain_gap > 0 and cg.removed_gas > 0
        assert cg.spans == [(0.0, cg.z_bubble_bottom), (cg.z_bubble_top, cg.z_cavity_tip)]
        removed.append(cg.removed_gas)
        bridges.append(cg.bridge / delta)
    deltas = 1.05 * np.array([v[2] for v in synthetic_scan.values()])
    slope = np.polyfit(np.log(deltas), np.log(removed), 1)[0]
    assert slope == pytest.approx(3.0, abs=0.15)
    assert max(bridges) / min(bridges) < 1.2


def test_cap_replacement_of_a_hairpin():
    """A planner-like chain whose fresh tip folds back (a needle and a connector) is rebuilt as a
    monotone arc from a surviving node."""
    delta = 1.0
    zz = np.linspace(10.0, 30.0, 201)  # surviving thread, r = 1.3 + 0.05 (z - 10)
    thread = np.column_stack([1.3 + 0.05 * (zz - 10.0), zz])
    needle = np.array([[0.0, 2.0], [0.3, 2.3], [0.6, 3.5], [0.9, 6.0], [1.05, 3.0], [1.2, 6.0],
                       [1.28, 8.0]])  # tip, needle, connector back towards the tip, thread
    pts = np.vstack([needle, thread])
    nc = SimpleNamespace(points=pts, sizes=np.full(len(pts), 0.2),
                         zeta=np.linspace(0.0, 1.0, len(pts)),
                         origin=np.r_[np.full(len(needle), -1), np.arange(len(thread))],
                         end_types=("axis", "fixed"))
    assert rc.cap_backtrack(needle) > 1.0
    edits = rc.replace_caps([nc], delta, rc.ReconnectSettings(delta=delta))
    (e,) = edits
    assert e.replaced_fresh == len(needle) and e.backtrack > 1.0
    assert nc.points[0, 0] == 0.0 and np.all(nc.origin[e.points:] >= 0)
    assert rc.cap_backtrack(nc.points[:e.points + 1]) == 0.0
    assert e.beta_deg == pytest.approx(math.degrees(math.atan(0.05)), abs=0.05)  # narrowing to the tip
    assert e.splice_angle > 175.0  # tangent: the first chord turns by half an arc step
    assert np.all(np.diff(nc.zeta) > 0) and nc.zeta[0] == 0.0


def test_post_cut_stops():
    s = rc.ReconnectSettings(delta=1e-3)
    bubble = rc.Chain(np.array([[0.0, 4.6], [0.01, 4.64], [0.0, 4.68]]), ("axis", "axis"))
    cavity = rc.Chain(np.array([[0.0, 4.69], [0.01, 4.7], [R_C, 6.0]]), ("axis", "fixed"))
    cut = rc.cut_geometry([bubble, cavity], 3.0, gas_before=1.0)
    assert cut.bridge == pytest.approx(0.01)
    assert rc.post_cut_stops([bubble, cavity], cut, s, h_min=1e-5) is None
    closing = rc.Chain(cavity.points - [0.0, 0.0095], ("axis", "fixed"))
    assert rc.post_cut_stops([bubble, closing], cut, s, 1e-5)["reason"] == "bridge-closing"
    small = rc.Chain(np.array([[0.0, 4.6], [1e-3, 4.60025], [0.0, 4.6005]]), ("axis", "axis"))
    assert rc.post_cut_stops([small, cavity], cut, s, 1e-5)["reason"] == "bubble-jets-approach"


# --- stored profiles of a real neck --------------------------------------------------------------

stored = pytest.mark.skipif(not PROFILES or not Path(PROFILES).is_dir(),
                            reason="FARADAY_RECONNECT_PROFILES not set")


def _stored(step: int):
    path = Path(PROFILES) / f"profile_{step:06d}.txt"
    header = json.loads(path.read_text().splitlines()[0][2:])
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1], header


@stored
def test_neck_radius_is_the_driver_minimum():
    r, z, header = _stored(1076)
    r_n, z_n, _ = rc.neck_radius(r, z)
    assert r_n == header["ell"]  # the stop profile: the local scale is the neck
    # a second, nearly equal local minimum sits beside it
    i = np.arange(1, len(r) - 1)
    loc = i[(r[i] < r[i - 1]) & (r[i] <= r[i + 1])]
    close = sorted(r[loc][r[loc] < 1.01 * r_n])
    assert len(close) >= 2 and close[0] == r_n


@stored
@pytest.mark.parametrize("step", [1028, 1055, 1076])
def test_planner_classes_on_stored_necks(step):
    r, z, _ = _stored(step)
    r_n, _, _ = rc.neck_radius(r, z)
    s = rc.ReconnectSettings(delta=1.0)
    assert rc.plan_on_profile(r, z, r_n, s)[0] == "none"
    outcomes = {f: rc.plan_on_profile(r, z, f * r_n, s)[0] for f in (1.005, 1.03, 1.05, 1.1, 1.2)}
    assert outcomes[1.005] == "deferral"
    assert set(outcomes.values()) <= {"none", "deferral", "unplannable", "refused", "plan"}
    assert all(outcomes[f] == "plan" for f in (1.03, 1.05, 1.1, 1.2))


@stored
@pytest.mark.parametrize("step", [1055, 1076])
def test_planner_caps_fold_and_replaced_caps_do_not(step):
    """The planner's caps on this neck double back by several cut radii; the replaced caps are
    monotone from the tip, without knife-edge angles, and tangent at the splice."""
    from shapely.geometry import LineString

    r, z, _ = _stored(step)
    r_n, _, _ = rc.neck_radius(r, z)
    for f in (1.03, 1.2, 2.0):
        delta = f * r_n
        s = rc.ReconnectSettings(delta=delta)
        outcome, plan = rc.plan_on_profile(r, z, delta, s)
        assert outcome == "plan"
        # the planner's caps fold, yet pass the simplicity check the meshing relies on
        for nc in plan.new_chains:
            assert LineString(nc.points).is_simple
        edits = rc.replace_caps(plan.new_chains, delta, s)
        assert len(edits) == 2
        assert max(e.backtrack for e in edits) > 1.0
        for e in edits:
            assert e.min_angle >= 30.0 and e.splice_angle > 170.0 and e.r_s >= delta
        cg = rc.cut_geometry(rc.plan_chains(plan.new_chains), s.reservoir_depth,
                             sum(plan.fragment_volumes_before))
        assert cg.bridge > 2 * delta and 0 < cg.removed_gas < 100 * delta**3


@stored
def test_volume_correction_is_refused_on_the_stop_profile():
    from pyoomph.meshes import axisymm_topology as topo

    r, z, _ = _stored(1076)
    r_n, _, _ = rc.neck_radius(r, z)
    chain = rc.profile_chain(r, z, 3.0)
    with pytest.raises(RuntimeError, match="self-intersecting"):
        topo.detect_and_plan([chain], 1.05 * r_n, None, volume_conservation=True)
