"""Contract tests of the reconnection geometry: settings, chains, liquid axis spans, the planner on
gas necks and the replacement of its caps, the gas volume near a cut, and the stops after a cut.
No finite-element problem is built.

Two synthetic gas necks stand in for the necks of a run: a hyperbolic neck of half-angle 10 degrees
(``synthetic_neck``), and a slender thread (``synthetic_thread``), nearly cylindrical within about
8 neck radii of the waist and resolved by 25 nodes per neck radius, as the neck of a collapse is at
its stop. On the thread the planner behaves as on the stored necks of a run: it defers just above
the neck radius, plans from ``min_landing_ratio`` on, and its caps fold back.

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
from shapely.geometry import LineString

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
                                 {"delta": 1e-4, "max_landing_ratio": 1.03},
                                 {"delta": 1e-4, "max_landing_ratio": math.inf},
                                 # the window must hold every landing state a run keeps
                                 {"delta": 1e-4, "min_landing_ratio": 1.07},
                                 {"delta": 1e-4, "max_landing_ratio": 1.08},
                                 {"delta": 1e-4, "volume_correction": True},
                                 {"delta": 1e-4, "cap_splice_factor": 1.0},  # removed
                                 {"delta": 1e-4, "tip_elements": 8.0},  # removed
                                 {"delta": 1e-4, "tip_size": 0.3}, {"delta": 1e-4, "bridge_size": 0.0},
                                 {"delta": 1e-4, "focus_radius": -1.0},
                                 {"delta": 1e-4, "projection": 1}])
def test_settings_reject_invalid(bad):
    with pytest.raises(ValueError):
        rc.ReconnectSettings.from_dict(bad)


def test_settings_admit_every_landing_state_a_run_keeps():
    s = rc.ReconnectSettings(delta=1e-4)
    assert s.min_landing_ratio <= rc.LANDING_RATIO < rc.MAX_LANDING_RATIO <= s.max_landing_ratio
    for kw in ({"min_landing_ratio": rc.LANDING_RATIO}, {"max_landing_ratio": rc.MAX_LANDING_RATIO},
               {"max_landing_ratio": 1.2}):
        rc.ReconnectSettings(delta=1e-4, **kw)


def test_a_continuation_may_change_only_where_the_run_after_the_cut_ends():
    import json

    s = rc.ReconnectSettings(delta=1e-3, tau_end=4e-5)
    recorded = json.loads(json.dumps(s.to_dict()))  # as the cut record holds it
    assert rc.CONTINUATION_KEYS == ("tau_end",)
    for tau_end in (4e-5, 1e-4, None):
        assert rc.ReconnectSettings.from_dict(dict(recorded, tau_end=tau_end)).continues_cut(
            recorded)
    for change in ({"dt0": 2e-9}, {"delta": 1.1e-3}, {"cap_window_factor": 8.0},
                   {"bubble_stop": 0.1}):
        assert not rc.ReconnectSettings.from_dict(dict(recorded, **change)).continues_cut(recorded)
    assert not s.continues_cut(None) and not s.continues_cut({})
    # where the driver stops a run after a cut, to round-off
    assert rc.tau_end_reached(4e-5, 4e-5) and rc.tau_end_reached(4e-5 * (1 - 1e-15), 4e-5)
    assert not rc.tau_end_reached(3.9e-5, 4e-5) and not rc.tau_end_reached(1e3, math.inf)


def test_cut_size_field_as_the_mesh_after_a_cut_builds_it():
    """The finer of the tip sizing about every end on the axis and the bridge sizing along the
    liquid span between the bubble's top and the cavity's tip, graded linearly."""
    sizing = {"h_tip": 1e-3, "h_bridge": 2e-3, "curvature": 20, "grading": 0.3}
    tips = [4.69, 4.28, 4.68]  # the cavity tip, the bubble's bottom and top, in any order
    r = np.array([0.0, 0.0, 0.01, 0.1, 0.002])
    z = np.array([4.69, 4.685, 4.69, 5.0, 4.0])
    h = rc.cut_size_field(r, z, tips, sizing)
    d = math.hypot(0.1, 0.31)
    expected = [1e-3,  # at the cavity tip
                min(1e-3 + 0.3 * 0.005, 2e-3),  # on the bridge, halfway between two tips
                min(1e-3 + 0.3 * 0.01, 2e-3 + 0.3 * 0.01),  # beside the cavity tip
                1e-3 + 0.3 * d,  # far above it: the tip sizing, the bridge's beyond it
                1e-3 + 0.3 * math.hypot(0.002, 0.28)]  # below the bubble, no bridge there
    assert h == pytest.approx(expected, rel=1e-12)
    # before a cut (one end on the axis) there is no bridge
    assert rc.cut_size_field([0.0, 0.3], [1.0, 1.4], [1.0], sizing) == pytest.approx(
        [1e-3, 1e-3 + 0.3 * 0.5], rel=1e-12)


def test_collapse_block_split_leaves_the_driver_block_unchanged():
    block = {"neck_stop": 3e-4, "h_floor": 2e-5}
    rest, recon, keep, landing = rc.split_collapse_block(block)
    assert rest == block and recon is None and keep is False and landing == ()
    rest, recon, keep, landing = rc.split_collapse_block(
        dict(block, reconnect={"delta": 3e-4}, keep_stop_state=True, landing=[2.5e-4, 1e-3, 5e-4]))
    assert rest == block and recon == {"delta": 3e-4} and keep is True
    assert landing == (1e-3, 5e-4, 2.5e-4)
    for bad in ({"keep_stop_state": "yes"}, {"landing": 1e-3}, {"landing": [1e-3, 0.0]},
                {"landing": [1e-3, True]}, {"landing": [1e-3, math.nan]},
                {"landing": [1e-3, 1e-3]}):
        with pytest.raises(ValueError):
            rc.split_collapse_block(bad)


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
                   per_radius: float = 10.0, plateau: float = 0.0):
    """A gas boundary from the bottom of a bubble on the axis, through a hyperbolic neck of radius
    ``r_n`` and asymptotic half-angle ``beta``, to the wall: ``r(z)`` ordered from the axis.

    Within about 20 r_n of the neck the shape and the node spacing (``r_n/per_radius`` within
    ``plateau r_n`` of the neck, growing beyond) scale with ``r_n``, so a cut at a fixed
    ``delta/r_n`` is geometrically similar for every ``r_n``."""
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
        return min(0.02, max(r_n / per_radius, 0.1 * max(0.0, d - plateau * r_n)))

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


def synthetic_thread(r_n: float):
    """A slender gas thread like the neck of a collapse at its stop: ``r = sqrt(r_n^2 + (dz/8)^2)``
    (1.6 r_n at 10 r_n from the waist, a cone of 7 degrees beyond), uniformly resolved by 25 nodes
    per neck radius within 20 r_n of the waist."""
    return synthetic_neck(r_n, beta_deg=math.degrees(math.atan(1.0 / 8.0)), a=0.03,
                          per_radius=25.0, plateau=20.0)


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
    for r, z, rn in synthetic_scan.values():
        assert rc.plan_on_profile(r, z, 0.97 * rn, s)[0] == "none"
        assert rc.plan_on_profile(r, z, 1.0 * rn, s)[0] == "none"
        # just above the neck the waist is shorter than 2 rmin: the planner defers
        assert [rc.plan_on_profile(r, z, f * rn, s)[0] for f in (1.005, 1.01, 1.02)] == \
            ["deferral", "deferral", "plan"]
        outcome, plan = rc.plan_on_profile(r, z, 1.05 * rn, s)
        assert outcome == "plan"
        ends = sorted(tuple(c.end_types) for c in plan.new_chains)
        assert ends == [("axis", "axis"), ("axis", "fixed")]
        assert [e.kind for e in plan.events] == ["pinch"]


def test_replaced_caps_on_a_synthetic_neck_scale_with_the_cut(synthetic_scan):
    removed, bridges = [], []
    for r, z, rn in synthetic_scan.values():
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


@pytest.fixture(scope="module")
def thread_scan():
    out = {}
    for r_n in (1e-3, 3e-4):
        r, z = synthetic_thread(r_n)
        rn, _, _ = rc.neck_radius(r, z)
        assert rn == pytest.approx(r_n, rel=1e-5)
        out[r_n] = (r, z, rn)
    return out


def test_planner_classes_on_a_slender_thread(thread_scan):
    """As on the stored necks of a run: a deferral just above the neck radius, plans from
    ``min_landing_ratio`` up."""
    s = rc.ReconnectSettings(delta=1.0)
    for r, z, rn in thread_scan.values():
        assert rc.plan_on_profile(r, z, rn, s)[0] == "none"
        assert rc.plan_on_profile(r, z, 1.005 * rn, s)[0] == "deferral"
        lo = s.min_landing_ratio
        for f in (lo, 1.05, s.max_landing_ratio, 1.2):
            assert rc.plan_on_profile(r, z, f * rn, s)[0] == "plan", f


def _cut_on(r, z, delta):
    s = rc.ReconnectSettings(delta=delta)
    outcome, plan = rc.plan_on_profile(r, z, delta, s)
    assert outcome == "plan"
    folded = [rc.cap_backtrack(nc.points[:rc.fresh_tip_run(nc.origin, end) + 1] if end == 0 else
                               nc.points[::-1][:rc.fresh_tip_run(nc.origin, end) + 1]) / delta
              for nc in plan.new_chains for end in (0, -1)
              if nc.end_types[end] == "axis" and int(nc.origin[end]) < 0]
    simple = [LineString(nc.points).is_simple for nc in plan.new_chains]
    edits = rc.replace_caps(plan.new_chains, delta, s)
    chains = [rc.orient_chain(c.points, R_C, 1e-7) for c in rc.plan_chains(plan.new_chains)]
    cg = rc.cut_geometry(chains, s.reservoir_depth, sum(plan.fragment_volumes_before))
    return s, folded, simple, edits, chains, cg



@pytest.mark.parametrize("f", [1.03, 1.05, 1.2])
def test_planner_caps_fold_on_a_slender_thread_and_replaced_caps_do_not(thread_scan, f):
    """The planner's caps double back by several cut radii yet pass the simplicity check the
    meshing relies on; the replaced caps are monotone, tangent at the splice and without knife
    edges, spliced at the first survivor beyond the planner's window; the bridge is 13-18 delta
    long and the liquid gains 53-69 delta^3 (delta/r_n from 1.03 to 1.2); and the state right after
    the cut meets no stop."""
    for r, z, rn in thread_scan.values():
        delta = f * rn
        s, folded, simple, edits, chains, cg = _cut_on(r, z, delta)
        assert all(simple) and len(folded) == 2 and min(folded) > 1.0
        assert len(edits) == 2
        for e in edits:
            assert e.backtrack > 1.0 and e.splice_set_by == "planner-window"
            assert e.min_angle >= 30.0 and e.splice_angle > 170.0 and e.r_s >= delta
        assert 12 * delta < cg.bridge < 19 * delta
        assert 50 * delta**3 < cg.removed_gas < 72 * delta**3
        assert rc.post_cut_stops(chains, cg, s, rc.tip_size(delta, s),
                                 fold_stop=3e-4) is None


def test_the_planner_window_sets_the_cut(thread_scan):
    """On a slender thread the first survivor beyond the planner's window lies at about 1.4 cut
    radii, so the floor at the cut radius drops no survivor (``splice_set_by`` ``planner-window``
    for both caps), and the window factor moves the splices, the bridge and the removed gas."""
    r, z, rn = thread_scan[3e-4]
    delta = 1.05 * rn

    def summary(**kw):
        s = rc.ReconnectSettings(delta=delta, **kw)
        outcome, plan = rc.plan_on_profile(r, z, delta, s)
        assert outcome == "plan"
        edits = rc.replace_caps(plan.new_chains, delta, s)
        cg = rc.cut_geometry(rc.plan_chains(plan.new_chains), s.reservoir_depth,
                             sum(plan.fragment_volumes_before))
        return rc.in_cut_radii(edits, cg, delta, s)

    base, window4, window8 = (summary(), summary(cap_window_factor=4.0),
                              summary(cap_window_factor=8.0))
    for g in (base, window4, window8):
        assert g["splice_set_by"] == ["planner-window"] * 2 and g["replaced_survivors"] == [0, 0]
    assert min(base["r_s"]) > 1.3 and "cap_splice_factor" not in base
    assert 13 < base["bridge"] < 15.5 and 54 < base["removed_gas"] < 60  # measured 14.3 and 57.1
    assert window4["bridge"] < 0.8 * base["bridge"] and window8["bridge"] > 1.15 * base["bridge"]
    assert window4["removed_gas"] < base["removed_gas"] < window8["removed_gas"]
    assert min(window4["r_s"]) < min(base["r_s"]) < min(window8["r_s"])


def test_the_splice_floor_at_the_cut_radius():
    """Where the survivors beyond the planner's window are narrower than the cut radius, they are
    dropped up to the first one at least as wide (``delta-floor``), so no fresh cap sits on a thread
    narrower than the post-cut neck limit."""
    delta = 1.0
    zz = np.linspace(10.0, 30.0, 201)  # surviving thread, r = 0.8025 + 0.05 (z - 10)
    thread = np.column_stack([0.8025 + 0.05 * (zz - 10.0), zz])
    cap = np.array([[0.0, 9.0], [0.4, 9.3], [0.7, 9.7]])  # the planner's tip
    pts = np.vstack([cap, thread])
    nc = SimpleNamespace(points=pts, sizes=np.full(len(pts), 0.2),
                         zeta=np.linspace(0.0, 1.0, len(pts)),
                         origin=np.r_[np.full(len(cap), -1), np.arange(len(thread))],
                         end_types=("axis", "fixed"))
    (e,) = rc.replace_caps([nc], delta, rc.ReconnectSettings(delta=delta))
    assert e.splice_set_by == "delta-floor" and e.replaced_survivors == 40  # r < 1 below z = 14
    assert e.r_s >= delta and e.r_s == pytest.approx(1.0025)
    assert nc.points[0, 0] == 0.0 and rc.cap_backtrack(nc.points[:e.points + 1]) == 0.0


@pytest.mark.parametrize("fold_over_delta", [1.0, 3.0, 20.0])
def test_a_fold_stop_far_above_delta_finds_no_fold_on_freshly_cut_chains(thread_scan,
                                                                          fold_over_delta):
    """The fold stops after a cut keep the landing run's absolute ``fold_stop``, which may be many
    cut radii. They need a fold pair (two nodes closer than ``1/fold_ratio`` of the arc between
    them), and a cap with the thread behind it has none, so neither the bubble's fold stop nor the
    driver's on the open chain fires on the new thread however large ``fold_stop`` is."""
    from faraday_singularity.selfcontact import fold_gap

    for r, z, rn in thread_scan.values():
        delta = 1.05 * rn
        s, _, _, _, chains, cg = _cut_on(r, z, delta)
        fold_stop = fold_over_delta * delta
        assert rc.post_cut_stops(chains, cg, s, rc.tip_size(delta, s), fold_stop=fold_stop) is None
        for c in chains:
            f = fold_gap(c.points[:, 0], c.points[:, 1], ratio=4.0, stop=fold_stop,
                         stop_elements=3.0, search_factor=4.0)
            assert math.isinf(f.gap) and not f.crossed


def test_cut_geometry_in_delta_units_depends_on_the_landing_ratio(thread_scan):
    """Within the admitted band of delta/r_n the bridge and the removed gas, in units of delta,
    vary by less than a quarter; at twice the neck radius they do not: a cut there is refused."""
    r, z, rn = thread_scan[3e-4]
    s = rc.ReconnectSettings(delta=1.0)
    geo = {}
    for f in (s.min_landing_ratio, s.max_landing_ratio, 2.0):
        cg = _cut_on(r, z, f * rn)[-1]
        geo[f] = (cg.bridge / (f * rn), cg.removed_gas / (f * rn) ** 3)
    lo, hi, far = (geo[f] for f in (s.min_landing_ratio, s.max_landing_ratio, 2.0))
    assert hi[0] / lo[0] < 1.25 and hi[1] / lo[1] < 1.25
    assert far[0] / lo[0] > 1.5 and far[1] / lo[1] > 1.3


def test_gas_volume_in_a_box():
    # a gas cylinder of radius a from the axis tip at z = 1 up to the wall closure
    a = 0.1
    open_chain = rc.Chain(np.array([[0.0, 1.0], [a, 1.0], [a, 2.0], [R_C, 2.0]]), ("axis", "fixed"))
    box = (0.5, 1.2, 1.7)
    assert rc.gas_volume_in_box([open_chain], 3.0, box) == pytest.approx(math.pi * a**2 * 0.5)
    # a bubble (a sphere of radius b) inside the box, and one cut by its lower edge
    b = 0.05
    th = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 2001)
    sphere = rc.Chain(np.column_stack([b * np.cos(th), 1.45 + b * np.sin(th)]), ("axis", "axis"))
    sphere.points[[0, -1], 0] = 0.0
    v = rc.gas_volume_in_box([open_chain, sphere], 3.0, box)
    assert v == pytest.approx(math.pi * a**2 * 0.5 + 4 / 3 * math.pi * b**3, rel=1e-5)
    half = rc.Chain(sphere.points - [0.0, 0.25], ("axis", "axis"))  # centre on z = 1.2
    v = rc.gas_volume_in_box([open_chain, half], 3.0, box)
    assert v == pytest.approx(math.pi * a**2 * 0.5 + 2 / 3 * math.pi * b**3, rel=1e-5)
    # moving the surface outside the box changes nothing
    moved = rc.Chain(np.array([[0.0, 1.0], [a, 1.0], [a, 2.0], [1.0, 2.3], [R_C, 2.1]]),
                     ("axis", "fixed"))
    assert rc.gas_volume_in_box([moved], 3.0, box) == rc.gas_volume_in_box([open_chain], 3.0, box)


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


def _bubble(z0=4.6, length=0.08, a=0.01, waist=0.0, n=801):
    """A bubble from the axis to the axis, ``r = a sin(pi s)`` with ``s`` from 0 to 1, narrowed at
    mid-height by the fraction ``waist``."""
    s = np.linspace(0.0, 1.0, n)
    r = a * np.sin(math.pi * s) * (1.0 - waist * np.exp(-(((s - 0.5) / 0.05) ** 2)))
    r[[0, -1]] = 0.0
    return rc.Chain(np.column_stack([r, z0 + length * s]), ("axis", "axis"))


def _cavity(z_tip=4.69, n=400):
    """An open chain from the axis just above the bubble, along a thread of radius 2e-3 that
    widens to the wall."""
    z = np.linspace(z_tip, 4.75, n)
    r = 2e-3 * np.sqrt(np.clip((z - z_tip) / 1e-3, 0.0, 1.0))
    pts = np.vstack([np.column_stack([r, z]), [[0.5, 5.0], [R_C, 6.0]]])
    return rc.Chain(pts, ("axis", "fixed"))


def test_post_cut_stops():
    s = rc.ReconnectSettings(delta=1e-3)
    bubble, cavity = _bubble(), _cavity()
    cut = rc.cut_geometry([bubble, cavity], 3.0, gas_before=1.0)
    assert cut.bridge == pytest.approx(0.01)
    assert rc.post_cut_stops([bubble, cavity], cut, s, h_min=1e-5) is None
    closing = rc.Chain(cavity.points - [0.0, 0.0095], ("axis", "fixed"))
    assert rc.post_cut_stops([bubble, closing], cut, s, 1e-5)["reason"] == "bridge-closing"
    small = rc.Chain(np.array([[0.0, 4.6], [1e-3, 4.60025], [0.0, 4.6005]]), ("axis", "axis"))
    assert rc.post_cut_stops([small, cavity], cut, s, 1e-5)["reason"] == "bubble-jets-approach"
    assert rc.post_cut_stops([bubble], cut, s, 1e-5) == {"reason": "topology-changed", "chains": 1}
    assert rc.post_cut_stops([bubble, cavity, small], cut, s, 1e-5)["chains"] == 3
    # the cavity coming down beside the bubble
    beside = rc.Chain(np.array([[0.0, 4.69], [0.0103, 4.6905], [0.0102, 4.64], [0.0105, 4.6],
                                [0.02, 4.6], [0.5, 5.0], [R_C, 6.0]]), ("axis", "fixed"))
    stop = rc.post_cut_stops([bubble, beside], cut, s, 1e-4)
    assert stop["reason"] == "chain-contact-approach" and stop["chain_gap"] < 3e-4
    # the bubble drains under its constant pressure
    thin = rc.Chain(bubble.points * [0.05, 1.0], ("axis", "axis"))
    assert rc.post_cut_stops([thin, cavity], cut, s, 1e-5)["reason"] == "bubble-vanishing"


def test_post_cut_stops_on_a_second_neck_of_either_chain():
    s = rc.ReconnectSettings(delta=1e-3)
    bubble, cavity = _bubble(), _cavity()
    cut = rc.cut_geometry([bubble, cavity], 3.0, gas_before=1.0)
    # a waist of the bubble wider than delta is no stop; narrower, it is a second pinch-off
    assert rc.post_cut_stops([_bubble(waist=0.5), cavity], cut, s, 1e-5) is None
    necked = _bubble(waist=0.95)
    stop = rc.post_cut_stops([necked, cavity], cut, s, 1e-5)
    assert stop["reason"] == "secondary-pinch-approach" and stop["chain"] == "bubble"
    assert stop["r_neck"] == pytest.approx(5e-4, rel=1e-3) and stop["z_neck"] == pytest.approx(4.64)
    assert stop["neck_limit"] == s.delta
    # the limit is at least neck_elements elements
    stop = rc.post_cut_stops([_bubble(waist=0.5), cavity], cut, s, 3e-3)
    assert stop["reason"] == "secondary-pinch-approach"
    assert stop["neck_limit"] == pytest.approx(9e-3)
    # the bubble's axis ends are no necks: the measured neck is interior on both sides
    assert rc.chain_neck(bubble.points, bubble.end_types)[2] == -1
    # a waist on the open chain, behind the cavity's fresh tip
    P = cavity.points.copy()
    k = 300
    P[k - 5:k + 6, 0] = np.minimum(P[k - 5:k + 6, 0], 8e-4 + 1e-5 * np.abs(np.arange(-5, 6)))
    stop = rc.post_cut_stops([bubble, rc.Chain(P, cavity.end_types)], cut, s, 1e-5)
    assert stop["reason"] == "secondary-pinch-approach" and stop["chain"] == "open"
    assert stop["r_neck"] == pytest.approx(8e-4)


def test_post_cut_stops_on_a_fold_of_the_bubble():
    """A slot of gap 5e-4 in the bubble's surface (the groove beside a jet), with nodes 2.5e-4
    apart: below 3 element lengths, it is a fold to stop at; a bubble that crosses itself raises."""
    from faraday_singularity.selfcontact import InterfaceSelfIntersection

    s = rc.ReconnectSettings(delta=1e-3)
    cavity = _cavity()
    h = 2.5e-4

    def line(a, b):
        n = max(2, math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / h) + 1)
        return np.linspace(a, b, n)[:-1]

    def slotted(gap):
        corners = [(0.0, 4.6), (0.01, 4.6), (0.01, 4.6 + gap), (0.002, 4.6 + gap),
                   (0.002, 4.65), (0.008, 4.66), (0.0, 4.68)]
        pts = np.vstack([line(corners[i], corners[i + 1]) for i in range(len(corners) - 1)]
                        + [np.array([corners[-1]])])
        return rc.Chain(pts, ("axis", "axis"))

    cut = rc.cut_geometry([slotted(0.01), cavity], 3.0, gas_before=1.0)
    assert rc.post_cut_stops([slotted(0.01), cavity], cut, s, 1e-5, fold_stop=3e-4) is None
    stop = rc.post_cut_stops([slotted(5e-4), cavity], cut, s, 1e-5, fold_stop=3e-4)
    assert stop["reason"] == "self-contact-approach" and stop["chain"] == "bubble"
    assert stop["trigger"] == "fold-gap" and stop["contact"] == "fold"  # resolved, ~16 elements deep
    assert stop["fold_gap"] == pytest.approx(5e-4) and stop["fold_limit"] == pytest.approx(3 * 2 * h)
    assert stop["fold_arc"] >= 4 * stop["fold_h_elem"]
    crossed = slotted(5e-4)
    crossed.points[[200, 201]] = crossed.points[[201, 200]]  # two nodes of the bottom swapped
    with pytest.raises(InterfaceSelfIntersection):
        rc.post_cut_stops([crossed, cavity], cut, s, 1e-5, fold_stop=3e-4)


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
    r_n, _, _ = rc.neck_radius(r, z)
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
    monotone from the tip, without knife-edge angles, and tangent at the splice; the liquid gains
    as much as on the synthetic thread."""
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
        # measured 12.7-22.9 delta and 50-76 delta^3 for delta/r_n from 1.03 to 2
        assert 12 * delta < cg.bridge < 24 * delta
        assert 48 * delta**3 < cg.removed_gas < 78 * delta**3


@stored
@pytest.mark.parametrize("step", [1028, 1055, 1076])
def test_cut_of_a_stored_neck_at_the_landing_ratio(step):
    """At delta/r_n = LANDING_RATIO the splices on a stored neck are the first survivors beyond the
    planner's window (the floor at delta drops none), the bridge is 13-14.5 delta long and the cut
    removes 54-59 delta^3 of gas, and no stop after the cut fires with the fold settings of the
    run the profiles come from (fold_stop 3e-4, fold_stop_elements 3, fold_ratio 4); the open chain
    keeps a neck wider than delta (``open_neck``), which ``post_cut_stops`` watches."""
    r, z, _ = _stored(step)
    r_n, _, _ = rc.neck_radius(r, z)
    delta = rc.LANDING_RATIO * r_n
    s = rc.ReconnectSettings(delta=delta)
    outcome, plan = rc.plan_on_profile(r, z, delta, s)
    assert outcome == "plan"
    edits = rc.replace_caps(plan.new_chains, delta, s)
    chains = [rc.orient_chain(c.points, R_C, 1e-7) for c in rc.plan_chains(plan.new_chains)]
    cg = rc.cut_geometry(chains, s.reservoir_depth, sum(plan.fragment_volumes_before))
    g = rc.in_cut_radii(edits, cg, delta, s)
    assert g["splice_set_by"] == ["planner-window"] * 2 and min(g["r_s"]) > 1.3
    assert 12.5 < g["bridge"] < 15 and 53 < g["removed_gas"] < 60
    h_min = rc.tip_size(delta, s)
    assert rc.post_cut_stops(chains, cg, s, h_min, neck_elements=3.0, fold_ratio=4.0,
                             fold_stop=3e-4, fold_stop_elements=3.0) is None
    open_neck = rc.chain_neck(next(c for c in chains if c.is_open).points, ("axis", "fixed"))[0]
    assert not open_neck < delta  # a neck narrower than delta would stop the run at the cut


@stored
def test_volume_correction_is_refused_on_the_stop_profile():
    from pyoomph.meshes import axisymm_topology as topo

    r, z, _ = _stored(1076)
    r_n, _, _ = rc.neck_radius(r, z)
    chain = rc.profile_chain(r, z, 3.0)
    with pytest.raises(RuntimeError, match="self-intersecting"):
        topo.detect_and_plan([chain], 1.05 * r_n, None, volume_conservation=True)


# --- after the cut: the jets of the bubble's tips ----------------------------------------------


def _jet_bubble(n=4001, R_b=0.1, zc=4.58, D=0.01, w=2e-3):
    """A spherical bubble whose top hangs down into a jet, ``z -> z - D exp(-(r/w)^2)`` on its upper
    half, as ``FaradayProblem.interface_segments`` gives it: upwards from its bottom tip."""
    th = np.linspace(0.0, math.pi, n)
    r = R_b * np.sin(th)
    r[[0, -1]] = 0.0
    z = zc - R_b * np.cos(th)
    upper = th > 0.5 * math.pi
    z[upper] -= D * np.exp(-((r[upper] / w) ** 2))
    return {"open": False, "end_types": ("axis", "axis"), "r": r, "z": z,
            "u_r": np.zeros_like(r), "u_z": np.where(upper, -1.0, 0.5)}


JET_KW = {"r_search": 1.5, "rise_tol": 1e-6, "flat_aspect": 0.01, "jet_aspect": 0.1,
          "jet_max_radius": 0.3}


def test_the_bubble_ends_are_taken_in_the_frame_of_their_jets():
    """Each end runs from its tip to the bubble's widest point; the top is mirrored, z and u_z, so
    that the liquid lies below its tip as at the cavity's."""
    seg = _jet_bubble()
    top, bottom = rc.bubble_end(seg, "bubble_top"), rc.bubble_end(seg, "bubble_bottom")
    k = int(np.argmax(seg["r"]))
    assert bottom["sign"] == 1.0 and top["sign"] == -1.0
    assert np.array_equal(bottom["r"], seg["r"][:k + 1]) and np.array_equal(bottom["z"],
                                                                            seg["z"][:k + 1])
    assert np.array_equal(top["r"], seg["r"][k:][::-1])
    assert np.array_equal(top["z"], -seg["z"][k:][::-1])
    # the widest node, the last of the top, lies on the lower half of this bubble
    assert np.all(top["u_z"][:-1] == 1.0) and top["u_z"][-1] == -0.5
    assert np.all(bottom["u_z"] == 0.5)


def test_the_jet_hanging_from_the_bubble_top_is_measured_as_one_rising_from_a_cavity():
    """The base of the jet the bridge drives down into the bubble is the highest point of the
    bubble's top near the axis; its height is returned in the frame of the axis. The bottom of the
    bubble, a sphere, has no jet: its tip is the lowest point there."""
    R_b, zc, D, w = 0.1, 4.58, 0.01, 2e-3
    seg = _jet_bubble(R_b=R_b, zc=zc, D=D, w=w)
    jet = rc.tip_jet(rc.bubble_end(seg, "bubble_top"), **JET_KW)
    r = np.linspace(1e-6, 0.05, 500001)
    z = zc + np.sqrt(R_b**2 - r**2) - D * np.exp(-((r / w) ** 2))
    k = int(np.argmax(z))
    assert jet["jet"] and jet["bump"] and not jet["flat_bump"] and jet["index"] > 0
    assert jet["r_j"] == pytest.approx(r[k], rel=1e-3)
    assert jet["z_b"] == pytest.approx(z[k], abs=1e-7)
    assert jet["z_tip"] == seg["z"][-1]
    assert jet["rise"] == pytest.approx(z[k] - seg["z"][-1], abs=1e-7)
    none = rc.tip_jet(rc.bubble_end(seg, "bubble_bottom"), **JET_KW)
    assert none["index"] == 0 and none["r_j"] == 0.0 and not none["bump"] and not none["jet"]
    assert none["z_b"] == none["z_tip"] == seg["z"][0]
    # a dimple barely deeper than the sphere's curvature allows (2 D/w^2 > 1/R_b) is a bump
    # but no jet (its rise is below jet_aspect r_j)
    flat = rc.tip_jet(rc.bubble_end(_jet_bubble(D=4e-5, w=2e-3), "bubble_top"), **JET_KW)
    assert flat["bump"] and flat["index"] > 0 and not flat["jet"]
    assert flat["rise"] < JET_KW["jet_aspect"] * flat["r_j"]


# --- after the cut: self-contact the remesh would meet, and the mesh of each feature -------------


def kinked(P, k, eps=0.02):
    """Nodes ``k + 1`` and ``k + 2`` of a polyline pulled to within ``eps`` of its spacing of node
    ``k``, with a zigzag: a cluster of fine nodes beside coarse ones, as gmsh's sizing from
    curvature left on the bubble's bottom of a run. The polygon stays simple; its uniform
    Catmull-Rom spline, the curve a remesh is built on, loops."""
    Q = np.array(P, dtype=float)
    t = Q[k + 3] - Q[k]
    t /= np.hypot(*t)
    n = np.array([-t[1], t[0]])
    d = eps * np.hypot(*(Q[k + 1] - Q[k]))
    Q[k + 1] = Q[k] + d * t + 0.3 * d * n
    Q[k + 2] = Q[k] + 2 * d * t - 0.3 * d * n
    return Q


def test_a_kink_whose_spline_loops_is_a_clean_stop_before_the_remesh_meets_it():
    """A kink of three nodes whose spline crosses itself, while the nodes do not, is the contact a
    remesh from those nodes raises on (check_chains, InterfaceSelfIntersection: a failed run). The
    stops after the cut make the same test first and stop cleanly (self-contact-approach, trigger
    spline-contact), on either chain, and report it as a kink at the scale of the mesh, not a fold:
    its two sides are three node intervals apart, beside a jump in node spacing."""
    from faraday_singularity.selfcontact import InterfaceSelfIntersection, first_contact

    s = rc.ReconnectSettings(delta=1e-3)
    bubble, cavity = _bubble(), _cavity()
    cut = rc.cut_geometry([bubble, cavity], 3.0, gas_before=1.0)
    bad = rc.Chain(kinked(bubble.points, 300), bubble.end_types)
    contact = first_contact(bad.points)
    assert contact["what"] == "spline" and contact["nodes"] == (299, 300, 301, 302)
    with pytest.raises(InterfaceSelfIntersection, match="spline through the free surface"):
        rc.check_chains([bad, cavity])
    stop = rc.post_cut_stops([bad, cavity], cut, s, h_min=1e-5)
    assert stop["reason"] == "self-contact-approach" and stop["trigger"] == "spline-contact"
    assert stop["chain"] == "bubble" and stop["contact"] == "kink" and stop["span"] == 3
    assert stop["nodes"] == [299, 300, 301, 302] and stop["spacing_ratio"] > 20
    assert stop["contact_r"] == pytest.approx(bad.points[299, 0], rel=1e-2)
    # on the open chain, beyond the region of the driver's own fold stop
    k = 200
    bad_open = rc.Chain(kinked(cavity.points, k), cavity.end_types)
    with pytest.raises(InterfaceSelfIntersection):
        rc.check_chains([bubble, bad_open])
    stop = rc.post_cut_stops([bubble, bad_open], cut, s, h_min=1e-5)
    assert stop["trigger"] == "spline-contact" and stop["chain"] == "open"
    assert stop["contact"] == "kink"
    # nodes that have crossed are no state to stop on
    crossed = bubble.points.copy()
    crossed[[200, 201]] = crossed[[201, 200]]
    with pytest.raises(InterfaceSelfIntersection, match="bubble"):
        rc.post_cut_stops([rc.Chain(crossed, bubble.end_types), cavity], cut, s, h_min=1e-5)


def test_a_fold_spanning_few_elements_is_a_kink():
    assert rc.fold_contact(arc=4e-4, h_elem=1e-4) == "fold"
    assert rc.fold_contact(arc=3.9e-4, h_elem=1e-4) == "kink"


def _post_cut_segs(bottom=None, n=4001):
    """Both chains after a cut as ``FaradayProblem.interface_segments`` gives them: the jet bubble
    (its top hanging into a jet) with, with ``bottom = (D, w)``, its bottom rising into a jet,
    ``z -> z + D exp(-(r/w)^2)`` on its lower half, and the cavity above it."""
    seg = _jet_bubble(n=n)
    if bottom is not None:
        D, w = bottom
        lower = np.arange(n) < n // 2
        seg["z"][lower] += D * np.exp(-((seg["r"][lower] / w) ** 2))
    cav = _cavity(z_tip=seg["z"][-1] + 0.01)
    open_ = {"open": True, "end_types": ("axis", "fixed"), "r": cav.points[:, 0],
             "z": cav.points[:, 1], "u_r": np.zeros(len(cav.points)),
             "u_z": np.zeros(len(cav.points))}
    return [seg, open_]


def _regions(segs, cut, s, cavity=None, **kw):
    bubble = segs[0]
    tips = {name: rc.tip_jet(rc.bubble_end(bubble, name), **JET_KW) for name in rc.BUBBLE_TIP_SIGNS}
    cavity = cavity or {"z_tip": float(segs[1]["z"][0]), "kappa": 1.0 / 1e-4, "r_j": 0.0,
                        "rise": 0.0, "bump": False, "flat_bump": False}
    kw = {"h_floor": 1e-6, "h_axis": 0.05, "elements_per_scale": 16.0, "focus_extent": 20.0, **kw}
    return {g["name"]: g for g in rc.feature_regions(segs, cavity, tips, rc.post_cut_record(segs),
                                                     cut, s, **kw)}


def test_the_tips_the_cut_made_are_meshed_at_their_own_scale_never_coarser_than_the_cut():
    """The cavity's tip and the bubble's top, which the cut made, are meshed at their own scale,
    clipped to between the landing run's h_floor and the cut's focus size (focus_size cut radii),
    the coarsest size they may have: an apex radius of 1e-4 at the cavity's tip asks for
    1e-4/16, the floor here, within 20 of its radii, and the jet hanging from the bubble's top
    (base radius about 3e-3, 0.01 deep, its tip of radius about 2e-4) for its tip's radius over
    16 within 20 of those; beside each, ``<tip>_cut`` keeps the cut's focus size over the cut's
    focus radius and the jet. A broad tip (radius 0.05) is meshed at the cut's focus size over
    that region alone, as the cut sized it."""
    s = rc.ReconnectSettings(delta=1e-3)
    h_cut, r_cut = s.focus_size * 1e-3, s.focus_radius * 1e-3
    segs = _post_cut_segs()
    cut = {"delta": 1e-3, "rmin": 1e-3}
    g = _regions(segs, cut, s, h_floor=6.25e-6)
    rec = rc.post_cut_record(segs)
    top = rc.tip_jet(rc.bubble_end(segs[0], "bubble_top"), **JET_KW)
    reach = 2 * math.hypot(top["r_j"], top["rise"])
    assert g["cavity_tip"]["target"] == pytest.approx(6.25e-6) and g["cavity_tip"]["ell"] == 1e-4
    assert g["cavity_tip"]["radius"] == pytest.approx(20 * 1e-4)
    R_top = rec["R_bubble_top"]
    assert R_top < top["r_j"] and g["bubble_top"]["ell"] == pytest.approx(R_top)
    assert g["bubble_top"]["target"] == pytest.approx(R_top / 16)
    assert g["bubble_top"]["radius"] == pytest.approx(20 * R_top)
    for name in rc.FRESH_TIPS:
        assert g[name]["r"] == 0.0 and g[name]["floor"] == 6.25e-6
        assert g[name + "_cut"]["target"] == pytest.approx(h_cut) == g[name + "_cut"]["floor"]
        assert g[name + "_cut"]["z"] == g[name]["z"]
    assert g["cavity_tip"]["z"] == segs[1]["z"][0] and g["bubble_top"]["z"] == segs[0]["z"][-1]
    assert g["cavity_tip_cut"]["radius"] == pytest.approx(r_cut)
    assert g["bubble_top_cut"]["radius"] == pytest.approx(reach) and reach > r_cut
    assert "bubble_bottom" not in g  # a sphere's bottom, no landing focus: no feature
    # the floor is the landing run's h_floor
    assert _regions(segs, cut, s, h_floor=2e-5)["cavity_tip"]["target"] == pytest.approx(2e-5)
    # a broad tip: the cut's focus size over its focus radius, no second region
    broad = _regions(segs, cut, s, h_floor=6.25e-6,
                     cavity={"z_tip": float(segs[1]["z"][0]), "kappa": 1.0 / 0.05, "r_j": 0.0,
                             "rise": 0.0, "bump": False, "flat_bump": False})
    assert broad["cavity_tip"]["target"] == pytest.approx(h_cut)
    assert broad["cavity_tip"]["radius"] == pytest.approx(r_cut) and "cavity_tip_cut" not in broad


def test_the_bubble_bottom_keeps_its_landing_resolution_far_from_the_cut():
    """The bubble's bottom, the landing's cavity bottom with its jet (base radius about 8e-3, a
    size of 5e-4 by its own scale), keeps the finer resolution it had at the landing (a focus of
    4e-5 within 0.02 of the axis) over the region the landing resolved, within its half of the
    bubble, far below the cut; the cut's focus size (6.25e-5) would coarsen it."""
    s = rc.ReconnectSettings(delta=1e-3)
    segs = _post_cut_segs(bottom=(2e-3, 5e-3))
    z_bb, z_bt = float(segs[0]["z"][0]), float(segs[0]["z"][-1])
    kept = rc.kept_region({"size": 4e-5, "radius": 0.02, "grading": 0.2}, 6.25e-5, z_bb, z_bt)
    assert kept["size"] == 4e-5 and kept["radius"] == 0.02
    cut = {"delta": 1e-3, "rmin": 1e-3, "focus": {"kept": kept}}
    g = _regions(segs, cut, s, h_floor=6.25e-6)["bubble_bottom"]
    bottom = rc.tip_jet(rc.bubble_end(segs[0], "bubble_bottom"), **JET_KW)
    assert bottom["bump"] and 3e-3 < bottom["r_j"] < 1e-2
    assert g["target"] == 4e-5 and g["kept"] == 4e-5 and g["floor"] == 6.25e-6
    assert (g["r"], g["z"]) == (0.0, z_bb)
    assert g["radius"] == pytest.approx(min(20 * bottom["r_j"], 0.5 * (z_bt - z_bb)))
    assert g["radius"] >= kept["radius"]
    # a narrow jet there asks for its own, finer size, floored at the landing run's h_floor
    narrow = _regions(_post_cut_segs(bottom=(2e-4, 1e-4), n=40001), cut, s, h_floor=6.25e-6)
    r_j = narrow["bubble_bottom"]["ell"]
    assert narrow["bubble_bottom"]["target"] == pytest.approx(max(r_j / 16, 6.25e-6))
    assert narrow["bubble_bottom"]["target"] < 4e-5
    # within its half of the bubble only, so that it never reaches the bubble's fresh top
    wide = rc.kept_region({"size": 4e-5, "radius": 1.0, "grading": 0.2}, 6.25e-5, z_bb, z_bt)
    assert wide["radius"] == pytest.approx(0.5 * (z_bt - z_bb))
    # the landing coarser than the cut: the cut's focus size; no landing focus: none
    assert rc.kept_region({"size": 1e-4, "radius": 0.02}, 6.25e-5, z_bb, z_bt)["size"] == 6.25e-5
    assert rc.kept_region(None, 6.25e-5, z_bb, z_bt) is None


def test_a_neck_after_the_cut_is_meshed_at_its_own_scale():
    s = rc.ReconnectSettings(delta=1e-3)
    segs = _post_cut_segs()
    P = np.column_stack([segs[1]["r"], segs[1]["z"]])
    k = 300
    P[k - 5:k + 6, 0] = np.minimum(P[k - 5:k + 6, 0], 1.6e-3 + 1e-5 * np.abs(np.arange(-5, 6)))
    segs[1]["r"] = P[:, 0]
    g = _regions(segs, {"delta": 1e-3, "rmin": 1e-3}, s)["neck_open"]
    assert g["ell"] == pytest.approx(1.6e-3) and g["target"] == pytest.approx(1e-4)
    assert (g["r"], g["z"]) == pytest.approx((1.6e-3, P[k, 1])) and g["radius"] == pytest.approx(0.032)
    assert g["floor"] == pytest.approx(s.focus_size * 1e-3)


def test_region_spacing_reads_the_longest_and_the_free_shortest_element():
    segs = _post_cut_segs()
    region = {"r": 0.0, "z": float(segs[0]["z"][0]), "radius": 0.02}
    h_max, h_min = rc.region_spacing(segs, region, [float(segs[0]["z"][0])], None, 1e-4)
    r, z = segs[0]["r"], segs[0]["z"]
    inside = np.hypot(r - region["r"], z - region["z"]) <= region["radius"]
    h = 2 * np.hypot(np.diff(r), np.diff(z))[inside[:-1] & inside[1:]]
    assert h_max == pytest.approx(h.max()) and h_min == pytest.approx(h.min())
    assert rc.region_spacing(segs, {"r": 1.0, "z": 0.0, "radius": 1e-3}, [], None, 1e-4) == (
        pytest.approx(math.nan, nan_ok=True), pytest.approx(math.nan, nan_ok=True))


# --- after the cut: the interface on its quadratic elements, the bases of the jets ---------------


def _p2_line(n=41, L=1e-4, z=4.6):
    """A straight chain of ``n`` quadratic elements of chord ``L``: vertices and midside nodes."""
    r = np.linspace(0.0, n * L, 2 * n + 1)
    return np.column_stack([r, np.full_like(r, z)])


def _bent(P, k, turn_deg=None, f=None):
    """The chain ``P`` with the midside node of element ``k`` moved off the chord so that its
    tangent turns by ``turn_deg`` across it (``s/L = tan(turn/2)/4`` at mid-chord), or along the
    chord to ``f`` of it."""
    Q = np.array(P, dtype=float)
    a, b = Q[2 * k], Q[2 * k + 2]
    c = b - a
    L = math.hypot(*c)
    n = np.array([-c[1], c[0]]) / L
    s = 0.0 if turn_deg is None else L * math.tan(math.radians(turn_deg) / 2) / 4
    Q[2 * k + 1] = a + (0.5 if f is None else f) * c + s * n
    return Q


def test_p2_edges_on_a_circle_a_hairpin_and_a_sliding_midside_node():
    """On a circle of radius R the bulge of an element is s/L = tan(phi/2)/2, about L/(8R), and its
    tangent turns by the angle it subtends; a midside node pushed 0.55 chords off the chord turns
    its element by 2 atan(2.2) = 131 degrees, the hairpin of the failed run; a midside node slid
    along the chord to 0.28 of it is read as that, with no turn. A chain with an even number of
    nodes is no chain of quadratic elements."""
    R, phi = 1e-3, math.radians(3.0)  # each element subtends 2 phi
    th = np.arange(41) * phi
    e = rc.p2_edges(np.column_stack([R * np.sin(th), 4.6 - R * np.cos(th)]))
    L = 2 * R * math.sin(phi)
    assert e["L"] == pytest.approx(np.full(20, L), rel=1e-12)
    assert e["bulge"] == pytest.approx(np.full(20, math.tan(phi / 2) / 2), rel=1e-9)
    assert e["bulge"] == pytest.approx(np.full(20, L / (8 * R)), rel=2e-3)
    exact = math.degrees(2 * math.atan(2 * math.tan(phi / 2)))
    assert e["turn"] == pytest.approx(np.full(20, exact), rel=1e-9)
    assert e["turn"] == pytest.approx(np.full(20, math.degrees(2 * phi)), rel=1e-3)
    assert e["f"] == pytest.approx(np.full(20, 0.5), abs=1e-9)
    P = _p2_line()
    Q = P.copy()
    Q[2 * 7 + 1] += [0.0, 0.55 * 1e-4]
    e = rc.p2_edges(Q)
    assert e["bulge"][7] == pytest.approx(0.55) and e["turn"][7] == pytest.approx(131.1, abs=0.05)
    assert e["turn"][7] == pytest.approx(math.degrees(2 * math.atan(2.2)), rel=1e-9)
    assert np.all(e["turn"][np.arange(41) != 7] == pytest.approx(0.0, abs=1e-9))
    rd = rc.edge_reading(Q)
    assert rd["turn"] == e["turn"][7] and rd["turn_index"] == 15
    assert rd["turn_L"] == pytest.approx(1e-4)
    assert (rd["turn_r"], rd["turn_z"]) == pytest.approx(tuple(Q[15]))
    slid = rc.p2_edges(_bent(P, 12, f=0.28))
    assert slid["f"][12] == pytest.approx(0.28) and slid["turn"][12] == pytest.approx(0.0, abs=1e-9)
    assert rc.edge_reading(_bent(P, 12, f=0.28))["midside"] == pytest.approx(0.22)
    with pytest.raises(ValueError):
        rc.p2_edges(P[:-1])
    assert math.isnan(rc.edge_reading(P[:-1])["turn"]) and rc.edge_reading(P[:-1])["edges"] is None


def _guard(P, fresh, h_floor=1e-6, **kw):
    s = rc.ReconnectSettings(delta=1e-3, **kw)
    return rc.interface_guard({"bubble": rc.edge_reading(P)}, s, fresh=fresh, h_floor=h_floor,
                              refine_ratio=1.5)


@pytest.mark.parametrize("fresh", [True, False])
def test_the_interface_guard_refines_beyond_30_degrees_and_stops_beyond_90_or_120(fresh):
    """A rebuild refines where an element's tangent turns by more than 30 degrees across it or its
    midside node lies more than a tenth of the chord from mid-chord; the run stops
    (interface-unresolved) beyond 90 degrees on a freshly built mesh and 120 on any, and where the
    midside node lies more than a fifth of the chord off mid-chord, and also where an element that
    asks for a refinement is already at h_floor (feature-below-floor)."""
    P = _p2_line()
    for turn, refine in ((29.0, False), (31.0, True)):
        out = _guard(_bent(P, 9, turn), fresh)
        assert out["stop"] is None and bool(out["refine"]) == refine, turn
    assert list(_guard(_bent(P, 9, 31.0), fresh)["refine"]["bubble"]) == [9]
    for turn in (89.0, 91.0, 119.0, 121.0):
        out = _guard(_bent(P, 9, turn), fresh)
        stops = turn > (90.0 if fresh else 120.0)
        assert (out["stop"] is not None) == stops, turn
        if stops:
            stop = out["stop"]
            assert stop["reason"] == "interface-unresolved" and stop["trigger"] == "edge-turn"
            assert stop["chain"] == "bubble" and stop["turn"] == pytest.approx(turn)
            assert stop["limit"] == (90.0 if fresh else 120.0) and stop["fresh"] is fresh
            assert stop["index"] == 19 and stop["L"] == pytest.approx(1e-4)
        else:
            assert list(out["refine"]["bubble"]) == [9]
    out = _guard(_bent(P, 9, f=0.31), fresh)  # 0.19 off mid-chord: a refinement
    assert out["stop"] is None and list(out["refine"]["bubble"]) == [9]
    stop = _guard(_bent(P, 9, f=0.29), fresh)["stop"]  # 0.21 off: a stop
    assert stop["trigger"] == "edge-midside" and stop["midside"] == pytest.approx(0.21)
    # an element at the floor that asks for a refinement: no rebuild can resolve it
    stop = _guard(_bent(P, 9, 40.0), fresh, h_floor=1e-4)["stop"]
    assert stop["trigger"] == "feature-below-floor" and stop["h_floor"] == 1e-4
    assert _guard(_bent(P, 9, 40.0), fresh, h_floor=6e-5)["stop"] is None  # 1e-4 > 1.5 x 6e-5
    # the thresholds are settings of the cut
    assert _guard(_bent(P, 9, 31.0), fresh, edge_turn_refine=35.0)["refine"] == {}


def _as_mesh_nodes(points, samples=2):
    """A chain as a rebuild would mesh it: its points as vertices and the midside node of each
    element on the uniform Catmull-Rom spline through them (pyoomph's and gmsh's), half way in
    its parameter."""
    from faraday_singularity.selfcontact import catmull_rom

    return catmull_rom(points[:, 0], points[:, 1], samples=samples)


def test_the_guard_does_not_fire_on_freshly_cut_chains(thread_scan):
    """The chains a cut makes on a slender thread, meshed as a rebuild would (their nodes as
    vertices, midside nodes on the spline through them), bend by a few degrees per element and
    keep their midside nodes at mid-chord within the cut's region (focus_radius cut radii of the
    neck: the caps, the splices, the thread behind them): the guard neither refines nor stops
    there. (Farther out the synthetic thread's bubble bottom is sampled at 0.02 on a radius of
    0.03, which the guard rightly reads as too coarse.)"""
    for r, z, rn in thread_scan.values():
        delta = 1.05 * rn
        s, _, _, _, chains, cg = _cut_on(r, z, delta)
        z_n = 0.5 * (cg.z_bubble_top + cg.z_cavity_tip)
        readings = {}
        for c in chains:
            P = _as_mesh_nodes(c.points)
            near = np.flatnonzero(np.hypot(P[:, 0], P[:, 1] - z_n) <= s.focus_radius * delta)
            lo, hi = near.min() // 2 * 2, -(-near.max() // 2) * 2  # whole elements
            readings["open" if c.is_open else "bubble"] = rc.edge_reading(P[lo:hi + 1])
        for rd in readings.values():
            assert rd["turn"] < 15.0 and rd["midside"] < 0.05, rd
        out = rc.interface_guard(readings, s, fresh=True, h_floor=6.25e-6, refine_ratio=1.5)
        assert out == {"stop": None, "refine": {}}


def _groove(w=1e-3, r_jet=1e-3, h=2.5e-5, depth=5.0, z_c=4.70, z_tip=4.71):
    """An open chain from the axis whose jet (a cylinder of radius ``r_jet`` with a rounded top at
    ``z_tip``) stands in a groove of width ``w`` with a semicircular bottom (radius ``w/2``,
    centre at ``z_c``) and walls ``depth`` widths high, then runs out to the wall: the nodes of
    quadratic elements ``h`` long, coarser beyond the groove. Returns the chain and the index of
    the groove's apex."""
    R0 = 0.5 * w

    def seg(a, b, spacing=0.5 * h):
        n = max(1, math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / spacing))
        return [np.array(a) + (np.array(b) - np.array(a)) * k / n for k in range(n)]

    def arc(c, R, a0, a1):
        n = max(2, math.ceil(abs(a1 - a0) * R / (0.5 * h)))
        return [np.array([c[0] + R * math.cos(a), c[1] + R * math.sin(a)])
                for a in np.linspace(a0, a1, n + 1)[:-1]]

    top = z_tip - r_jet
    out = (r_jet + w + 14.0 * w, z_c + depth * w + 14.0 * w)
    pts = (arc((0.0, top), r_jet, 0.5 * math.pi, 0.0)  # the jet's rounded top
           + seg((r_jet, top), (r_jet, z_c))  # down its side
           + arc((r_jet + R0, z_c), R0, math.pi, 2 * math.pi)  # the groove's bottom
           + seg((r_jet + w, z_c), (r_jet + w, z_c + depth * w))  # up the cavity wall
           + seg((r_jet + w, z_c + depth * w), out)  # fine for 20 widths beyond the groove,
           + seg(out, (0.5, 5.0), 2e-3) + [np.array([R_C, 5.2])])  # then coarse to the wall
    P = np.array(pts)
    if len(P) % 2 == 0:
        P = np.vstack([P[:-1], [[0.5 * (P[-2, 0] + R_C), 5.1]], P[-1:]])
    P[0, 0] = 0.0
    apex = int(np.argmin(np.where(P[:, 0] < 0.1, P[:, 1], np.inf)))
    return P, apex


def test_the_apex_and_the_gap_of_an_analytic_groove():
    """At the base of a jet standing in a groove of width w with a semicircular bottom, the apex
    radius read on the vertices (the smaller of the fits over 2 and 4 of them on either side) is
    the bottom's radius w/2, and the groove's fold pair (straddling the apex, closer than a
    quarter of the arc between them) lies across its width: both within 5 percent; the walls
    being parallel, its opening angle is small."""
    from faraday_singularity.collapse import Profile, jet_base

    for w in (1e-3, 4e-4):
        P, apex = _groove(w=w, h=w / 40)
        jb = jet_base(Profile(P[:, 0], P[:, 1]), 1.5, 1e-6)
        assert abs(jb.index - apex) <= 1 and jb.jet_born
        a = rc.jet_base_apex(P[:, 0], P[:, 1], jb.index)
        assert a["apex"] % 2 == 0 and a["R_apex"] == pytest.approx(w / 2, rel=0.05)
        assert a["apex_z"] == pytest.approx(4.70 - w / 2, abs=w / 20)
        # the chords to the vertices 6 away: 6 elements of w/40 along the bottom of radius w/2
        assert a["groove_angle"] == pytest.approx(180.0 - math.degrees(0.3), abs=0.5)
        g = rc.groove_gap(P[:, 0], P[:, 1], a["apex"], ratio=4.0, stop=1e-4, stop_elements=3.0)
        assert g["gap"] == pytest.approx(w, rel=0.05)
        i, j = g["pair"]
        assert i < a["apex"] < j and 4.0 * g["gap"] < g["arc"]
    # met by few elements (w/4), the chords reach up its walls: its opening angle is small
    P, apex = _groove(w=1e-3, h=2.5e-4)
    assert rc.jet_base_apex(P[:, 0], P[:, 1], apex)["groove_angle"] < 50.0
    # an open groove (a wedge of 120 degrees) has no fold pair
    s = np.linspace(0.0, 4e-3, 161)
    V = np.column_stack([1e-3 + np.r_[-s[::-1], s[1:]] * math.sin(math.radians(60)),
                         4.7 + np.abs(np.r_[-s[::-1], s[1:]]) * math.cos(math.radians(60))])
    g = rc.groove_gap(V[:, 0], V[:, 1], 160, ratio=4.0, stop=1e-4, stop_elements=3.0)
    assert math.isinf(g["gap"])
    a = rc.jet_base_apex(V[:, 0], V[:, 1], 160)
    # a corner: its radius is that of the fit, about its window (two vertices, 1e-4, each side)
    assert a["groove_angle"] == pytest.approx(120.0, abs=0.5) and a["R_apex"] < 2e-4


def test_the_annular_stop_fires_only_on_a_closing_groove_across_a_jet_base():
    """A fold pair across the groove at the base of a jet whose gap falls below the fold limit
    max(fold_stop, 3 h) is the annular-contact-approach, with the jet, its apex and R_apex; the
    same fold away from a tracked apex is a self-contact-approach, a groove wider than the limit no
    stop at all, and a fresh cut has none."""
    s = rc.ReconnectSettings(delta=1e-3)
    bubble = _bubble()
    for w, stops in ((4e-4, True), (2e-3, False)):
        P, apex = _groove(w=w, h=w / 40)
        cavity = rc.Chain(P, ("axis", "fixed"))
        cut = rc.cut_geometry([bubble, cavity], 3.0, gas_before=1.0)
        a = rc.jet_base_apex(P[:, 0], P[:, 1], apex)
        apexes = {"open": [{"tip": "cavity_tip", "index": a["apex"], "r": 1e-3 + w / 2,
                            "z": a["apex_z"], "R_apex": a["R_apex"],
                            "groove_angle": a["groove_angle"]}]}
        stop = rc.post_cut_stops([bubble, cavity], cut, s, 1e-5, fold_stop=1e-3, apexes=apexes)
        if not stops:
            assert stop is None
            continue
        assert stop["reason"] == "annular-contact-approach" and stop["trigger"] == "fold-gap"
        assert stop["chain"] == "open" and stop["jet"] == "cavity_tip"
        assert stop["fold_gap"] == pytest.approx(w, rel=0.05) and stop["fold_limit"] == 1e-3
        assert stop["apex_index"] == a["apex"] and stop["R_apex"] == pytest.approx(w / 2, rel=0.05)
        i, j = stop["fold_pair"]
        assert i < a["apex"] < j and stop["contact"] == "fold"
        # not straddling a tracked apex: an ordinary self-contact
        elsewhere = {"open": [dict(apexes["open"][0], index=len(P) - 3)]}
        other = rc.post_cut_stops([bubble, cavity], cut, s, 1e-5, fold_stop=1e-3, apexes=elsewhere)
        assert other["reason"] == "self-contact-approach" and other["trigger"] == "fold-gap"
        assert rc.post_cut_stops([bubble, cavity], cut, s, 1e-5,
                                 fold_stop=1e-3)["reason"] == "self-contact-approach"


def test_the_annular_stop_does_not_fire_on_a_fresh_cut(thread_scan):
    for r, z, rn in thread_scan.values():
        s, _, _, _, chains, cg = _cut_on(r, z, 1.05 * rn)
        apexes = {"open": [{"tip": "cavity_tip", "index": 5}],
                  "bubble": [{"tip": "bubble_top", "index": len(chains[0].points) - 6}]}
        assert rc.post_cut_stops(chains, cg, s, rc.tip_size(1.05 * rn, s), fold_stop=3e-4,
                                 apexes=apexes) is None


def test_the_bases_of_the_jets_are_meshed_in_their_own_regions():
    """The base of each jet, the apex of the groove around it, is meshed at its radius predicted
    ahead over 16, clipped to between h_floor and the cut's focus size, within max(8 R_apex, 4
    target) of the base: none where that would be the cut's focus size (the tip's region covers
    it); at the bubble's bottom, far from the cut, the cap is the size its own region asks for."""
    s = rc.ReconnectSettings(delta=1e-3)
    segs = _post_cut_segs()
    cut = {"delta": 1e-3, "rmin": 1e-3}
    base = {"r": 3e-3, "z": 4.67, "R_apex": 4e-4, "R_pred": 3e-4}
    g = _regions(segs, cut, s, h_floor=6.25e-6, apexes={"bubble_top": base})["jet_base_bubble_top"]
    assert g["target"] == pytest.approx(3e-4 / 16) and g["ell"] == 4e-4
    assert (g["r"], g["z"]) == (3e-3, 4.67) and g["radius"] == pytest.approx(8 * 4e-4)
    assert g["floor"] == 6.25e-6
    floored = _regions(segs, cut, s, h_floor=6.25e-6,
                       apexes={"bubble_top": dict(base, R_apex=5e-5, R_pred=2.5e-5)})
    assert floored["jet_base_bubble_top"]["target"] == 6.25e-6
    assert floored["jet_base_bubble_top"]["radius"] == pytest.approx(8 * 5e-5)
    broad = _regions(segs, cut, s, apexes={"bubble_top": dict(base, R_apex=2e-3, R_pred=2e-3)})
    assert "jet_base_bubble_top" not in broad  # 2e-3/16 is above the cut's focus size
    # the bubble's bottom: capped at its own region's target
    segs = _post_cut_segs(bottom=(2e-3, 5e-3))
    kept = rc.kept_region({"size": 4e-5, "radius": 0.02, "grading": 0.2}, 6.25e-5,
                          float(segs[0]["z"][0]), float(segs[0]["z"][-1]))
    cut = {"delta": 1e-3, "rmin": 1e-3, "focus": {"kept": kept}}
    for R, present in ((1.6e-3, False), (3.2e-4, True)):
        g = _regions(segs, cut, s, h_floor=6.25e-6,
                     apexes={"bubble_bottom": {"r": 5e-3, "z": 4.49, "R_apex": R, "R_pred": R}})
        assert g["bubble_bottom"]["target"] == 4e-5
        assert ("jet_base_bubble_bottom" in g) == present
        if present:
            assert g["jet_base_bubble_bottom"]["target"] == pytest.approx(R / 16)


def test_kink_regions_follow_the_guard_and_are_released_after_their_hold():
    """Where the guard asks for a refinement, a region about the worst element of the chain at a
    quarter of its chord (never below h_floor) within four chords; asked again elsewhere, it grows
    to hold both; once the chain reads no such element for its hold it is released, its target the
    chord it met."""
    P = _bent(_p2_line(), 9, 40.0)
    readings = {"bubble": rc.edge_reading(P)}
    s = rc.ReconnectSettings(delta=1e-3)
    refine = rc.interface_guard(readings, s, fresh=False, h_floor=1e-6, refine_ratio=1.5)["refine"]
    k = rc.kink_regions({}, readings, refine, t=1.0, hold=lambda L: 0.5, h_floor=1e-6)["bubble"]
    assert k["name"] == "kink_bubble" and (k["r"], k["z"]) == pytest.approx(tuple(P[19]))
    assert k["target"] == pytest.approx(2.5e-5) and k["radius"] == pytest.approx(4e-4)
    assert k["h_before"] == pytest.approx(1e-4) and not k["released"] and k["t_set"] == 1.0
    assert rc.kink_regions({}, readings, refine, t=1.0, hold=lambda L: 0.5,
                           h_floor=5e-5)["bubble"]["target"] == 5e-5
    Q = _bent(_p2_line(), 30, 40.0)
    rq = {"bubble": rc.edge_reading(Q)}
    grown = rc.kink_regions({"bubble": k}, rq, {"bubble": np.array([30])}, t=1.2,
                            hold=lambda L: 0.5, h_floor=1e-6)["bubble"]
    assert grown["radius"] == pytest.approx(math.hypot(*(Q[61] - P[19])) + 4e-4)
    assert (grown["r"], grown["z"]) == (k["r"], k["z"]) and grown["t_set"] == 1.2
    calm = {"bubble": rc.edge_reading(_p2_line())}
    held = rc.kink_regions({"bubble": grown}, calm, {}, t=1.5, hold=lambda L: 0.5, h_floor=1e-6)
    assert not held["bubble"]["released"]
    gone = rc.kink_regions({"bubble": grown}, calm, {}, t=1.8, hold=lambda L: 0.5, h_floor=1e-6)
    assert gone["bubble"]["released"] and gone["bubble"]["target"] == pytest.approx(1e-4)


def _jacobian_sampled(X, n=60):
    """``det(dx/ds)`` of a quadratic triangle (nodes in oomph-lib's order) on a grid of its
    reference triangle, from the derivatives of its Lagrange shape functions."""
    out = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            s1, s2 = i / n, j / n
            s3 = 1.0 - s1 - s2
            # N0 = s1(2s1-1), N1 = s2(2s2-1), N2 = s3(2s3-1), N3 = 4s1s2, N4 = 4s2s3, N5 = 4s3s1
            d1 = np.array([4 * s1 - 1, 0.0, -(4 * s3 - 1), 4 * s2, -4 * s2, 4 * (s3 - s1)])
            d2 = np.array([0.0, 4 * s2 - 1, -(4 * s3 - 1), 4 * s1, 4 * (s3 - s2), -4 * s1])
            a, b = d1 @ X, d2 @ X
            out.append(a[0] * b[1] - a[1] * b[0])
    return np.array(out)


def test_the_bezier_bound_of_the_jacobian_of_a_quadratic_triangle():
    """The bound is 1 on a straight-sided triangle, exact at the vertices, never above the
    Jacobian sampled densely over the element (and close to its minimum for mildly curved ones),
    zero for a midside node at the quarter point of its edge and negative past it (or for an
    element inside out)."""
    tri = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    mid = np.array([0.5 * (tri[0] + tri[1]), 0.5 * (tri[1] + tri[2]), 0.5 * (tri[2] + tri[0])])
    straight = np.vstack([tri, mid])
    assert rc.p2_jacobian_bounds(straight, [range(6)]) == pytest.approx([1.0])
    for f, expected in ((0.5, 1.0), (0.3, 0.2), (0.25, 0.0), (0.2, -0.2)):
        X = straight.copy()
        X[3] = tri[0] + f * (tri[1] - tri[0])
        b = rc.p2_jacobian_bounds(X, [range(6)])[0]
        assert b == pytest.approx(expected, abs=1e-12), f
        assert b <= _jacobian_sampled(X).min() + 1e-12
    rng = np.random.default_rng(3)
    for scale in (0.05, 0.2, 0.35):
        for _ in range(40):
            X = straight.copy()
            X[3:] += scale * rng.uniform(-1.0, 1.0, (3, 2))
            b = rc.p2_jacobian_bounds(X, [range(6)])[0]
            J = _jacobian_sampled(X)
            assert b <= J.min() + 1e-12
            corners = _jacobian_sampled(X, n=1)  # at s = (0, 0), (0, 1), (1, 0): the vertices
            assert b <= corners.min() + 1e-12
            if scale == 0.05:
                assert J.min() - b < 0.15
    inside_out = straight[[1, 0, 2, 3, 5, 4]]
    assert rc.p2_jacobian_bounds(inside_out, [range(6)])[0] < 0.0


def test_the_settings_of_the_interface_after_the_cut_are_checked_and_are_cut_keys():
    s = rc.ReconnectSettings(delta=1e-3)
    assert (s.edge_turn_refine, s.edge_turn_stop_fresh, s.edge_turn_stop) == (30.0, 90.0, 120.0)
    assert (s.midside_refine, s.midside_stop, s.jacobian_min) == (0.1, 0.2, 0.2)
    assert (s.high_order_threshold, s.quality_floor, s.inverted_element_remesh) == (0.3, 0.15, True)
    assert (s.apex_extent, s.apex_lookahead, s.rebuild_spline) == (8.0, 20, "nodes")
    for bad in ({"edge_turn_refine": 95.0}, {"edge_turn_stop": 80.0}, {"edge_turn_stop": 180.0},
                {"midside_stop": 0.3}, {"midside_refine": 0.25}, {"jacobian_min": 1.0},
                {"quality_floor": 0.0}, {"apex_lookahead": -1}, {"apex_lookahead": 2.5},
                {"rebuild_spline": "smooth"}, {"inverted_element_remesh": 1},
                {"high_order_threshold": 1.5}):
        with pytest.raises(ValueError):
            rc.ReconnectSettings.from_dict({"delta": 1e-3, **bad})
    assert rc.ReconnectSettings.from_dict({"delta": 1e-3, "rebuild_spline": "vertices"}
                                          ).rebuild_spline == "vertices"
    recorded = json.loads(json.dumps(s.to_dict()))
    for key, value in (("edge_turn_refine", 25.0), ("apex_lookahead", 10),
                       ("rebuild_spline", "vertices"), ("inverted_element_remesh", False)):
        assert not rc.ReconnectSettings(delta=1e-3, **{key: value}).continues_cut(recorded), key
    # a cut recorded before these settings existed is not continued: the run is cut again
    old = {k: v for k, v in recorded.items() if k not in ("edge_turn_refine", "jacobian_min")}
    assert not s.continues_cut(old)


def test_the_resolution_after_the_cut_is_a_cut_key_that_defaults_to_the_collapse():
    s = rc.ReconnectSettings(delta=1e-3)
    assert s.elements_per_scale is None
    assert rc.ReconnectSettings.from_dict({"delta": 1e-3, "elements_per_scale": 8}).elements_per_scale == 8
    assert rc.ReconnectSettings.from_dict({"delta": 1e-3, "elements_per_scale": None}).elements_per_scale is None
    for bad in (0, -1.0, float("inf"), True, "8"):
        with pytest.raises(ValueError):
            rc.ReconnectSettings.from_dict({"delta": 1e-3, "elements_per_scale": bad})
    recorded = json.loads(json.dumps(s.to_dict()))
    assert not rc.ReconnectSettings(delta=1e-3, elements_per_scale=8).continues_cut(recorded)
    assert s.continues_cut(recorded)
