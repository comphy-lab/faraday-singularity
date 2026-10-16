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


def test_the_tips_the_cut_made_are_meshed_at_the_cut_focus_size_whatever_their_scale():
    """The cavity's tip and the bubble's top, which the cut made, are meshed at the cut's focus
    size, focus_size cut radii: an apex radius of 1e-4 (a size of 6.25e-6 by the rule, the landing
    run's h_floor or above) does not refine them, and their regions cover the cut's focus radius
    and the jet hanging from the top (base radius about 3e-3, 0.01 deep)."""
    s = rc.ReconnectSettings(delta=1e-3)
    segs = _post_cut_segs()
    g = _regions(segs, {"delta": 1e-3, "rmin": 1e-3}, s, h_floor=6.25e-6)
    for name in rc.FRESH_TIPS:
        assert g[name]["target"] == pytest.approx(s.focus_size * 1e-3) == g[name]["floor"]
        assert g[name]["r"] == 0.0
    assert g["cavity_tip"]["z"] == segs[1]["z"][0] and g["bubble_top"]["z"] == segs[0]["z"][-1]
    assert g["cavity_tip"]["radius"] == pytest.approx(s.focus_radius * 1e-3)
    top = rc.tip_jet(rc.bubble_end(segs[0], "bubble_top"), **JET_KW)
    assert g["bubble_top"]["radius"] == pytest.approx(2 * math.hypot(top["r_j"], top["rise"]))
    assert g["bubble_top"]["radius"] > 0.02  # the jet, 0.01 long, reaches beyond 20 cut radii
    assert "bubble_bottom" not in g  # a sphere's bottom, no landing focus: no feature


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
