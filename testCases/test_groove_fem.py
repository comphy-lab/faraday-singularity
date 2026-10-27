"""Finite-element tests of the fill of a closing gas groove (``groovesurgery.groove_fill``) on
synthetic states.

* RF, a fold of the ring: the synthetic pierced state of ``ring_fem_state.py`` after its gas cut,
  put in the ring stage with an elliptic ring in the liquid whose boundary has a narrow, smooth
  finger of gas pushed out into the liquid (``test_groove_geometry.bumped_ellipse``), a solenoidal
  flow about it. The stops of the ring stage request ``ring-fold`` (``RF``) with the fill as its
  handler; refusals (no last step, the part off, a fill beyond ``max_fill``) leave the problem as it
  was; the fill rebuilds the mesh with one ring, the finger gone, the ring's volume less by the
  filled volume, the transfer as predicted (interface nodes on the closure, bulk nodes inside the
  filled region, none of the axis), the closures G1-G3 and ``G3_box`` within their limits (floored
  at the noise of the site's disc in ordinary rebuilds of the state), and the run steps after it.
* A-groove, a slot of the open chain before the ring: the synthetic necked cavity of
  ``test_reconnect_fem.py`` after its gas cut, its open chain given a narrow slot of gas on its flat
  part (the template's planned geometry, built outside the ring stage as the fill's own rebuild
  is), the stops of the events branch request ``groove`` (``A-groove``); the fill keeps both chains
  and the bridge, the slot filled, the closures within their limits, steps after.

Each problem compiles in seconds; the module takes a few minutes.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

pytest.importorskip("pyoomph")

import ring_fem_state as S  # noqa: E402
import test_groove_geometry as G  # noqa: E402
import test_reconnect_fem as fem  # noqa: E402
from test_ring_fem import problem_state  # noqa: E402

from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import groovesurgery, ring, ringsurgery, surgery  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402
from faraday_singularity.mesh import Focus, FocusRegion, RingSizing  # noqa: E402
from faraday_singularity.reconnect import ReconnectSettings  # noqa: E402

KAPPA = 0.4  # kappa_g: the trigger kappa_g delta above three of the finger's element lengths
H_FINE = 8e-4  # the mesh about the groove
FOLD = {"ratio": 4.0, "stop_elements": 3.0}
KW = {"neck_elements": 3.0, "fold_ratio": 4.0, "fold_stop": 1e-3, "fold_stop_elements": 3.0,
      "fold_search": 4.0, "apexes": None, "guard": None}
CUT_KW = {"dt_prev": 1e-5, "elements_per_scale": 8.0, "h_floor": 1e-4}
DT = 1e-5


def stream_flow(r, z, *, c, L=0.1, W=1.0):
    """The solenoidal flow of the stream function ``psi = r^2/2 W exp(-|x - c|^2/L^2)``:
    ``u_r = r W g (z - z_c)/L^2``, ``u_z = W g (1 - r (r - r_c)/L^2)``, zero radial on the axis."""
    r, z = np.asarray(r, dtype=float), np.asarray(z, dtype=float)
    g = np.exp(-((r - c[0]) ** 2 + (z - c[1]) ** 2) / L**2)
    return r * W * g * (z - c[1]) / L**2, W * g * (1.0 - r * (r - c[0]) / L**2)


def impose(pb, c):
    m = pb.get_mesh("liquid")
    fi = m.get_nodal_field_indices()
    nodes = list(m.nodes())
    ur, uz = stream_flow([n.x(0) for n in nodes], [n.x(1) for n in nodes], c=c)
    for n, a, b in zip(nodes, ur, uz):
        n.set_value(fi["velocity_x"], float(a))
        n.set_value(fi["velocity_y"], float(b))
    pb.reapply_boundary_conditions()
    pb.assign_initial_values_impulsive(DT)
    pb.timestepper.set_num_unsteady_steps_done(0)
    pb.invalidate_cached_mesh_data()


def site_noise(pb, centre, R, rebuilds=3) -> dict:
    """The noise of the disc of radius ``R`` about ``centre`` (the fill's budgets, ``surgery.
    site_grid``) in ordinary rebuilds of the state: the largest change one of ``rebuilds`` makes."""
    def budgets():
        named, _ = surgery._named_chains(pb)
        loops = [rc.Chain(ring.closed(np.column_stack([x["r"], x["z"]])), ("loop", "loop"))
                 for x in pb.interface_loops()]
        poly = surgery.box_polyline_budgets([c for _, c in named] + loops, centre[1], R,
                                            S.RECON.reservoir_depth, r_centre=centre[0])
        grid = surgery.box_integrals(pb, surgery.site_grid(centre[0], centre[1], R))
        return {"V": poly["V_liquid"], "Pz": grid["Pz"], "KE": grid["KE"]}

    out = {"V": 0.0, "Pz": 0.0, "KE": 0.0}
    before = budgets()
    for _ in range(rebuilds):
        pb.force_remesh({pb.active_mesh_template()})
        pb._domains_to_remesh.clear()
        pb.invalidate_cached_mesh_data()
        after = budgets()
        out = {k: max(out[k], abs(after[k] - before[k])) for k in out}
        before = after
    return out


def block(noise, **groove):
    sigma = {"sigma_V": 2.0 * noise["V"], "sigma_Pz": 2.0 * noise["Pz"],
             "sigma_KE": 2.0 * noise["KE"]}
    return ev.EventSettings.from_dict({"ring": {"enabled": True}, "gap": {"kappa_g": KAPPA},
                                       "groove": {"enabled": True, "max_fill": 1.0, **sigma,
                                                  **groove}})


def fold_of(delta):
    return {**FOLD, "stop": KAPPA * delta}


# --- RF: a finger of gas of the ring --------------------------------------------------------------

RING_C, RING_A, RING_B = (1.2, 0.5), 0.08, 0.05


def finger_loop():
    """The ring: an ellipse in the liquid of the pierced state with a finger of gas 0.03 long and
    about 0.004 wide pushed out at its outermost point, counter-clockwise, an even count."""
    t = np.linspace(-math.pi, math.pi, 200001)
    kw = {"c": RING_C, "a": RING_A, "b": RING_B, "A": 0.03, "sigma": 0.048}
    dense = G.bumped_ellipse(t, **kw)
    tip = G.bumped_ellipse(np.array([0.0]), **kw)[0]
    L = G.resample(dense, G.spacing(tip, fine=H_FINE, coarse=4e-3, reach=0.05))[:-1]
    return L[:-1] if len(L) % 2 else L


def ring_record(pb, L):
    return {"k_cut": 0, "t_cut_ring": pb.get_current_time(dimensional=False, as_float=True),
            "V_b": float(pb.cut["geometry"]["bubble_volume"]), "V0": rc.revolved_volume(L),
            "rho": 0.01, "r_w": 0.005, "z_mid": RING_C[1], "removed": None,
            "sizing": {"h_ring": 4e-3, "h_col": 4e-3, "grading": 0.3, "frozen": [],
                       "ring_cut": None},
            "fill": {"V": 0.0, "Pz": 0.0, "KE": 0.0}, "settings_sha256": ""}


def reload(pb, state: Path, record: dict, ring_rec: dict | None, focus: Focus):
    pb.load_state(str(state))
    surgery.restore_cut(pb, record)
    if ring_rec is not None:
        ringsurgery.restore_ring(pb, ring_rec)
    pb.active_mesh_template().focus = focus
    pb.events = []
    pb.invalidate_cached_mesh_data()


def ring_request(pb, events):
    named, _ = surgery._named_chains(pb)
    loops = [np.column_stack([x["r"], x["z"]]) for x in pb.interface_loops()]
    return ring.ring_stage_events([c for _, c in named], loops,
                                  rc.CutGeometry(**pb.cut["geometry"]), S.RECON,
                                  pb.cut["sizing"]["h_tip"], events, pb.ring, **KW)


@pytest.fixture(scope="module")
def fingered(tmp_path_factory):
    out = tmp_path_factory.mktemp("finger")
    res = {"out": out}
    with S.pierce_problem(out / "pb") as pb:
        pb.initialise()
        record = S.gas_cut_record(pb)
        tmpl = pb.active_mesh_template()
        named = dict(surgery._named_chains(pb)[0])
        L = finger_loop()
        tip = G.bumped_ellipse(np.array([0.0]), c=RING_C, a=RING_A, b=RING_B, A=0.03,
                               sigma=0.048)[0]
        focus = Focus(size=H_FINE, radius=0.2, grading=0.3, regions=(
            FocusRegion("cavity_tip", size=0.004, r=0.0, z=0.95, radius=0.1),
            FocusRegion("finger", size=H_FINE, r=float(tip[0]) - 0.015, z=float(tip[1]),
                        radius=0.05)))
        # the ring stage, entered with the ring of the test in place of the bubble
        tmpl.ring_stage = True
        tmpl.ring_sizing = RingSizing(h_ring=4e-3, grading=0.3)
        tmpl.focus = focus
        tmpl.pending_ring_geometry = {"chains": [named["open"]], "loops": [L]}
        pb._ring_transfer = {"axis_bulk": True, "reach": 0.1}
        try:
            pb.force_remesh({tmpl})
        finally:
            pb._ring_transfer = None
        pb._domains_to_remesh.clear()
        loops = pb.interface_loops()
        assert len(loops) == 1 and len(pb.interface_segments()) == 1
        pb.ring = ring_record(pb, np.column_stack([loops[0]["r"], loops[0]["z"]]))
        res["ring_rec"] = dict(pb.ring)
        impose(pb, (RING_C[0] + 0.05, RING_C[1]))
        initial = Path(surgery._keep(pb, None, out, "initial", {"ring": pb.ring})["path"])
        events0 = block({"V": 0.0, "Pz": 0.0, "KE": 0.0})
        req0 = ring_request(pb, events0)
        res["request0"] = req0
        # the noise of the disc of the fill in ordinary rebuilds of the state
        g = groovesurgery.fill_trigger(np.column_stack([loops[0]["r"], loops[0]["z"]]),
                                       req0["fold_pair"], loop=True,
                                       fold=fold_of(S.DELTA), mouth=1.5)
        res["trigger"] = g
        noise = site_noise(pb, (float(req0["fold_r"]), float(req0["fold_z"])), 4.0 * S.DELTA)
        res["noise"] = noise
        reload(pb, initial, record, res["ring_rec"], focus)
        events = block(noise)
        req = ring_request(pb, events)
        res["request"] = req
        loops_before = pb.interface_loops()
        res["V_ring_before"] = rc.revolved_volume(np.column_stack([loops_before[0]["r"],
                                                                   loops_before[0]["z"]]))
        res["volume_before"] = pb.cut_observables()["volume"]
        # the refusals, which leave the problem as it was
        refused = {}
        for name, kw, blk in (("dt", dict(CUT_KW, dt_prev=0.0), events),
                              ("disabled", CUT_KW, ev.EventSettings.from_dict(
                                  {"ring": {"enabled": True}, "gap": {"kappa_g": KAPPA}})),
                              ("volume", CUT_KW, block(noise, max_fill=1e-6))):
            before = problem_state(pb)
            with pytest.raises(surgery.CutRefused) as err:
                groovesurgery.groove_fill(pb, blk, S.RECON, out / f"refused-{name}", req, k=1,
                                          fold=fold_of(S.DELTA), **kw)
            refused[name] = {"record": err.value.record, "precut": err.value.precut,
                             "same": problem_state(pb) == before}
        res["refused"] = refused
        made = groovesurgery.groove_fill(pb, events, S.RECON, out / "fill", req, k=1,
                                         fold=fold_of(S.DELTA), log=lambda *_: None, **CUT_KW)
        res["made"] = made
        res["loops"] = pb.interface_loops()
        res["segs"] = pb.interface_segments()
        res["volume_after"] = pb.cut_observables()["volume"]
        res["after_request"] = ring_request(pb, events)
        for _ in range(3):
            fem._step(pb, DT)
        res["t_end"] = pb.get_current_time(dimensional=False, as_float=True)
        res["t_fill"] = made["t"]
        res["loops_end"] = pb.interface_loops()
    return res


def test_the_finger_of_the_ring_is_requested_and_the_refusals_change_nothing(fingered):
    req = fingered["request"]
    assert (req["kind"], req["class"], req["handler"]) == ("ring-fold", "RF", ev.GROOVE_HANDLER)
    assert req["chain"] == "ring" and req["contact"] == "fold"
    assert req["fold_gap"] < req["fold_limit"] == pytest.approx(KAPPA * S.DELTA)
    assert fingered["trigger"]["holds"]
    refused = fingered["refused"]
    for name, got in refused.items():
        assert got["same"], name
        assert got["record"]["outcome"] == "refused", name
    assert refused["dt"]["record"]["failed_at"] == "admission" and refused["dt"]["precut"] is None
    assert refused["disabled"]["record"]["failed_at"] == "admission"
    vol = refused["volume"]
    assert vol["record"]["failed_at"] == "geometry" and Path(vol["precut"]).is_file()
    assert "volume" in vol["record"]["geometry"]["failed"]


def test_the_finger_is_filled_the_ring_kept_and_its_budgets_close(fingered):
    rec = fingered["made"]
    assert (rec["kind"], rec["class"], rec["handler"], rec["outcome"]) == (
        "groove-fill", "RF", ev.GROOVE_HANDLER, "made")
    assert rec["stage"] == "ring" and rec["chain"] == "ring"
    geo = rec["geometry"]
    assert geo["side"] == "gas" and geo["mouth"] >= geo["width"]
    V_fill = geo["V_fill"]
    assert V_fill > 0.0 and rec["filled"]["V"] == V_fill
    # one open chain and one ring; the ring's volume less by the filled volume, within the
    # resampling of its quadratic surface by the rebuild
    assert len(fingered["segs"]) == 1 and len(fingered["loops"]) == 1
    L = np.column_stack([fingered["loops"][0]["r"], fingered["loops"][0]["z"]])
    V_after = rc.revolved_volume(L)
    assert V_after == pytest.approx(fingered["V_ring_before"] - V_fill,
                                    abs=0.02 * V_fill + 5 * fingered["noise"]["V"])
    # the liquid gains the filled volume
    assert fingered["volume_after"] - fingered["volume_before"] == pytest.approx(
        V_fill, rel=0.05)
    tr = rec["transfer"]
    assert abs(tr["reported_unlocated"] - tr["predicted_unlocated"]) <= surgery.UNLOCATED_SLACK
    assert tr["predicted_off_closure"] == 0 and tr["closure_nodes"] > 0
    gas = tr["former_gas"]
    assert gas["predicted_bulk"] > 0 and gas["axis"] == 0
    assert abs(gas["bulk"] - gas["predicted_bulk"]) <= max(0.1 * gas["predicted_bulk"], 5)
    assert tr["no_value"] == {} and tr["corners"]["new"] == 0
    c = rec["closures"]
    assert c["hard"] == [] and set(c["limits"]) == {f"G{i}" for i in range(1, 7)} | {"G3_box"}
    for x in ("G1", "G2", "G3", "G3_box"):
        assert c[x] <= c["limits"][x], x
    assert c["G3_projected"] and math.isfinite(c["G3_phi"]["phi_relative_error"])
    # the box's integrals weighted by its cutoff (2026-10-25), the unweighted G2 reported beside
    # the closures and never among them; the weighted G3_box is the identity's left side
    assert rec["budgets"]["box"]["weight"] == "cutoff"
    grid = rec["budgets"]["grid"]
    assert all(grid[x]["weight"] == "cutoff" and "unweighted" in grid[x] for x in grid)
    u = c["G2_unweighted"]
    assert u["limit"] == c["limits"]["G2"] and u["weighted"] == c["G2"] and u["value"] >= 0.0
    assert "G2_unweighted" not in c["flags"] + c["hard"]
    assert c["G3_box"] == pytest.approx(abs(c["G3_phi"]["phi_lhs"]), rel=1e-6,
                                        abs=1e-9 * c["scales"]["P_ref"])
    assert c["scales"]["U"] > 0.0 and c["scales"]["P_ref"] > 0.0
    assert rec["gas_removed"]["V"] == V_fill
    # the fold the fill met is gone
    after = fingered["after_request"]
    assert after is None or after.get("class") != "RF" or after["fold_gap"] >= after["fold_limit"]
    # the steps after the fill
    assert fingered["t_end"] == pytest.approx(fingered["t_fill"] + 3 * DT, rel=1e-9)
    assert len(fingered["loops_end"]) == 1


# --- A-groove: a slot of gas of the open chain before the ring ------------------------------------

SLOT_R, SLOT_W, SLOT_A, SLOT_RF = 0.6, 0.004, 0.03, 0.006


def slotted(open_chain):
    """The open chain of the necked cavity after its cut with a slot of gas at ``SLOT_R``: walls
    vertical ``SLOT_W`` apart and ``SLOT_A`` deep, a half-circle bottom, fillets ``SLOT_RF`` at its
    mouth; the chain's own nodes away from it. Returns the chain and the height of the slot's
    bottom."""
    P = np.asarray(open_chain.points, dtype=float)
    z_of = lambda r: float(np.interp(r, P[:, 0], P[:, 1]))  # noqa: E731
    x0, x1 = SLOT_R - 0.5 * SLOT_W, SLOT_R + 0.5 * SLOT_W
    lo, hi = x0 - SLOT_RF - 0.05, x1 + SLOT_RF + 0.05
    z0, z1 = z_of(x0 - SLOT_RF), z_of(x1 + SLOT_RF)
    zb = 0.5 * (z0 + z1) - SLOT_A
    left = np.linspace(lo, x0 - SLOT_RF, 4001)
    right = np.linspace(x1 + SLOT_RF, hi, 4001)
    parts = [np.column_stack([left, [z_of(r) for r in left]]),
             G._arc_points((x0 - SLOT_RF, z0 - SLOT_RF), SLOT_RF, 0.5 * math.pi, 0.0),
             np.column_stack([np.full(2001, x0), np.linspace(z0 - SLOT_RF, zb, 2001)]),
             G._arc_points((SLOT_R, zb), 0.5 * SLOT_W, math.pi, 2.0 * math.pi),
             np.column_stack([np.full(2001, x1), np.linspace(zb, z1 - SLOT_RF, 2001)]),
             G._arc_points((x1 + SLOT_RF, z1 - SLOT_RF), SLOT_RF, math.pi, 0.5 * math.pi),
             np.column_stack([right, [z_of(r) for r in right]])]
    inner = G.resample(np.vstack(parts), G.spacing((SLOT_R, zb), fine=H_FINE, coarse=0.004,
                                                   reach=0.04))
    keep_lo, keep_hi = P[:, 0] < lo - 0.004, P[:, 0] > hi + 0.004
    Q = np.vstack([P[keep_lo], inner, P[keep_hi]])
    return rc.Chain(Q, tuple(open_chain.end_types)), zb - 0.5 * SLOT_W


@pytest.fixture(scope="module")
def slot(tmp_path_factory):
    out = tmp_path_factory.mktemp("slot")
    res = {"out": out}
    s = ReconnectSettings(**fem.FAST_CUT)
    with fem.neck_problem(out / "pb", s) as pb:
        pb.initialise()
        fem.impose_closing_flow(pb)
        surgery.cut(pb, s, out / "cut", log=lambda *_: None)
        tmpl = pb.active_mesh_template()
        named = dict(surgery._named_chains(pb)[0])
        new, zb = slotted(named["open"])
        old_focus = tmpl.focus
        regions = tuple(old_focus.regions) if old_focus is not None and old_focus.regions else ()
        tmpl.focus = Focus(size=H_FINE, radius=0.2, grading=0.3, regions=regions + (
            FocusRegion("slot", size=H_FINE, r=SLOT_R, z=zb + 0.5 * SLOT_A, radius=0.05),))
        # the slot through the template's planned geometry, outside the ring stage
        tmpl.pending_ring_geometry = {"chains": [named["bubble"], new], "loops": []}
        pb._ring_transfer = {"axis_bulk": False, "reach": 0.0}
        try:
            pb.force_remesh({tmpl})
        finally:
            pb._ring_transfer = None
        pb._domains_to_remesh.clear()
        assert tmpl.ring_stage is False and tmpl.pending_ring_geometry is None
        segs = pb.interface_segments()
        res["segs_before"] = [(c["open"], tuple(c["end_types"])) for c in segs]
        impose(pb, (SLOT_R, zb + 0.01))
        named2, _ = surgery._named_chains(pb)
        request = rc.post_cut_stops([c for _, c in named2], rc.CutGeometry(**pb.cut["geometry"]),
                                    s, pb.cut["sizing"]["h_tip"],
                                    events=block({"V": 0.0, "Pz": 0.0, "KE": 0.0}),
                                    fold_stop=1e-3)
        res["request"] = request
        noise = site_noise(pb, (float(request["fold_r"]), float(request["fold_z"])),
                           4.0 * s.delta)
        res["noise"] = noise
        # the state the noise left: the request read again on it
        named3, _ = surgery._named_chains(pb)
        request = rc.post_cut_stops([c for _, c in named3], rc.CutGeometry(**pb.cut["geometry"]),
                                    s, pb.cut["sizing"]["h_tip"], events=block(noise),
                                    fold_stop=1e-3)
        res["request_after_noise"] = request
        gas_before = rc.revolved_volume(rc.closed_section(dict(named3)["open"].points,
                                                          ("axis", "fixed"), s.reservoir_depth))
        made = groovesurgery.groove_fill(pb, block(noise), s, out / "fill", request, k=1,
                                         fold=fold_of(s.delta), log=lambda *_: None, **CUT_KW)
        res["made"] = made
        named4, _ = surgery._named_chains(pb)
        res["gas_change"] = rc.revolved_volume(rc.closed_section(
            dict(named4)["open"].points, ("axis", "fixed"), s.reservoir_depth)) - gas_before
        res["segs_after"] = [(c["open"], tuple(c["end_types"])) for c in pb.interface_segments()]
        res["spans"] = surgery._axis_spans_of_mesh(pb)
        for _ in range(3):
            fem._step(pb, DT)
        res["steps_t"] = pb.get_current_time(dimensional=False, as_float=True) - made["t"]
    return res


def test_the_slot_of_the_open_chain_is_requested_as_a_groove(slot):
    req = slot["request"]
    assert (req["kind"], req["class"], req["handler"]) == ("groove", "A-groove",
                                                           ev.GROOVE_HANDLER)
    assert req["chain"] == "open" and req["contact"] == "fold"
    assert abs(req["fold_r"] - SLOT_R) < 2 * SLOT_W
    assert slot["segs_before"] == [(False, ("axis", "axis")), (True, ("axis", "fixed"))] or \
        sorted(slot["segs_before"]) == sorted([(False, ("axis", "axis")),
                                               (True, ("axis", "fixed"))])


def test_the_slot_is_filled_both_chains_kept_and_its_budgets_close(slot):
    rec = slot["made"]
    assert (rec["kind"], rec["class"], rec["outcome"]) == ("groove-fill", "A-groove", "made")
    assert "stage" not in rec and rec["chain"] == "open"
    V_fill = rec["geometry"]["V_fill"]
    assert V_fill > 0.0
    assert slot["gas_change"] == pytest.approx(-V_fill, abs=0.02 * V_fill + 5 * slot["noise"]["V"])
    assert sorted(slot["segs_after"]) == sorted(slot["segs_before"])
    assert len(slot["spans"]) == 2  # below the bubble, and the bridge
    tr = rec["transfer"]
    assert abs(tr["reported_unlocated"] - tr["predicted_unlocated"]) <= surgery.UNLOCATED_SLACK
    assert tr["predicted_off_closure"] == 0 and tr["closure_nodes"] > 0
    gas = tr["former_gas"]
    assert gas["predicted_bulk"] > 0 and gas["axis"] == 0
    c = rec["closures"]
    assert c["hard"] == []
    for x in ("G1", "G2", "G3", "G3_box"):
        assert c[x] <= c["limits"][x], x
    assert slot["steps_t"] == pytest.approx(3 * DT, rel=1e-6)
