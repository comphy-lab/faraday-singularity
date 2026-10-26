"""Local ring contacts: phase, route, hard balances and rollback (synthetic states)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "testCases")]

from faraday_singularity import ringcontact as contact
from faraday_singularity.events import EventSettings
from faraday_singularity.driver import CollapseDriver
from faraday_singularity.ring import loop_spline_contact
from test_floor_fill import _fake_driver, teardrop
from test_groove_geometry import finger_ring


def wedge_loop():
    """A synthetic sharp gas cusp: simple node polygon, crossed rebuild spline."""
    return np.array([(4, 2), (3, 1), (2, .8), (1, .3), (0, 0), (1.2, .35),
                     (2.6, -.04), (4, -1), (7, -3), (10, 0), (9, 2), (7, 3)]) * 1e-6 + [1e-3, 4.6]


def spline_stop(P):
    hit = loop_spline_contact(P)
    return dict(reason="self-contact-approach", trigger="spline-contact", chain="ring",
                contact=hit["what"], crossings=hit["count"], nodes=list(hit["nodes"]))


def one_interval_loop():
    return np.array([(1.05, .4), (.65, .25), (.3, .12), (0, 0), (.003, -.017), (.26, -.18),
                  (.65, -.3), (1.1, -.4), (4, -1.5), (6, 0), (4, 1.5), (2, .8)]) * 1e-6 + [1e-3, 4.6]


def test_a_crossing_within_one_cubic_interval_reads_its_four_control_nodes(monkeypatch):
    from faraday_singularity import groove

    P = one_interval_loop()
    for roll in (0, 1):
        L = np.roll(P, roll, axis=0)
        stop = spline_stop(L)
        assert stop["nodes"][:2] == stop["nodes"][2:]
        w = contact.spline_wedge(L, stop, stop_elements=3)
        assert w is not None
        req = CollapseDriver._floor_fill_request(_fake_driver(L), stop)
        assert req["spline_fill"]["wedge"] == w and req["side"] == "gas"
    monkeypatch.setattr(groove, "gas_share", lambda *a: 0.0)
    assert contact.spline_wedge(P, spline_stop(P), stop_elements=3) is None


def test_the_split_is_the_even_vertex_when_the_half_count_is_odd(monkeypatch):
    from faraday_singularity import ring

    P = one_interval_loop()
    P = np.insert(P, 9, [(P[8]*2+P[9])/3, (P[8]+P[9]*2)/3], axis=0)
    P = np.roll(P, 4, axis=0)
    assert len(P) == 14 and len(ring.loop_halves(P)[0])-1 == 6
    stop = spline_stop(P)
    assert stop["nodes"] == [7, 8, 7, 8]  # ordinary interval, not the split at 6
    assert contact.spline_wedge(P, stop, stop_elements=3) is not None
    # A stop on the last interval of the first half must not use P[7] as its
    # fourth control: that interval has a phantom control instead.
    fake = dict(loop_spline_contact(P), nodes=(5, 6, 5, 6))
    monkeypatch.setattr(ring, "loop_spline_contact", lambda p: fake)
    assert contact.spline_wedge(P, dict(stop, nodes=[5, 6, 5, 6]), stop_elements=3) is None


def test_a_local_spline_crossing_is_certified_as_gas_before_it_is_filled(monkeypatch):
    for roll in (0, 2, -4):  # seam on a vertex, including a crossing through the seam
        P = np.roll(wedge_loop(), roll, axis=0)
        stop = spline_stop(P)
        w = contact.spline_wedge(P, stop, stop_elements=3)
        assert w is not None and w["gap"] < w["limit"]
        req = CollapseDriver._floor_fill_request(_fake_driver(P), stop)
        assert req["handler"] == "groove-fill" and req["contact"] == "spline"
        assert req["side"] == "gas" and "floor" not in req
        assert req["spline_fill"]["guard"] == stop and req["spline_fill"]["wedge"] == w
        json.dumps(req)
        for extra in ({"contact": "polygon"}, {"crossings": 2}, {"nodes": [0, 1, 8, 9]},
                      {"chain": "open"}, {"reason": "interface-unresolved"}):
            assert contact.spline_wedge(P, dict(stop, **extra), stop_elements=3) is None
        for kw in ({"removed": True}, {"ring": False}, {"enabled": False},
                   {"jet": None}, {"jet": "grow"}, {"loops": 2}):
            assert CollapseDriver._floor_fill_request(_fake_driver(P, **kw), stop) is None
    # Four node intervals can touch three quadratic elements when the first
    # interval starts at a midside node. That is outside the local handler.
    from faraday_singularity import ring
    P = wedge_loop()
    hit = dict(loop_spline_contact(P), nodes=(1, 2, 4, 5))
    stop = dict(spline_stop(P), nodes=list(hit["nodes"]))
    monkeypatch.setattr(ring, "loop_spline_contact", lambda p: hit)
    assert contact.spline_wedge(P, stop, stop_elements=3) is None


def test_the_spline_route_passes_the_wedges_limits_and_checks_the_base(tmp_path, monkeypatch):
    from faraday_singularity import groovesurgery
    from test_bookkeeping import CutProblem
    from test_events_plumbing import _events
    from test_groove_driver import GROOVE_ON, _run, fake_fill

    P = wedge_loop()
    req = CollapseDriver._floor_fill_request(_fake_driver(P), spline_stop(P))
    calls, checks = [], []
    inner = fake_fill(calls)

    def fill(*a, base_check=None, **kw):
        checks.append(base_check)
        return inner(*a, **kw)

    drv = _run(tmp_path, CutProblem(), GROOVE_ON, req, monkeypatch, calls)
    monkeypatch.setattr(groovesurgery, "groove_fill", fill)
    result = drv.run(1e-9)
    drv.close()
    assert result["stop"]["reason"] == "max-steps" and result["events"]["spline_fills"] == 1
    assert "floor_fills" not in result["events"] and len(calls) == 1
    assert calls[0]["fold"] == req["spline_fill"]["fold"] and callable(checks[0])
    assert checks[0]()["ok"]
    log = [x for x in _events(tmp_path) if x["event"] == "spline-fill"]
    assert len(log) == 1 and log[0]["guard"]["nodes"] == req["spline_fill"]["guard"]["nodes"]


def liquid_stop(P):
    from faraday_singularity.groove import side_folds
    f = side_folds(P, loop=True, ratio=4, stop=1e-4, stop_elements=3, search_factor=4)["liquid"]["fold"]
    return dict(reason="self-contact-approach", trigger="fold-gap", chain="ring", contact="kink",
                side="liquid", fold_pair=list(f.pair), fold_gap=f.gap, fold_arc=f.arc,
                fold_h_elem=f.h_elem, fold_r=f.r, fold_z=f.z)


def liquid_driver(P, **kw):
    fake = _fake_driver(P, **kw)
    fake.events = EventSettings.from_dict({"ring": {"enabled": True, "sigma_V": 1e-6, "sigma_Pz": 1e-6},
                                          "groove": {"enabled": True, "sheets": "log"}})
    return fake


def test_a_liquid_kink_asks_only_for_refinement_and_keeps_a_resolved_contact_stopped():
    P = finger_ring(A=-3e-4)
    stop = liquid_stop(P)
    read = contact.liquid_reading(P, stop)
    assert read["trigger"] == "liquid-kink" and read["L"] == pytest.approx(stop["fold_h_elem"])
    assert read["liquid_contact"] == stop
    fake = liquid_driver(P)
    assert CollapseDriver._ring_refine_due(fake, stop)
    assert CollapseDriver._floor_fill_request(fake, stop) is None
    for extra in ({"contact": "fold"}, {"side": "gas"}, {"chain": "open"},
                  {"fold_pair": []}, {"side": None}, {"trigger": "spline-contact"}):
        other = dict(stop, **extra)
        assert contact.liquid_reading(P, other) == other
    assert not CollapseDriver._ring_refine_due(liquid_driver(P, h_floor=1), stop)


def test_uncalibrated_noise_keeps_the_original_liquid_stop_with_an_explanation():
    P = finger_ring(A=-3e-4)
    for V, Pz in ((0, 1e-6), (1e-6, 0), (0, 0)):
        fake = liquid_driver(P)
        fake.events = EventSettings.from_dict({"ring": {"enabled": True, "sigma_V": V, "sigma_Pz": Pz}})
        stop = liquid_stop(P)
        assert not CollapseDriver._ring_refine_due(fake, stop)
        assert stop["reason"] == "self-contact-approach" and stop["side"] == "liquid"
        assert "not calibrated" in stop["liquid_refine_unavailable"]


def test_a_real_liquid_kink_reading_dispatches_to_the_audited_handler():
    P = finger_ring(A=-3e-4)
    fake = liquid_driver(P)
    stop = liquid_stop(P)
    fake._ring_refine_due = lambda st: CollapseDriver._ring_refine_due(fake, st)
    fake._ring_refine_once = lambda *args: pytest.fail("the audit must wrap this route")
    calls = []

    def audited(m, reading, attempt):
        calls.append((reading, attempt))
        return {"refined": True}, None

    fake._ring_liquid_refine = audited
    m, found = CollapseDriver._ring_refine(fake, {}, stop)
    assert m == {"refined": True} and found is None
    assert len(calls) == 1 and calls[0][0]["liquid_contact"] == stop


def test_the_run_loop_routes_a_ring_stage_liquid_stop_to_the_audit(tmp_path, monkeypatch):
    from test_events_plumbing import _events_driver
    from test_ring_driver import RingProblem, made_ring_cut
    from faraday_singularity import ring
    from faraday_singularity.reconnect import ReconnectSettings

    P = finger_ring(A=-3e-4)
    class SheetProblem(RingProblem):
        def interface_loops(self):
            if not self.ringed:
                return []
            return [{"r": P[:, 0], "z": P[:, 1], "u_r": np.zeros(len(P)),
                     "u_z": np.zeros(len(P)), "vertex": np.arange(len(P)) % 2 == 0}]

    pb = SheetProblem()
    block = liquid_driver(P).events.to_dict()
    block["jetcore"] = {"enabled": True, "far_kinks": "recentre"}
    block["post"] = {"h_floor": 1e-7}
    drv = _events_driver(tmp_path, pb, ReconnectSettings(delta=1e-3, dt0=1e-6),
                         events=EventSettings.from_dict(block), max_steps=0)
    made_ring_cut(drv, pb)
    calls, stops = [], []
    raw = liquid_stop(P)
    assert drv._ring_refine_due(raw)

    def stage(*args, **kw):
        stops.append(True)
        return dict(raw) if not calls else None

    def audit(m, reading, attempt):
        calls.append(reading)
        pb.remesh_count += 1
        drv._post_cut_cache = drv._stop_cache = None
        return drv.measure(), None

    monkeypatch.setattr(ring, "ring_stage_events", stage)
    monkeypatch.setattr(drv, "_ring_liquid_refine", audit)
    result = drv.run(1e-9)
    drv.close()
    assert result["stop"]["reason"] == "max-steps" and len(calls) == 1 and stops
    assert calls[0]["liquid_contact"]["side"] == "liquid"


def budget(v=1.0, p=1.0, loops=1):
    return {"poly": {"V_liquid": v}, "box": {"Pz": p}, "loops": loops}


def test_zero_removal_balances_are_hard_and_nonfinite_values_fail():
    sigma = SimpleNamespace(sigma_V=1e-6, sigma_Pz=1e-5)
    before = budget()
    good = contact.closures(before, budget(1 + 1e-6, 1 + 1e-5), sigma)
    assert good["hard"] == [] and good["V_added"] == good["V_removed"] == 0
    assert contact.closures(before, budget(1.01, 1), sigma)["hard"] == ["C_V"]
    assert contact.closures(before, budget(1, 1.01), sigma)["hard"] == ["C_P"]
    assert contact.closures(before, budget(loops=2), sigma)["hard"] == ["C_loop"]
    assert contact.closures(before, budget(float("nan"), float("inf")), sigma)["hard"] == ["C_V", "C_P"]


def _box_sides(grid, *, gas_of, found_of, u_of):
    """A sample of ``ringcontact.sample`` on a synthetic field: the polylines' gas ``gas_of``,
    the finite-element liquid ``found_of(r, z)`` and the axial velocity ``u_of(r, z)``."""
    w = grid["dV"] * grid["phi"]

    def sampler(r, z):
        found = found_of(np.asarray(r), np.asarray(z))
        return np.where(found, u_of(np.asarray(r), np.asarray(z)), 0.0), found

    cells = contact.cell_readings(grid, gas_of, sampler)
    return {"poly": {"V_liquid": 1e-6}, "loops": 1,
            "box": {"Pz": float(np.sum(w * cells["uz"]))}, "_cells": cells}


def test_a_cell_moved_across_a_free_surface_counts_its_liquid_part_not_its_whole_volume():
    """The lip refine's C_P (``ringcontact.momentum_change``): r1 at step 7817 and r1s4 at 8208
    (bf80e74) stopped on a refine whose box grid moved one heavy cell's midpoint across the free
    surface of the plug, its liquid fraction 0.502 to 0.502 and 0.498 to 0.498: its whole cell
    counted by the midpoint rule, 3.9e-9 and 3.6e-9 against 1.62e-9, the cells in the liquid both
    times 1.8e-10 and -4.1e-11. Read consistently, a crossed cell counts the change of its liquid
    part's momentum (its fraction from each side's polylines, its velocity at its liquid part's
    centroid on each side): the artefact goes, and a real change of velocity, in the liquid or in
    the crossed cell itself, stays hard; a crossed cell that a side cannot read so counts the
    midpoint rule's change; a cell that a thin film crosses twice reads its fraction."""
    from shapely.geometry import box as rect

    from faraday_singularity.surgery import site_grid

    grid = site_grid(1.4e-3, 4.612, 4e-3)
    w = grid["dV"] * grid["phi"]
    k = int(np.argmax(w))
    zc = float(grid["zz"][k])
    sigma = SimpleNamespace(sigma_V=1.13e-13, sigma_Pz=3.24e-10)
    limit = 5.0 * sigma.sigma_Pz
    cell = contact._cells(grid, [k])[0]

    def plane(eps, du=0.0, du_cell=0.0):  # liquid below the plane z = zc + eps, 377 + du in it
        import shapely

        def u_of(r, z):
            inside = shapely.contains_xy(cell, r, z) if du_cell else np.zeros(len(r), bool)
            return 377.0 + du + du_cell * inside
        return _box_sides(grid, gas_of=rect(0.0, zc + eps, 1.0, 10.0),
                          found_of=lambda r, z: z < zc + eps, u_of=u_of)

    before, after = plane(1e-9), plane(-1e-9)
    assert before["_cells"]["found"][k] and not after["_cells"]["found"][k]
    c = contact.closures(before, after, sigma, grid)
    split = c["C_P_split"]
    big = split["largest_crossed"]
    assert split["cells_crossed"] >= 1 and split["cells_unresolved"] == 0
    assert big["resolved"] and 0.4 < big["fraction"][0] < 0.6 and 0.4 < big["fraction"][1] < 0.6
    assert c["C_P_grid"] > limit  # the midpoint rule's whole cell
    assert c["values"]["C_P"] < 1e-3 * c["C_P_grid"] and c["hard"] == []
    assert c["values"]["C_P"] == pytest.approx(abs(split["matched"] + split["crossed"]))
    # a change of the velocity in the liquid: hard, as before
    moved = contact.closures(before, plane(-1e-9, du=0.05), sigma, grid)
    assert moved["hard"] == ["C_P"] and moved["C_P_split"]["matched"] > limit
    # a real change of the velocity in the crossed cell itself: counted there, hard
    du = 3.0 * limit / (0.5 * w[k])
    local = contact.closures(before, plane(-1e-9, du_cell=du), sigma, grid)
    lk = local["C_P_split"]["largest_crossed"]
    assert local["hard"] == ["C_P"]
    assert lk["value"] == pytest.approx(w[k] * lk["fraction"][1] * du, rel=1e-3,
                                        abs=1e-3 * limit)
    # the side that holds the liquid's sample is read: the reverse crossing, the same reading
    back = contact.closures(after, before, sigma, grid)
    assert back["hard"] == [] and back["values"]["C_P"] == pytest.approx(c["values"]["C_P"])
    # a crossing the polylines do not see (the finite-element surface alone moved across the
    # midpoint): unread, the midpoint rule's change counted, hard
    unseen = _box_sides(grid, gas_of=rect(0.0, zc + 1.0, 1.0, 10.0),
                        found_of=lambda r, z: ~((np.abs(r - grid["r"][k]) < 1e-12)
                                                & (np.abs(z - zc) < 1e-12)),
                        u_of=lambda r, z: np.full(len(r), 377.0))
    clear = _box_sides(grid, gas_of=rect(0.0, zc + 1.0, 1.0, 10.0),
                       found_of=lambda r, z: np.ones(len(r), bool),
                       u_of=lambda r, z: np.full(len(r), 377.0))
    miss = contact.closures(clear, unseen, sigma, grid)
    assert miss["C_P_split"]["cells_unresolved"] == 1 and miss["hard"] == ["C_P"]
    assert miss["values"]["C_P"] == pytest.approx(miss["C_P_grid"])
    # a film of gas across the cell (two crossings): its fraction read, the artefact gone
    film_w = 0.2 * grid["R"] / grid["n"]

    def film(z0):
        return _box_sides(grid, gas_of=rect(0.0, z0, 1.0, z0 + film_w),
                          found_of=lambda r, z: ~((z > z0) & (z < z0 + film_w)),
                          u_of=lambda r, z: np.full(len(r), 377.0))

    a, b = film(zc + 1e-9), film(zc - 1e-9)  # the midpoint below the film, then within it
    two = contact.closures(a, b, sigma, grid)
    tk = two["C_P_split"]["largest_crossed"]
    assert a["_cells"]["found"][k] and not b["_cells"]["found"][k] and tk["resolved"]
    assert 0.5 < tk["fraction"][0] < 1.0 and 0.5 < tk["fraction"][1] < 1.0
    assert two["hard"] == [] and two["C_P_grid"] > limit
    # samples without their cells' readings: the midpoint rule's change, as before
    plain = contact.closures(contact.plain_sample(before), contact.plain_sample(after), sigma,
                             grid)
    assert plain["values"]["C_P"] == pytest.approx(c["C_P_grid"]) and plain["hard"] == ["C_P"]
    assert "_cells" not in contact.plain_sample(before)


def crossed_teardrop():
    P, _ = teardrop(arc_points=121)
    Q = P.copy()
    h, theta = 5e-7, np.deg2rad(25)
    Q[0] = P[0]+[1.6*h*np.cos(theta), 2*h*np.sin(theta)]
    return P, Q


def test_a_crossed_candidate_is_read_on_its_simple_pre_step_gas_pocket(monkeypatch):
    from faraday_singularity import groove
    P, Q = crossed_teardrop()
    reading = contact.step_contact(Q)
    assert reading is not None and reading["crossings"] == 1
    wedge, phase = contact.step_wedge(P, reading, stop_elements=3)
    assert phase == "gas" and wedge is not None
    assert contact.step_wedge(Q, reading, stop_elements=3) == (None, "unreadable")
    monkeypatch.setattr(groove, "gas_share", lambda *a: 0.0)
    assert contact.step_wedge(P, reading, stop_elements=3) == (None, "liquid")


def test_the_step_check_precedes_acceptance_and_does_not_map_nodes_across_a_rebuild():
    P, Q = crossed_teardrop()
    loop = {"r": Q[:, 0], "z": Q[:, 1], "node_ids": np.arange(len(Q))}
    fake = SimpleNamespace(pb=SimpleNamespace(remesh_count=3, interface_loops=lambda **kw: [loop]),
                           state=SimpleNamespace(step=7, t_prev=1.0), _t=lambda: 2.0)
    before = {"P": P, "node_ids": np.arange(len(P)), "remesh_count": 3}
    with pytest.raises(contact.RingStepContact) as exc:
        CollapseDriver._ring_step_check(fake, before)
    assert exc.value.reading["attempted_step"] == 8 and fake.state.step == 7
    fake.pb.remesh_count += 1
    assert CollapseDriver._ring_step_check(fake, before) is None


@pytest.mark.parametrize("liquid", [False, True])
def test_a_crossed_step_is_restored_before_the_local_fill_and_unknown_phases_stay_stopped(monkeypatch, liquid):
    from faraday_singularity import groove
    P, Q = crossed_teardrop()
    reading = contact.step_contact(Q)
    reading.update(before_local=P[reading["local_indices"]].tolist(), attempted_time=2.0, attempted_step=8,
                   node_ids=list(range(len(P))), local_node_ids=reading["local_indices"])
    fake = liquid_driver(P)
    restored, calls = [], []
    fake._restore_step_snapshot = lambda: restored.append(True)
    fake.measure = lambda: {"prestate": True, "time": 1.0}
    fake._distortion_flags = lambda: []
    fake.pb = SimpleNamespace(interface_loops=lambda **kw: [{
        "r": P[:, 0], "z": P[:, 1], "node_ids": np.arange(len(P))}])
    fake._event = lambda *args, **kw: None
    fake._meet_event = lambda m, req: (calls.append(req) or m, None)
    if liquid:
        monkeypatch.setattr(groove, "gas_share", lambda *a: 0.0)
    _, stop = CollapseDriver._meet_ring_step_contact(fake, {}, contact.RingStepContact(reading))
    assert restored == [True]
    if liquid:
        assert not calls and stop["trigger"] == "stepped-contact-unhandled"
    else:
        assert stop is None and len(calls) == 1 and calls[0]["step_fill"]["guard"]["pre_step_phase"] == "gas"


def test_a_crossed_cell_near_the_axis_counts_its_revolved_liquid_momentum():
    """A crossed cell's liquid part is weighted by its volume of revolution, not its area (the
    review of batch A, F1): a cell near the axis that a plane r = r_k crosses holds less volume on
    its inner side than its area says. Its counted momentum change, the velocity changed in it
    too, is the one a fine midpoint integration of 2 pi r u over the cell's liquid part gives."""
    import shapely
    from shapely.geometry import box as rect

    from faraday_singularity.surgery import site_grid

    grid = site_grid(1.4e-3, 4.612, 4e-3)
    w = grid["dV"] * grid["phi"]
    # the cell nearest the axis, wholly at r > 0, whose radial extent is largest against its
    # distance from it: its inner half holds much less volume than its outer half
    cand = np.flatnonzero((grid["r"] < 2e-4) & (grid["rho"] > 1.5e-3))
    polys = contact._cells(grid, cand)
    x0 = np.array([p.bounds[0] for p in polys])
    span = np.array([p.bounds[2] - p.bounds[0] for p in polys])
    ok = np.flatnonzero(x0 > 0.0)
    j = ok[np.argmax(span[ok] / x0[ok])]
    k, cell = int(cand[j]), polys[j]
    rk = float(grid["r"][k])
    sigma = SimpleNamespace(sigma_V=1.13e-13, sigma_Pz=3.24e-10)

    def u_of(r, z, du):
        return 100.0 + 1e5 * r + du * shapely.contains_xy(cell, r, z)

    def side(eps, du):  # the liquid inside r < r_k + eps, the gas beyond
        return _box_sides(grid, gas_of=rect(rk + eps, 0.0, 1.0, 10.0),
                          found_of=lambda r, z: r < rk + eps, u_of=lambda r, z: u_of(r, z, du))

    def fine(eps, du, n=600):  # 2 pi int r u dA over the cell's liquid part, and its volume
        x0, y0, x1, y1 = cell.bounds
        hx, hy = (x1 - x0) / n, (y1 - y0) / n
        X, Y = np.meshgrid(x0 + (np.arange(n) + 0.5) * hx, y0 + (np.arange(n) + 0.5) * hy)
        X, Y = X.ravel(), Y.ravel()
        inside = shapely.contains_xy(cell, X, Y)
        liquid = inside & (X < rk + eps)
        dV = 2.0 * math.pi * X * hx * hy
        return (float(np.sum(dV[liquid] * u_of(X[liquid], Y[liquid], du))),
                float(np.sum(dV[liquid])), float(np.sum(dV[inside])))

    du = 50.0
    before, after = side(1e-9, 0.0), side(-1e-9, du)
    assert before["_cells"]["found"][k] and not after["_cells"]["found"][k]
    c = contact.closures(before, after, sigma, grid)
    lk = c["C_P_split"]["largest_crossed"]
    assert c["C_P_split"]["cells_crossed"] >= 1 and lk["r"] == pytest.approx(rk, abs=1e-15)
    (Pb, Vb, Vc), (Pa, Va, _) = fine(1e-9, 0.0), fine(-1e-9, du)
    expected = w[k] / Vc * (Pa - Pb)
    assert lk["value"] == pytest.approx(expected, rel=2e-3)
    planar = shapely.difference(cell, rect(rk, 0.0, 1.0, 10.0)).area / cell.area
    assert lk["fraction"][0] == pytest.approx(Vb / Vc, rel=5e-3)  # the lattice's own error
    assert abs(lk["fraction"][0] - planar) > 0.05  # the area fraction would miscount it


def _two_crossed(after_P=(0.0, 0.0)):
    """Box cells of two crossed cells (in the liquid before, in the gas after) for
    ``momentum_change``, the second side's liquid momenta ``after_P``."""
    grid = {"dV": np.ones(2), "phi": np.ones(2), "r": np.array([1.0, 2.0]),
            "zz": np.zeros(2), "rho": np.ones(2)}

    def cells(found, P):
        return {"found": np.asarray(found), "uz": np.where(found, 1.0, 0.0),
                "near": np.arange(2), "V_cell": np.ones(2), "V_liquid": np.full(2, 0.5),
                "P_liquid": np.asarray(P, dtype=float), "read_liquid": np.ones(2, bool)}
    return grid, cells([True, True], [0.5, 0.5]), cells([False, False], after_P)


def test_a_crossed_midpoint_whose_velocity_is_not_finite_fails_the_momentum_balance():
    """A crossed cell whose midpoint enters the liquid with a velocity that is not finite, its
    liquid part's moments finite (the review of batch A, F4): the reading never replaces a change
    that is not finite; C_P fails."""
    grid, b, a = _two_crossed()
    a = dict(a, found=np.array([True, False]), uz=np.array([math.nan, 0.0]))  # gas -> liquid
    b = dict(b, found=np.array([False, True]), uz=np.array([0.0, 1.0]))
    sigma = SimpleNamespace(sigma_V=1e-6, sigma_Pz=1e-5)
    before = dict(budget(p=1.0), _cells=b)
    after = dict(budget(p=math.nan), _cells=a)  # the midpoint rule's sum holds the NaN
    split = contact.momentum_change(grid, before, after)
    assert math.isnan(split["change"])
    c = contact.closures(before, after, sigma, grid)
    assert c["hard"] == ["C_P"] and math.isnan(c["values"]["C_P"])
    # a finite midpoint rule's change but a NaN sample in the liquid: the reading fails too
    after_f = dict(budget(p=1.0), _cells=a)
    assert contact.closures(before, after_f, sigma, grid)["hard"] == ["C_P"]


def test_opposite_infinite_crossed_cells_fail_the_momentum_balance():
    grid, b, a = _two_crossed((math.inf, -math.inf))
    sigma = SimpleNamespace(sigma_V=1e-6, sigma_Pz=1e-5)
    before = dict(budget(), _cells=b)
    after = dict(budget(), _cells=a)
    split = contact.momentum_change(grid, before, after)
    assert math.isnan(split["change"]) and math.isnan(split["crossed"])
    c = contact.closures(before, after, sigma, grid)
    assert c["hard"] == ["C_P"] and math.isnan(c["values"]["C_P"])


@pytest.mark.parametrize("miss", [None, "volume", "momentum", "base", "nonfinite",
                                  "nan-midpoint"])
@pytest.mark.parametrize("fresh", [False, True])
def test_refinement_is_accepted_only_with_closed_balances_and_an_unchanged_base(tmp_path, monkeypatch, miss, fresh):
    from faraday_singularity import driver as dv
    from faraday_singularity import surgery
    from faraday_singularity.mesh import Focus

    before, after = budget(), budget()
    grid = {}
    if miss == "volume": after["poly"]["V_liquid"] += 1
    if miss == "momentum": after["box"]["Pz"] += 1
    if miss == "nonfinite":  # opposite infinities in two crossed cells: C_P fails, rolled back
        grid, b, a = _two_crossed((math.inf, -math.inf))
        before, after = dict(before, _cells=b), dict(after, _cells=a)
    if miss == "nan-midpoint":  # a crossed midpoint entering the liquid at NaN: rolled back
        grid, b, a = _two_crossed()
        a = dict(a, found=np.array([True, False]), uz=np.array([math.nan, 0.0]))
        b = dict(b, found=np.array([False, True]), uz=np.array([0.0, 1.0]))
        before, after = dict(before, _cells=b), dict(after, _cells=a)
        after["box"]["Pz"] = math.nan
    samples = iter([before, after])
    monkeypatch.setattr(contact, "sample", lambda *a: next(samples))
    monkeypatch.setattr(surgery, "site_grid", lambda *a: grid)
    monkeypatch.setattr(dv, "base_guard_checked", lambda *a: {
        "ok": miss != "base", "checked": False, "change": {"r_j": float("nan")}})
    focus = Focus(size=.01, radius=.1, grading=.3)
    template = SimpleNamespace(focus=focus)
    restored, logged, invalidated = [], [], []
    state = SimpleNamespace(step=5, kinks={"old": 1}, far_yields=2,
                            last_remesh_step=1, snapshots=0, events=[])
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    gated = miss == "base" and not fresh

    def refine(*a):
        template.focus = None
        pb.remesh_count += 1
        state.kinks["new"] = 2
        state.far_yields += 1
        state.last_remesh_step = state.step
        state.events.append("ring-refine")
        (profiles / f"profile_{state.snapshots:06d}.txt").write_text("rejected or accepted rebuild")
        state.snapshots += 1
        if gated:
            gate = tmp_path / "states" / "gate_00000005.dump"
            gate.parent.mkdir()
            gate.write_bytes(b"pre-refine state")
            gate.with_suffix(".json").write_text('{"before":true}')
            fake._gate_stop = {"reason": "box-gate", "precut": str(gate)}
        return {"after": True}, None

    def restore(x):
        restored.append(x)
        pb.remesh_count += 1  # loading the earlier mesh also changes the rebuild count

    pb = SimpleNamespace(remesh_count=3, _snapshot_state=lambda: b"state", active_mesh_template=lambda: template,
                         _restore_state=restore, invalidate_cached_mesh_data=lambda: invalidated.append(True))
    fake = SimpleNamespace(events=SimpleNamespace(ring=SimpleNamespace(
        sigma_V=1e-6, sigma_Pz=1e-5, box_radius=4)),
        pb=pb, reconnect=SimpleNamespace(delta=.001, reservoir_depth=3), state=state,
        _ring_refine_once=refine, _taus=lambda t: {}, _t=lambda: 1,
        outdir=tmp_path, measure=lambda: {"rolled_back": True},
        _post_cut_cache=None, _stop_cache=None, _gate_stop=None, _stepped_rc=None if fresh else 3)
    def event(*args, **kw):
        fake.state.events.append(args[0])
        logged.append((args, kw))
    fake._event = event
    fake._fresh_mesh = lambda: CollapseDriver._fresh_mesh(fake)
    m, stop = CollapseDriver._ring_liquid_refine(fake, {}, {"r": .1, "z": 1}, 1)
    record = json.loads(next((tmp_path / "events").glob("ring_contact_*.json")).read_text(),
                        parse_constant=lambda value: pytest.fail(f"non-finite JSON: {value}"))
    assert fake.state is state and state.snapshots == 1
    assert state.events == ["ring-refine", "ring-contact-refine"]
    assert record["base"]["change"]["r_j"] is None
    if miss is None:
        assert stop is None and not restored and record["closures"]["hard"] == []
    else:
        assert restored == [b"state"] and template.focus is focus
        assert invalidated == [True] and fake._fresh_mesh() is fresh
        assert fake.state.kinks == {"old": 1} and m == {"rolled_back": True}
        assert (state.far_yields, state.last_remesh_step) == (2, 1)
        assert stop["trigger"] == "ring-contact-unclosed" and record["closures"]["hard"]
        assert not Path(stop["record"]).is_absolute()
        (profiles / f"profile_{state.snapshots:06d}.txt").write_text("stop on restored state")
        assert (profiles / "profile_000000.txt").read_text() == "rejected or accepted rebuild"
        if gated:
            assert fake._gate_stop is None and not (tmp_path / "states/gate_00000005.dump").exists()
            assert (tmp_path / record["rolled_back_gate"]["prestate"]).read_bytes() == b"pre-refine state"
    assert logged[-1][0][0] == "ring-contact-refine"


def loop_reader(P, monkeypatch):
    """Use the real reader's order check, orientation, seam and node identities."""
    from faraday_singularity.problem import FaradayProblem
    from pyoomph.meshes import zeta
    monkeypatch.setattr(zeta, "_find_closed_segments", lambda *a: [True, False])
    n = len(P)
    cache = SimpleNamespace(
        get_interface_line_segments=lambda: ([list(range(n))+[0], [0, 1]], None),
        get_coordinates=lambda: P.T, get_data=lambda key: np.zeros(n),
        elem_indices=np.array([[i, i+1, (i+2) % n] for i in range(0, n, 2)]))
    pb = SimpleNamespace(get_cached_mesh_data=lambda *a, **kw: cache)
    return lambda **kw: FaradayProblem.interface_loops(pb, **kw)


def test_a_graded_crossing_with_a_changed_seam_maps_the_actual_nodes(monkeypatch):
    from faraday_singularity.ring import LOOP_STEP_RATIO, loop_steps
    P, Q = crossed_teardrop()
    # Grade a remote midside interval, while retaining a simple input loop.
    k = len(P)//2+11
    k += 1-k % 2
    for X in (P, Q):
        X[k] = X[k-1]+.14*(X[k+1]-X[k-1])
    reader = loop_reader(P, monkeypatch)
    before_loop = reader(with_node_ids=True)[0]
    seam = int(before_loop["node_ids"][0])
    other = (seam+2) % len(Q)
    Q[other, 0] = Q[seam, 0]+1e-12
    pb = SimpleNamespace(remesh_count=3, interface_loops=loop_reader(Q, monkeypatch))
    with pytest.raises(RuntimeError, match="not geometrically ordered"):
        pb.interface_loops()
    candidate = pb.interface_loops(validate=False, with_node_ids=True)[0]
    assert candidate["node_ids"][0] != seam
    lengths = loop_steps(np.column_stack([candidate["r"], candidate["z"]]))
    assert max(lengths / np.minimum(np.roll(lengths, 1), np.roll(lengths, -1))) > LOOP_STEP_RATIO
    before = {"P": np.column_stack([before_loop["r"], before_loop["z"]]),
              "node_ids": before_loop["node_ids"], "remesh_count": 3}
    fake = SimpleNamespace(pb=pb, state=SimpleNamespace(step=7, t_prev=1), _t=lambda: 2)
    with pytest.raises(contact.RingStepContact) as exc:
        CollapseDriver._ring_step_check(fake, before)
    reading = exc.value.reading
    np.testing.assert_array_equal(reading["before_local"], P[reading["local_node_ids"]])
    pb.interface_loops = reader
    fake._distortion_flags = lambda: [{"discarded": True}]
    fake._restore_step_snapshot = lambda: None
    fake.measure = lambda: {"time": 1}
    logged, requests = [], []
    fake._event = lambda *a, **kw: logged.append(kw)
    fake._local_ring_fill_request = lambda *a, **kw: CollapseDriver._local_ring_fill_request(fake, *a, **kw)
    fake.events = liquid_driver(P).events
    fake.s = SimpleNamespace(fold_stop_elements=3)
    fake._taus = lambda t: {}
    fake._meet_event = lambda m, req: (requests.append(req) or m, None)
    _, stop = CollapseDriver._meet_ring_step_contact(fake, {}, exc.value)
    assert stop is None and requests[0]["step_fill"]["guard"]["pre_step_phase"] == "gas"
    assert logged[0]["discarded_distortion"] == [{"discarded": True}]
    assert logged[0]["rejected_diagnostics"] == {"after": 1, "through": 2}
    assert contact.loop_node_map([0, 1, 2], [1, 2, 0]).tolist() == [2, 0, 1]
    for ids in ([0, 2, 1], [0, 1, 1], [0, 1, 3]):
        assert contact.loop_node_map([0, 1, 2], ids) is None


def test_an_unreadable_candidate_rolls_back_as_a_contact_not_a_newton_failure():
    def unreadable(**kw):
        raise RuntimeError("synthetic unreadable walk")
    fake = SimpleNamespace(pb=SimpleNamespace(remesh_count=1, interface_loops=unreadable), _t=lambda: 2)
    with pytest.raises(contact.RingStepContact, match="local part") as exc:
        CollapseDriver._ring_step_check(fake, {"remesh_count": 1})
    assert "synthetic unreadable" in exc.value.reading["unreadable"]


def test_the_run_loop_restores_and_fills_a_crossed_step_then_accepts_the_next(tmp_path, monkeypatch):
    from test_events_plumbing import _events_driver
    from test_ring_driver import RingProblem, made_ring_cut
    from test_step_failure import NetProblem
    from test_groove_driver import fake_fill
    from faraday_singularity import groovesurgery
    from faraday_singularity.reconnect import ReconnectSettings

    P, Q = crossed_teardrop()
    class CrossProblem(RingProblem, NetProblem):
        flags = []
        crossed = False
        def interface_loops(self, **kw):
            X = Q if self.crossed else P
            return [{"r": X[:, 0], "z": X[:, 1], "u_r": np.zeros(len(X)),
                     "u_z": np.zeros(len(X)), "vertex": np.arange(len(X)) % 2 == 0,
                     **({"node_ids": np.arange(len(X))} if kw.get("with_node_ids") else {})}]
        def run(self, *a, **kw):
            super().run(*a, **kw)
            self.crossed = self.calls == 1
            if self.crossed:
                self.flags.append({"synthetic_distortion": True})
        def _restore_state(self, snap):
            super()._restore_state(snap)
            self.crossed = False
        def take_distortion_flags(self):
            result, self.flags = self.flags, []
            return result

    pb = CrossProblem()
    block = liquid_driver(P).events.to_dict()
    block["jetcore"] = {"enabled": True, "far_kinks": "recentre"}
    drv = _events_driver(tmp_path, pb, ReconnectSettings(delta=1e-3, dt0=1e-6),
                         events=EventSettings.from_dict(block), max_steps=1)
    made_ring_cut(drv, pb)
    drv._stop = lambda m: None
    drv._remesh_checks = lambda m: (m, None)
    drv._quality_floor_flags = lambda: []
    drv._distortion_remesh = lambda flags: pytest.fail(f"stale distortion: {flags}")
    calls = []
    inner = fake_fill(calls)
    def fill(*a, base_check=None, **kw):
        assert not pb.crossed and len(pb.restored) == 1 and not pb.flags
        assert base_check()["ok"]
        return inner(*a, **kw)
    monkeypatch.setattr(groovesurgery, "groove_fill", fill)
    result = drv.run(1e-9)
    drv.close()
    assert result["stop"]["reason"] == "max-steps" and result["steps"] == 1
    assert pb.calls == 2 and len(calls) == 1 and result["events"]["step_contact_fills"] == 1
    log = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    rolled = [e for e in log if e["event"] == "ring-contact-step-rollback"]
    assert len(rolled) == 1 and rolled[0]["discarded_distortion"] == [{"synthetic_distortion": True}]
    assert not any(e["event"] == "step-failed" for e in log)


@pytest.mark.parametrize("geometry", ["two-crossings", "three-elements", "overlap"])
def test_an_ineligible_crossed_step_is_a_resumable_stop_on_its_input(tmp_path, monkeypatch, geometry):
    from test_events_plumbing import _events_driver
    from test_ring_driver import RingProblem, made_ring_cut
    from test_step_failure import NetProblem
    from faraday_singularity.reconnect import ReconnectSettings
    from faraday_singularity.ring import loop_crossings
    from shapely.geometry import LinearRing

    P, Q = crossed_teardrop()
    if geometry == "three-elements":
        Q[0] = P[0]+1.5*(Q[0]-P[0])
        assert len(loop_crossings(Q)) == 1
    elif geometry == "two-crossings":
        Q = P.copy()
        Q[3] = 2*P[-3]-P[3]
        assert len(loop_crossings(Q)) == 2
    else:
        Q = P.copy()
        Q[1] = P[-1]
    assert not LinearRing(Q).is_simple and contact.step_contact(Q) is None

    class CrossProblem(RingProblem, NetProblem):
        crossed = False
        def interface_loops(self, **kw):
            X = Q if self.crossed else P
            return [{"r": X[:, 0], "z": X[:, 1], "u_r": np.zeros(len(X)),
                     "u_z": np.zeros(len(X)), "vertex": np.arange(len(X)) % 2 == 0,
                     **({"node_ids": np.arange(len(X))} if kw.get("with_node_ids") else {})}]
        def run(self, *a, **kw):
            super().run(*a, **kw)
            self.crossed = True
        def _restore_state(self, snap):
            super()._restore_state(snap)
            self.crossed = False
        def save_state(self, path, quiet=True):
            Path(path).write_text(json.dumps({"time": self.t, "P": (Q if self.crossed else P).tolist()}))
            self.saved.append(self.t)

    pb = CrossProblem()
    block = liquid_driver(P).events.to_dict()
    block["jetcore"] = {"enabled": True, "far_kinks": "recentre"}
    drv = _events_driver(tmp_path, pb, ReconnectSettings(delta=1e-3, dt0=1e-6),
                         events=EventSettings.from_dict(block), max_steps=1)
    made_ring_cut(drv, pb)
    def stop(m):
        assert not pb.crossed, "a crossed candidate must never reach measurement or stop guards"
        return None
    original_measure = drv.measure
    def measure():
        assert not pb.crossed, "a crossed candidate must be restored before measurement"
        return original_measure()
    monkeypatch.setattr(drv, "measure", measure)
    monkeypatch.setattr(drv, "_stop", stop)
    monkeypatch.setattr(drv, "_remesh_checks", lambda m: (m, None))
    monkeypatch.setattr(drv, "_meet_event", lambda *a: pytest.fail("no fill is allowed"))
    result = drv.run(1e-9)
    drv.close()
    assert result["steps"] == 0 and len(pb.restored) == 1
    assert result["stop"]["trigger"] == "stepped-contact-unhandled"
    assert "outside the local fill rule" in result["stop"]["contact"]["unreadable"]
    saved = json.loads(Path(result["stop"]["end_state"]).read_text())
    assert saved["time"] == pb.restored[0]["t"]
    np.testing.assert_array_equal(saved["P"], P)
    meta = json.loads(Path(result["stop"]["end_state"]).with_suffix(".json").read_text())
    assert meta["step"] == 0 and meta["time"] == pb.restored[0]["t"]
