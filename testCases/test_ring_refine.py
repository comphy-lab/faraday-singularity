"""The ring's refinement on a freshly built mesh, without a finite-element problem
(``driver.CollapseDriver._ring_refine``, ``_ring_refine_due``, ``_ring_refine_once`` on stand-ins;
``jetcore.far_kink_regions``' ``refined`` mark; ``driver.base_guard``): the rim of the thin sheet
that the downward jet and the plug drive out of the ring's column at the collision of their
liquids, as r1 of the delta triple met it at tau/delta^2 24.74, eight steps after its ring cut (an
element of the ring turning by 91.1 degrees on the mesh rebuilt with the ring's far kink yielded to
half the column's size, chord 9.4e-6).

* the stop met: the guard's ``interface-unresolved`` on the ring, an ``edge-turn`` on a fresh mesh
  (r1's) or on a stepped one (r1h's, 123.3 degrees three steps after its ring cut) up to
  ``RING_REFINE_TURN_MAX``, or an ``edge-midside``, at an element above the guard's floor, in the
  ring stage with jetcore's recentred far kinks; nothing else (the open chain, a turn beyond the
  cap, ``feature-below-floor``, a curved Jacobian, the floor, no ring, the ring removed, no jetcore
  or kinks that grow);
* the floor: an element of the ring at the guard's floor that asks is carried there with
  jetcore's recentred far kinks (``ring.ring_floored_ok``, ``reconnect.interface_guard``'s
  ``floored_ok``), within the guard's turn limits, and still asks; the open chain's, and the ring's
  without them, stop the run as before;
* the rebuild: the element in a far kink of its own (``kink_ring_r<n>_far``, ``refined``; the
  refined one whose reach holds it refined further, never coarser; a feature elsewhere a second
  one) at ``min(1/2, edge_turn_refine/turn)`` of its chord (never below ``h_floor``), the radius of
  the far kink it lies in kept, every other region of the focus as it is and in its order, a
  yield; the base of the upward
  jet read before and after it, a move beyond ``HEAD_FOLD_BASE_TOL`` the stop
  ``ring-refine-base``, a state without a born jet not checked; the stops of the new mesh;
* at most ``RING_REFINE_TRIES`` rebuilds per state, the stop that stays carrying their count;
* the far kink's rule after a refine: every far kink of the ring follows its own feature (the
  elements that ask within its reach: the rim moving on), a refined one at its target, a new
  feature in a region of its own (``kink_ring_m<n>_far``, at most ``MULTI_FAR_KINKS``), a yield
  halving the finer of a region's target and its size, each released by its own hold; without a
  refined region every region is as at e923ead;
* ``base_guard``: the reading of ``_head_fold_check`` (e923ead's formula, value for value).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

from faraday_singularity import driver as dv  # noqa: E402
from faraday_singularity.driver import CollapseDriver  # noqa: E402
from faraday_singularity.jetcore import far_kink_regions  # noqa: E402
from faraday_singularity.mesh import Focus, FocusRegion  # noqa: E402
from faraday_singularity.reconnect import ReconnectSettings  # noqa: E402

# r1's stop (state_00007415 of the delta triple's member r1, fix1): the guard's reading
STOP = {"reason": "interface-unresolved", "trigger": "edge-turn", "chain": "ring",
        "turn": 91.1441404384896, "limit": 90.0, "fresh": True, "r": 1.8257050256382403e-4,
        "z": 4.61186639837446, "L": 9.39902376642992e-06, "index": 2597}
H_FLOOR = 7.8125e-7
H_COL = 2.077275537176241e-05
KINK = {"name": "kink_ring_far", "chain": "ring", "r": 1.6071831003016467e-4,
        "z": 4.611886158785072, "radius": 8.060938268988124e-05, "target": H_COL,
        "h_before": 2.894478669127556e-05, "t_set": 10.954720328524774, "released": False,
        "turn": 73.5040251788492, "yields": 0, "yield_t": None, "far": True}
# r1h's stop (the far floor halved, delta/128), three steps after its ring cut, on a stepped mesh
STOP_R1H = {"reason": "interface-unresolved", "trigger": "edge-turn", "chain": "ring",
            "turn": 123.3, "limit": 120.0, "fresh": False, "r": 1.114e-4, "z": 4.61192,
            "L": 2.30e-5, "index": 2593}
BASE = {"r_j": 1.290061417830118e-3, "z_b": 4.716019972998573, "V_j": 830.4721688598765,
        "Q_j": 4.3420590885911246e-3}


def _due_driver(*, ring=True, removed=False, jet="recentre", h_floor=H_FLOOR):
    recon = ReconnectSettings(delta=1e-3, dt0=1e-9)
    fake = SimpleNamespace(
        reconnect=recon, s=SimpleNamespace(refine_ratio=1.5),
        _ring=lambda: ({"removed": 3 if removed else None} if ring else None),
        _jetcore=lambda: (SimpleNamespace(far_kinks=jet) if jet is not None else None),
        _h_floor=lambda: h_floor, _post_cut_state=lambda: {"edges": {}})
    fake._ring_jacobian_reading = lambda st: CollapseDriver._ring_jacobian_reading(fake, st)
    fake._ring_liquid_reading = lambda st: CollapseDriver._ring_liquid_reading(fake, st)
    fake._ring_sheet_reading = lambda st: CollapseDriver._ring_sheet_reading(fake, st)
    fake._ring_kink_reading = lambda st: CollapseDriver._ring_kink_reading(fake, st)
    fake._ring_reading = lambda st: CollapseDriver._ring_reading(fake, st)
    fake._refined_chain = lambda: CollapseDriver._refined_chain(fake)
    return fake


def test_the_ring_refine_meets_the_rings_stops_and_no_other():
    due = CollapseDriver._ring_refine_due
    fake = _due_driver()
    assert due(fake, dict(STOP))
    assert due(fake, dict(STOP_R1H))  # r1h's: on a stepped mesh, beyond edge_turn_stop
    assert due(fake, dict(STOP, trigger="edge-midside", midside=0.24, limit=0.2))
    assert due(fake, dict(STOP, turn=dv.RING_REFINE_TURN_MAX))  # the cap, included
    # a rebuilt mesh's curved element read on the ring's element (_ring_jacobian_reading; 63293b0's
    # successor: before it this stop was not met)
    assert due(fake, dict(STOP, trigger="curved-jacobian", limit=0.2, jacobian_bound=0.09))
    for change in ({"chain": "open"}, {"chain": "bubble"}, {"turn": dv.RING_REFINE_TURN_MAX + 0.5},
                   {"trigger": "feature-below-floor"}, {"trigger": "curved-jacobian", "turn": 151.0},
                   {"reason": "self-contact-approach"}, {"L": 1.5 * H_FLOOR}, {"L": math.nan}):
        assert not due(fake, dict(STOP, **change)), change
        assert not due(fake, dict(STOP_R1H, **change)), change
    assert not due(fake, None)
    for kw in ({"ring": False}, {"removed": True}, {"jet": None}, {"jet": "grow"}):
        assert not due(_due_driver(**kw), dict(STOP)), kw
        assert not due(_due_driver(**kw), dict(STOP_R1H)), kw


def _floored_loop(n=16, a=2e-6, b=1e-6, c=(0.01, 4.68)):
    """A small ring of ``n/2`` quadratic elements, at most about 1.2e-6 long, the sharper ones
    turning by more than 30 degrees: at the floor of 1e-6 (``refine_ratio`` 1.5) those ask and none
    can be refined."""
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([c[0] + a * np.cos(t), c[1] + b * np.sin(t)])


def test_a_ring_element_at_the_floor_is_carried_there_with_recentred_far_kinks():
    from faraday_singularity import events as ev
    from faraday_singularity import reconnect as rc
    from faraday_singularity import ring as rg

    from test_ring_stops import KW, flat_open, record, stops

    S = ReconnectSettings(delta=1e-3, dt0=1e-9)
    L = _floored_loop()
    rd = {"open": rc.edge_reading(flat_open().points), "ring": rg.loop_edges(L)}
    plain = rc.interface_guard(rd, S, fresh=False, h_floor=1e-6, refine_ratio=1.5)
    assert plain["stop"]["trigger"] == "feature-below-floor" and plain["stop"]["chain"] == "ring"
    carried = rc.interface_guard(rd, S, fresh=False, h_floor=1e-6, refine_ratio=1.5,
                                 floored_ok=("ring",))
    asks = carried["refine"]["ring"]
    e = rd["ring"]["edges"]
    assert carried["stop"] is None and len(asks) >= 2
    assert np.all(e["L"][asks] <= 1.5e-6) and np.all(e["turn"][asks] > S.edge_turn_refine)
    # the limits of the turn still stop it, and the open chain's floor is not carried
    sharp = {"ring": dict(rd["ring"], turn=125.0)}
    assert rc.interface_guard(sharp, S, fresh=False, h_floor=1e-6, refine_ratio=1.5,
                              floored_ok=("ring",))["stop"]["trigger"] == "edge-turn"
    other = {"open": rd["ring"]}
    assert rc.interface_guard(other, S, fresh=False, h_floor=1e-6, refine_ratio=1.5,
                              floored_ok=("ring",))["stop"]["trigger"] == "feature-below-floor"
    # the ring stage's stops: carried with jetcore's recentred far kinks, a stop without them
    guard = {"fresh": False, "h_floor": 1e-6, "refine_ratio": 1.5}
    ringed = ev.EventSettings.from_dict({"ring": {"enabled": True}})
    recentred = ev.EventSettings.from_dict({"ring": {"enabled": True},
                                            "jetcore": {"enabled": True,
                                                        "far_kinks": "recentre"}})
    assert rg.ring_floored_ok(ringed) == () and rg.ring_floored_ok(recentred) == ("ring",)
    without = stops([flat_open()], [L], events=ringed, guard=guard)
    assert without["trigger"] == "feature-below-floor" and without["chain"] == "ring"
    assert stops([flat_open()], [L], events=recentred, guard=guard) is None
    assert stops([flat_open()], [L], events=recentred, guard=dict(guard, fresh=True)) is None
    assert KW["guard"] is None and record()["sizing"]


def test_the_rebuilds_are_bounded_and_a_stop_that_stays_carries_their_count():
    calls = []

    def run(outcomes):
        calls.clear()
        fake = _due_driver()
        fake._ring_refine_due = lambda s: CollapseDriver._ring_refine_due(fake, s)

        def once(m, stop, attempt):
            calls.append(attempt)
            return {"m": attempt}, outcomes[attempt - 1]
        fake._ring_refine_once = once
        return CollapseDriver._ring_refine(fake, {"m": 0}, dict(STOP))

    m, stop = run([None])
    assert stop is None and m == {"m": 1} and calls == [1]
    event = {"reason": "event-request", "kind": "ring-fold"}
    m, stop = run([dict(STOP, turn=95.0), event])
    assert stop == event and calls == [1, 2] and "ring_refines" not in stop
    m, stop = run([dict(STOP, turn=95.0 + k) for k in range(dv.RING_REFINE_TRIES)])
    assert calls == list(range(1, dv.RING_REFINE_TRIES + 1))
    assert stop["ring_refines"] == dv.RING_REFINE_TRIES and stop["reason"] == "interface-unresolved"
    # a stop it does not meet is returned as it is
    fake = _due_driver()
    fake._ring_refine_due = lambda s: CollapseDriver._ring_refine_due(fake, s)
    fake._ring_refine_once = lambda *a: pytest.fail("no rebuild for a stop not met")
    other = dict(STOP, chain="open")
    assert CollapseDriver._ring_refine(fake, {"m": 0}, other) == ({"m": 0}, other)


class _Template:
    def __init__(self, focus):
        self.focus = focus


def _once_driver(tmp_kinks, focus, *, after=None, stop_after=None):
    """A stand-in of the driver for one rebuild: its state, focus, rebuild, measurements and
    events recorded."""
    log = {"remesh": 0, "events": [], "snapshots": [], "stops": 0}
    after = dict(BASE) if after is None else after
    template = _Template(focus)

    def force_remesh():
        log["remesh"] += 1

    def stop(m):
        log["stops"] += 1
        return stop_after

    fake = SimpleNamespace(
        state=SimpleNamespace(kinks=tmp_kinks, far_yields=2, step=7415, last_remesh_step=7411),
        reconnect=ReconnectSettings(delta=1e-3, dt0=1e-9),
        pb=SimpleNamespace(active_mesh_template=lambda: template, force_remesh=force_remesh),
        _h_floor=lambda: H_FLOOR, _t=lambda: 10.954720338063346,
        _box_noise_before=lambda: None, _gate_before=lambda: None,
        _box_noise_after=lambda noise, reason: None, _gate_after=lambda gate, reason: {},
        measure=lambda: dict(after), _keep_landings=lambda m: None,
        _event=lambda kind, m, **kw: log["events"].append((kind, kw)),
        _axis_repair_note=lambda: {}, _snapshot=lambda m, reason: log["snapshots"].append(reason),
        _taus=lambda t: {"tau": 2.474e-5, "tau_p": 2.3018e-5}, _stop=stop)
    return fake, template, log


def _focus():
    regions = (FocusRegion("cavity_tip", size=3.9e-5, r=0.0, z=4.75, radius=6.3e-3),
               FocusRegion("kink_ring_far", size=1.0386e-5, r=1.7397e-4, z=4.611875,
                           radius=8.99e-5),
               FocusRegion("ring_cut", size=H_COL, r=0.0, z=4.61195, radius=1.19e-3))
    return Focus(size=1.0386e-5, radius=6.3e-3, grading=0.2, regions=regions)


def test_one_rebuild_refines_the_element_in_a_far_kink_of_its_own():
    kinks = {"ring_far": dict(KINK), "open": {"name": "kink_open", "chain": "open"}}
    fake, template, log = _once_driver(kinks, _focus())
    m, stop = CollapseDriver._ring_refine_once(fake, dict(BASE), dict(STOP), 1)
    target = 30.0 / STOP["turn"] * STOP["L"]
    g = kinks["ring_r1_far"]
    assert g["name"] == "kink_ring_r1_far" and g["chain"] == "ring"
    assert g["target"] == pytest.approx(target, rel=1e-14) and g["refined"] and g["far"]
    assert (g["r"], g["z"]) == (STOP["r"], STOP["z"])
    # the yielded far kink holds the element: its radius and the chord it met carried over
    assert g["radius"] == KINK["radius"] and g["h_before"] == KINK["h_before"]
    assert g["yields"] == 1 and not g["released"]
    assert kinks["ring_far"] == KINK  # every other region as it was, the core's too
    assert kinks["open"] == {"name": "kink_open", "chain": "open"}
    assert fake.state.far_yields == 3 and fake.state.last_remesh_step == 7415
    assert log["remesh"] == 1 and log["snapshots"] == ["ring-refine"] and log["stops"] == 1
    names = [x.name for x in template.focus.regions]
    assert names == ["cavity_tip", "kink_ring_far", "ring_cut", "kink_ring_r1_far"]
    new = template.focus.region("kink_ring_r1_far")
    assert (new.size, new.r, new.z, new.radius) == (g["target"], g["r"], g["z"], g["radius"])
    assert template.focus.regions[:3] == _focus().regions
    assert template.focus.size == g["target"] and template.focus.grading == 0.2
    (kind, rec), = log["events"]
    assert kind == "ring-refine" and rec["attempt"] == 1 and rec["target"] == g["target"]
    assert rec["name"] == "kink_ring_r1_far" and rec["reading"]["turn"] == STOP["turn"]
    assert rec["base"]["ok"] and rec["base"]["checked"]
    assert stop is None and m == BASE


def test_a_rebuild_halves_at_least_refines_its_own_region_further_and_floors_its_size():
    # no far kink yet: the region added at edge_turn_refine/turn of the chord
    kinks = {}
    focus = Focus(size=3.9e-5, radius=6.3e-3, grading=0.2,
                  regions=(FocusRegion("cavity_tip", size=3.9e-5, r=0.0, z=4.75, radius=6.3e-3),))
    fake, template, log = _once_driver(kinks, focus)
    CollapseDriver._ring_refine_once(fake, dict(BASE), dict(STOP, turn=95.0, L=8e-6), 1)
    g = kinks["ring_r1_far"]
    assert g["target"] == pytest.approx(30.0 / 95.0 * 8e-6)
    assert g["radius"] == 4.0 * 8e-6 and g["yields"] == 1 and g["h_before"] == 8e-6
    assert [x.name for x in template.focus.regions] == ["cavity_tip", "kink_ring_r1_far"]
    # the same feature again, within the reach of that region (its radius and four chords): refined
    # further in it, never coarser, a second yield
    fake, template, _ = _once_driver(kinks, template.focus)
    CollapseDriver._ring_refine_once(fake, dict(BASE), dict(STOP, turn=95.0, L=2.5e-6,
                                                            r=STOP["r"] + 4e-5), 2)
    assert set(kinks) == {"ring_r1_far"} and kinks["ring_r1_far"]["yields"] == 2
    assert kinks["ring_r1_far"]["target"] == pytest.approx(30.0 / 95.0 * 2.5e-6)
    assert [x.name for x in template.focus.regions] == ["cavity_tip", "kink_ring_r1_far"]
    # a feature elsewhere: a second region of its own, the first as it was
    first = dict(kinks["ring_r1_far"])
    fake, template, _ = _once_driver(kinks, template.focus)
    CollapseDriver._ring_refine_once(fake, dict(BASE), dict(STOP, turn=95.0, L=8e-6, r=4e-4,
                                                            z=4.6115), 1)
    assert kinks["ring_r1_far"] == first and kinks["ring_r2_far"]["r"] == 4e-4
    assert [x.name for x in template.focus.regions] == ["cavity_tip", "kink_ring_r1_far",
                                                         "kink_ring_r2_far"]
    # a turn below twice edge_turn_refine (a fresh limit set low): halved at least
    kinks = {}
    fake, _, _ = _once_driver(kinks, focus)
    CollapseDriver._ring_refine_once(fake, dict(BASE), dict(STOP, turn=55.0, L=8e-6), 1)
    assert kinks["ring_r1_far"]["target"] == 0.5 * 8e-6
    # a midside node: half the chord; the ring's far kink elsewhere: not its radius, unchanged
    kinks = {"ring_far": dict(KINK, r=0.5, z=0.5)}
    fake, template, log = _once_driver(kinks, _focus())
    midside = {k: v for k, v in STOP.items() if k != "turn"}
    CollapseDriver._ring_refine_once(fake, dict(BASE), dict(midside, trigger="edge-midside",
                                                            midside=0.25, limit=0.2), 2)
    g = kinks["ring_r1_far"]
    assert g["target"] == 0.5 * STOP["L"] and g["radius"] == 4.0 * STOP["L"]
    assert math.isnan(g["turn"]) and g["yields"] == 1 and kinks["ring_far"] == dict(KINK, r=0.5,
                                                                                     z=0.5)
    kinks = {"ring_far": dict(KINK, released=True)}  # a released kink holds nothing
    fake, _, _ = _once_driver(kinks, _focus())
    CollapseDriver._ring_refine_once(fake, dict(BASE), dict(STOP, L=2.0 * H_FLOOR), 1)
    g = kinks["ring_r1_far"]
    assert g["target"] == H_FLOOR and g["h_before"] == 2 * H_FLOOR and g["radius"] == 8 * H_FLOOR


def test_a_base_moved_by_the_rebuild_stops_the_run_and_none_is_checked_without_a_jet():
    moved = dict(BASE, V_j=BASE["V_j"] * (1 + 2e-3))
    fake, _, log = _once_driver({}, _focus(), after=moved, stop_after={"never": "read"})
    m, stop = CollapseDriver._ring_refine_once(fake, dict(BASE), dict(STOP), 2)
    assert stop["reason"] == "interface-unresolved" and stop["trigger"] == "ring-refine-base"
    assert stop["chain"] == "ring" and stop["attempt"] == 2 and not stop["base"]["ok"]
    assert stop["base"]["change"]["V_j"] == pytest.approx(2e-3)
    assert log["stops"] == 0  # the stop is the base's, not the new mesh's
    jetless = {"r_j": 0.2, "z_b": 0.9, "V_j": math.nan, "Q_j": math.nan}
    fake, _, log = _once_driver({}, _focus(), after=jetless)
    m, stop = CollapseDriver._ring_refine_once(fake, jetless, dict(STOP), 1)
    (_, rec), = log["events"]
    assert stop is None and rec["base"]["ok"] and not rec["base"]["checked"]


def _hold(L):
    return 1.0


def _ring_reading(turn, L, mid=(1.8e-4, 4.6119)):
    return {"ring": {"edges": {"turn": np.array([float(turn)]), "L": np.array([float(L)]),
                               "mid": np.array([list(mid)]), "f": np.array([0.5]),
                               "index": np.array([1])}}}


def _ring_readings(*elements):
    """A reading of the ring's quadratic elements, each ``(turn, L, (r, z))``."""
    turns, Ls, mids = zip(*elements)
    return {"ring": {"edges": {"turn": np.array(turns, dtype=float),
                               "L": np.array(Ls, dtype=float),
                               "mid": np.array(mids, dtype=float),
                               "f": np.full(len(turns), 0.5),
                               "index": 2 * np.arange(len(turns)) + 1}}}


def test_after_a_refine_every_far_kink_of_the_ring_follows_its_own_feature():
    from faraday_singularity.jetcore import MULTI_FAR_KINKS

    kw = dict(hold=_hold, h_far=1.5625e-5, h_floor=H_FLOOR, yield_turn=75.0,
              floors={"ring": H_COL})
    ROOT_RZ, RIM_RZ = (9e-5, 4.6118), (1.8e-4, 4.6119)
    ordinary = dict(KINK, r=ROOT_RZ[0], z=ROOT_RZ[1], radius=4e-5, t_set=0.0)
    rim = {"name": "kink_ring_r1_far", "chain": "ring", "r": RIM_RZ[0], "z": RIM_RZ[1],
           "radius": 1.2e-5, "target": 3.09e-6, "h_before": 2.9e-5, "t_set": 0.0,
           "released": False, "turn": 91.1, "yields": 1, "yield_t": 0.0, "far": True,
           "refined": True}
    prev = {"ring_far": ordinary, "ring_r1_far": rim}
    # the rim alone asks, moved on within the region's reach: the refined region follows it at
    # its size, four of its chords wide; the ring's far kink is not moved, nor released
    d = rim["radius"] + 4.0 * 3e-6
    one = far_kink_regions(prev, _ring_readings((44.0, 3e-6, (RIM_RZ[0] + 0.99 * d, RIM_RZ[1]))),
                           {"ring": np.array([0])}, t=0.1, sizes={}, **kw)
    k = one["ring_r1_far"]
    assert k["target"] == 3.09e-6 and k["refined"] and k["r"] == RIM_RZ[0] + 0.99 * d
    assert k["radius"] == 4.0 * 3e-6 and k["h_before"] == 2.9e-5 and k["t_set"] == 0.1
    assert one["ring_far"] == ordinary and set(one) == {"ring_far", "ring_r1_far"}
    # just beyond every reach: a region of its own, at the floor, the others as they were
    x = (RIM_RZ[0] + 1.01 * d, RIM_RZ[1])
    out = far_kink_regions(prev, _ring_readings((44.0, 3e-6, x)), {"ring": np.array([0])},
                           t=0.1, sizes={}, **kw)
    assert out["ring_r1_far"] == rim and out["ring_far"] == ordinary
    m1 = out["ring_m1_far"]
    assert (m1["name"], m1["r"], m1["target"], m1["multi"]) == ("kink_ring_m1_far", x[0], H_COL,
                                                                 True)
    assert "refined" not in m1 and m1["yields"] == 0
    # the rim and the column's root both ask: each followed by its own region
    both = far_kink_regions(prev, _ring_readings((44.0, 3e-6, RIM_RZ), (35.0, 3e-5, ROOT_RZ)),
                            {"ring": np.array([0, 1])}, t=0.2, sizes={}, **kw)
    assert both["ring_r1_far"]["r"] == RIM_RZ[0] and both["ring_r1_far"]["target"] == 3.09e-6
    assert both["ring_far"]["r"] == ROOT_RZ[0] and both["ring_far"]["target"] == H_COL
    assert set(both) == {"ring_far", "ring_r1_far"}
    # past the yield turn, met by the mesh: halved from the finer of its target and its size
    y = far_kink_regions(prev, _ring_readings((80.0, 3e-6, RIM_RZ)), {"ring": np.array([0])},
                         t=0.2, sizes={"kink_ring_r1_far": 3.0e-6}, **kw)["ring_r1_far"]
    assert y["target"] == pytest.approx(1.5e-6) and y["yields"] == 2 and y["refined"]
    # an ordinary region finer in the mesh than its target (once refined, its target back at the
    # floor): a yield halves its size, where without a refined region (e923ead's rule) it halves
    # its target, coarser than the mesh is there
    sharp = _ring_readings((84.0, 9e-6, ROOT_RZ))
    after = far_kink_regions(prev, sharp, {"ring": np.array([0])}, t=0.2,
                             sizes={"kink_ring_far": 4.8e-6}, **kw)["ring_far"]
    assert after["target"] == pytest.approx(2.4e-6) and after["yields"] == 1
    single = far_kink_regions({"ring_far": ordinary}, sharp, {"ring": np.array([0])}, t=0.2,
                              sizes={"kink_ring_far": 4.8e-6}, **kw)["ring_far"]
    assert single["target"] == pytest.approx(0.5 * H_COL)
    # only the root asks: the rim's region kept within its hold, released after it (its target the
    # chord it met), while the ring's far kink follows the root
    kept = far_kink_regions(prev, _ring_readings((35.0, 3e-5, ROOT_RZ)), {"ring": np.array([0])},
                            t=0.5, sizes={}, **kw)
    assert not kept["ring_r1_far"]["released"] and kept["ring_r1_far"]["target"] == 3.09e-6
    gone = far_kink_regions(prev, _ring_readings((35.0, 3e-5, ROOT_RZ)), {"ring": np.array([0])},
                            t=5.0, sizes={}, **kw)
    assert gone["ring_r1_far"]["released"] and gone["ring_r1_far"]["target"] == 2.9e-5
    assert gone["ring_far"]["r"] == ROOT_RZ[0] and not gone["ring_far"]["released"]
    # nothing asks: each region released by its own hold
    quiet = far_kink_regions(prev, {}, {}, t=5.0, sizes={}, **kw)
    assert quiet["ring_r1_far"]["released"] and quiet["ring_far"]["released"]
    # at most MULTI_FAR_KINKS regions on the chain: features far apart, the worst first
    spots = [(30.5 + j, 3e-5, (1e-3 * (j + 1), 4.62)) for j in range(MULTI_FAR_KINKS + 2)]
    many = far_kink_regions({"ring_r1_far": rim}, _ring_readings(*spots),
                            {"ring": np.arange(len(spots))}, t=0.1, sizes={}, **kw)
    assert len(many) == MULTI_FAR_KINKS and many["ring_r1_far"] == rim
    assert sorted(g["r"] for k_, g in many.items() if g.get("multi")) == pytest.approx(
        [1e-3 * (j + 1) for j in range(3, MULTI_FAR_KINKS + 2)])
    # without a refined region every region is e923ead's: the worst element, at the floor
    plain = far_kink_regions({"ring_far": ordinary},
                             _ring_readings((44.0, 3e-6, RIM_RZ), (35.0, 3e-5, ROOT_RZ)),
                             {"ring": np.array([0, 1])}, t=0.2, sizes={}, **kw)
    assert set(plain) == {"ring_far"} and plain["ring_far"]["r"] == RIM_RZ[0]
    assert plain["ring_far"]["target"] == H_COL


def test_base_guard_is_the_head_fold_checks_reading_value_for_value():
    """``base_guard`` against e923ead's own formula in ``_head_fold_check``, on readings within
    and beyond the tolerance, with a zero and a NaN."""
    def e923ead(before_m, after_m):
        keys = ("r_j", "z_b", "V_j", "Q_j")
        before = {k: float(before_m.get(k, math.nan)) for k in keys}
        after = {k: float(after_m.get(k, math.nan)) for k in keys}
        r_j = before["r_j"]
        change = {k: (after[k] - before[k]) / abs(before[k]) if before[k] else math.nan
                  for k in ("r_j", "V_j", "Q_j")}
        change["z_b"] = (after["z_b"] - before["z_b"]) / r_j if r_j > 0.0 else math.nan
        ok = all(math.isfinite(v) and abs(v) <= dv.HEAD_FOLD_BASE_TOL for v in change.values())
        return {"before": before, "after": after, "change": change,
                "tol": dv.HEAD_FOLD_BASE_TOL, "ok": bool(ok)}

    rng = np.random.default_rng(7)
    cases = [(BASE, BASE), (BASE, dict(BASE, z_b=BASE["z_b"] + 2e-3 * BASE["r_j"])),
             (dict(BASE, V_j=0.0), BASE), (dict(BASE, Q_j=math.nan), BASE), ({}, BASE)]
    for _ in range(50):
        after = {k: v * (1.0 + rng.normal(0.0, 1e-3)) for k, v in BASE.items()}
        cases.append((BASE, after))
    seen = set()
    for before, after in cases:
        got, want = dv.base_guard(before, after), e923ead(before, after)
        assert list(got) == list(want) and list(got["change"]) == list(want["change"])
        assert repr(got) == repr(want)
        seen.add(got["ok"])
    assert seen == {True, False}
