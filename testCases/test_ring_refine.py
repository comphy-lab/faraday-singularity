"""The ring's refinement on a freshly built mesh, without a finite-element problem
(``driver.CollapseDriver._ring_refine``, ``_ring_refine_due``, ``_ring_refine_once`` on stand-ins;
``jetcore.far_kink_regions``' ``refined`` mark; ``driver.base_guard``): the rim of the thin sheet
that the downward jet and the plug drive out of the ring's column at the collision of their
liquids, as r1 of the delta triple met it at tau/delta^2 24.74, eight steps after its ring cut (an
element of the ring turning by 91.1 degrees on the mesh rebuilt with the ring's far kink yielded to
half the column's size, chord 9.4e-6).

* the stop met: the guard's ``interface-unresolved`` on the ring, on a fresh mesh, an ``edge-turn``
  within the limit on any mesh or an ``edge-midside``, at an element above the guard's floor, in
  the ring stage with jetcore's recentred far kinks; nothing else (the open chain, a stepped mesh,
  a turn beyond ``edge_turn_stop``, ``feature-below-floor``, a curved Jacobian, the floor, no ring,
  the ring removed, no jetcore or kinks that grow);
* the rebuild: the ring's far kink recentred on the element at ``min(1/2, edge_turn_refine/turn)``
  of its chord (never below ``h_floor``), the radius of the region it lies in kept, the other
  regions of the focus as they are and in their order, ``refined``, a yield; the base of the upward
  jet read before and after it, a move beyond ``HEAD_FOLD_BASE_TOL`` the stop
  ``ring-refine-base``, a state without a born jet not checked; the stops of the new mesh;
* at most ``RING_REFINE_TRIES`` rebuilds per state, the stop that stays carrying their count;
* the far kink's rule: a ``refined`` region keeps its target and its mark while the ring asks for
  one, yields past ``far_yield_turn`` as before, and is released by the hold, after which a new
  request is an ordinary region at the floor, as is every region the driver did not refine
  (the rule of e923ead);
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
BASE = {"r_j": 1.290061417830118e-3, "z_b": 4.716019972998573, "V_j": 830.4721688598765,
        "Q_j": 4.3420590885911246e-3}


def _due_driver(*, ring=True, removed=False, jet="recentre", h_floor=H_FLOOR):
    recon = ReconnectSettings(delta=1e-3, dt0=1e-9)
    fake = SimpleNamespace(
        reconnect=recon, s=SimpleNamespace(refine_ratio=1.5),
        _ring=lambda: ({"removed": 3 if removed else None} if ring else None),
        _jetcore=lambda: (SimpleNamespace(far_kinks=jet) if jet is not None else None),
        _h_floor=lambda: h_floor)
    return fake


def test_the_ring_refine_meets_the_rings_fresh_stops_and_no_other():
    due = CollapseDriver._ring_refine_due
    fake = _due_driver()
    assert due(fake, dict(STOP))
    assert due(fake, dict(STOP, trigger="edge-midside", midside=0.24, limit=0.2))
    assert due(fake, dict(STOP, turn=120.0))  # the limit on any mesh, included
    for change in ({"chain": "open"}, {"chain": "bubble"}, {"fresh": False},
                   {"turn": 120.5}, {"trigger": "feature-below-floor"},
                   {"trigger": "curved-jacobian"}, {"reason": "self-contact-approach"},
                   {"L": 1.5 * H_FLOOR}, {"L": math.nan}):
        assert not due(fake, dict(STOP, **change)), change
    assert not due(fake, None)
    for kw in ({"ring": False}, {"removed": True}, {"jet": None}, {"jet": "grow"}):
        assert not due(_due_driver(**kw), dict(STOP)), kw


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


def test_one_rebuild_recentres_the_rings_far_kink_at_the_size_its_turn_asks_for():
    kinks = {"ring_far": dict(KINK), "open": {"name": "kink_open", "chain": "open"}}
    fake, template, log = _once_driver(kinks, _focus())
    m, stop = CollapseDriver._ring_refine_once(fake, dict(BASE), dict(STOP), 1)
    target = 30.0 / STOP["turn"] * STOP["L"]
    g = kinks["ring_far"]
    assert g["target"] == pytest.approx(target, rel=1e-14) and g["refined"] and g["far"]
    assert (g["r"], g["z"]) == (STOP["r"], STOP["z"])
    assert g["radius"] == KINK["radius"]  # the element lies in it: its radius kept (> 4 L)
    assert g["h_before"] == KINK["h_before"] and g["yields"] == 1 and not g["released"]
    assert kinks["open"] == {"name": "kink_open", "chain": "open"}  # the core's untouched
    assert fake.state.far_yields == 3 and fake.state.last_remesh_step == 7415
    assert log["remesh"] == 1 and log["snapshots"] == ["ring-refine"] and log["stops"] == 1
    names = [x.name for x in template.focus.regions]
    assert names == ["cavity_tip", "kink_ring_far", "ring_cut"]
    new = template.focus.region("kink_ring_far")
    assert (new.size, new.r, new.z, new.radius) == (g["target"], g["r"], g["z"], g["radius"])
    for a, b in zip(template.focus.regions, _focus().regions):
        if a.name != "kink_ring_far":
            assert a == b
    assert template.focus.size == g["target"] and template.focus.grading == 0.2
    (kind, rec), = log["events"]
    assert kind == "ring-refine" and rec["attempt"] == 1 and rec["target"] == g["target"]
    assert rec["reading"]["turn"] == STOP["turn"] and rec["base"]["ok"] and rec["base"]["checked"]
    assert stop is None and m == BASE


def test_a_rebuild_halves_at_least_adds_the_region_and_floors_its_size():
    # no far kink yet: the region added at edge_turn_refine/turn of the chord
    kinks = {}
    focus = Focus(size=3.9e-5, radius=6.3e-3, grading=0.2,
                  regions=(FocusRegion("cavity_tip", size=3.9e-5, r=0.0, z=4.75, radius=6.3e-3),))
    fake, template, log = _once_driver(kinks, focus)
    CollapseDriver._ring_refine_once(fake, dict(BASE), dict(STOP, turn=95.0, L=8e-6), 1)
    assert kinks["ring_far"]["target"] == pytest.approx(30.0 / 95.0 * 8e-6)
    assert kinks["ring_far"]["radius"] == 4.0 * 8e-6 and kinks["ring_far"]["yields"] == 1
    assert kinks["ring_far"]["h_before"] == 8e-6
    assert [x.name for x in template.focus.regions] == ["cavity_tip", "kink_ring_far"]
    # a turn below twice edge_turn_refine (a fresh limit set low): halved at least
    kinks = {}
    fake, _, _ = _once_driver(kinks, focus)
    CollapseDriver._ring_refine_once(fake, dict(BASE), dict(STOP, turn=55.0, L=8e-6), 1)
    assert kinks["ring_far"]["target"] == 0.5 * 8e-6
    # a midside node: half the chord; a region elsewhere: not its radius
    kinks = {"ring_far": dict(KINK, r=0.5, z=0.5)}
    fake, template, log = _once_driver(kinks, _focus())
    midside = {k: v for k, v in STOP.items() if k != "turn"}
    CollapseDriver._ring_refine_once(fake, dict(BASE), dict(midside, trigger="edge-midside",
                                                            midside=0.25, limit=0.2), 2)
    g = kinks["ring_far"]
    assert g["target"] == 0.5 * STOP["L"] and g["radius"] == 4.0 * STOP["L"]
    assert math.isnan(g["turn"]) and g["yields"] == 1
    kinks = {"ring_far": dict(KINK, released=True)}  # released: a region of its own again
    fake, _, _ = _once_driver(kinks, _focus())
    CollapseDriver._ring_refine_once(fake, dict(BASE), dict(STOP, L=2.0 * H_FLOOR), 1)
    assert kinks["ring_far"]["target"] == H_FLOOR and kinks["ring_far"]["h_before"] == 2 * H_FLOOR


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


def test_a_refined_kink_keeps_its_size_while_the_ring_asks_and_is_released_by_the_hold():
    kw = dict(hold=_hold, h_far=1.5625e-5, h_floor=H_FLOOR, yield_turn=75.0,
              floors={"ring": H_COL})
    refined = {"ring_far": dict(KINK, target=3.09e-6, refined=True, yields=1, t_set=0.0,
                                h_before=2.9e-5)}
    ask = {"ring": np.array([0])}
    g = far_kink_regions(refined, _ring_reading(44.0, 3e-6), ask, t=0.1, sizes={}, **kw)
    k = g["ring_far"]
    assert k["target"] == 3.09e-6 and k["refined"] and k["r"] == 1.8e-4 and k["yields"] == 1
    assert k["radius"] == 4.0 * 3e-6 and k["h_before"] == 2.9e-5
    # the same region not refined by the driver: back to the floor at once (e923ead's rule)
    plain = {"ring_far": {k_: v for k_, v in refined["ring_far"].items() if k_ != "refined"}}
    p = far_kink_regions(plain, _ring_reading(44.0, 3e-6), ask, t=0.1, sizes={}, **kw)
    assert p["ring_far"]["target"] == H_COL and "refined" not in p["ring_far"]
    # past the yield turn, met by the mesh: halved, still refined
    y = far_kink_regions(g, _ring_reading(80.0, 3e-6), ask, t=0.2,
                         sizes={"kink_ring_far": 3.0e-6}, **kw)["ring_far"]
    assert y["target"] == pytest.approx(1.545e-6) and y["yields"] == 2 and y["refined"]
    # no element asks within the hold: released with its mark, its target the chord it met
    gone = far_kink_regions({"ring_far": y}, {}, {}, t=5.0, sizes={}, **kw)["ring_far"]
    assert gone["released"] and gone["target"] == gone["h_before"] == 2.9e-5
    again = far_kink_regions({"ring_far": gone}, _ring_reading(44.0, 3e-6), ask, t=6.0,
                             sizes={}, **kw)["ring_far"]
    assert again["target"] == H_COL and "refined" not in again and not again["released"]
    # a refined region whose chain asks no more stays within its hold
    held = far_kink_regions(g, {}, {}, t=0.5, sizes={}, **kw)["ring_far"]
    assert not held["released"] and held["refined"] and held["target"] == 3.09e-6


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
