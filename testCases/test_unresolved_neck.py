"""An unresolved liquid neck on a far head, without a finite-element problem
(``events.unresolved_neck_request``; the driver's route, ``driver.CollapseDriver.
_unresolved_neck_request``, on stand-ins): the downward jet of the air-water run after its cut at
tau/delta^2 39.4, a slender thread of radius about 0.12 delta stretched into the bubble with a bulb
of radius 0.36 delta at its tip, a drop's neck (0.09 delta) between the bulb and the thread, and a
shallow minimum of the thread 2.9 delta above the tip (0.122 delta) that is a tracked neck too (its
head the bulb and the thread below it) but lies outside the one neck region of its chain, which
follows the narrower neck, so that a freshly built mesh meets it at fewer than ``liquid.resolution``
element lengths (``liquid-neck-unresolved``) although it is not closing.

* the stop on a far head is the far-head cut of that head at the unresolved neck (kind
  ``far-head``, class ``F-head``), the stop as ``far_event`` and its reading as ``unresolved``,
  the neck's pinch readings carried;
* none for another stop, a neck not among the state's, one without a single head or with a head
  too large, and on the open chain for a head that reaches into the core or without a core; a head
  of the open chain off the core (the upward jet's cut-born head) is cut as a far head;
* the driver: on a stand-in with that thread, the stops of the fresh mesh are the unresolved neck,
  the driver turns it into the cut with the check of the upward jet's base (``base_check``) and
  meets it (an event ``far-head`` with ``far_kind`` ``liquid-neck-unresolved``); without
  ``pinch.far_head`` the stop stays.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402

DELTA = 1e-3
Z_TIP = 4.60  # the downward jet's tip on the axis, below the open chain of test_head_fold
LIQUID = ev.EventSettings.from_dict({"liquid": {"enabled": True},
                                     "post": {"open_neck_stop": False, "neck_floor": 2e-6}})
# the thread above its bulb, in units of delta: (height above the tip, radius), read off the
# downward jet's end at the stop (the drop's neck at 0.946, the thread's minimum at 2.86)
THREAD = ((0.36, 0.36), (0.52, 0.334), (0.65, 0.267), (0.76, 0.174), (0.88, 0.0985),
          (0.946, 0.0908), (1.03, 0.0975), (1.17, 0.1137), (1.32, 0.1246), (1.46, 0.1285),
          (1.61, 0.1283), (1.90, 0.125), (2.19, 0.123), (2.48, 0.1223), (2.86, 0.12207),
          (3.23, 0.1222), (3.79, 0.1225), (4.91, 0.1236), (7.2, 0.1281), (12.0, 0.1404),
          (20.0, 0.1718), (28.8, 0.2263), (36.0, 0.2947), (45.0, 0.55), (52.0, 0.95),
          (55.0, 1.4))


def _resample(dense, spacing):
    """Nodes along the dense polyline ``dense`` at the local spacing ``spacing`` (one value per
    dense point, in its units), the ends kept."""
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(dense, axis=0).T))]
    out, x = [0.0], 0.0
    while True:
        x += float(np.interp(x, s, spacing))
        if x >= s[-1] - 0.3 * float(np.interp(s[-1], s, spacing)):
            break
        out.append(x)
    out.append(s[-1])
    out = np.asarray(out)
    return np.column_stack([np.interp(out, s, dense[:, 0]), np.interp(out, s, dense[:, 1])])


def thread_bubble(delta=DELTA, z_tip=Z_TIP, fine=0.006, coarse=0.02):
    """The bubble chain, from its bottom tip on the axis (14 delta below the jet's tip) out to its
    widest point (9.7 delta, 9 delta above the tip), in along its ceiling to the base of the
    downward jet (1.4 delta wide, 55 delta above the tip), and down the jet to its tip on the
    axis: the thread of :data:`THREAD` above a bulb of radius 0.36 delta. Node spacing ``fine``
    delta up to 2.3 delta above the tip, ``coarse`` delta from 2.5 delta to 6 delta (so that the
    thread's minimum at 2.86 delta, 0.122 delta wide, has 3 elements of ``2 coarse`` across its
    radius and the drop's neck 7.6 of ``2 fine``), coarser beyond. Returns the chain and the nodes
    of the two necks (the thread's, the drop's)."""
    from scipy.interpolate import PchipInterpolator

    zt, rt = np.asarray(THREAD).T
    f = PchipInterpolator(zt, rt)
    th = np.linspace(0.0, 0.5 * math.pi, 400)
    bulb = np.column_stack([0.36 * np.sin(th), 0.36 - 0.36 * np.cos(th)])  # tip up to (0.36, 0.36)
    z_up = np.unique(np.r_[np.linspace(0.36, 3.0, 4000), np.linspace(3.0, 55.0, 4000)])
    thread = np.column_stack([f(z_up), z_up])
    jet = np.vstack([bulb, thread[1:]])[::-1]  # from the base down to the tip
    ceil = np.column_stack([np.linspace(9.7, 1.4, 600), np.linspace(9.0, 55.0, 600)])
    t = np.linspace(0.0, 0.5 * math.pi, 600)
    side = np.column_stack([9.7 * np.cos(t), 9.0 - 23.0 * np.sin(t)])[::-1]  # bottom tip up
    dense = np.vstack([side, ceil[1:], jet[1:]])
    z = dense[:, 1]
    on_jet = np.arange(len(dense)) >= len(side) + len(ceil) - 2
    h = np.where(on_jet, np.interp(z, [0.0, 2.3, 2.5, 6.0, 8.0, 55.0],
                                   [fine, fine, coarse, coarse, 0.08, 0.08]), 0.4)
    P = _resample(dense, h)
    P[0, 0] = P[-1, 0] = 0.0
    P = np.column_stack([delta * P[:, 0], z_tip + delta * P[:, 1]])
    zz = (P[:, 1] - z_tip) / delta
    tail = np.arange(len(P)) > len(P) // 2
    i_thread = int(np.argmin(np.where(tail & (zz > 2.0) & (zz < 4.0), P[:, 0], np.inf)))
    i_drop = int(np.argmin(np.where(tail & (zz > 0.6) & (zz < 1.4), P[:, 0], np.inf)))
    return rc.Chain(P, ("axis", "axis")), i_thread, i_drop


def unresolved_stop(x, resolution=4.0):
    """The stop ``post_cut_events`` makes of the tracked neck ``x`` on a fresh mesh."""
    return {"reason": ev.LIQUID_NECK_UNRESOLVED, **ev._neck_fields(x),
            "limit": resolution * x["h_loc"], "resolution": resolution, "fresh": True,
            "tau": 3.94e-5, "tau_p": 3.71e-5}


@pytest.fixture(scope="module")
def state():
    chain, i_thread, i_drop = thread_bubble()
    necks = ev.liquid_necks([("bubble", chain)], LIQUID.liquid)
    by = {x["index"]: x for x in necks}
    return {"chain": chain, "necks": necks, "thread": by.get(i_thread), "drop": by.get(i_drop),
            "i_thread": i_thread, "i_drop": i_drop}


def test_the_thread_has_two_tracked_necks_and_only_the_drops_is_resolved(state):
    from faraday_singularity.selfcontact import check_simple_interface

    check_simple_interface(np.asarray(state["chain"].points))
    up, drop = state["thread"], state["drop"]
    assert up is not None and drop is not None, [(x["index"], x["r_n"]) for x in state["necks"]]
    assert {x["index"] for x in state["necks"] if x["status"] == "head"} == {up["index"],
                                                                             drop["index"]}
    assert up["r_n"] == pytest.approx(0.12207 * DELTA, rel=2e-3)
    assert drop["r_n"] == pytest.approx(0.0908 * DELTA, rel=2e-2)
    assert up["s"] == pytest.approx(0.36 * DELTA, rel=1e-3) == drop["s"]
    assert up["r_n"] < LIQUID.liquid.resolution * up["h_loc"]  # unresolved
    assert drop["r_n"] > LIQUID.liquid.resolution * drop["h_loc"]  # resolved
    n = len(state["chain"].points)
    assert up["head"] == [up["index"], n - 1] and drop["head"] == [drop["index"], n - 1]
    assert up["direction"] == -1.0 and up["side"] == "after"
    assert 3.0 < up["head_volume_over_s3"] < 20.0


def test_the_unresolved_neck_of_a_far_head_is_the_cut_of_that_head(state):
    up = state["thread"]
    stop = unresolved_stop(up)
    got = ev.unresolved_neck_request(stop, state["necks"])
    assert (got["reason"], got["kind"], got["class"], got["handler"]) == (
        ev.EVENT_REQUEST, "far-head", "F-head", "liquid-cut")
    assert got["chain"] == "bubble" and got["index"] == up["index"] and got["head"] == up["head"]
    assert got["r_n"] == up["r_n"] and got["s"] == up["s"]
    assert got["far_event"]["kind"] == ev.LIQUID_NECK_UNRESOLVED
    assert got["far_event"]["limit"] == stop["limit"] and "reason" not in got["far_event"]
    assert got["unresolved"] == {"r_n": up["r_n"], "h_loc": up["h_loc"],
                                 "limit": stop["limit"], "resolution": 4.0}
    assert "head_fold" not in got
    # the trigger's readings carried where the necks have them (not closing here)
    necks = [dict(x) for x in state["necks"]]
    for x in necks:
        x["pinch"] = {"fired": False, "T_p_over_tau_c": 38.9}
    got = ev.unresolved_neck_request(stop, necks)
    assert got["pinch"] == {"fired": False, "T_p_over_tau_c": 38.9}
    # the event the liquid cut admits: the request's neck is the state's
    assert ev.event_neck(got)["index"] == up["index"]


def test_no_cut_for_another_stop_or_a_neck_without_a_single_head(state):
    up = state["thread"]
    stop = unresolved_stop(up)
    assert ev.unresolved_neck_request(None, state["necks"]) is None
    assert ev.unresolved_neck_request(dict(stop, reason="interface-unresolved"),
                                      state["necks"]) is None
    assert ev.unresolved_neck_request(dict(stop, index=up["index"] + 7), state["necks"]) is None
    assert ev.unresolved_neck_request(dict(stop, chain="open"), state["necks"]) is None
    for status in ("no-head", "two-heads", "head-too-large"):
        necks = [dict(x, status=status, **({"head": None} if status != "head-too-large" else {}))
                 if x["index"] == up["index"] else x for x in state["necks"]]
        assert ev.unresolved_neck_request(stop, necks) is None, status


def test_on_the_open_chain_only_a_head_off_the_core_is_cut():
    from test_head_fold import folded_head_chain

    chain, i_n, i_a, rim, wall = folded_head_chain()
    necks = ev.liquid_necks([("open", chain)], LIQUID.liquid)
    head = next(x for x in necks if x["status"] == "head" and x["head"][0] == 0)
    core = np.zeros(len(chain.points), dtype=bool)
    core[head["index"]:] = True  # the core from the neck on
    stop = unresolved_stop(head)
    got = ev.unresolved_neck_request(stop, necks, core)
    assert got["chain"] == "open" and got["index"] == head["index"] and got["head"][0] == 0
    reach = core.copy()
    reach[head["index"] - 20] = True  # the head reaching into the core
    assert ev.unresolved_neck_request(stop, necks, reach) is None
    assert ev.unresolved_neck_request(stop, necks, None) is None  # no core: none of it far


# --- the driver's route, on stand-ins -------------------------------------------------------------


def _driver_run(tmp_path, monkeypatch, block):
    """A stand-in driver right after a cut, its bubble :func:`thread_bubble` and its open chain the
    folded head's of ``test_head_fold`` at delta 1e-3 (a jet standing on the base, an axial speed
    1000 everywhere): the stops of the fresh mesh, the request the driver makes of them and, where
    it makes one, the event met with a stand-in liquid cut that reads the base check. Returns the
    stop, the request, what the cut saw and the driver's events."""
    from test_bookkeeping import CutProblem
    from test_events_plumbing import _events, _events_driver
    from test_events_run import fake_surgery
    from test_head_fold import folded_head_chain

    from faraday_singularity import surgery
    from faraday_singularity.reconnect import ReconnectSettings

    cut = fake_surgery()
    seen = {}
    O = np.asarray(folded_head_chain(delta=DELTA)[0].points)
    B = np.asarray(thread_bubble()[0].points)

    class Thread(CutProblem):
        def interface_segments(self):
            return [self._chain("bubble", B[:, 0].copy(), B[:, 1].copy(), ("axis", "axis")),
                    self._chain("open", O[:, 0].copy(), O[:, 1].copy(), ("axis", "fixed"))]

        def axial_velocity_at(self, points):
            return [1000.0] * len(points), True

    def liquid_cut(problem, events, reconnect, outdir, request, *, base_check=None, **kw):
        seen["base_check"] = base_check is not None
        seen["request"] = request
        if base_check is not None:
            seen["guard"] = base_check()
        return cut.liquid_cut(problem, events, reconnect, outdir, request, **kw)

    monkeypatch.setattr(surgery, "liquid_cut", liquid_cut)
    recon = ReconnectSettings(delta=DELTA, dt0=1e-6)
    drv = _events_driver(tmp_path, Thread(), recon, ev.EventSettings.from_dict(block),
                         state_dt=0.0, state_keep=10)
    m = drv.measure()
    stop = drv._stop(m)
    req = drv._unresolved_neck_request(stop)
    met = None
    if req is not None:
        _, met = drv._meet_event(m, req)
    events = _events(tmp_path)
    drv.close()
    return {"stop": stop, "request": req, "met": met, "seen": seen, "events": events, "m": m}


BLOCK = {"liquid": {"enabled": True}, "post": {"open_neck_stop": False, "neck_floor": 2e-6},
         "jetcore": {"enabled": True}, "pinch": {"enabled": True, "far_head": True}}


def test_the_driver_meets_the_unresolved_neck_by_the_cut_with_the_base_check(tmp_path,
                                                                              monkeypatch):
    out = _driver_run(tmp_path, monkeypatch, BLOCK)
    stop, req, seen = out["stop"], out["request"], out["seen"]
    assert stop["reason"] == ev.LIQUID_NECK_UNRESOLVED and stop["chain"] == "bubble"
    _, i_thread, _ = thread_bubble()
    assert stop["index"] == i_thread
    assert (req["kind"], req["class"], req["index"]) == ("far-head", "F-head", i_thread)
    assert req["pinch"]["fired"] is False  # no history: no closing neck
    assert out["met"] is None  # made
    assert seen["base_check"] is True and seen["request"]["unresolved"]["limit"] == stop["limit"]
    g = seen["guard"]
    assert out["m"]["r_j"] > 0.0 and g["ok"] is True and set(g["change"]) == {"r_j", "z_b", "V_j",
                                                                              "Q_j"}
    far = [e for e in out["events"] if e["event"] == "far-head"]
    assert len(far) == 1 and far[0]["far_kind"] == ev.LIQUID_NECK_UNRESOLVED
    assert far[0]["chain"] == "bubble" and far[0]["r_n"] == pytest.approx(stop["r_n"])
    rec = json.loads((tmp_path / "events" / "event_001.json").read_text())
    assert (rec["kind"], rec["class"]) == ("far-head", "F-head")


def test_without_far_head_the_stop_stays(tmp_path, monkeypatch):
    block = json.loads(json.dumps(BLOCK))
    block["pinch"]["far_head"] = False  # (far_head needs jetcore: the block refuses it without)
    out = _driver_run(tmp_path, monkeypatch, block)
    assert out["stop"]["reason"] == ev.LIQUID_NECK_UNRESOLVED
    assert out["request"] is None and out["seen"] == {}
