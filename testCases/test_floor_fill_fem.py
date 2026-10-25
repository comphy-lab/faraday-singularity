"""Finite-element test of the fill at the floor of a wedge of gas of the ring
(``driver.CollapseDriver._floor_fill_request`` and ``_meet_groove``; ``groovesurgery.groove_fill``
with ``base_check``), on the ring of ``test_groove_fem.py``: the synthetic pierced state put in the
ring stage with an elliptic ring whose boundary has a narrow finger of gas pushed out into the
liquid, the solenoidal flow about it, and the run's floor (``post.h_floor``) set so that the
finger's tip lies at an element at the floor.

* the guard's reading at the tip (an edge turn on the ring, at the floor) is the request of the
  fill at the floor (``RF``, ``contact`` ``floor``, the wedge read at the scale of the floor); the
  driver meets it: the fill made with the floor's fold limits, its hard closures within their
  limits, the base of the upward jet read (none stands on this state: not checked), the finger
  gone and the ring kept, an event ``floor-fill`` and ``floor_fills`` in the run's events;
* a check of the base that finds it moved is the hard closure ``G_base``: the stop
  ``event-unclosed`` with the state before the event.

The module takes a few minutes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

pytest.importorskip("pyoomph")

import ring_fem_state as S  # noqa: E402
import test_groove_fem as GF  # noqa: E402
import test_groove_geometry as G  # noqa: E402

from faraday_singularity import driver as dv  # noqa: E402
from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import surgery  # noqa: E402
from faraday_singularity.driver import CollapseDriver, CollapseSettings  # noqa: E402
from faraday_singularity.mesh import Focus, FocusRegion, RingSizing  # noqa: E402
from faraday_singularity.reconnect import revolved_volume  # noqa: E402
from faraday_singularity.ring import FLOOR_FOLD_RATIO, loop_edges  # noqa: E402

H_FLOOR = 6e-4  # the run's floor: the finger's tip, meshed at GF.H_FINE, at most 1.5 of it


def _block(noise):
    sigma = {"sigma_V": 2.0 * noise["V"], "sigma_Pz": 2.0 * noise["Pz"],
             "sigma_KE": 2.0 * noise["KE"]}
    return ev.EventSettings.from_dict({
        "ring": {"enabled": True}, "gap": {"kappa_g": GF.KAPPA},
        "groove": {"enabled": True, "max_fill": 1.0, **sigma},
        "jetcore": {"enabled": True, "far_kinks": "recentre", "ring_kink_floor": True},
        "post": {"h_floor": H_FLOOR}})


def _tip_reading(pb):
    """The guard's reading at the finger's tip: its element on the ring (``loop_edges``), an edge
    turn on a fresh mesh as the guard reports one."""
    lp = pb.interface_loops()[0]
    L = np.column_stack([lp["r"], lp["z"]])
    e = loop_edges(L)["edges"]
    tip = G.bumped_ellipse(np.array([0.0]), c=GF.RING_C, a=GF.RING_A, b=GF.RING_B, A=0.03,
                           sigma=0.048)[0]
    k = int(np.argmin(np.hypot(e["mid"][:, 0] - tip[0], e["mid"][:, 1] - tip[1])))
    return {"reason": "interface-unresolved", "trigger": "edge-turn", "chain": "ring",
            "turn": 95.0, "limit": 90.0, "fresh": True, "r": float(e["mid"][k, 0]),
            "z": float(e["mid"][k, 1]), "L": float(e["L"][k]), "index": int(e["index"][k])}


@pytest.fixture(scope="module")
def floored(tmp_path_factory):
    out = tmp_path_factory.mktemp("floor")
    res = {}
    with S.pierce_problem(out / "pb") as pb:
        pb.initialise()
        record = S.gas_cut_record(pb)
        tmpl = pb.active_mesh_template()
        named = dict(surgery._named_chains(pb)[0])
        L = GF.finger_loop()
        tip = G.bumped_ellipse(np.array([0.0]), c=GF.RING_C, a=GF.RING_A, b=GF.RING_B, A=0.03,
                               sigma=0.048)[0]
        focus = Focus(size=GF.H_FINE, radius=0.2, grading=0.3, regions=(
            FocusRegion("cavity_tip", size=0.004, r=0.0, z=0.95, radius=0.1),
            FocusRegion("finger", size=GF.H_FINE, r=float(tip[0]) - 0.015, z=float(tip[1]),
                        radius=0.05)))
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
        pb.ring = GF.ring_record(pb, np.column_stack([loops[0]["r"], loops[0]["z"]]))
        ring_rec = dict(pb.ring)
        GF.impose(pb, (GF.RING_C[0] + 0.05, GF.RING_C[1]))
        initial = Path(surgery._keep(pb, None, out, "initial", {"ring": pb.ring})["path"])
        noise = GF.site_noise(pb, (float(tip[0]) - 0.015, float(tip[1])), 4.0 * S.DELTA)
        res["noise"] = noise
        events = _block(noise)
        for name in ("made", "moved"):
            GF.reload(pb, initial, record, ring_rec, focus)
            V_ring = revolved_volume(np.column_stack([pb.interface_loops()[0]["r"],
                                                      pb.interface_loops()[0]["z"]]))
            drv = CollapseDriver(pb, CollapseSettings(t_end=10.0, dt_max=0.01, max_steps=2,
                                                      state_dt=0.0, state_keep=10),
                                 out / name, reconnect=S.RECON, events=events)
            drv.state.dt_prev = GF.DT
            m = drv.measure()
            stop = _tip_reading(pb)
            req = drv._floor_fill_request(stop)
            got = {"stop": stop, "request": req, "V_ring": V_ring, "h_floor": drv._h_floor()}
            if name == "moved":  # a base found moved by the fill
                real = dv.base_guard_checked

                def moved(before, after):
                    g = real(before, after)
                    g.update(ok=False, checked=True, change={"r_j": 1.0})
                    return g
                dv.base_guard_checked = moved
            try:
                m2, after = drv._meet_event(m, req)
            finally:
                dv.base_guard_checked = real if name == "moved" else dv.base_guard_checked
            got.update(after=after, events_made=[dict(e) for e in pb.events],
                       loops=pb.interface_loops(),
                       summary=drv._events_summary({}),
                       log=[json.loads(x) for x in (out / name / "events.jsonl").read_text()
                            .splitlines()])
            rec_path = out / name / "events" / "event_001.json"
            got["record"] = json.loads(rec_path.read_text()) if rec_path.is_file() else None
            drv.close()
            res[name] = got
    return res


def test_the_wedge_at_the_floor_is_filled_by_the_driver_with_the_floors_limits(floored):
    got = floored["made"]
    stop, req = got["stop"], got["request"]
    assert stop["L"] <= 1.5 * got["h_floor"]
    assert req is not None and (req["kind"], req["class"], req["contact"]) == (
        "ring-fold", "RF", "floor")
    assert req["floor"]["fold"]["ratio"] == FLOOR_FOLD_RATIO
    assert req["floor"]["wedge"]["turn"] > 0.0 and req["floor"]["guard"]["index"] == stop["index"]
    assert got["after"] is None  # made
    rec = got["record"]
    assert (rec["kind"], rec["class"], rec["outcome"], rec["chain"]) == (
        "groove-fill", "RF", "made", "ring")
    assert rec["request"]["floor"]["wedge"]["pair"] == req["fold_pair"]
    assert rec["trigger"]["holds"] and rec["trigger"]["ratio"] == FLOOR_FOLD_RATIO
    assert rec["trigger"]["limit"] == pytest.approx(req["fold_limit"])
    c = rec["closures"]
    assert c["hard"] == [] and "G_base" not in c["flags"]
    for x in ("G1", "G2", "G3", "G3_box"):
        assert c[x] <= c["limits"][x], x
    base = rec["base_guard"]
    assert base["ok"] and base["checked"] is False  # no jet stands on this state
    # the finger gone, the ring kept less the filled volume
    assert rec["geometry"]["side"] == "gas" and rec["filled"]["V"] > 0.0
    assert len(got["loops"]) == 1
    L = np.column_stack([got["loops"][0]["r"], got["loops"][0]["z"]])
    assert revolved_volume(L) == pytest.approx(got["V_ring"] - rec["filled"]["V"],
                                               abs=0.02 * rec["filled"]["V"]
                                               + 5 * floored["noise"]["V"])
    # countable: its own event and the run's count
    floors = [e for e in got["log"] if e["event"] == "floor-fill"]
    assert len(floors) == 1 and floors[0]["k"] == 1 and floors[0]["V_fill"] == rec["filled"]["V"]
    assert got["summary"]["floor_fills"] == 1 and got["summary"]["by_kind"] == {"groove-fill": 1}


def test_a_base_moved_by_the_fill_at_the_floor_is_a_hard_closure_and_a_resumable_stop(floored):
    got = floored["moved"]
    after = got["after"]
    assert after["reason"] == ev.EVENT_UNCLOSED and "G_base" in after["hard"]
    assert Path(after["precut"]).name == "precut_e001.dump" and Path(after["precut"]).is_file()
    rec = got["record"]
    assert rec["outcome"] == "unclosed" and "G_base" in rec["closures"]["hard"]
    assert rec["base_guard"]["ok"] is False
    assert got["events_made"] == [] and "floor_fills" not in got["summary"]
    assert not any(e["event"] == "floor-fill" for e in got["log"])
