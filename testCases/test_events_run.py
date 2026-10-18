"""Contract tests of the events a run after the cut makes with the events block
(``collapse.reconnect_events``): the tip of an event, the monitor of a cascade of events at a tip
and the ghosts of the removed heads (``events.py``). No finite-element problem is built."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faraday_singularity import events as ev  # noqa: E402

DELTA = 1e-3


# --- the tip of an event, and the cascade at a tip ------------------------------------------------


def test_the_tip_of_an_event_is_the_end_on_the_axis_its_head_holds():
    assert ev.event_tip({"chain": "open", "side": "before"}) == "cavity_tip"
    assert ev.event_tip({"chain": "bubble", "side": "after"}) == "bubble_top"
    assert ev.event_tip({"chain": "bubble", "side": "before"}) == "bubble_bottom"
    assert ev.FAR_TIPS == {"bubble_top": "bubble_bottom", "bubble_bottom": "bubble_top",
                           "cavity_tip": None}


def _monitor(**kw):
    s = ev.EventSettings.from_dict(kw)
    return ev.CascadeMonitor.for_run(s, DELTA, T=2e-6)


def _sequence(monitor, tip, intervals, volumes, t0=0.0):
    """Events at ``tip`` at the cumulative ``intervals`` (in delta^2) with heads of the
    ``volumes``: the verdict of each."""
    t, out = t0, []
    for k, (dt, V) in enumerate(zip(intervals, volumes), start=1):
        t += dt * DELTA**2
        out.append(monitor.add(tip, k, t, V))
    return out


def test_events_at_a_steady_rate_are_a_stationary_cascade():
    """Events 4 delta^2 apart with heads of one volume: the first at its tip, then undetermined
    until three pairs of ratios (the fifth event), then stationary; the interval in delta^2 and
    as events per T."""
    got = _sequence(_monitor(), "cavity_tip", [1.0, 4.0, 4.0, 4.0, 4.0, 4.0], [1.0] * 6)
    assert [g["verdict"] for g in got] == ["first", "undetermined", "undetermined",
                                           "undetermined", "stationary", "stationary"]
    assert [g["generation"] for g in got] == [1, 2, 3, 4, 5, 6]
    assert math.isnan(got[0]["interval"])
    assert got[1]["interval_over_delta2"] == pytest.approx(4.0)
    assert got[1]["events_per_T"] == pytest.approx(2e-6 / (4.0 * DELTA**2))
    assert got[2]["interval_ratio"] == pytest.approx(1.0) and got[2]["volume_ratio"] == 1.0


def test_events_that_shrink_in_interval_and_volume_are_a_cascade_that_stops():
    """Intervals and heads each 0.6 of the one before, below the ratio 0.7 of the test: the three
    pairs of ratios of the fifth event make it a shrinking cascade; an interval below
    cascade_min_interval delta^2 makes one at once, from the second event at a tip."""
    shrink = [4.0 * 0.6**j for j in range(5)]
    got = _sequence(_monitor(), "bubble_top", shrink, [0.6**j for j in range(5)])
    assert [g["verdict"] for g in got] == ["first", "undetermined", "undetermined",
                                           "undetermined", "shrink"]
    assert got[-1]["shrinking"] == 3 and got[-1]["interval_ratio"] == pytest.approx(0.6)
    got = _sequence(_monitor(), "bubble_top", [1.0, 0.2], [1.0, 1.0])
    assert got[1]["verdict"] == "shrink" and got[1]["interval_over_delta2"] == pytest.approx(0.2)
    # a shorter interval with heads that do not shrink is no cascade
    got = _sequence(_monitor(), "bubble_top", shrink, [1.0] * 5)
    assert got[-1]["verdict"] == "stationary" and got[-1]["shrinking"] == 0
    # nor with the test's keys set apart: cascade_min_interval 0 never stops on the interval
    got = _sequence(_monitor(cascade_min_interval=0.0), "bubble_top", [1.0, 0.01], [1.0, 1.0])
    assert got[1]["verdict"] == "undetermined"


def test_events_that_grow_in_interval_and_volume_are_a_growing_cascade():
    grow = [1.0 * 1.6**j for j in range(5)]
    got = _sequence(_monitor(), "cavity_tip", grow, [1.6**j for j in range(5)])
    assert got[-1]["verdict"] == "grow" and got[-1]["growing"] == 3
    got = _sequence(_monitor(cascade_count=1), "cavity_tip", grow[:3], [1.0, 1.6, 2.56])
    assert [g["verdict"] for g in got] == ["first", "undetermined", "grow"]


def test_the_tips_are_followed_apart_and_a_run_rebuilds_its_monitor_from_its_records():
    """The events of each tip are their own sequence; a monitor made from the records of the
    events a run made (a restart) gives the same verdicts, which a check of the next event
    continues."""
    m = _monitor()
    _sequence(m, "cavity_tip", [1.0, 4.0, 4.0], [1.0, 1.0, 1.0])
    first = m.check("bubble_top", 1e-5, 1.0)
    assert first["verdict"] == "first" and first["generation"] == 1
    records = [{"k": k, "kind": "liquid-neck", "outcome": "made", "tip": e["tip"], "t": e["t"],
                "trigger": {"head_volume": e["V"]}}
               for k, e in enumerate(({"tip": "cavity_tip", "t": x["t"], "V": x["V"]}
                                      for x in m.history["cavity_tip"]), start=1)]
    again = ev.CascadeMonitor.for_run(ev.EventSettings(), DELTA, T=2e-6, records=records)
    assert again.verdicts() == m.verdicts()
    nxt = m.history["cavity_tip"][-1]["t"] + 4.0 * DELTA**2
    assert again.check("cavity_tip", nxt, 1.0) == m.check("cavity_tip", nxt, 1.0)
    assert again.check("cavity_tip", nxt, 1.0)["generation"] == 4


# --- the ghosts of the removed heads --------------------------------------------------------------


def _ghost(**kw):
    """A head hanging from the bubble's top, cut at a neck at z = 1: it ended at 0.9 and the cap's
    tip, the stump's, is at 0.99; it moves down at 50, its rear retracting at 20 more."""
    return {"t0": 0.0, "U": -50.0, "direction": -1.0, "v_ret": 20.0, "z_front0": 0.9,
            "z_rear0": 0.99, "tip": "bubble_top", "far_tip": "bubble_bottom", "step0": 10,
            "recontacts": [], **kw}


def test_a_ghost_moves_rigidly_and_its_rear_retracts_towards_its_head():
    g = _ghost()
    assert ev.ghost_place(g, 0.0) == (0.9, 0.99)
    front, rear = ev.ghost_place(g, 1e-3)
    assert front == pytest.approx(0.9 - 0.05) and rear == pytest.approx(0.99 - 0.07)


def test_a_ghost_flags_the_stump_that_catches_it_and_the_far_tip_it_reaches_once_each():
    """Rear: the stump's tip past the ghost's rear; front: the ghost's front at the bubble's
    bottom; neither before two steps after the cut, each once; the head of the cavity's tip rises
    into the open gas and reaches no tip."""
    g = _ghost()
    tips = {"bubble_top": 0.99 - 0.2, "bubble_bottom": 0.5, "cavity_tip": 1.2}
    assert ev.ghost_recontacts(g, 1e-3, 11, tips) == []  # one step after the cut: not yet
    (rear,) = ev.ghost_recontacts(g, 1e-3, 12, tips)  # the stump's tip is below the rear
    assert rear["kind"] == "rear" and rear["tip"] == "bubble_top"
    assert rear["z_ghost"] == pytest.approx(0.92) and rear["z_tip"] == pytest.approx(0.79)
    g["recontacts"].append(rear)
    assert ev.ghost_recontacts(g, 1e-3, 13, tips) == []
    (front,) = ev.ghost_recontacts(g, 8e-3, 14, tips)  # 0.9 - 0.4 = 0.5: the bubble's bottom
    assert front["kind"] == "front" and front["tip"] == "bubble_bottom"
    up = _ghost(U=50.0, direction=1.0, z_front0=1.3, z_rear0=1.21, tip="cavity_tip",
                far_tip=None)
    assert ev.ghost_recontacts(up, 1.0, 100, {"cavity_tip": 0.21}) == []
    assert [x["kind"] for x in ev.ghost_recontacts(up, 1.0, 100, {"cavity_tip": 1e3})] == ["rear"]
