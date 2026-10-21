"""The stops and events of the ring stage (``ring.ring_stage_events``) on synthetic states, without
a finite-element problem: an open chain (a flat free surface, or one whose tip is the head of a
jet rising from a groove, ``test_events_stops.cavity_with_head``) and a ring, a closed loop in the
liquid below it.

* a quiet state calls for nothing;
* the ring below ``bubble_stop`` of the bubble's volume at the gas cut is the request of its
  removal (``ring-collapse``, ``W2``, handler ``ring-removal``);
* the ring coming up to the axis is ``ring-axis`` (``RA``), a waist of the ring closing ``ring-fold``
  (``RF``), the ring within ``3 h_min`` of the open chain ``chain-contact`` (``X``), each without a
  handler;
* not one open chain, two loops, no loop before the removal or one after it: ``topology-changed``;
  a ring whose nodes cross raises;
* parity with ``events.post_cut_events``: the open chain's landed liquid neck gives the same request,
  field for field, as the branch before the ring gives it beside a far round bubble.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402
from faraday_singularity import ring  # noqa: E402
from faraday_singularity.selfcontact import InterfaceSelfIntersection  # noqa: E402

from test_events_stops import LIQUID, R_C, cavity_with_head  # noqa: E402

DELTA = 1e-3
S = rc.ReconnectSettings(delta=DELTA)
H_MIN = S.tip_size * DELTA
KW = {"neck_elements": 3.0, "fold_ratio": 4.0, "fold_stop": 1e-3, "fold_stop_elements": 3.0,
      "fold_search": 4.0, "apexes": None, "guard": None}
RING = ev.EventSettings.from_dict({"ring": {"enabled": True}})


def flat_open(z=4.69, h=2e-4):
    """A flat free surface from the axis to the wall at height ``z``."""
    r, d = [0.0], h
    while r[-1] + d < R_C:  # graded, as a mesh is: the spline through collinear nodes doubles
        r.append(r[-1] + d)  # back where their spacing jumps
        d = min(1.05 * d, 0.02) if r[-1] > 0.05 else d
    r = np.array(r[:-1] + [R_C])
    return rc.Chain(np.column_stack([r, np.full(len(r), z)]), ("axis", "fixed"))


def ellipse(c, a, b, n=200):
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([c[0] + a * np.cos(t), c[1] + b * np.sin(t)])


def cut_of(V_b):
    return rc.CutGeometry(z_bubble_bottom=4.5, z_bubble_top=4.6, z_cavity_tip=4.69, bridge=0.09,
                          chain_gap=0.01, bubble_volume=V_b)


def record(**kw):
    return {"V_b": 1.0, "r_w": 1e-4, "sizing": {"h_col": 1e-5}, "removed": None, **kw}


def stops(chains, loops, V_b=None, events=RING, rec=None, **kw):
    L = loops[0] if loops else None
    V_b = (10.0 * rc.revolved_volume(L) if L is not None else 1.0) if V_b is None else V_b
    return ring.ring_stage_events(chains, loops, cut_of(V_b), S, H_MIN, events,
                                  rec or record(V_b=V_b), **{**KW, **kw})


def test_a_quiet_state_of_the_ring_stage_calls_for_nothing():
    L = ellipse((0.01, 4.68), 2e-3, 1e-3)
    assert stops([flat_open()], [L]) is None
    # after the ring's removal, with no loop
    assert stops([flat_open()], [], rec=record(removed={"k": 5})) is None


def test_the_ring_below_bubble_stop_is_the_request_of_its_removal():
    L = ellipse((0.01, 4.68), 2e-3, 1e-3)
    V = rc.revolved_volume(L)
    req = stops([flat_open()], [L], V_b=200.0 * V)
    assert (req["reason"], req["kind"], req["class"], req["handler"]) == (
        ev.EVENT_REQUEST, "ring-collapse", "W2", "ring-removal")
    assert req["ring_volume"] == pytest.approx(V) and req["ratio"] == pytest.approx(1 / 200)
    assert req["limit"] == pytest.approx(0.01 * 200.0 * V) and req["chain"] == "ring"
    # just above it: nothing
    assert stops([flat_open()], [L], V_b=99.0 * V) is None


def test_the_ring_at_the_axis_a_closing_fold_and_the_open_chain_are_events_without_handlers():
    # the axis: the ring's innermost radius below axis_ratio r_w (0.5 r_w = 2e-3)
    L = ellipse((3e-3, 4.68), 2e-3, 1e-3)  # r_min 1e-3
    req = stops([flat_open()], [L], rec=record(r_w=4e-3, V_b=1.0))
    assert (req["kind"], req["class"], req["handler"]) == ("ring-axis", "RA", None)
    assert req["r_min"] == pytest.approx(1e-3) and req["limit"] == pytest.approx(2e-3)
    assert ring.ring_axis_limits(record(r_w=4e-3), RING) == pytest.approx((2e-3, 3e-5))
    # a waist of the ring closing below max(kappa_g delta, fold_stop_elements h_elem)
    t = np.linspace(0.0, 2.0 * math.pi, 400, endpoint=False)
    rho = 0.02 + np.cos(t) ** 2
    peanut = np.column_stack([0.01 + 1e-3 * rho * np.cos(t), 4.68 + 1e-3 * rho * np.sin(t)])
    req = stops([flat_open()], [peanut])
    assert (req["kind"], req["class"], req["handler"]) == ("ring-fold", "RF", None)
    assert req["chain"] == "ring" and req["contact"] == "fold"
    assert req["fold_gap"] == pytest.approx(4e-5, rel=0.05) and req["fold_gap"] < req["fold_limit"]
    # the ring within 3 h_min of the open chain (2 h_min below it)
    gap = 2.0 * H_MIN
    L = ellipse((0.02, 4.69 - gap - 1e-3), 4e-3, 1e-3)
    req = stops([flat_open()], [L])
    assert (req["kind"], req["class"], req["handler"]) == ("chain-contact", "X", None)
    assert req["chain_gap"] == pytest.approx(gap, rel=1e-3) and req["limit"] == 3.0 * H_MIN


def test_a_changed_topology_stops_and_a_crossed_ring_raises():
    L = ellipse((0.01, 4.68), 2e-3, 1e-3)
    two = stops([flat_open()], [L, ellipse((0.02, 4.68), 1e-3, 5e-4)])
    assert two == {"reason": "topology-changed", "chains": 1, "loops": 2}
    assert stops([flat_open()], [])["reason"] == "topology-changed"  # no ring before removal
    assert stops([flat_open()], [L], rec=record(removed={"k": 5}))["reason"] == \
        "topology-changed"
    bubble = rc.Chain(np.array([[0.0, 4.5], [1e-2, 4.55], [0.0, 4.6]]), ("axis", "axis"))
    assert stops([bubble, flat_open()], [L])["chains"] == 2
    t = np.linspace(0.0, 2.0 * math.pi, 101, endpoint=False) + 0.013
    eight = np.column_stack([0.01 + 3e-3 * np.sin(t), 4.68 + 3e-3 * np.sin(t) * np.cos(t)])
    with pytest.raises(InterfaceSelfIntersection, match="ring"):
        stops([flat_open()], [eight])
    with pytest.raises(RuntimeError, match="touches"):
        stops([flat_open()], [ellipse((0.01, 4.69), 2e-3, 1e-3)])


def test_the_open_chains_liquid_neck_is_requested_as_before_the_ring():
    """The open chain's landed neck (a jet rising from a groove, its neck 0.05 delta under a head
    0.5 delta wide): the ring stage's request is the branch's before the ring beside a far round
    bubble, field for field; with the ring far below the open chain."""
    cav = cavity_with_head(DELTA, r_n=0.05)
    th = np.linspace(0.0, math.pi, 801)
    sphere = rc.Chain(np.column_stack([0.05 * np.sin(th), 4.3 - 0.05 * np.cos(th)]),
                      ("axis", "axis"))
    sphere.points[[0, -1], 0] = 0.0
    before = rc.post_cut_stops([sphere, cav], rc.cut_geometry([sphere, cav], 3.0, gas_before=1.0),
                               S, H_MIN, events=LIQUID)
    assert (before["kind"], before["class"], before["chain"]) == ("liquid-neck", "L", "open")
    liquid_ring = ev.EventSettings.from_dict({**LIQUID.to_dict(), "ring": {"enabled": True}})
    L = ellipse((0.1, 4.4), 2e-2, 1e-2)
    after = stops([cav], [L], events=liquid_ring)
    assert after == before
