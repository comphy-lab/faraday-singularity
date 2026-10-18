"""Contract tests of the stops and events of a state after the cut with the events block
(``reconnect.post_cut_stops`` with ``events``; ``events.post_cut_events``). No finite-element
problem is built.

The synthetic state is the one the cuts leave at the root of a jet they launch: a bubble whose top
holds an annular gas groove around a liquid jet that hangs from it, the jet a root from the groove's
apex, a neck, and a bulb (its head) whose tip is the bubble's top on the axis; the open chain above
is that of ``test_bookkeeping.CutProblem``. Every length of the jet and the groove, and the node
spacing there, is in units of the cut radius ``delta``, inside a bubble and a cavity of fixed size,
so that the same state at another ``delta`` must give the same stops in units of ``delta``.

The tests on the stop states of runs after a cut run when ``FARADAY_POSTCUT_RUNS`` names their run
directories (``os.pathsep``-separated; each with ``parameters.json``, ``run_status.json`` and the
profiles of the state its run stopped on).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402

R_C = 3.8317059702075125
Z_APEX = 4.68  # the height of the groove's apex, the base of the hanging jet
LIQUID = ev.EventSettings.from_dict({"liquid": {"enabled": True},
                                     "post": {"open_neck_stop": False, "neck_floor": 2e-6}})
RUNS = [Path(p) for p in os.environ.get("FARADAY_POSTCUT_RUNS", "").split(os.pathsep) if p]


# --- the synthetic state --------------------------------------------------------------------------


def _hermite(z0, r0, z1, r1, n):
    """``r(z)`` from ``(r0, z0)`` to ``(r1, z1)`` with zero slope ``dr/dz`` at both ends."""
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack([r0 + (r1 - r0) * (3 * t**2 - 2 * t**3), z0 + (z1 - z0) * t])


def head_feature(*, r_n, s, r_w, r_root, m=4000):
    """The groove and the hanging jet in units of delta, the apex at ``z = 0``, from the roof of the
    bubble at ``r = 3`` to the jet's tip on the axis: the roof (horizontal), a rounded corner, the
    groove's outer wall (``r = r_w``) up to a semicircular top whose highest point is the apex,
    down the jet's root (``r_root`` at the top) to the neck ``r_n`` 0.4 below the top, the bulb
    widening to its largest radius ``s`` 0.5 lower and closing as a sphere onto the axis. Returns
    the dense polyline and the index of the apex."""
    a = 0.5 * (r_w - r_root)
    zeta_n, zeta_b = a + 0.4, a + 0.9
    D, q = zeta_b + s + 0.5, 0.25
    roof = -D - q
    t = np.linspace(-0.5 * math.pi, -math.pi, m)
    arc = np.linspace(0.0, math.pi, m)
    tt = np.linspace(0.0, 0.5 * math.pi, m)
    pieces = [np.column_stack([np.linspace(3.0, r_w + q, m), np.full(m, roof)]),
              np.column_stack([r_w + q + q * np.cos(t), roof + q + q * np.sin(t)]),
              np.column_stack([np.full(m, r_w), np.linspace(roof + q, -a, m)]),
              np.column_stack([0.5 * (r_w + r_root) + a * np.cos(arc), -a + a * np.sin(arc)]),
              _hermite(-a, r_root, -zeta_n, r_n, m), _hermite(-zeta_n, r_n, -zeta_b, s, m),
              np.column_stack([s * np.cos(tt), -zeta_b - s * np.sin(tt)])]
    P = np.vstack([pieces[0]] + [p[1:] for p in pieces[1:]])
    apex = 3 * (m - 1) + (m - 1) // 2  # the top of the semicircle
    return P, apex


def _nodes(dense, start: float, delta: float, h: float, h_far: float, grading: float = 0.05):
    """Nodes of quadratic elements along the dense polyline (an odd count, each midside node half
    way along its element): ``h`` from the arclength ``start`` on to the end, growing by
    ``grading`` per unit distance before it, up to ``h_far``. Returns the arclengths."""
    sarc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(dense, axis=0).T))])
    nodes = [sarc[-1]]
    while True:
        x = nodes[-1] - min(h_far, h + grading * max(0.0, start - nodes[-1]))
        if x <= 0.0:
            break
        nodes.append(x)
    nodes = nodes[:-10]  # the far end, at h_far: spaced evenly down to the start
    n = max(2, round(nodes[-1] / h_far))
    if (len(nodes) + n) % 2 == 0:
        n += 1
    nodes = np.array(nodes[::-1])
    return sarc, np.concatenate([nodes[0] * np.arange(n) / n, nodes])


def bubble_with_head(delta, *, r_n=0.3, s=0.5, r_w=1.0, r_root=0.45, R_b=0.05, h=1 / 256,
                     h_far=1e-3):
    """The bubble chain, from its bottom up a sphere of radius ``R_b`` to the roof, then
    :func:`head_feature` at ``delta`` (spacing ``h delta`` there); ``(chain, apex index)``."""
    local, apex = head_feature(r_n=r_n, s=s, r_w=r_w, r_root=r_root)
    feat = np.column_stack([delta * local[:, 0], Z_APEX + delta * local[:, 1]])
    r1, zroof = feat[0]
    zs = zroof - math.sqrt(R_b**2 - r1**2)
    th = np.linspace(0.0, math.pi - math.asin(r1 / R_b), 80000)
    sphere = np.column_stack([R_b * np.sin(th), zs - R_b * np.cos(th)])
    dense = np.vstack([sphere, feat[1:]])
    sarc, nodes = _nodes(dense, float(np.sum(np.hypot(*np.diff(sphere, axis=0).T))), delta,
                         h * delta, h_far)
    P = np.column_stack([np.interp(nodes, sarc, dense[:, 0]), np.interp(nodes, sarc, dense[:, 1])])
    P[0, 0] = P[-1, 0] = 0.0
    k_apex = int(np.argmin(np.abs(nodes - sarc[len(sphere) - 1 + apex])))
    return rc.Chain(P, ("axis", "axis")), k_apex


def cavity():
    """The open chain of ``test_bookkeeping.CutProblem``: a cap of tip radius 1e-3 at 4.69, a line,
    and the wall."""
    phi = np.linspace(0.0, 0.5 * math.pi, 201)
    A, B = 2e-3, 4e-3
    cr, cz = A * np.sin(phi), 4.69 + B * (1.0 - np.cos(phi))
    t = np.linspace(0.0, 1.0, 401)[1:]
    line = np.column_stack([A + (0.5 - A) * t, cz[-1] + (5.0 - cz[-1]) * t])
    t = np.linspace(0.0, 1.0, 101)[1:]
    wall = np.column_stack([0.5 + (R_C - 0.5) * t, 5.0 + t])
    return rc.Chain(np.vstack([np.column_stack([cr, cz]), line, wall]), ("axis", "fixed"))


def cavity_with_head(delta, *, r_n=0.3, s=0.5, r_w=1.0, r_root=0.45, h=1 / 256, h_far=2e-3):
    """An open chain whose tip is the head of a jet rising from the trough of a groove: the
    :func:`head_feature` mirrored (its apex at ``4.70``, the lowest point of the groove), from its
    tip on the axis to the roof, then out to the wall."""
    local, apex = head_feature(r_n=r_n, s=s, r_w=r_w, r_root=r_root)
    feat = np.column_stack([delta * local[::-1, 0], 4.70 - delta * local[::-1, 1]])
    t = np.linspace(0.0, 1.0, 20000)[1:, None]  # out to the wall, leaving the roof tangentially
    far = ((1 - t) ** 2 * feat[-1] + 2 * t * (1 - t) * np.array([0.25, feat[-1, 1]])
           + t**2 * np.array([0.5, 5.0]))
    dense = np.vstack([feat, far, [[R_C, 5.2]]])
    # the spacing is graded away from the feature, which here comes first: build it reversed
    rev = dense[::-1]
    sarc, nodes = _nodes(rev, float(np.sum(np.hypot(*np.diff(rev, axis=0).T))) -
                         float(np.sum(np.hypot(*np.diff(feat, axis=0).T))), delta, h * delta,
                         h_far)
    P = np.column_stack([np.interp(nodes, sarc, rev[:, 0]), np.interp(nodes, sarc, rev[:, 1])])
    P = P[::-1].copy()
    P[0, 0], P[-1] = 0.0, [R_C, 5.2]
    return rc.Chain(P, ("axis", "fixed"))


def _stops(chains, delta=1e-3, events=LIQUID, cut=None, settings=None, **kw):
    s = settings or rc.ReconnectSettings(delta=delta)
    cut = cut or rc.cut_geometry(chains, 3.0, gas_before=1.0)
    return rc.post_cut_stops(chains, cut, s, s.tip_size * s.delta, events=events, **kw)


def _apexes(apex_index, chain="bubble", tip="bubble_top"):
    return {chain: [{"tip": tip, "index": apex_index, "r": 0.0, "z": Z_APEX, "R_apex": 1e-4}]}


# --- head sides and tracked necks -----------------------------------------------------------------


def test_the_head_of_a_neck_is_the_side_whose_far_end_is_on_the_axis():
    """The head of the hanging jet is the side of its neck from the neck to the bubble's top: it
    ends on the axis, holds no widest node of the bubble, lies below the neck and bounds a simple
    section of 3-4.5 s^3; the other side, with the bubble's widest node, is none. On the open
    chain the head of a rising jet is the side from its tip to the neck, the other ends on the
    wall."""
    bubble, _ = bubble_with_head(1e-3, r_n=0.2)
    nk = next(x for x in rc.chain_necks(bubble.points, bubble.end_types) if x["side"] == "liquid")
    before, after = ev.head_sides(bubble.points, bubble.end_types, nk["index"], is_bubble=True)
    assert before["holds_widest"] and not before["qualifies"]
    assert after["qualifies"] and after["direction"] == -1.0 and after["one_sided"]
    assert after["hi"] == len(bubble.points) - 1 and after["lo"] == nk["index"]
    assert after["s"] == pytest.approx(5e-4, rel=1e-6)
    ratio = after["volume"] / after["s"] ** 3
    assert 3.0 < ratio < 4.5
    open_ = cavity_with_head(1e-3, r_n=0.2)
    nk = next(x for x in rc.chain_necks(open_.points, open_.end_types) if x["side"] == "liquid")
    before, after = ev.head_sides(open_.points, open_.end_types, nk["index"], is_bubble=False)
    assert before["qualifies"] and before["direction"] == 1.0 and before["lo"] == 0
    assert not after["axis_end"] and not after["qualifies"]
    assert before["volume"] / before["s"] ** 3 == pytest.approx(ratio, rel=1e-3)
    # a side that rises above its neck's height, or whose section is not simple, is no head
    P = bubble.points.copy()
    i = next(x for x in rc.chain_necks(P, bubble.end_types) if x["side"] == "liquid")["index"]
    P[i + 40, 1] = P[i, 1] + 0.1 * P[i, 0]
    assert not ev.head_sides(P, bubble.end_types, i, is_bubble=True)[1]["one_sided"]
    crossed = np.array([[0.0, 0.0], [0.2, 0.3], [1.0, 0.8], [1.0, 0.2], [0.5, 1.0], [1.2, 1.4],
                        [0.0, 1.6]])
    sides = ev.head_sides(crossed, ("axis", "axis"), 4, is_bubble=False)
    assert sides[0]["one_sided"] and not sides[0]["simple"] and not sides[0]["qualifies"]


def test_a_liquid_neck_is_tracked_from_its_depth_and_lands_at_eps_d_of_its_head():
    """A neck on the liquid side is tracked once it is track_depth below its shoulder, with its
    head, its largest radius s, the element length there (twice the shorter adjacent segment), and
    lands once r_n <= eps_d s: the bubble's neck at 0.09 delta, with s = 0.5 delta, at eps_d 0.2."""
    for r_n, landed in ((0.3, False), (0.1001, False), (0.09, True)):
        bubble, _ = bubble_with_head(1e-3, r_n=r_n)
        (x,) = ev.liquid_necks((("bubble", bubble),), LIQUID.liquid)
        assert x["chain"] == "bubble" and x["status"] == "head" and x["landed"] == landed
        assert x["r_n"] == pytest.approx(r_n * 1e-3, rel=1e-3) and x["s"] == pytest.approx(5e-4)
        assert x["ratio"] == pytest.approx(x["r_n"] / (0.2 * x["s"]))
        assert x["h_loc"] == pytest.approx(2e-3 / 256, rel=0.01) and x["direction"] == -1.0
        assert x["head"] == [x["index"], len(bubble.points) - 1]
    shallow = ev.LiquidEventSettings(enabled=True, track_depth=0.5)
    assert ev.liquid_necks((("bubble", bubble_with_head(1e-3, r_n=0.3)[0]),), shallow) == []
    # the open chain's neck, at the root of a jet rising from a groove
    (x,) = ev.liquid_necks((("open", cavity_with_head(1e-3, r_n=0.09)),), LIQUID.liquid)
    assert x["chain"] == "open" and x["landed"] and x["head"][0] == 0 and x["direction"] == 1.0
    # a head larger than max_head_volume s^3 is not handled
    big = ev.LiquidEventSettings(enabled=True, max_head_volume=2.0)
    (x,) = ev.liquid_necks((("bubble", bubble_with_head(1e-3, r_n=0.09)[0]),), big)
    assert x["status"] == "head-too-large" and x["landed"]


# --- the events branch ----------------------------------------------------------------------------


def test_without_the_block_the_stops_are_the_cuts():
    """``events=None`` is the stops of the cut, unchanged: the state with a landed liquid neck stops
    on it as a second pinch-off below delta; the check of curved elements belongs to the branch."""
    bubble, _ = bubble_with_head(1e-3, r_n=0.09)
    chains = [bubble, cavity()]
    cut = rc.cut_geometry(chains, 3.0, gas_before=1.0)
    s = rc.ReconnectSettings(delta=1e-3)
    legacy = rc.post_cut_stops(chains, cut, s, 6.25e-5)
    assert legacy == rc.post_cut_stops(chains, cut, s, 6.25e-5, events=None)
    assert legacy["reason"] == "secondary-pinch-approach" and legacy["neck_side"] == "liquid"
    with pytest.raises(TypeError, match="events branch"):
        rc.post_cut_stops(chains, cut, s, 6.25e-5, invalid=lambda: None)


def test_a_landed_liquid_neck_is_an_event_with_its_head():
    bubble, _ = bubble_with_head(1e-3, r_n=0.3)
    assert _stops([bubble, cavity()]) is None  # tracked, not landed: no second pinch-off
    bubble, _ = bubble_with_head(1e-3, r_n=0.09)
    req = _stops([bubble, cavity()])
    assert req["reason"] == ev.EVENT_REQUEST and req["kind"] == "liquid-neck"
    assert req["class"] == "L" and req["handler"] == "liquid-cut" and req["chain"] == "bubble"
    assert req["eps_d"] == 0.2 and req["ratio"] == pytest.approx(0.9, rel=1e-3)
    assert req["status"] == "head" and 3.0 < req["head_volume_over_s3"] < 4.5
    # a head too large to delete: the event has no handler (L-unhandled)
    big = ev.EventSettings.from_dict({"liquid": {"enabled": True, "max_head_volume": 2.0},
                                      "post": {"open_neck_stop": False, "neck_floor": 2e-6}})
    req = _stops([bubble, cavity()], events=big)
    assert req["class"] == "L-unhandled" and req["status"] == "head-too-large"
    assert req["kind"] == "liquid-neck" and req["handler"] is None
    # the nearest its landing first, on either chain
    req = _stops([bubble_with_head(1e-3, r_n=0.095)[0], cavity_with_head(1e-3, r_n=0.05)])
    assert req["chain"] == "open" and req["ratio"] == pytest.approx(0.5, rel=1e-3)
    # liquid events off: a liquid neck at the cut's second-pinch test, below delta, whose head is
    # never classified: no handler (L-unhandled), and no head, s or direction
    off = ev.EventSettings.from_dict({})
    req = _stops([bubble_with_head(1e-3, r_n=0.3)[0], cavity()], events=off)
    assert req["kind"] == "liquid-neck" and req["detected_by"] == "secondary-pinch"
    assert req["neck_depth"] == pytest.approx(0.4, rel=1e-3)
    assert req["class"] == "L-unhandled" and req["handler"] is None
    assert not {"head", "s", "direction"} & set(req)


def test_the_handler_of_an_event_is_its_classs_and_an_unhandled_class_has_none():
    """The handler a request names is that of its class (``events.HANDLERS``), whatever found it:
    a fold that reaches a head too large to delete lands its neck as an L-unhandled event (no
    handler), not as a groove to fill."""
    assert ev.HANDLERS == {"W1": "ring", "L": "liquid-cut", "A-head": "liquid-cut",
                           "L-unhandled": None, "A-groove": "groove-fill", "G": "n-chain-split",
                           "B": None, "X": None}
    big = ev.EventSettings.from_dict({"liquid": {"enabled": True, "max_head_volume": 2.0},
                                      "post": {"open_neck_stop": False, "neck_floor": 2e-6}})
    bubble, apex = bubble_with_head(1e-3, r_n=0.2, r_w=0.54)
    req = _stops([bubble, cavity()], events=big, apexes=_apexes(apex))
    assert (req["kind"], req["class"], req["handler"]) == ("a-head", "L-unhandled", None)
    assert req["a_head"]["status"] == "head-too-large"
    assert req["a_head"]["r_n_over_s"] == pytest.approx(0.4, rel=1e-3)
    # beyond eps_max the head's neck does not land: the fold is a groove, whatever the head
    bubble, apex = bubble_with_head(1e-3, r_n=0.3, r_w=0.54)
    req = _stops([bubble, cavity()], events=big, apexes=_apexes(apex))
    assert (req["kind"], req["class"], req["handler"]) == ("groove", "A-groove", "groove-fill")


def test_a_fold_at_the_scale_of_the_mesh_stops_the_run_as_the_cuts_stops_do():
    """A spike of one node three spacings off the bubble (its spline simple) is a fold two node
    intervals long, a kink at the scale of the mesh: no groove to fill nor head to cut, but the
    cut's own stop (``self-contact-approach``, trigger ``fold-gap``, contact ``kink``), with and
    without liquid events."""
    from test_reconnect_geometry import _bubble, _cavity

    from faraday_singularity.selfcontact import first_contact

    bubble, cav = _bubble(), _cavity()
    P, k = bubble.points.copy(), 300
    t = P[k + 1] - P[k]
    h = math.hypot(*t)
    t /= h
    P[k + 1] = P[k] + 0.5 * h * t + 3.0 * h * np.array([-t[1], t[0]])
    P[k + 2] = P[k] + 1.3 * h * t
    spike = rc.Chain(P, bubble.end_types)
    assert first_contact(spike.points) is None
    cut = rc.cut_geometry([bubble, cav], 3.0, gas_before=1.0)
    legacy = _stops([spike, cav], cut=cut, events=None)
    assert legacy["reason"] == "self-contact-approach" and legacy["contact"] == "kink"
    for events in (ev.EventSettings(), LIQUID):
        stop = _stops([spike, cav], cut=cut, events=events)
        assert stop["reason"] == "self-contact-approach" and stop["trigger"] == "fold-gap"
        assert stop["contact"] == "kink" and stop["fold_pair"] == [k, k + 2]
        assert stop["fold_limit"] == legacy["fold_limit"]  # 3 h_elem, above kappa_g delta


def test_a_fold_closes_at_kappa_g_delta_and_one_that_reaches_a_head_lands_its_neck():
    """A groove 0.04 delta wide between the jet's bulb and the groove's outer wall closes at
    kappa_g delta = 0.05 delta (not at a narrower kappa_g): a fold pair across the apex of the
    groove, one node on the bulb, the head of the neck. With r_n/s = 0.4 <= eps_max the head's neck
    lands at once (a-head); at 0.6 it is a closing groove that names the head it reached."""
    bubble, apex = bubble_with_head(1e-3, r_n=0.2, r_w=0.54)
    req = _stops([bubble, cavity()], apexes=_apexes(apex))
    assert req["kind"] == "a-head" and req["class"] == "A-head" and req["handler"] == "liquid-cut"
    assert req["fold_gap"] == pytest.approx(4e-5, rel=1e-2) and req["fold_limit"] == 5e-5
    a, b = req["fold_pair"]
    assert a < apex < b and req["a_head"]["head"][0] < b <= req["a_head"]["head"][1]
    assert req["a_head"]["r_n_over_s"] == pytest.approx(0.4, rel=1e-3)
    assert req["jet"] == "bubble_top" and req["apex_index"] == apex
    bubble, apex = bubble_with_head(1e-3, r_n=0.3, r_w=0.54)
    req = _stops([bubble, cavity()], apexes=_apexes(apex))
    assert (req["kind"], req["class"], req["handler"]) == ("groove", "A-groove", "groove-fill")
    assert req["a_head"]["r_n_over_s"] == pytest.approx(0.6, rel=1e-3)
    narrow = ev.EventSettings.from_dict({"liquid": {"enabled": True}, "gap": {"kappa_g": 0.02},
                                         "post": {"open_neck_stop": False, "neck_floor": 2e-6}})
    assert _stops([bubble, cavity()], events=narrow) is None
    # kappa_g null: the collapse's absolute fold_stop in its place (liquid events off, a neck too
    # shallow for the cut's second-pinch test)
    legacy = ev.EventSettings.from_dict({"gap": {"kappa_g": None}})
    shallow, _ = bubble_with_head(1e-3, r_n=0.42, r_w=0.54)
    assert _stops([shallow, cavity()], events=legacy, fold_stop=1e-5) is None
    req = _stops([shallow, cavity()], events=legacy, fold_stop=1e-4)
    assert req["kind"] == "groove" and req["fold_limit"] == 1e-4 and "a_head" not in req


def _finger(chain, z_at, *, w=2e-4, L=2e-3, h=1e-4):
    """The bubble chain with a gas finger ``L`` long and ``w`` wide (node spacing ``h``) out of its
    lower sphere at the node nearest the height ``z_at``: a fold of the resolved surface, far from
    the jet, narrower relative to its limit than the groove around the jet. Returns the chain and
    the number of nodes it adds."""
    P = chain.points
    lower = np.arange(len(P)) < len(P) // 2
    k = int(np.argmin(np.where(lower, np.abs(P[:, 1] - z_at), np.inf)))
    t = P[k + 1] - P[k - 1]
    t /= np.hypot(*t)
    out = np.array([t[1], -t[0]]) * (1.0 if t[1] > 0 else -1.0)  # into the liquid, r growing
    base = P[k]
    side1 = [base - 0.5 * w * t + x * out for x in np.arange(h, L, h)]
    tip = [base + L * out + 0.5 * w * (math.cos(a) * t + math.sin(a) * out)
           for a in np.linspace(math.pi, 0.0, 5)[1:-1]]
    side2 = [base + 0.5 * w * t + x * out for x in np.arange(L, 0.5 * h, -h)]
    Q = np.vstack([P[:k], [base - 0.5 * w * t], side1, tip, side2, [base + 0.5 * w * t],
                   P[k + 1:]])
    return rc.Chain(Q, chain.end_types), len(Q) - len(P)


def test_a_fold_that_lands_a_head_comes_before_any_groove_on_either_chain():
    """Spec order: a fold that reaches the head of a tracked neck with r_n/s <= eps_max (A-head)
    before any closing groove (A-groove), whichever is closer to its limit: a groove on the other
    chain, or a closer fold on the same chain, which the fold of the whole chain (fold_gap's, one
    per chain) selects instead, does not hide it. Each head's own fold is that of fold_gap among
    the pairs with a node on it."""
    from faraday_singularity.selfcontact import fold_gap

    kw = {"ratio": 4.0, "stop": 5e-5, "stop_elements": 3.0, "search_factor": 4.0}
    bubble, apex = bubble_with_head(1e-3, r_n=0.2, r_w=0.54)
    alone = _stops([bubble, cavity()], apexes=_apexes(apex))
    assert alone["kind"] == "a-head" and alone["fold_gap"] / alone["fold_limit"] > 0.75
    (neck,) = ev.liquid_necks((("bubble", bubble),), LIQUID.liquid)
    whole = fold_gap(bubble.points[:, 0], bubble.points[:, 1], **kw)
    own = ev.head_fold(bubble.points, neck["head"], neck["index"], **kw)
    assert own == whole and list(own.pair) == alone["fold_pair"]
    # a groove closer to its limit on the open chain, the head's neck there not landing (0.6 s)
    open_ = cavity_with_head(1e-3, r_n=0.3, r_w=0.53)
    other = _stops([bubble_with_head(1e-3, r_n=0.3)[0], open_])
    assert other["kind"] == "groove" and other["chain"] == "open"
    assert other["fold_gap"] / other["fold_limit"] < 0.65
    req = _stops([bubble, open_], apexes=_apexes(apex))
    assert (req["kind"], req["class"], req["chain"]) == ("a-head", "A-head", "bubble")
    assert req["fold_pair"] == alone["fold_pair"]
    # a closer fold of the same chain, far from the jet: the chain's own fold, but not the event
    fingered, added = _finger(bubble, Z_APEX - 0.08)
    whole = fold_gap(fingered.points[:, 0], fingered.points[:, 1], **kw)
    assert whole.gap / max(5e-5, 3.0 * whole.h_elem) < 0.1 and whole.pair[1] < apex
    assert rc.fold_contact(whole.arc, whole.h_elem) == "fold"
    req = _stops([fingered, cavity()], apexes=_apexes(apex + added))
    assert (req["kind"], req["class"]) == ("a-head", "A-head")
    assert req["fold_pair"] == [k + added for k in alone["fold_pair"]]
    # with the head's neck too wide to land, the closer fold is the groove
    wide, apex_w = bubble_with_head(1e-3, r_n=0.3, r_w=0.54)
    fingered, added = _finger(wide, Z_APEX - 0.08)
    req = _stops([fingered, cavity()], apexes=_apexes(apex_w + added))
    assert req["kind"] == "groove" and req["fold_pair"] == list(fold_gap(
        fingered.points[:, 0], fingered.points[:, 1], **kw).pair) and "a_head" not in req


def test_the_branch_orders_invalid_states_then_the_window_then_the_events():
    """Precedence within one state: a state that is not valid stops first (a spline that contacts
    itself, then on a fresh mesh an unresolved tracked neck, the guard, the driver's check of the
    curved elements), then the end of the window (the bubble emptied), then the events (the
    bubble's tips meeting, a landed liquid neck, a closing fold, a gas neck, the bridge)."""
    from test_reconnect_geometry import kinked

    bubble, apex = bubble_with_head(1e-3, r_n=0.09, r_w=0.54)  # landed, and the groove closing
    chains = [bubble, cavity()]
    cut = rc.cut_geometry(chains, 3.0, gas_before=1.0)
    assert _stops(chains, apexes=_apexes(apex))["kind"] == "liquid-neck"  # L before the fold
    # a spline contact far from the jet: an invalid state first
    bad = rc.Chain(kinked(bubble.points, 300), bubble.end_types)
    stop = _stops([bad, chains[1]], cut=cut)
    assert stop["reason"] == "self-contact-approach" and stop["trigger"] == "spline-contact"
    # the driver's check of the curved elements: after the guard, before the window and events
    calls = []

    def invalid():
        calls.append(1)
        return {"reason": "interface-unresolved", "trigger": "curved-jacobian"}

    guard = {"fresh": True, "h_floor": 1e-9, "refine_ratio": 1.5}
    assert _stops(chains, guard=guard, invalid=invalid)["trigger"] == "curved-jacobian"
    assert _stops([bad, chains[1]], cut=cut, guard=guard,
                  invalid=invalid)["trigger"] == "spline-contact"
    assert len(calls) == 1
    # an unresolved neck on a fresh mesh (r_n below 4 element lengths), on a fresh mesh only
    thin, _ = bubble_with_head(1e-3, r_n=0.02)
    stop = _stops([thin, chains[1]], guard=guard, invalid=invalid)
    assert stop["reason"] == "liquid-neck-unresolved" and stop["fresh"]
    assert stop["limit"] == pytest.approx(4 * stop["h_loc"]) and stop["r_n"] < stop["limit"]
    assert _stops([thin, chains[1]], guard=dict(guard, fresh=False))["kind"] == "liquid-neck"
    # the bubble emptied: the end of the window, before any event
    full = rc.CutGeometry(**dict(cut.to_dict(), bubble_volume=1e3 * cut.bubble_volume))
    stop = _stops(chains, cut=full)
    assert stop["reason"] == "bubble-vanishing" and stop["window_end"] is True
    # the bubble's tips within delta: the jet piercing the bubble, before the landed neck
    length = float(bubble.points[-1, 1] - bubble.points[0, 1])
    req = _stops(chains, settings=rc.ReconnectSettings(delta=1.01 * length))
    assert req["kind"] == "piercing" and req["class"] == "W1" and req["handler"] == "ring"
    assert req["bubble_length"] == pytest.approx(length)


def test_the_guard_of_a_mesh_stepped_on_comes_after_the_events_as_in_the_cuts_stops():
    """An element of the jet's root bent by 130 degrees, beyond edge_turn_stop on any mesh, in a
    state with a landed liquid neck: on a freshly built mesh the guard stops the run first (a
    rebuild that is not valid); on a mesh a step has run on the event comes first, as the cut's
    own stops report the neck before the guard, and the guard stops a state without an event."""
    from test_reconnect_geometry import _bent

    bubble, apex = bubble_with_head(1e-3, r_n=0.09)
    bent = rc.Chain(_bent(bubble.points, (apex + 200) // 2, turn_deg=130.0), bubble.end_types)
    chains = [bent, cavity()]
    cut = rc.cut_geometry([bubble, chains[1]], 3.0, gas_before=1.0)
    stepped = {"fresh": False, "h_floor": 1e-3 / 512, "refine_ratio": 1.5}
    fresh = dict(stepped, fresh=True)
    legacy = _stops(chains, cut=cut, events=None, guard=stepped)
    assert legacy["reason"] == "secondary-pinch-approach" and legacy["neck_side"] == "liquid"
    for events, cls in ((LIQUID, "L"), (ev.EventSettings(), "L-unhandled")):
        req = _stops(chains, cut=cut, events=events, guard=stepped)
        assert (req["reason"], req["kind"], req["class"]) == (ev.EVENT_REQUEST, "liquid-neck", cls)
        stop = _stops(chains, cut=cut, events=events, guard=fresh)
        assert (stop["reason"], stop["trigger"]) == ("interface-unresolved", "edge-turn")
    # no event: the guard of the stepped mesh, last
    bubble, apex = bubble_with_head(1e-3, r_n=0.3)
    bent = rc.Chain(_bent(bubble.points, (apex + 200) // 2, turn_deg=130.0), bubble.end_types)
    assert _stops([bubble, chains[1]], cut=cut, guard=stepped) is None
    stop = _stops([bent, chains[1]], cut=cut, guard=stepped)
    assert (stop["reason"], stop["trigger"]) == ("interface-unresolved", "edge-turn")
    assert not stop["fresh"]


def test_every_change_of_topology_the_cut_stopped_on_is_an_event_with_the_block():
    """On the states of ``test_reconnect_geometry``'s stops after a cut: the bridge closing, the
    chains approaching, the bubble's tips meeting, a gas neck, a liquid neck (liquid events off)
    and a closing fold are events, each a request for its handler; the chains not two and the
    bubble emptying stay stops."""
    from test_reconnect_geometry import _bubble, _cavity

    off = ev.EventSettings.from_dict({})
    s = rc.ReconnectSettings(delta=1e-3)
    bubble, cav = _bubble(), _cavity()
    cut = rc.cut_geometry([bubble, cav], 3.0, gas_before=1.0)

    def stops(chains, h_min=1e-5):
        return rc.post_cut_stops(chains, cut, s, h_min, events=off)

    assert stops([bubble, cav]) is None
    closing = rc.Chain(cav.points - [0.0, 0.0095], ("axis", "fixed"))
    req = stops([bubble, closing])
    assert (req["kind"], req["class"], req["handler"]) == ("bridge-closing", "B", None)
    beside = rc.Chain(np.array([[0.0, 4.69], [0.0103, 4.6905], [0.0102, 4.64], [0.0105, 4.6],
                                [0.02, 4.6], [0.5, 5.0], [R_C, 6.0]]), ("axis", "fixed"))
    req = stops([bubble, beside], 1e-4)
    assert (req["kind"], req["class"]) == ("chain-contact", "X") and req["chain_gap"] < 3e-4
    small = rc.Chain(np.array([[0.0, 4.6], [1e-3, 4.60025], [0.0, 4.6005]]), ("axis", "axis"))
    req = rc.post_cut_stops([small, cav], rc.cut_geometry([small, cav], 3.0, gas_before=1.0), s,
                            1e-5, events=off)
    assert (req["kind"], req["class"], req["handler"]) == ("piercing", "W1", "ring")
    assert stops([bubble]) == {"reason": "topology-changed", "chains": 1}
    thin = rc.Chain(bubble.points * [0.05, 1.0], ("axis", "axis"))
    assert stops([thin, cav])["reason"] == "bubble-vanishing"
    req = stops([_bubble(waist=0.95), cav])
    assert (req["kind"], req["class"], req["handler"]) == ("gas-neck", "G", "n-chain-split")
    assert req["neck_side"] == "gas" and req["r_neck"] == pytest.approx(5e-4, rel=1e-3)
    from test_reconnect_geometry import _bubble_with_jet

    jet_cav = _cavity(z_tip=4.70)
    req = rc.post_cut_stops([_bubble_with_jet(dent=0.6), jet_cav],
                            rc.cut_geometry([_bubble_with_jet(), jet_cav], 3.0, gas_before=1.0), s,
                            1e-6, events=off)
    assert req["kind"] == "liquid-neck" and req["neck_side"] == "liquid"
    # with liquid events on the same neck is tracked, its head (a jet 80 radii long) too large to
    # delete: no event until it lands, then one without a handler
    req = rc.post_cut_stops([_bubble_with_jet(dent=0.6), jet_cav],
                            rc.cut_geometry([_bubble_with_jet(), jet_cav], 3.0, gas_before=1.0), s,
                            1e-6, events=LIQUID)
    assert req is None
    req = rc.post_cut_stops([_bubble_with_jet(dent=0.9), jet_cav],
                            rc.cut_geometry([_bubble_with_jet(), jet_cav], 3.0, gas_before=1.0), s,
                            1e-6, events=LIQUID)
    assert req["class"] == "L-unhandled" and req["status"] == "head-too-large"


def test_the_events_are_the_same_in_units_of_delta_at_every_delta():
    """The jet, its neck and its groove scaled with delta by 2^-k (k = 0..7) inside a bubble and a
    cavity of fixed size, with the node spacing there: the same tracked neck, landing, unresolved
    neck on a fresh mesh, and fold (A-head and A-groove) in units of delta, at every delta."""
    ref = None
    for k in range(8):
        d = 1e-3 * 2.0 ** -k
        out = []
        for r_n, r_w in ((0.3, 1.0), (0.09, 1.0), (0.2, 0.54), (0.3, 0.54)):
            bubble, apex = bubble_with_head(d, r_n=r_n, r_w=r_w)
            req = _stops([bubble, cavity()], delta=d, apexes=_apexes(apex))
            out.append(None if req is None else (
                req["kind"], round(req.get("ratio", 0.0), 6),
                round(req.get("fold_gap", 0.0) / d, 6), round(req.get("fold_limit", 0.0) / d, 6),
                (req["fold_pair"][1] if "fold_pair" in req else req["index"]) - apex))
        thin, _ = bubble_with_head(d, r_n=0.02)
        stop = _stops([thin, cavity()], delta=d,
                      guard={"fresh": True, "h_floor": d / 512, "refine_ratio": 1.5})
        out.append((stop["reason"], round(stop["r_n"] / stop["h_loc"], 6)))
        assert out[0] is None and out[1][0] == "liquid-neck"
        assert out[2][0] == "a-head" and out[3][0] == "groove"
        assert out[4][0] == "liquid-neck-unresolved"
        if ref is None:
            ref = out
        assert out == ref, (k, out, ref)


# --- the stop states of runs after a cut ----------------------------------------------------------


runs = pytest.mark.skipif(not RUNS, reason="FARADAY_POSTCUT_RUNS not set")


def _stop_state(run: Path):
    """The chains of the state a run after a cut stopped on (its ``stop`` profiles, as the driver
    hands them to the stops: the ends on the axis put on it), its stop, its cut radius and fold
    settings."""
    status = json.loads((run / "run_status.json").read_text())
    params = json.loads((run / "parameters.json").read_text())
    stop = status["collapse"]["stop"]
    path = None
    for p in sorted((run / "profiles").glob("profile_*[0-9].txt")):
        head = json.loads(p.read_text().splitlines()[0][2:])
        if head.get("reason") == "stop":
            path = p
    assert path is not None, run
    o, b = np.loadtxt(path)[:, :2], np.loadtxt(path.with_name(path.stem + "_bubble.txt"))[:, :2]
    o[0, 0], b[0, 0], b[-1, 0] = 0.0, 0.0, 0.0
    chains = [rc.Chain(b, ("axis", "axis")), rc.Chain(o, ("axis", "fixed"))]
    return chains, stop, params


@runs
@pytest.mark.parametrize("run", RUNS, ids=[f"run{k}" for k in range(len(RUNS))])
def test_the_stop_states_of_runs_after_a_cut_go_on_with_the_block(run):
    """Each stop state of a run after a cut, on a liquid neck at the root of a cut-born jet
    (``secondary-pinch-approach``, liquid side) or on the groove around such a jet
    (``annular-contact-approach``): the stops of the cut find it again, while with liquid events
    on and the fold limit scaled with delta (0.05 delta) the run goes on: the neck is tracked with
    its head (3-10 s^3), not landed, resolved on a fresh mesh. A fold limit as wide as the run's
    absolute one finds the run's fold again, an A-groove whose bulb side lies on the head of the
    neck between the apex and it; a neck landed at a wider eps_d is a liquid-neck event."""
    chains, stop, params = _stop_state(run)
    record = json.loads((run / "cut_diagnostics.json").read_text())
    cut, h_tip = rc.CutGeometry(**record["geometry"]), record["sizing"]["h_tip"]
    s = rc.ReconnectSettings.from_dict(params["reconnect"])
    c = params["collapse"]
    kw = {"neck_elements": c["neck_stop_elements"], "fold_ratio": c["fold_ratio"],
          "fold_stop": c["fold_stop"], "fold_stop_elements": c["fold_stop_elements"],
          "fold_search": 4.0}
    apexes = ({stop["chain"]: [{"tip": stop["jet"], "index": stop["apex_index"]}]}
              if stop["reason"] == "annular-contact-approach" else None)
    legacy = rc.post_cut_stops(chains, cut, s, h_tip, apexes=apexes, **kw)
    assert legacy is not None and legacy["reason"] == stop["reason"]
    if stop["reason"] == "secondary-pinch-approach":
        assert legacy["r_neck"] == pytest.approx(stop["r_neck"], rel=1e-9)
    else:
        assert legacy["fold_pair"] == stop["fold_pair"]
    assert rc.post_cut_stops(chains, cut, s, h_tip, events=LIQUID, **kw) is None
    guard = {"fresh": True, "h_floor": c["h_floor"], "refine_ratio": c["refine_ratio"]}
    assert rc.post_cut_stops(chains, cut, s, h_tip, events=LIQUID, guard=guard, **kw) is None
    named = (("bubble", chains[0]), ("open", chains[1]))
    necks = ev.liquid_necks(named, LIQUID.liquid)
    assert necks and all(x["status"] == "head" and not x["landed"] for x in necks)
    assert all(3.0 < x["head_volume_over_s3"] < 10.0 for x in necks)
    if stop["reason"] == "annular-contact-approach":
        wide = ev.EventSettings.from_dict({"liquid": {"enabled": True},
                                           "gap": {"kappa_g": c["fold_stop"] / s.delta},
                                           "post": {"open_neck_stop": False, "neck_floor": 2e-6}})
        req = rc.post_cut_stops(chains, cut, s, h_tip, events=wide, apexes=apexes, **kw)
        assert req["kind"] == "groove" and req["fold_pair"] == stop["fold_pair"]
        assert req["jet"] == stop["jet"]
        head = req["a_head"]
        assert head["chain"] == stop["chain"] and head["r_n_over_s"] > wide.liquid.eps_max
        assert stop["apex_index"] < head["index"] < stop["fold_pair"][1] <= head["head"][1]
    neck = min(necks, key=lambda x: x["r_n"] / x["s"])
    eps = 1.01 * neck["r_n"] / neck["s"]
    landed = ev.EventSettings.from_dict({"liquid": {"enabled": True, "eps_d": eps,
                                                    "eps_max": max(eps, 0.5)},
                                         "post": {"open_neck_stop": False, "neck_floor": 2e-6}})
    req = rc.post_cut_stops(chains, cut, s, h_tip, events=landed, **kw)
    assert req["kind"] == "liquid-neck" and req["index"] == neck["index"]
    assert req["chain"] == neck["chain"]


# --- the floors after the cut, in units of delta --------------------------------------------------


def test_the_floor_of_the_guard_is_the_same_in_units_of_delta_with_a_floor_scaled_with_delta():
    """An element of the jet's root bent by 40 degrees asks for a refinement; the guard stops the
    run on it (feature-below-floor) once its chord lies within 1.5 floors. With the floor scaled
    with delta (delta/512 none, delta/128 every one of the 2^-k scalings stops) the outcome is the
    same at every delta; with the collapse's absolute floor it changes with delta."""
    from test_reconnect_geometry import _bent

    outcomes = {"d/512": set(), "d/128": set(), "absolute": []}
    for k in range(8):
        d = 1e-3 * 2.0 ** -k
        bubble, apex = bubble_with_head(d, r_n=0.3)
        i = apex + 200  # down the jet's root, between the apex and the neck
        bent = rc.Chain(_bent(bubble.points, i // 2, turn_deg=40.0), bubble.end_types)
        chains = [bent, cavity()]
        cut = rc.cut_geometry([bubble, chains[1]], 3.0, gas_before=1.0)
        for name, h_floor in (("d/512", d / 512), ("d/128", d / 128), ("absolute", 7.8125e-7)):
            stop = _stops(chains, delta=d, cut=cut,
                          guard={"fresh": False, "h_floor": h_floor, "refine_ratio": 1.5})
            got = None if stop is None else (stop["reason"], stop["trigger"])
            if name == "absolute":
                outcomes[name].append(got)
            else:
                outcomes[name].add(got)
    assert outcomes["d/512"] == {None}
    assert outcomes["d/128"] == {("interface-unresolved", "feature-below-floor")}
    assert outcomes["absolute"][0] is None and outcomes["absolute"][-1] is not None


def test_a_neck_after_the_cut_is_followed_below_the_cut_radius_with_the_block():
    """The region of a neck is floored at the cut's focus size, delta/16, without the block's
    post.neck_floor; with it a neck of 0.1 delta is meshed at r_n/eps, and post.
    neck_elements_per_scale sets eps there alone."""
    from test_reconnect_geometry import _post_cut_segs, _regions

    s = rc.ReconnectSettings(delta=1e-3)
    segs = _post_cut_segs()
    P = np.column_stack([segs[1]["r"], segs[1]["z"]])
    k = 300
    P[k - 5:k + 6, 0] = np.minimum(P[k - 5:k + 6, 0], 1e-4 + 1e-6 * np.abs(np.arange(-5, 6)))
    segs[1]["r"] = P[:, 0]
    cut = {"delta": 1e-3, "rmin": 1e-3}
    legacy = _regions(segs, cut, s)["neck_open"]
    assert legacy["target"] == pytest.approx(6.25e-5) and legacy["floor"] == 6.25e-5
    assert _regions(segs, cut, s, neck_floor=None, neck_elements_per_scale=None)["neck_open"] \
        == legacy
    g = _regions(segs, cut, s, neck_floor=1e-6)["neck_open"]
    assert g["target"] == pytest.approx(1e-4 / 16) and g["floor"] == 1e-6
    g = _regions(segs, cut, s, neck_floor=1e-6, neck_elements_per_scale=32.0)
    assert g["neck_open"]["target"] == pytest.approx(1e-4 / 32)
    assert g["cavity_tip"] == _regions(segs, cut, s)["cavity_tip"]  # the rest at its own eps
