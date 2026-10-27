"""A curled head, without a finite-element problem (``events.head_sides``, ``events.liquid_necks``,
``events.far_head_request``): a cut-born head whose rim curls down and back towards its stem, past
the height of its root neck (the narrowest point of the stem below it), with a pocket of gas
between the rim's lip and the stem, as r1 of the delta triple met it at tau/delta^2 6.45.

* on the open chain (the upward jet's head) and on the bubble (the downward jet's head, the curl
  mirrored): the root neck is a tracked neck with a head (``status`` ``head``, ``curled``, its
  ``curl_depth`` the rim's depth below the neck in neck radii), the head the chain from the tip to
  the root neck, its volume the half-section's with the curled rim;
* the same head with its rim above the neck is a head as before, with no ``curled`` key; a side
  whose nodes pass the neck's height inside the neck's radius is no head;
* the routing to the far-head cut (``far_head_request``): a gas neck or a closing groove at the
  pocket's mouth, on the curled head, is the cut of its root neck; never where the site or any node
  of the head but its neck is a core node of the open chain; a site off the head is no cut.
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

from test_events_stops import R_C, Z_APEX, _nodes  # noqa: E402

from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402

DELTA = 1e-3
LIQUID = ev.EventSettings.from_dict({"liquid": {"enabled": True},
                                     "post": {"open_neck_stop": False, "neck_floor": 2e-6}})
R_N, Z_N = 0.28, -1.15  # the root neck, in delta, z from the head's tip


def _arc(c, rad, a0, a1, m):
    t = np.linspace(a0, a1, m)
    return np.column_stack([c[0] + rad * np.cos(t), c[1] + rad * np.sin(t)])


def _hermite(p0, p1, m):
    """``r(z)`` from ``p0`` to ``p1`` (``(r, z)``) with zero slope ``dr/dz`` at both ends."""
    t = np.linspace(0.0, 1.0, m)
    return np.column_stack([p0[0] + (p1[0] - p0[0]) * (3 * t**2 - 2 * t**3),
                            p0[1] + (p1[1] - p0[1]) * t])


def curled_head(*, rim_bottom=-1.2, m=4000):
    """The head in units of delta, its tip on the axis at ``z = 0``, from the tip: the cap out to
    the rim (``r`` 0.85 at ``z`` -0.8), the rim down to -1.0, its bottom round to ``rim_bottom``
    at ``r`` 0.65 and back up to the lip (``r`` 0.45), the lip up to -0.5, the pocket's ceiling
    over to the stem (``r`` 0.30), the stem down to the root neck (``R_N`` at ``Z_N``) and on,
    widening, to ``r`` 0.35 at -1.8. The pocket's mouth is 0.15 wide."""
    th = np.linspace(0.0, 0.5 * math.pi, m)
    depth = -1.0 - rim_bottom
    pieces = [np.column_stack([0.85 * np.sin(th), -0.8 * (1.0 - np.cos(th))]),
              np.column_stack([np.full(m, 0.85), np.linspace(-0.8, -1.0, m)]),
              # the rim's bottom: half an ellipse from (0.85, -1) down to rim_bottom and up to the lip
              np.column_stack([0.65 + 0.2 * np.cos(np.linspace(0.0, -math.pi, m)),
                               -1.0 + depth * np.sin(np.linspace(0.0, -math.pi, m))]),
              np.column_stack([np.full(m, 0.45), np.linspace(-1.0, -0.5, m)]),
              _arc((0.375, -0.5), 0.075, 0.0, math.pi, m),
              _hermite((0.30, -0.5), (R_N, Z_N), m), _hermite((R_N, Z_N), (0.35, -1.8), m)]
    return np.vstack([pieces[0]] + [p[1:] for p in pieces[1:]])


def _resample(dense, delta, h=1 / 128, h_far=1e-3, start=None):
    sarc, nodes = _nodes(dense, sarc_at(dense, start) if start is not None else 0.0, delta,
                         h * delta, h_far)
    return np.column_stack([np.interp(nodes, sarc, dense[:, 0]),
                            np.interp(nodes, sarc, dense[:, 1])])


def sarc_at(dense, k):
    return float(np.sum(np.hypot(*np.diff(dense[:k + 1], axis=0).T)))


def open_chain_with_curled_head(delta=DELTA, **kw):
    """The open chain from the upward jet's curled head on the axis (:func:`curled_head`, the tip at
    ``z = Z_APEX + 4 delta``) down its stem to the jet's base (``r`` 0.5 delta), round the groove's
    apex and up the cavity's wall to the container's wall. Returns the chain and the index of the
    root neck (the stem's narrowest node below the head)."""
    head = curled_head(**kw)
    m = 2000
    stem = _hermite((0.35, -1.8), (0.5, -3.0), m)
    base = _arc((0.6, -3.0), 0.1, math.pi, 2.0 * math.pi, m)  # the groove's apex, below
    wall = np.column_stack([np.linspace(0.7, 3.0, m), np.linspace(-3.0, -1.0, m)])
    local = np.vstack([head, stem[1:], base[1:], wall[1:]])
    z_tip = Z_APEX + 4.0 * delta
    feat = np.column_stack([delta * local[:, 0], z_tip + delta * local[:, 1]])
    far = np.column_stack([np.linspace(feat[-1, 0], R_C, 400)[1:],
                           np.linspace(feat[-1, 1], feat[-1, 1] + 0.3, 400)[1:]])
    dense = np.vstack([feat, far])
    # nodes from the far end at 1e-3 down to delta/128 over the feature
    rev = dense[::-1]
    sarc, nodes = _nodes(rev, sarc_at(rev, len(far)), delta, delta / 128, 1e-3)
    P = np.column_stack([np.interp(nodes, sarc, rev[:, 0]), np.interp(nodes, sarc, rev[:, 1])])
    P = P[::-1].copy()
    P[0, 0] = 0.0
    return rc.Chain(P, ("axis", "fixed")), _root_index(P, z_tip + delta * Z_N)


def bubble_with_curled_head(delta=DELTA, R_b=0.05, **kw):
    """The bubble chain, from its bottom up a sphere of radius ``R_b`` to the roof, inwards along the
    roof to the downward jet's stem and down it to the jet's curled head, mirrored
    (:func:`curled_head`, its rim curling up past the root neck), the tip on the axis. Returns the
    chain and the index of the root neck."""
    head = curled_head(**kw)[::-1] * np.array([1.0, -1.0])  # from the stem's end to the tip
    m = 2000
    top = 2.3  # the roof above the head's tip, in delta
    q = 0.2
    roof = np.column_stack([np.linspace(3.0, 0.35 + q, m), np.full(m, top)])
    corner = _arc((0.35 + q, top - q), q, 0.5 * math.pi, math.pi, m)
    stem = np.column_stack([np.full(m, 0.35), np.linspace(top - q, 1.8, m)])
    local = np.vstack([roof, corner[1:], stem[1:], head[1:]])
    z_tip = Z_APEX - top * delta
    feat = np.column_stack([delta * local[:, 0], z_tip + delta * local[:, 1]])
    r1, zroof = feat[0]
    zs = zroof - math.sqrt(R_b**2 - r1**2)
    th = np.linspace(0.0, math.pi - math.asin(r1 / R_b), 80000)
    sphere = np.column_stack([R_b * np.sin(th), zs - R_b * np.cos(th)])
    dense = np.vstack([sphere, feat[1:]])
    sarc, nodes = _nodes(dense, sarc_at(dense, len(sphere) - 1), delta, delta / 128, 1e-3)
    P = np.column_stack([np.interp(nodes, sarc, dense[:, 0]), np.interp(nodes, sarc, dense[:, 1])])
    P[0, 0] = P[-1, 0] = 0.0
    return rc.Chain(P, ("axis", "axis")), _root_index(P, z_tip - delta * Z_N)


def _root_index(P, z_root):
    """The stem's narrowest node near the root neck's height."""
    near = np.flatnonzero(np.abs(P[:, 1] - z_root) < 0.2 * DELTA)
    near = near[P[near, 0] < 0.32 * DELTA]
    return int(near[np.argmin(P[near, 0])])


def _neck(chain, name, index):
    named = [(name, chain)]
    necks = ev.liquid_necks(named, LIQUID.liquid)
    return next(x for x in necks if abs(x["index"] - index) <= 2), necks


@pytest.mark.parametrize("which", ["open", "bubble"])
def test_a_head_whose_rim_curls_past_its_root_neck_is_a_curled_head(which):
    chain, i_root = (open_chain_with_curled_head() if which == "open"
                     else bubble_with_curled_head())
    x, _ = _neck(chain, which, i_root)
    assert x["status"] == "head" and x["curled"] is True
    assert x["r_n"] == pytest.approx(R_N * DELTA, rel=0.02)
    # the rim's bottom 0.05 delta past the neck: 0.18 neck radii, beyond the tolerance 0.05
    assert x["curl_depth"] == pytest.approx(0.05 / R_N, rel=0.1)
    assert x["curl_depth"] > ev.HEAD_SIDE_TOL
    lo, hi = x["head"]
    n = len(chain.points)
    assert (lo, hi) == ((0, x["index"]) if which == "open" else (x["index"], n - 1))
    assert x["direction"] == (1.0 if which == "open" else -1.0)
    assert x["s"] == pytest.approx(0.85 * DELTA, rel=0.01)
    assert 0.0 < x["head_volume"] <= LIQUID.liquid.max_head_volume * x["s"] ** 3
    assert x["r_n"] / x["s"] == pytest.approx(R_N / 0.85, rel=0.03)


def test_the_same_head_with_its_rim_above_the_neck_is_a_head_as_before():
    chain, i_root = open_chain_with_curled_head(rim_bottom=-1.1)  # 0.05 delta above the neck
    x, _ = _neck(chain, "open", i_root)
    assert x["status"] == "head" and "curled" not in x and "curl_depth" not in x


def test_a_side_passing_the_neck_inside_its_radius_is_no_head():
    """The head's side bent below the neck's height at a radius inside the neck's (a chain that
    turns back under the neck rather than a rim curling outside it) is no head."""
    sides = ev.head_sides(_bent_under(), ("axis", "fixed"), 2, is_bubble=False)
    before = sides[0]
    assert not before["one_sided"] and not before["curled"] and not before["qualifies"]


def _bent_under():
    """A short open chain from the axis whose first node lies below the neck (node 2) at a radius
    inside it."""
    return np.array([[0.0, 0.99], [0.5e-3, 1.02], [1e-3, 1.0], [2e-3, 0.95], [3e-3, 0.9],
                     [4e-3, 0.95], [5e-3, 1.0]])


def _gas_neck(chain_name, index):
    return {"reason": ev.EVENT_REQUEST, "kind": "gas-neck", "class": "G",
            "handler": ev.HANDLERS["G"], "chain": chain_name, "index": int(index)}


def _lip_node(P, delta=DELTA, which="open"):
    """A node of the lip's inner side, the pocket's mouth (``r`` 0.45 delta, half way up)."""
    z_mid = (Z_APEX + 4.0 * delta - 0.75 * delta) if which == "open" else \
        (Z_APEX - 2.3 * delta + 0.75 * delta)
    d = np.hypot(P[:, 0] - 0.45 * delta, P[:, 1] - z_mid)
    return int(np.argmin(d))


@pytest.mark.parametrize("cls", ["G", "A-groove"])
def test_an_event_at_the_pocket_of_a_curled_far_head_is_the_cut_of_its_root_neck(cls):
    chain, i_root = open_chain_with_curled_head()
    x, necks = _neck(chain, "open", i_root)
    P = np.asarray(chain.points)
    k = _lip_node(P)
    req = _gas_neck("open", k) if cls == "G" else {
        "reason": ev.EVENT_REQUEST, "kind": "groove", "class": "A-groove",
        "handler": ev.GROOVE_HANDLER, "chain": "open", "fold_pair": [k, x["index"] - 3]}
    core = np.zeros(len(P), dtype=bool)
    core[x["index"]:] = True  # the core from the root neck on: the stem, the base, the walls
    got = ev.far_head_request(req, necks, core)
    assert (got["kind"], got["class"], got["handler"]) == ("far-head", "F-head", "liquid-cut")
    assert got["index"] == x["index"] and got["far_event"]["class"] == cls
    assert got["curled"] is True and got["head"] == x["head"]
    # bubble: every node is far
    bub, j_root = bubble_with_curled_head()
    y, bnecks = _neck(bub, "bubble", j_root)
    kb = _lip_node(np.asarray(bub.points), which="bubble")
    assert y["head"][0] <= kb <= y["head"][1]
    got = ev.far_head_request(_gas_neck("bubble", kb), bnecks)
    assert got["index"] == y["index"] and got["chain"] == "bubble"


def test_the_cut_of_a_far_head_never_takes_a_node_of_the_core():
    chain, i_root = open_chain_with_curled_head()
    x, necks = _neck(chain, "open", i_root)
    P = np.asarray(chain.points)
    k = _lip_node(P)
    n = len(P)
    on_site = np.zeros(n, dtype=bool)
    on_site[k] = True
    assert ev.far_head_request(_gas_neck("open", k), necks, on_site) is None
    on_head = np.zeros(n, dtype=bool)
    on_head[x["index"] - 5] = True  # a node of the stem inside the head, not the site
    assert ev.far_head_request(_gas_neck("open", k), necks, on_head) is None
    assert ev.far_head_request(_gas_neck("open", k), necks, None) is None  # no core: no cut
    # a site off the head (the cavity's wall beyond the root neck) is no cut, nor the neck itself
    beyond = np.zeros(n, dtype=bool)
    assert ev.far_head_request(_gas_neck("open", x["index"] + 40), necks, beyond) is None
    assert ev.far_head_request(_gas_neck("open", x["index"]), necks, beyond) is None
