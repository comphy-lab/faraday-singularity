"""Contract tests of the geometry of the cut of a liquid neck (``events.truncate_chain``): the head
removed, the stump closed by a half-spheroid at the neck, the hard checks of the cap, and the
removed region. No finite-element problem is built.

The synthetic states are those of ``test_events_stops``: a jet hanging from the bubble's top and
one rising from the cavity's tip, each with a neck at its root and a bulb (its head) whose tip is
on the axis, every length of the jet in units of the cut radius ``delta``; and a jet whose head is
a hemisphere on a cylinder on a cone frustum, whose removed volume and areas are known exactly.

The tests on the stop states of runs after a cut run when ``FARADAY_POSTCUT_RUNS`` names their run
directories (``os.pathsep``-separated), as in ``test_events_stops``.
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

from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402

import test_events_stops as stops  # noqa: E402

R_C = stops.R_C
CUT = {"eta": 1.0, "elements_per_scale": 8.0, "min_angle": 30.0, "reservoir_depth": 3.0}


def _necks(named):
    return ev.liquid_necks(named, stops.LIQUID.liquid)


def _neck(named, chain):
    return next(x for x in _necks(named) if x["chain"] == chain)


# --- the cap ----------------------------------------------------------------------------------------


def test_the_cap_is_a_half_spheroid_tangent_to_the_neck_and_square_to_the_axis():
    r_n, z_n = 2e-4, 4.6
    for direction, eta in ((-1.0, 1.0), (1.0, 1.0), (1.0, 0.5)):
        cap = ev.hemispheroid_cap(r_n, z_n, direction, eta, 32)
        assert len(cap) == 32 and tuple(cap[-1]) == (0.0, z_n + direction * eta * r_n)
        # on the spheroid, between the neck's plane and the tip, on the side of the head
        assert np.allclose((cap[:, 0] / r_n) ** 2 + ((cap[:, 1] - z_n) / (eta * r_n)) ** 2, 1.0)
        assert np.all(direction * (cap[:, 1] - z_n) > 0.0)
        # leaving the neck parallel to the axis, meeting the axis at a right angle: the chords of
        # the first and the last interval within half the angle the interval subtends
        d0 = cap[0] - (r_n, z_n)
        assert abs(d0[0]) < 0.1 * abs(d0[1])
        d1 = cap[-1] - cap[-2]
        assert abs(d1[1]) < 0.05 * abs(d1[0])
        # evenly spaced along the arc
        seg = np.hypot(*np.diff(np.vstack([[r_n, z_n], cap]), axis=0).T)
        assert np.ptp(seg) < 1e-3 * seg.mean()
    assert ev.cap_arc_length(r_n, 1.0) == pytest.approx(0.5 * math.pi * r_n, rel=1e-7)
    # one interval per r_n/eps of arc, at least CAP_MIN_POINTS
    assert ev.cap_point_count(r_n, 1.0, 8.0) == ev.CAP_MIN_POINTS == 16
    assert ev.cap_point_count(r_n, 1.0, 16.0) == math.ceil(0.5 * math.pi * 16)


def test_the_points_of_a_cap_grow_from_the_spacing_of_the_neck_to_its_own():
    """From the neck's node spacing the intervals grow (or shrink) by at most CAP_GROWTH per point to
    the cap's own spacing; the last is between half and one and a half times the one before it."""
    L, h = 1.0, 0.05
    for h0 in (0.002, 0.05, 0.2):
        u = ev.cap_spacing(L, h0, h)
        d = np.diff(np.concatenate([[0.0], u]))
        assert u[-1] == L and np.all(d > 0.0) and d[0] == pytest.approx(min(h0, L))
        ratio = d[1:-1] / d[:-2]
        assert np.all(ratio <= ev.CAP_GROWTH * (1 + 1e-12))
        assert np.all(ratio >= 1.0 / ev.CAP_GROWTH * (1 - 1e-12))
        assert 0.5 <= d[-1] / d[-2] <= 1.5
        assert np.all(d[:-1] <= max(h0, h) * (1 + 1e-12))
    with pytest.raises(ValueError, match="positive"):
        ev.cap_spacing(L, 0.0, h)


# --- the cut of either chain ------------------------------------------------------------------------


def test_a_liquid_cut_removes_the_head_of_either_chain_and_keeps_the_neck_as_the_splice():
    """The jet hanging from the bubble's top loses its head, the nodes after its neck (the bubble's
    top moves up to z_n - r_n); the jet rising from the cavity's tip loses the nodes before its neck
    (the cavity's tip moves down to z_n + r_n). The nodes away from the head are kept as they were,
    the neck node is the splice, the cap ends on the axis, its points start at the spacing of the
    neck on the side kept, and the chains a mesh is built on stay valid."""
    d = 1e-3
    bubble, _ = stops.bubble_with_head(d, r_n=0.09)
    cavity = stops.cavity_with_head(d, r_n=0.09)
    named = (("bubble", bubble), ("open", cavity))
    for name, end, tip, direction in (("bubble", -1, "bubble_top", -1.0),
                                      ("open", 0, "cavity_tip", 1.0)):
        neck = _neck(named, name)
        cut = ev.truncate_chain(named, neck, **CUT)
        old = dict(named)[name].points
        i = neck["index"]
        assert (cut.chain, cut.end, cut.tip, cut.direction) == (name, end, tip, direction)
        assert cut.r_n == neck["r_n"] and cut.z_n == neck["z_n"] and cut.s == neck["s"]
        kept = cut.origin >= 0
        assert np.array_equal(cut.points[kept], old[cut.origin[kept]])
        assert np.array_equal(cut.origin[kept], np.arange(i + 1) if end == -1 else
                              np.arange(i, len(old)))
        assert np.array_equal(cut.points[~kept], cut.cap if end == -1 else cut.cap[::-1])
        assert tuple(cut.points[end]) == (0.0, neck["z_n"] + direction * neck["r_n"])
        assert cut.z_tip_new == pytest.approx(neck["z_n"] + direction * neck["r_n"])
        assert cut.z_tip_old == old[end, 1]
        assert direction * (cut.z_tip_old - cut.z_tip_new) == pytest.approx(
            cut.L_rem - neck["r_n"], rel=1e-9)
        assert cut.summary()["tip_shift"] == pytest.approx(cut.L_rem - neck["r_n"], rel=1e-9)
        # the hard checks as read: the cap inside the old liquid, monotone, tangent at the splice
        ch = cut.checks
        assert ch["failed"] == [] and ch["chains"] == "ok" and ch["removed_simple"]
        assert ch["clearance"] >= ev.CAP_CLEARANCE * neck["r_n"] and ch["clearance_skipped"] > 0
        assert ch["backtrack"] == 0.0 and ch["min_angle"] > 150.0
        j = i - 1 if end == -1 else i + 1  # the neck's neighbour on the side kept
        assert ch["first_spacing"] == pytest.approx(np.hypot(*(old[j] - old[i])), rel=1e-9)
        assert 2.0 < cut.V_rem / neck["s"] ** 3 < 10.0
        assert cut.V_rem < neck["head_volume"]  # the stump keeps the cap
        new = [rc.Chain(cut.points, c.end_types) if k == name else c for k, c in named]
        assert rc.check_chains(new) > 0.0
    # both heads alike: the cavity's is the bubble's mirrored
    a = ev.truncate_chain(named, _neck(named, "bubble"), **CUT)
    b = ev.truncate_chain(named, _neck(named, "open"), **CUT)
    assert a.V_rem == pytest.approx(b.V_rem, rel=1e-6) and len(a.cap) == len(b.cap)
    assert a.A_rem == pytest.approx(b.A_rem, rel=1e-6)


def _sphere_on_cylinder(*, s, H, t, r_n, z_n=0.8, below=0.3, n=4000):
    """An open chain whose head is a hemisphere of radius ``s`` on a cylinder of radius ``s`` and
    height ``H`` on a cone frustum narrowing over ``t`` to the neck ``r_n`` at ``z_n``; below the neck
    a frustum widens to ``r = 0.05`` at ``z_n - below``, then a flat surface runs to the wall.
    Returns the chain and the index of the neck."""
    z1 = z_n + t
    z2 = z1 + H
    phi = np.linspace(0.5 * math.pi, 0.0, n)
    pieces = [np.column_stack([s * np.cos(phi), z2 + s * np.sin(phi)]),
              np.column_stack([np.full(n, s), np.linspace(z2, z1, n)]),
              np.column_stack([np.linspace(s, r_n, n), np.linspace(z1, z_n, n)])]
    k = sum(len(p) - 1 for p in pieces)
    pieces += [np.column_stack([np.linspace(r_n, 0.05, n), np.linspace(z_n, z_n - below, n)]),
               np.column_stack([np.linspace(0.05, R_C, n), np.full(n, z_n - below)])]
    P = np.vstack([pieces[0]] + [p[1:] for p in pieces[1:]])
    P[0, 0] = 0.0
    return rc.Chain(P, ("axis", "fixed")), k


def _hand_neck(chain, i):
    """The tracked neck of node ``i`` with its head, as ``events.liquid_necks`` gives it."""
    (h,) = [x for x in ev.head_sides(chain.points, chain.end_types, i, is_bubble=False)
            if x["qualifies"]]
    return {"chain": "open", "index": i, "r_n": float(chain.points[i, 0]),
            "z_n": float(chain.points[i, 1]), "s": h["s"], "head": [h["lo"], h["hi"]],
            "direction": h["direction"], "status": "head", "head_volume": h["volume"]}


def test_the_removed_volume_and_areas_are_those_of_a_sphere_on_a_cylinder():
    """The removed region of a head that is a hemisphere on a cylinder on a frustum: its volume is
    the head's above the neck's plane less the half-spheroid that closes the stump, and its areas are
    the head's surface and the cap's, to the polylines' accuracy."""
    s, H, t, r_n = 0.05, 0.08, 0.03, 0.01
    chain, i = _sphere_on_cylinder(s=s, H=H, t=t, r_n=r_n)
    head = (math.pi * t / 3.0 * (r_n**2 + r_n * s + s**2) + math.pi * s**2 * H
            + 2.0 / 3.0 * math.pi * s**3)
    area = (math.pi * (r_n + s) * math.hypot(t, s - r_n) + 2.0 * math.pi * s * H
            + 2.0 * math.pi * s**2)
    for eta in (1.0, 0.5):
        cut = ev.truncate_chain((("open", chain),), _hand_neck(chain, i), **dict(CUT, eta=eta))
        assert cut.V_rem == pytest.approx(head - 2.0 / 3.0 * math.pi * r_n**3 * eta, rel=1e-5)
        assert cut.A_rem == pytest.approx(area, rel=1e-5)
        assert cut.L_rem == pytest.approx(t + H + s, rel=1e-12)
        if eta == 1.0:  # a half sphere
            assert cut.A_cap == pytest.approx(2.0 * math.pi * r_n**2, rel=5e-3)
        assert rc.revolved_volume(cut.removed) == cut.V_rem


def test_a_cap_that_leaves_the_old_liquid_is_refused_and_a_flatter_one_may_pass():
    """A bulb of 1.5 r_n that ends 0.7 r_n above its neck: the half sphere reaches beyond its tip
    and out of the old liquid (refused, with the checks as read); the half-spheroid of height
    r_n/2 (CAP_RETRY_HEIGHT, the one retry) stays inside it. A neck that is no node of its chain, or
    one without a single head, is refused too."""
    r_n, s, b, z_n = 0.01, 0.015, 0.004, 0.8
    zc = z_n + b * math.sqrt(1 - (r_n / s) ** 2)
    phi = np.linspace(0.5 * math.pi, -math.asin(math.sqrt(1 - (r_n / s) ** 2)), 4000)
    bulb = np.column_stack([s * np.cos(phi), zc + b * np.sin(phi)])
    bulb[-1] = (r_n, z_n)
    below = np.column_stack([np.linspace(r_n, 0.05, 4000)[1:], np.linspace(z_n, 0.3, 4000)[1:]])
    flat = np.column_stack([np.linspace(0.05, R_C, 4000)[1:], np.full(3999, 0.3)])
    P = np.vstack([bulb, below, flat])
    P[0, 0] = 0.0
    chain, i = rc.Chain(P, ("axis", "fixed")), len(bulb) - 1
    neck = _hand_neck(chain, i)
    with pytest.raises(ev.CapRefused) as err:
        ev.truncate_chain((("open", chain),), neck, **CUT)
    assert {"tip", "inside"} <= set(err.value.checks["failed"])
    assert err.value.checks["clearance"] < 0.0
    cut = ev.truncate_chain((("open", chain),), neck, **dict(CUT, eta=ev.CAP_RETRY_HEIGHT))
    assert cut.checks["failed"] == [] and cut.eta == 0.5
    with pytest.raises(ev.CapRefused, match="not a node"):
        ev.truncate_chain((("open", chain),), dict(neck, r_n=1.01 * r_n), **CUT)
    with pytest.raises(ev.CapRefused, match="no single head"):
        ev.truncate_chain((("open", chain),), dict(neck, status="two-heads", head=None), **CUT)
    with pytest.raises(ev.CapRefused, match="does not end on the axis"):
        ev.truncate_chain((("open", chain),), dict(neck, head=[i, len(P) - 1]), **CUT)


def test_the_cut_of_a_neck_is_the_same_in_units_of_delta_at_every_delta():
    """The jet, its neck and its node spacing scaled with delta by 2^-k (k = 0..7) inside a bubble
    and a cavity of fixed size: the same cap and removed region in units of delta."""
    ref = None
    for k in range(8):
        d = 1e-3 * 2.0 ** -k
        bubble, _ = stops.bubble_with_head(d, r_n=0.09)
        named = (("bubble", bubble), ("open", stops.cavity()))
        neck = _neck(named, "bubble")
        cut = ev.truncate_chain(named, neck, **CUT)
        got = np.array([cut.V_rem / d**3, cut.A_rem / d**2, cut.A_cap / d**2, cut.L_rem / d,
                        (cut.z_tip_new - cut.z_n) / d, cut.checks["clearance"] / d,
                        cut.checks["first_spacing"] / d, len(cut.cap),
                        len(bubble.points) - neck["index"]])
        if ref is None:
            ref = got
        assert got == pytest.approx(ref, rel=1e-6), (k, got, ref)


# --- the stop states of runs after a cut ------------------------------------------------------------


@stops.runs
@pytest.mark.parametrize("run", stops.RUNS, ids=[f"run{k}" for k in range(len(stops.RUNS))])
def test_the_necks_of_the_stop_states_of_runs_after_a_cut_are_cut_cleanly(run):
    """Every tracked neck of a stop state of a run after a cut, at the root of a cut-born jet on
    either chain, is cut cleanly at the half sphere and at the retry's height: its cap inside the
    old liquid (the points that leave the neck too), monotone and tangent at the splice, the new
    chains valid, the removed head 3-10 s^3 and the tip moved by the head's reach less r_n."""
    chains, _, _ = stops._stop_state(run)
    named = (("bubble", chains[0]), ("open", chains[1]))
    necks = [x for x in _necks(named) if x["status"] == "head"]
    assert {x["chain"] for x in necks} == {"bubble", "open"}
    for neck in necks:
        for eta in (1.0, ev.CAP_RETRY_HEIGHT):
            cut = ev.truncate_chain(named, neck, **dict(CUT, eta=eta))
            ch = cut.checks
            assert ch["failed"] == [] and ch["clearance_skipped"] > 0.0
            assert ch["clearance"] > 0.05 * neck["r_n"] and ch["min_angle"] > 150.0
            assert 3.0 < cut.V_rem / neck["s"] ** 3 < 10.0
            assert cut.tip == ("bubble_top" if neck["chain"] == "bubble" else "cavity_tip")
            assert cut.summary()["tip_shift"] == pytest.approx(cut.L_rem - eta * neck["r_n"],
                                                              rel=1e-9)
