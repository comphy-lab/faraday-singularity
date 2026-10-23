"""The geometry of the fill of a closing gas groove (``groove.py``), without a finite-element
problem.

The synthetic states: an elliptic ring with a narrow, smooth finger of gas pushed out of it into the
liquid (a Gaussian bump of its boundary along its normal), or the same bump pushed in, a sheet of
liquid between two faces of gas (the replica of the sheet of event 5 of the ring stage); a flat open
chain with a narrow, smooth slot of gas down into the liquid; a bubble from the axis to the axis
with a finger of gas. Each is met as the stops meet it: its fold (``ring.loop_fold_gap``,
``selfcontact.fold_gap``) below ``max(kappa_g delta, fold_stop_elements h_elem)``.

* the fill of a gas finger of the ring, of a slot of the open chain and of a finger of the bubble:
  the pocket of gas, its mouth where the walls lie ``w_f`` apart, the closure C1 with both walls
  at the splices and inside the old gas, the filled region simple and inside the old gas, the gas
  before the fill the gas after it and the filled region together (to round-off), the new loop
  counter-clockwise and seamed on an old vertex, the new chain with the old ends, the result the
  same for every seam of the loop;
* the refusals: a sheet of liquid (``side``), the ring's lobe of a waist beyond ``max_fill V_b``
  (``volume``), a fill on the protected core (``core``), a pair at the chain's end (``pair``), walls
  that stay narrow to the chain's end (``mouth``).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faraday_singularity import groove  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402
from faraday_singularity import ring  # noqa: E402
from faraday_singularity.selfcontact import fold_gap  # noqa: E402

from test_ring_stops import flat_open  # noqa: E402

DELTA = 1e-3
KAPPA = 0.05
FOLD = {"ratio": 4.0, "stop": KAPPA * DELTA, "stop_elements": 3.0, "search_factor": 4.0}
R_C = 3.8317059702075125
RESERVOIR = rc.ReconnectSettings(delta=DELTA).reservoir_depth


def resample(dense, h_of):
    """Points along the dense polyline ``dense`` (closed if its ends coincide), spaced by
    ``h_of(point)``."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(dense, axis=0).T))])
    out, x = [0.0], 0.0
    while True:
        k = min(int(np.searchsorted(s, x)), len(dense) - 1)
        x += h_of(dense[k])
        if x > s[-1] - 0.5 * h_of(dense[-1]):
            break
        out.append(x)
    u = np.array(out + [s[-1]])
    return np.column_stack([np.interp(u, s, dense[:, 0]), np.interp(u, s, dense[:, 1])])


def bumped_ellipse(t, *, c=(0.01, 4.68), a=2e-3, b=1e-3, A=3e-4, sigma=1.8e-2, t0=0.0):
    """Points of the ellipse at the parameters ``t``, displaced along its outward normal by the
    bump ``A exp(-((t - t0)/sigma)^2)`` (``A`` negative: inwards)."""
    t = np.asarray(t, dtype=float)
    E = np.column_stack([c[0] + a * np.cos(t), c[1] + b * np.sin(t)])
    N = np.column_stack([b * np.cos(t), a * np.sin(t)])
    N /= np.hypot(*N.T)[:, None]
    return E + (A * np.exp(-(((t - t0 + math.pi) % (2 * math.pi) - math.pi) / sigma) ** 2))[:, None] * N


def spacing(centre, fine=4e-6, coarse=4e-5, reach=6e-4):
    def h(p):
        d = math.hypot(p[0] - centre[0], p[1] - centre[1])
        return fine if d < reach else min(coarse, fine + 0.05 * (d - reach))
    return h


def finger_ring(A=3e-4, sigma=1.8e-2, phase=0.0, t0=0.0):
    """The ring with its finger (``A`` positive: gas pushed out into the liquid; negative: a sheet
    of liquid pushed into the gas), counter-clockwise, an even count of nodes, seamed at the node
    ``phase`` of the way round (the seam a vertex: an even index)."""
    t = np.linspace(-math.pi, math.pi, 200001)
    dense = bumped_ellipse(t, A=A, sigma=sigma, t0=t0)
    tip = bumped_ellipse(np.array([t0]), A=A, sigma=sigma, t0=t0)[0]
    L = resample(dense, spacing(tip))[:-1]
    if len(L) % 2:
        L = L[:-1]
    k = 2 * int(phase * len(L) / 2)
    return np.roll(L, -k, axis=0)


def _arc_points(c, R, phi0, phi1, m=20001):
    phi = np.linspace(phi0, phi1, m)
    return np.column_stack([c[0] + R * np.cos(phi), c[1] + R * np.sin(phi)])


def slot_open(z=4.69, r1=0.02, A=3e-4, w=2.5e-5, R_f=5e-5):
    """The open chain with a narrow slot of gas at ``r1``, ``A`` deep and ``w`` wide, its walls
    vertical, its bottom a half circle, its mouth rounded by fillets of radius ``R_f`` (the liquid's
    corners), sampled finely about the slot."""
    x0, x1 = r1 - 0.5 * w, r1 + 0.5 * w
    parts = [np.column_stack([np.linspace(x0 - R_f - 3e-4, x0 - R_f, 2001), np.full(2001, z)]),
             _arc_points((x0 - R_f, z - R_f), R_f, 0.5 * math.pi, 0.0),
             np.column_stack([np.full(2001, x0), np.linspace(z - R_f, z - A, 2001)]),
             _arc_points((r1, z - A), 0.5 * w, math.pi, 2.0 * math.pi),
             np.column_stack([np.full(2001, x1), np.linspace(z - A, z - R_f, 2001)]),
             _arc_points((x1 + R_f, z - R_f), R_f, math.pi, 0.5 * math.pi),
             np.column_stack([np.linspace(x1 + R_f, x1 + R_f + 3e-4, 2001), np.full(2001, z)])]
    parts = [np.column_stack([np.linspace(0.0, parts[0][0, 0], 20001)[:-1],
                              np.full(20000, z)])] + parts
    parts.append(np.column_stack([np.linspace(parts[-1][-1, 0], R_C, 200001)[1:],
                                  np.full(200000, z)]))
    dense = np.vstack(parts)
    h = spacing((r1, z - 0.5 * A), fine=3e-6, coarse=4e-3, reach=3e-4)
    P = resample(dense, h)
    return rc.Chain(P, ("axis", "fixed"))


def chain_fold(P):
    f = fold_gap(P[:, 0], P[:, 1], ratio=FOLD["ratio"], stop=FOLD["stop"],
                 stop_elements=FOLD["stop_elements"], search_factor=FOLD["search_factor"])
    return f, max(FOLD["stop"], FOLD["stop_elements"] * f.h_elem)


def loop_fold(L):
    f = ring.loop_fold_gap(L, ratio=FOLD["ratio"], stop=FOLD["stop"],
                           stop_elements=FOLD["stop_elements"], search_factor=FOLD["search_factor"])
    return f, max(FOLD["stop"], FOLD["stop_elements"] * f.h_elem)


def fill(chain, P, pair, limit, *, ends=("loop", "loop"), V_b=None, others=(), **kw):
    V_b = (100.0 * rc.revolved_volume(P) if chain == "ring" else 1.0) if V_b is None else V_b
    args = dict(width=1.5 * limit, eta=1.0, elements_per_scale=8.0, min_angle=30.0,
                reservoir_depth=RESERVOIR, V_b=V_b, max_fill=0.01, others=others)
    args.update(kw)
    return groove.groove_fill_geometry(chain, P, ends, pair, **args)


def _tangent_jump(P, g, k_old, at_start):
    """The angle (rad) between the old wall's tangent at a splice and the closure's there."""
    k = int(np.flatnonzero(np.asarray(g.origin) == k_old)[0])
    n = len(g.points)
    d_wall = P[(k_old + 1) % len(P)] - P[(k_old - 1) % len(P)]
    B, _, _ = groove.closure_curve(P[g.i_s], P[(g.i_s + 1) % len(P)] - P[(g.i_s - 1) % len(P)],
                                   P[g.j_s], P[(g.j_s + 1) % len(P)] - P[(g.j_s - 1) % len(P)],
                                   g.eta)
    t = np.array([0.0, 1e-6]) if at_start else np.array([1.0 - 1e-6, 1.0])
    X = groove._bezier(B, t)
    d_cl = X[1] - X[0]
    cos = np.dot(d_wall, d_cl) / (np.hypot(*d_wall) * np.hypot(*d_cl))
    assert 0 <= k < n
    return math.acos(min(1.0, abs(cos)))


def test_a_gas_finger_of_the_ring_is_filled_at_its_mouth():
    L = finger_ring()
    f, limit = loop_fold(L)
    assert math.isfinite(f.gap) and f.gap < limit and rc.fold_contact(f.arc, f.h_elem) == "fold"
    g = fill("ring", L, f.pair, limit, others=[flat_open()])
    assert g.side == "gas" and g.gas_share == pytest.approx(1.0)
    # the mouth: the walls at least w_f apart at the splices, narrower within
    assert g.mouth >= g.width and g.width == pytest.approx(1.5 * limit)
    # the gas before is the gas after and the filled region together
    V0, V1 = rc.revolved_volume(L), rc.revolved_volume(g.points)
    assert V0 == pytest.approx(V1 + g.V_fill, rel=1e-12)
    assert g.V_gas_before == pytest.approx(V0) and g.V_gas_after == pytest.approx(V1)
    # most of the finger is filled: its volume beyond the bump's base
    finger = 2 * math.pi * 0.012 * 3e-4 * (2e-3 * 1.8e-2 * math.sqrt(math.pi) / 2)
    assert 0.3 * finger < g.V_fill < 1.5 * finger
    # C1 at both splices, the closure inside the old gas
    assert _tangent_jump(L, g, g.i_s, True) < 1e-3 and _tangent_jump(L, g, g.j_s, False) < 1e-3
    assert g.checks["clearance"] <= g.checks["clearance_limit"] < 0.0
    assert g.checks["fill_simple"] and g.checks["fill_share"] >= 1.0 - groove.SIDE_TOL
    # the new loop: counter-clockwise, simple, seamed on an old vertex, the closure in it
    assert ring.signed_area(g.points) > 0.0 and not ring.loop_crossings(g.points)
    assert g.origin[0] >= 0 and g.origin[0] % 2 == 0
    assert int(np.sum(g.origin < 0)) == len(g.closure) >= 16
    # the fold the stops met is gone from the new loop
    f2, limit2 = loop_fold(g.points)
    assert not (math.isfinite(f2.gap) and f2.gap < limit2)


def test_the_fill_of_the_ring_is_the_same_for_every_seam():
    a = finger_ring()
    b = finger_ring(phase=0.37)
    ga = fill("ring", a, loop_fold(a)[0].pair, loop_fold(a)[1])
    gb = fill("ring", b, loop_fold(b)[0].pair, loop_fold(b)[1])
    assert ga.V_fill == pytest.approx(gb.V_fill, rel=1e-9)
    assert ga.mouth == pytest.approx(gb.mouth, rel=1e-12)
    assert np.allclose(np.sort(ga.closure, axis=0), np.sort(gb.closure, axis=0), atol=1e-12)


def test_a_sheet_of_liquid_on_the_ring_is_refused():
    """The replica of the sheet of event 5: the ring's boundary pushed into the gas, a sheet of
    liquid whose two faces the fold test meets as a fold; its pocket is liquid."""
    L = finger_ring(A=-3e-4, t0=math.pi)  # from the ring's inner wall, out into the gas
    f, limit = loop_fold(L)
    assert math.isfinite(f.gap) and f.gap < limit
    with pytest.raises(groove.GrooveFillRefused) as e:
        fill("ring", L, f.pair, limit)
    assert e.value.checks["failed"] == ["side"] and e.value.checks["side"] == "liquid"
    assert e.value.checks["gas_share"] == pytest.approx(0.0, abs=1e-9)


def test_a_lobe_of_the_ring_beyond_max_fill_is_refused():
    t = np.linspace(0.0, 2.0 * math.pi, 400, endpoint=False)
    rho = 0.02 + np.cos(t) ** 2
    peanut = np.column_stack([0.01 + 1e-3 * rho * np.cos(t), 4.68 + 1e-3 * rho * np.sin(t)])
    f, limit = loop_fold(peanut)
    assert math.isfinite(f.gap) and f.gap < limit
    V = rc.revolved_volume(peanut)
    with pytest.raises(groove.GrooveFillRefused) as e:
        fill("ring", peanut, f.pair, limit, V_b=10.0 * V)
    assert "volume" in e.value.checks["failed"]
    assert e.value.checks["V_fill"] > 0.01 * 10.0 * V
    # with a generous max_fill the smaller lobe is filled, the ring left one loop
    g = fill("ring", peanut, f.pair, limit, V_b=10.0 * V, max_fill=1.0)
    assert g.V_fill < 0.5 * V + 1e-18 and rc.revolved_volume(g.points) > 0.4 * V


def test_a_slot_of_gas_in_the_open_chain_is_filled():
    c = slot_open()
    f, limit = chain_fold(c.points)
    assert math.isfinite(f.gap) and f.gap < limit and rc.fold_contact(f.arc, f.h_elem) == "fold"
    g = fill("open", c.points, f.pair, limit, ends=c.end_types)
    assert g.side == "gas" and g.mouth >= g.width
    gas0 = rc.revolved_volume(rc.closed_section(c.points, c.end_types, RESERVOIR))
    gas1 = rc.revolved_volume(rc.closed_section(g.points, c.end_types, RESERVOIR))
    assert gas0 == pytest.approx(gas1 + g.V_fill, rel=1e-12)
    assert g.points[0, 0] == 0.0 and g.points[-1, 0] == R_C
    assert _tangent_jump(c.points, g, g.i_s, True) < 1e-3
    assert _tangent_jump(c.points, g, g.j_s, False) < 1e-3
    rc.check_chains([rc.Chain(g.points, c.end_types)])
    f2, limit2 = chain_fold(g.points)
    assert not (math.isfinite(f2.gap) and f2.gap < limit2)
    # the core protected: refused
    mask = np.zeros(len(c.points), dtype=bool)
    mask[f.pair[0]] = True
    with pytest.raises(groove.GrooveFillRefused) as e:
        fill("open", c.points, f.pair, limit, ends=c.end_types, protected=mask)
    assert e.value.checks["failed"] == ["core"]


def test_a_finger_of_gas_of_the_bubble_is_filled():
    """A bubble from the axis to the axis (the right half of the ring's ellipse about the axis)
    with a finger of gas out into the liquid."""
    t = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 100001)
    dense = bumped_ellipse(t, c=(0.0, 4.5), a=2e-3, b=1e-3)
    tip = bumped_ellipse(np.array([0.0]), c=(0.0, 4.5), a=2e-3, b=1e-3)[0]
    P = resample(dense, spacing(tip))
    P[0, 0] = P[-1, 0] = 0.0
    bubble = rc.Chain(P, ("axis", "axis"))
    f, limit = chain_fold(P)
    assert math.isfinite(f.gap) and f.gap < limit
    g = fill("bubble", P, f.pair, limit, ends=bubble.end_types, others=[flat_open()],
             V_b=100.0 * rc.revolved_volume(P))
    assert g.V_gas_before == pytest.approx(g.V_gas_after + g.V_fill, rel=1e-12)
    rc.check_chains([rc.Chain(g.points, bubble.end_types), flat_open()])


def test_a_pair_at_the_end_or_walls_narrow_to_the_end_are_refused():
    c = slot_open()
    f, limit = chain_fold(c.points)
    with pytest.raises(groove.GrooveFillRefused) as e:
        fill("open", c.points, (0, f.pair[1]), limit, ends=c.end_types)
    assert e.value.checks["failed"] == ["pair"]
    # a fill width wider than the whole surface's gap to the axis end: no mouth within the chain
    with pytest.raises(groove.GrooveFillRefused) as e:
        fill("open", c.points, f.pair, limit, ends=c.end_types, width=10.0)
    assert e.value.checks["failed"] == ["mouth"]
    with pytest.raises(ValueError, match="no fill"):
        groove.groove_fill_geometry("cavity", c.points, c.end_types, f.pair, width=1.0, eta=1.0,
                                    elements_per_scale=8.0, min_angle=30.0,
                                    reservoir_depth=RESERVOIR, V_b=1.0, max_fill=0.01)
