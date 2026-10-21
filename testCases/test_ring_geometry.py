"""The geometry of the ring cut and the helpers of a closed loop (``ring.py``), without a
finite-element problem.

The synthetic pierced bubble (:func:`pierced_bubble`): a flat plug at ``z0`` from the axis out to a
rounded outer wall, a roof, and a jet hanging from the roof into the bubble, a neck of radius
``a_n`` above a head of radius ``a`` whose tip lies on the axis ``g`` above the plug; the head's
widest point is the jet's shoulder.

* the cut: its splices on the cap and on the plug, the closure tangent to both (C1), inside the old
  gas, its vertical segment at ``r_w = waist rho`` exactly; the filled region's volume against the
  integral of ``pi r^2 dz`` along the exact closure; the filled region and the ring simple, the
  ring counter-clockwise with its seam on a vertex; the ring and the filled region together the
  bubble, to round-off;
* a closing gap: the cut valid until the fillets of the closure no longer fit between the splices,
  refused then at ``closure``, and at ``splice`` once the tips meet;
* the other refusals: a splice radius beyond ``RHO_MAX`` shoulder radii, a jet whose shoulder is at
  its tip, a plug not monotone out to the splice, a splice radius the old mesh does not hold, a
  waist below a tenth of the splice radius, a ring born below ``bubble_stop`` of ``V_b``;
* the helpers of a loop: crossings (none on a circle, one on a figure-eight), the periodic fold
  (the waist of a ring, none at the seam, the same for every seam), the two halves split on
  vertices, the contact of their splines, the quadratic elements, the check of a loop's order.
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

from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402
from faraday_singularity import ring  # noqa: E402
from faraday_singularity.selfcontact import fold_gap  # noqa: E402

DELTA = 1e-3
Z0 = 4.6  # the plug
R_C = 3.8317059702075125


def _arc(c, q, psi0, psi1, h):
    """Points of a counter-clockwise arc of radius ``q`` about ``c`` (travel direction ``psi``),
    about ``h`` apart, the start excluded."""
    m = max(2, int(math.ceil(q * abs(psi1 - psi0) / h)))
    psi = np.linspace(psi0, psi1, m + 1)[1:]
    return np.column_stack([c[0] + q * np.sin(psi), c[1] - q * np.cos(psi)])


def pierced_bubble(g=0.8 * DELTA, *, a=DELTA / 6, a_n=None, R0=4 * DELTA, H=4 * DELTA,
                   h=None, h_far=None, z0=Z0):
    """The bubble chain of a pierced bubble (module docstring), upwards from the plug's tip on the
    axis to the jet's tip, ``2N + 1`` nodes about ``h`` apart within ``3 a`` of the axis and
    ``h_far`` beyond; ``(chain, parts)`` with the head's centre and radius."""
    a_n = 0.6 * a if a_n is None else a_n
    h = a / 40 if h is None else h
    h_far = 4 * h if h_far is None else h_far
    q = 0.5 * DELTA  # the fillets of the outer wall
    q2 = 0.5 * a  # the fillet of the roof into the neck
    z_roof = z0 + H
    z_c = z0 + g + a  # the head's centre
    th_n = math.acos(a_n / a)
    dense = 1e-2 * h
    pieces = [np.column_stack([np.linspace(0.0, R0 - q, 2000), np.full(2000, z0)])]
    pieces.append(_arc((R0 - q, z0 + q), q, 0.0, 0.5 * math.pi, dense))
    pieces.append(np.column_stack([np.full(2000, R0), np.linspace(z0 + q, z_roof - q, 2000)])[1:])
    pieces.append(_arc((R0 - q, z_roof - q), q, 0.5 * math.pi, math.pi, dense))
    pieces.append(np.column_stack([np.linspace(R0 - q, a_n + q2, 2000),
                                   np.full(2000, z_roof)])[1:])
    pieces.append(_arc((a_n + q2, z_roof - q2), q2, math.pi, 1.5 * math.pi, dense))
    z_neck = z_c + a * math.sin(th_n)
    pieces.append(np.column_stack([np.full(2000, a_n), np.linspace(z_roof - q2, z_neck,
                                                                    2000)])[1:])
    th = np.linspace(th_n, -0.5 * math.pi, 20000)[1:]
    pieces.append(np.column_stack([a * np.cos(th), z_c + a * np.sin(th)]))
    D = np.vstack(pieces)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(D, axis=0).T))])
    near = (D[:, 0] <= 3 * a)
    u, x = [0.0], 0.0
    while x < s[-1]:
        k = min(int(np.searchsorted(s, x)), len(D) - 1)
        x += h if near[k] else h_far
        u.append(min(x, s[-1]))
    u = np.unique(np.array(u))
    if len(u) % 2 == 0:  # an odd count: 2N + 1 nodes
        u = np.sort(np.append(u, 0.5 * (u[-1] + u[-2])))
    P = np.column_stack([np.interp(u, s, D[:, 0]), np.interp(u, s, D[:, 1])])
    P[0, 0] = P[-1, 0] = 0.0
    return rc.Chain(P, ("axis", "axis")), {"z_c": z_c, "a": a, "a_n": a_n, "z0": z0}


def open_chain():
    from test_events_stops import cavity

    return cavity()


def ring_settings(**kw):
    return ev.RingEventSettings(enabled=True, **kw)


def cut(bubble, settings=None, *, eps=16.0, V_b=None, bubble_stop=0.01):
    V_b = 30.0 * rc.revolved_volume(bubble.points) if V_b is None else V_b
    return ring.ring_cut_geometry(bubble, open_chain(), settings or ring_settings(),
                                  elements_per_scale=eps, min_angle=30.0, reservoir_depth=3.0,
                                  V_b=V_b, bubble_stop=bubble_stop)


def exact_fill_volume(rc_cut, P):
    """``|pi oint r^2 dz|`` around the filled region: the bubble's polyline inside the splices
    (exact for the polygon) and the closure's exact arcs and vertical segment (the closure as a
    curve, sampled finely)."""
    def poly(Q):  # the signed pi int r^2 dz along a polyline
        r0, z0, r1, z1 = Q[:-1, 0], Q[:-1, 1], Q[1:, 0], Q[1:, 1]
        return math.pi * float(np.sum((r0 * r0 + r0 * r1 + r1 * r1) / 3.0 * (z1 - z0)))

    n = len(P)
    curve = ring._Closure(P[rc_cut.i_t], math.radians(rc_cut.phi_deg), rc_cut.a_t, P[rc_cut.i_b],
                          math.radians(rc_cut.chi_deg), rc_cut.a_b, rc_cut.r_w)
    s = np.linspace(0.0, curve.length, 400001)
    exact = curve.at(s)
    exact[0], exact[-1] = P[rc_cut.i_t], P[rc_cut.i_b]
    total = poly(P[rc_cut.i_b::-1]) + poly(np.vstack([P[0], P[-1]])) \
        + poly(P[n - 1:rc_cut.i_t - 1:-1]) + poly(exact)
    return abs(total)


# --- the cut --------------------------------------------------------------------------------------


def test_the_cut_splices_on_the_cap_and_the_plug_tangentially_and_inside_the_gas():
    bubble, parts = pierced_bubble()
    P = bubble.points
    c = cut(bubble)
    assert c.s_t == pytest.approx(parts["a"], rel=1e-3)  # the head's widest point
    assert c.rho == pytest.approx(0.5 * c.s_t) and c.r_w == 0.5 * c.rho
    # the jet's splice on the head's lower half, beyond the shoulder; the plug's on the plug
    assert c.i_t > c.i_sh and c.r_t >= c.rho > P[c.i_t + 1, 0]
    assert math.hypot(c.r_t, c.z_t - parts["z_c"]) == pytest.approx(parts["a"], rel=1e-3)
    assert c.z_t < parts["z_c"] and c.z_b == Z0 and c.r_b >= c.rho > P[c.i_b - 1, 0]
    # C1 at both splices: the closure starts and ends at the splices, leaving each along the
    # chain's tangent there (the arcs' directions of travel, psi, at their ends), and its pieces
    # join with the same position and direction
    curve = ring._Closure(P[c.i_t], math.radians(c.phi_deg), c.a_t, P[c.i_b],
                          math.radians(c.chi_deg), c.a_b, c.r_w)
    assert np.allclose(curve.at([0.0, curve.length]), [P[c.i_t], P[c.i_b]], rtol=0.0,
                       atol=1e-15)
    d_t = ring._unit(P[c.i_t + 1] - P[c.i_t - 1])
    d_b = ring._unit(P[c.i_b + 1] - P[c.i_b - 1])
    for psi, want in ((curve.psi0, d_t), (2.0 * math.pi + math.radians(c.chi_deg), d_b)):
        got = np.array([math.cos(psi), math.sin(psi)])
        turn = math.atan2(got[0] * want[1] - got[1] * want[0], float(np.dot(got, want)))
        assert abs(turn) < 1e-12
    joints = curve.at([curve.L1, curve.L1 + curve.Lv])
    assert np.allclose(joints, [[c.r_w, curve.z_1], [c.r_w, curve.z_2]], rtol=0.0, atol=1e-15)
    # the points spaced as the splices are at each end, growing from there
    steps = np.hypot(*np.diff(np.vstack([P[c.i_t], c.closure, P[c.i_b]]), axis=0).T)
    assert steps[0] == pytest.approx(np.hypot(*(P[c.i_t] - P[c.i_t - 1])), rel=1e-2)
    assert steps[-1] == pytest.approx(np.hypot(*(P[c.i_b + 1] - P[c.i_b])), rel=1e-2)
    assert np.all(steps[1:] / steps[:-1] <= ev.CAP_GROWTH * 1.01 + 0.5)
    # the arcs meet the vertical segment at r_w, the segment exactly there
    on = np.isclose(c.closure[:, 0], c.r_w, rtol=0.0, atol=1e-15)
    assert on.sum() >= 3 and np.all(c.closure[on, 0] == c.r_w)
    assert c.closure[:, 0].min() >= c.r_w * (1 - 1e-12)
    assert c.checks["clearance"] <= -ev.CAP_CLEARANCE * c.rho and c.checks["failed"] == []
    assert len(c.closure) >= ev.CAP_MIN_POINTS


def test_the_filled_region_and_the_ring_make_the_bubble():
    """The filled region against ``pi oint r^2 dz`` along the exact closure: the closure's chords
    differ from its arcs by an error of at least the second order in their length, the longest
    those at the splices, at the chain's spacing there (within 2e-6 at 160 nodes per head radius
    and 256 points per waist radius, and more than three times as large at half the resolution of
    both); the region and the ring simple, the ring counter-clockwise, its seam a vertex of the old
    walk; the ring's volume and the region's the bubble's, to round-off."""
    from shapely.geometry import LinearRing

    errors = []
    for div, eps in ((80, 128.0), (160, 256.0)):
        bubble, _ = pierced_bubble(h=DELTA / 6 / div)
        c = cut(bubble, eps=eps)
        V_exact = exact_fill_volume(c, bubble.points)
        errors.append(abs(c.V_fill - V_exact) / V_exact)
    assert errors[1] < 2e-6 and errors[0] / errors[1] > 3.0, errors
    assert LinearRing(c.fill).is_simple and LinearRing(c.loop).is_simple
    assert ring.signed_area(c.loop) > 0.0
    assert c.origin[0] >= 0 and c.origin[0] % 2 == 0
    assert c.loop[0, 0] == pytest.approx(bubble.points[:, 0].max())
    assert np.array_equal(c.loop[c.origin >= 0], bubble.points[c.origin[c.origin >= 0]])
    V_bubble = rc.revolved_volume(bubble.points)
    assert rc.revolved_volume(ring.closed(c.loop)) + c.V_fill == pytest.approx(V_bubble,
                                                                              rel=1e-12)
    assert c.V_ring == pytest.approx(V_bubble - c.V_fill, rel=1e-12)
    summary = c.summary()
    assert summary["splice_jet"]["i"] == c.i_t and summary["closure_points"] == len(c.closure)
    assert "loop" not in summary and summary["checks"]["failed"] == []


def test_a_closing_gap_is_cut_until_the_fillets_no_longer_fit():
    """At gaps delta, delta/2, ...: the cut is made while the vertical segment between the two
    fillets has a length (``z_1 >= z_2``), refused at ``closure`` alone once it has none, and at
    ``splice`` once the tips meet or pass."""
    made, refused = [], []
    for k in range(9):
        g = DELTA * 0.5**k
        bubble, _ = pierced_bubble(g)
        try:
            c = cut(bubble)
            assert c.z_1 >= c.z_2
            made.append(g)
        except ring.RingCutRefused as e:
            assert e.checks["failed"] == ["closure"], (g, e.checks)
            assert e.checks["z_1"] < e.checks["z_2"]
            refused.append(g)
    assert made and refused and min(made) > max(refused)
    for g in (0.0, -0.1 * DELTA):
        bubble, _ = pierced_bubble(g)
        with pytest.raises(ring.RingCutRefused) as err:
            cut(bubble)
        assert err.value.checks["failed"] == ["splice"]


def _refused(bubble, settings=None, **kw):
    with pytest.raises(ring.RingCutRefused) as err:
        cut(bubble, settings, **kw)
    return err.value.checks


def test_the_other_refusals_of_the_cut():
    bubble, parts = pierced_bubble()
    with pytest.raises(ValueError, match="ring.rho"):
        ring_settings(rho=0.9)
    wide = SimpleNamespace(rho=0.9, waist=0.5)  # past the settings' own check
    assert _refused(bubble, wide)["failed"] == ["splice"]
    # a jet whose shoulder is at its tip: no node of the cap lies between them
    P = bubble.points.copy()
    P[-2, 0] = 2.0 * P[-3, 0]
    pointed = rc.Chain(P, bubble.end_types)
    checks = _refused(pointed)
    assert checks["failed"] == ["splice"] and checks["i_sh"] == len(P) - 2
    # a plug not monotone out to the splice
    P = bubble.points.copy()
    P[2, 0] = 0.5 * P[1, 0]
    assert _refused(rc.Chain(P, bubble.end_types))["failed"] == ["splice"]
    # a splice radius the old mesh does not hold: the cap and the plug meshed coarsely
    coarse, _ = pierced_bubble(h=parts["a"] / 4)
    checks = _refused(coarse, ring_settings(rho=0.2))
    assert checks["failed"] == ["splice"] and checks["rho"] < checks["rho_min"]
    # a waist below a tenth of the splice radius
    assert _refused(bubble, ring_settings(waist=0.05))["failed"] == ["closure"]
    # a ring born below bubble_stop of the bubble's volume at the gas cut
    checks = _refused(bubble, V_b=200.0 * rc.revolved_volume(bubble.points))
    assert checks["failed"] == ["volume"] and checks["V_ring"] < checks["V_ring_limit"]


# --- the helpers of a loop ------------------------------------------------------------------------


def circle(n=64, c=(1.0, 0.0), a=0.3, phase=0.0):
    t = phase + np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([c[0] + a * np.cos(t), c[1] + a * np.sin(t)])


def test_loop_crossings_none_on_a_circle_one_on_a_figure_eight():
    assert ring.loop_crossings(circle()) == []
    t = np.linspace(0.0, 2.0 * math.pi, 101, endpoint=False) + 0.013
    eight = np.column_stack([1.0 + 0.3 * np.sin(t), 0.3 * np.sin(t) * np.cos(t)])
    assert len(ring.loop_crossings(eight)) == 1
    # the closing chord is adjacent to the first: no contact at the seam
    assert (0, 63) not in ring.loop_crossings(circle())


def peanut(n=200, phase=0.0):
    t = phase + np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    rho = 0.05 + np.cos(t) ** 2
    return np.column_stack([1.5 + rho * np.cos(t), rho * np.sin(t)])


def test_the_fold_of_a_loop_is_periodic_and_the_same_for_every_seam():
    """The waist of a peanut is its fold, the pair across it 0.1 apart; a circle has none, where
    the open fold of its closed polyline would read the seam as one; rotating the seam moves the
    pair's indices only."""
    kw = {"ratio": 4.0, "stop": 0.2, "stop_elements": 3.0, "search_factor": 4.0}
    L = peanut()
    f = ring.loop_fold_gap(L, **kw)
    assert f.gap == pytest.approx(0.1, rel=2e-2) and not f.crossed
    assert f.r == pytest.approx(1.5, abs=1e-3) and abs(f.z) < 1e-3
    a, b = f.pair
    assert {round(float(L[a, 1]), 2), round(float(L[b, 1]), 2)} == {0.05, -0.05}
    for k in (1, 37, 99, 150):
        g = ring.loop_fold_gap(np.roll(L, -k, axis=0), **kw)
        assert (g.gap, g.r, g.z, g.arc) == pytest.approx((f.gap, f.r, f.z, f.arc), rel=1e-12)
    C = circle()
    assert math.isinf(ring.loop_fold_gap(C, **kw).gap)
    closed = ring.closed(C)
    assert math.isfinite(fold_gap(closed[:, 0], closed[:, 1], **kw).gap)  # the seam, read open


def test_the_halves_of_a_loop_split_on_vertices_and_their_splines_are_checked():
    for n in (64, 66, 130):
        L = circle(n)
        a, b = ring.loop_halves(L)
        m = len(a) - 1
        assert m % 2 == 0 and np.array_equal(a[0], L[0]) and np.array_equal(b[0], L[m])
        assert np.array_equal(b[-1], L[0]) and len(a) + len(b) == n + 2
        assert ring.loop_spline_contact(L) is None
    # a simple polygon whose spline loops in a short interval inside the first half
    hairpin = np.array([[1.0, 0.0], [1.2, 0.5], [1.4, 1.0], [2.0, 2.0], [2.1, 2.0], [2.7, 1.0],
                        [2.9, 0.5], [3.1, 0.0], [3.1, -1.0], [2.1, -1.5], [1.0, -1.0],
                        [1.0, -0.5]])
    assert ring.loop_crossings(hairpin) == []
    contact = ring.loop_spline_contact(hairpin)
    assert contact is not None and contact["what"] == "spline"
    figure = np.column_stack([1.0 + 0.3 * np.sin(np.linspace(0, 2 * math.pi, 41)[:-1] + 0.01),
                              0.3 * np.sin(np.linspace(0, 2 * math.pi, 41)[:-1] + 0.01)
                              * np.cos(np.linspace(0, 2 * math.pi, 41)[:-1] + 0.01)])
    assert ring.loop_spline_contact(figure)["what"] == "polygon"


def test_the_quadratic_elements_of_a_loop_and_the_check_of_its_order():
    L = circle(64)
    e = ring.loop_edges(L)
    assert e["edges"] is not None and len(e["edges"]["L"]) == 32  # 2N + 1 nodes closed
    assert e["turn"] == pytest.approx(360.0 / 32, rel=1e-2)
    assert ring.loop_edges(circle(63))["edges"] is None  # an odd count is no loop of elements
    ring.validate_loop_order(L)
    swapped = L.copy()
    swapped[[10, 42]] = swapped[[42, 10]]  # two jumps half way round
    with pytest.raises(RuntimeError, match="not geometrically ordered"):
        ring.validate_loop_order(swapped)
    with pytest.raises(RuntimeError, match="repeats its start"):
        ring.validate_loop_order(ring.closed(L))
    rec = ring.loop_record(L, u=np.tile([0.0, 2.0], (64, 1)))
    assert rec["V"] == pytest.approx(2 * math.pi**2 * 1.0 * 0.3**2, rel=2e-3)  # Pappus
    assert rec["r_c"] == pytest.approx(1.0, abs=1e-12) and rec["r_min"] == pytest.approx(0.7)
    assert rec["points"] == 64 and rec["u_max"] == 2.0
    assert rec["courant"] == pytest.approx(2.0 * rec["h_min"] / 2.0, rel=1e-9)
