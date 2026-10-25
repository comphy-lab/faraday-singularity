"""The fill at the floor of a wedge of gas of the ring, without a finite-element problem
(``ring.floor_wedge``; ``driver.CollapseDriver._floor_fill_request`` on a stand-in; the driver's
meeting of it, ``_meet_groove`` with ``groovesurgery.groove_fill`` stood in for): the corner r1 of
the delta triple reached at step 7506, 91 steps after its stop in the ring stage, a wedge of gas of
the ring opening at about 50 degrees whose apex the guard read at an element of the run's floor
(152.7 degrees, chord 7.2e-7, beyond the refine's cap), the finger a groove fill had left.

* the wedge, read at the scale of the floor: a vertex where the loop (counter-clockwise, its gas
  inside) turns left; the widest symmetric pair about it closer than ``fold_stop_elements``
  element lengths and with ``FLOOR_FOLD_RATIO`` times its gap below the arc between them; none
  for a vertex turning right (the tip of a finger of liquid) or a wedge too open to be a fold;
* the request: the guard's ``interface-unresolved`` on the ring at an element at the floor (an
  edge turn, a midside node, a feature below the floor), in the ring stage with jetcore's
  recentred far kinks and ``groove.enabled``: an ``RF`` of the fill with ``contact`` ``floor`` and
  ``floor`` (the reading, the wedge, the fold limits of the floor), the pair one the fill's own
  trigger accepts under those limits and refuses under the stops' own; nothing else;
* the former gas the fill's check of the transfer predicts (``groovesurgery.floor_bulk_inside``):
  its filled region read on the old elements' quadratic curve, less the band of the old elements
  that pyoomph's locator reaches;
* the driver: the fill is asked for with the floor's fold limits and a check of the upward jet's
  base (a state without a born jet not checked), an event ``floor-fill`` and ``floor_fills`` in
  the run's events; an ordinary groove's fill is asked for as before, without a base check.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import groovesurgery  # noqa: E402
from faraday_singularity.driver import CollapseDriver  # noqa: E402
from faraday_singularity.ring import FLOOR_FOLD_RATIO, floor_wedge, signed_area  # noqa: E402

H = 5e-7  # the node spacing about the apex: elements of 1e-6
APEX = (3.7e-4, 4.6115)


def teardrop(half=25.0, length=4e-5, h=H, apex=APEX, arc_points=120, roll=0):
    """A ring whose gas ends in a wedge of half-angle ``half`` degrees, its apex at ``apex``
    pointing to smaller ``r``: the lower wall out from the apex, an arc round, the upper wall
    back, counter-clockwise (the gas inside), rolled by ``roll``; and the apex's index."""
    t = math.radians(half)
    s = np.arange(h, length + 0.5 * h, h)
    a = np.asarray(apex, dtype=float)
    lower = a + np.column_stack([s * math.cos(t), -s * math.sin(t)])
    centre = a + np.array([length * math.cos(t), 0.0])
    rad = length * math.sin(t)
    phi = np.linspace(-0.5 * math.pi, 0.5 * math.pi, arc_points)[1:-1]
    arc = centre + rad * np.column_stack([np.cos(phi), np.sin(phi)])
    upper = (a + np.column_stack([s * math.cos(t), s * math.sin(t)]))[::-1]
    L = np.vstack([a[None, :], lower, arc, upper])
    assert signed_area(L) > 0.0
    return np.roll(L, roll, axis=0), roll % len(L)


def notched(depth=1.2e-5, h=H):
    """A ring with a notch of liquid pushed into its gas: its vertex turns right."""
    L, i = teardrop(half=60.0, length=4e-5, h=h)
    out = L.copy()
    k = len(L) // 2 + 30  # a node of the arc, pushed inward along -r
    out[k] = out[k] - np.array([depth, 0.0])
    return out, k


def test_the_wedge_of_gas_is_read_at_the_scale_of_the_floor():
    for roll in (0, 37, -11):  # wherever the seam lies
        L, i = teardrop(roll=roll)
        w = floor_wedge(L, i, stop_elements=3.0)
        assert w is not None and w["apex"] == i and w["ratio"] == FLOOR_FOLD_RATIO
        assert w["turn"] == pytest.approx(130.0, abs=1e-6) and w["angle"] == pytest.approx(50.0)
        a, b = w["pair"]
        n = len(L)
        assert {(i - w["m"]) % n, (i + w["m"]) % n} == {a, b}
        # the gap of the walls 2 s sin(25) below three element lengths (2 H each): s < 3.55 H
        assert w["m"] == 7 and w["gap"] == pytest.approx(2 * 7 * H * math.sin(math.radians(25)))
        assert w["h_elem"] == pytest.approx(2 * H) and w["limit"] == pytest.approx(6 * H)
        assert w["gap"] < w["limit"] and FLOOR_FOLD_RATIO * w["gap"] < w["arc"]
    # a wedge too open to be a fold at any scale (sin(55) FLOOR_FOLD_RATIO > 1), none
    L, i = teardrop(half=55.0)
    assert floor_wedge(L, i, stop_elements=3.0) is None
    # the tip of a finger of liquid: the loop turns right there
    L, k = notched()
    assert floor_wedge(L, k, stop_elements=3.0) is None


def _fake_driver(L, *, enabled=True, jet="recentre", ring=True, removed=False, h_floor=7.8125e-7,
                 loops=1, open_chain=None):
    from faraday_singularity.reconnect import edge_reading
    from faraday_singularity.ring import loop_edges

    events = ev.EventSettings.from_dict({"groove": {"enabled": enabled}})
    loop = {"r": L[:, 0], "z": L[:, 1]}
    edges = {"ring": loop_edges(L)} if loops == 1 else {}
    if open_chain is not None:
        edges["open"] = edge_reading(open_chain)
    fake = SimpleNamespace(
        events=events, s=SimpleNamespace(refine_ratio=1.5, fold_stop_elements=3.0),
        _ring=lambda: ({"removed": {"k": 9} if removed else None} if ring else None),
        _jetcore=lambda: (SimpleNamespace(far_kinks=jet) if jet is not None else None),
        _h_floor=lambda: h_floor,
        _post_cut_state=lambda: {"loops": [loop] * loops, "edges": edges},
        _t=lambda: 10.95, _taus=lambda t: {"tau": 2.5e-5, "tau_p": 2.3e-5})
    fake._ring_jacobian_reading = lambda st: CollapseDriver._ring_jacobian_reading(fake, st)
    fake._refined_chain = lambda: CollapseDriver._refined_chain(fake)
    return fake


def _guard(loop, i, **kw):
    out = {"reason": "interface-unresolved", "trigger": "edge-turn", "chain": "ring",
           "turn": 152.7, "limit": 90.0, "fresh": True, "r": float(loop[i, 0]),
           "z": float(loop[i, 1]), "L": 1.0e-6, "index": i, "tau": 2.5e-5}
    out.update(kw)
    return out


def test_a_wedge_at_the_floor_is_the_request_of_its_fill_and_nothing_else_is():
    L, i = teardrop(roll=23)
    req = CollapseDriver._floor_fill_request(_fake_driver(L), _guard(L, i))
    assert (req["reason"], req["kind"], req["class"], req["handler"]) == (
        ev.EVENT_REQUEST, "ring-fold", "RF", ev.GROOVE_HANDLER)
    assert req["chain"] == "ring" and req["contact"] == "floor" and req["side"] == "gas"
    w = floor_wedge(L, i, stop_elements=3.0)
    assert req["fold_pair"] == w["pair"] and req["fold_gap"] == w["gap"]
    assert req["fold_limit"] == w["limit"] and req["floor"]["wedge"] == w
    fold = req["floor"]["fold"]
    assert fold == {"ratio": FLOOR_FOLD_RATIO, "stop": 0.0, "stop_elements": 3.0}
    assert req["floor"]["guard"] == {k: v for k, v in _guard(L, i).items() if k != "reason"}
    assert (req["tau"], req["tau_p"]) == (2.5e-5, 2.3e-5)
    json.dumps(req)  # plain values: it goes into the dumps and records
    # the fill's own trigger, read again: a fold at the floor, none under the stops' limits
    trig = groovesurgery.fill_trigger(L, req["fold_pair"], loop=True, fold=fold, mouth=1.5)
    assert trig["holds"] and trig["width"] == pytest.approx(1.5 * w["limit"])
    plain = groovesurgery.fill_trigger(L, req["fold_pair"], loop=True,
                                       fold={"ratio": 4.0, "stop": 5e-5, "stop_elements": 3.0},
                                       mouth=1.5)
    assert not plain["holds"]
    # the other triggers at the floor
    for trig_ in ("edge-midside", "feature-below-floor", "curved-jacobian"):
        assert CollapseDriver._floor_fill_request(_fake_driver(L), _guard(L, i, trigger=trig_))
    # none of these
    nothing = [_guard(L, i, chain="open"), _guard(L, i, L=1.2e-6),
               _guard(L, i, trigger="absolute-quality"), dict(_guard(L, i), reason="event-request"),
               _guard(L, i, index=None), None]
    for stop in nothing:
        assert CollapseDriver._floor_fill_request(_fake_driver(L), stop) is None, stop
    for kw in ({"enabled": False}, {"jet": None}, {"jet": "grow"}, {"ring": False},
               {"removed": True}, {"loops": 0}, {"loops": 2}, {"h_floor": 6e-7}):
        assert CollapseDriver._floor_fill_request(_fake_driver(L, **kw), _guard(L, i)) is None, kw
    # the tip of a finger of liquid stays the stop
    N, k = notched()
    assert CollapseDriver._floor_fill_request(_fake_driver(N), _guard(N, k)) is None


def test_the_former_gas_of_a_fill_at_the_floor_is_read_beyond_the_band_of_the_old_elements():
    """``groovesurgery.floor_bulk_inside``: the bulk nodes a fill at the floor predicts unlocated
    are those of its filled region read on the old elements' quadratic curve, beyond
    ``LOCATOR_BAND`` of the old element's length from it (pyoomph's locator takes those nearer
    onto the old element's edge: 9 of the 64 nodes the second fill at the floor of the smoke from
    r1's step 7506 planned, all within 0.059 element lengths of the old surface); the chord
    reading shrunk by a twentieth of the arc's shortest chord keeps them."""
    import shapely

    from faraday_singularity.groovesurgery import LOCATOR_BAND, floor_bulk_inside
    from faraday_singularity.ringsurgery import _inside

    P, apex = teardrop(arc_points=121)
    n = len(P)
    assert n % 2 == 0 and apex == 0  # the seam on a vertex, the apex
    i_s, j_s = n - 10, 10  # the splices about the apex, the walk wrapping the seam
    C = np.linspace(P[i_s], P[j_s], 12)[1:-1]
    L_e = 2.0 * H  # the elements of the walls
    t = math.radians(25.0)
    u, nrm = np.array([math.cos(t), -math.sin(t)]), np.array([math.sin(t), math.cos(t)])
    base = np.asarray(APEX) + 6 * H * u  # a vertex of the lower wall, inside the fill
    depths = (0.03, 0.05, 0.1, 0.2)
    X = np.vstack([base + d * L_e * nrm for d in depths] + [base - 0.2 * L_e * nrm])  # + liquid
    mask, rec = floor_bulk_inside(X, P, i_s, j_s, C, 0.05 * H)
    assert mask.tolist() == [False, False, True, True, False]
    assert rec == {"mode": "floor-band", "band": LOCATOR_BAND, "samples": 8, "inside": 4,
                   "in_band": 2, "predicted": 2}
    walk = (i_s + np.arange((j_s - i_s) % n + 1)) % n
    chord = _inside(X, np.vstack([P[walk], C[::-1]]), 0.05 * H)
    assert chord.tolist() == [True, True, True, True, False]  # the nodes the locator takes kept
    # a midside node pushed into the gas: its element's curve bulges past its chords, a quarter
    # of the push, and a node a tenth of an element off the chord lies within the band of the curve
    Pc = P.copy()
    Pc[7] = Pc[7] + 0.3 * L_e * nrm  # the midside node of the element (6, 7, 8)
    d67 = (Pc[7] - Pc[6]) / np.hypot(*(Pc[7] - Pc[6]))
    y = 0.5 * (Pc[6] + Pc[7]) + 0.1 * L_e * np.array([-d67[1], d67[0]])
    chords = shapely.LineString(Pc[walk])
    assert shapely.distance(chords, shapely.Point(y)) > LOCATOR_BAND * L_e
    mask, rec = floor_bulk_inside(y[None, :], Pc, i_s, j_s, C, 0.05 * H)
    assert not mask[0] and rec["in_band"] == 1
    with pytest.raises(ValueError):
        floor_bulk_inside(X, P[:-1], i_s - 1, j_s, C, 0.05 * H)  # an odd loop


def test_a_curved_element_of_the_ring_is_read_as_the_rings_element_there():
    """``CollapseDriver._ring_jacobian_reading``: a rebuilt mesh's ``curved-jacobian`` stop whose
    element (its centroid) lies within a chord of the ring's nearest element, and no nearer to
    the open chain, is that element's reading on the ring (r1 at 63293b0 on two threads, step
    7501: a corner turning by 89.8 degrees within one element of 3.3e-6, rebuilt into an element
    whose Jacobian bound fell to 0.09); met by the refine above the floor and by the fill at the
    floor at a wedge of gas; any other stop as it is."""
    from faraday_singularity.ring import loop_edges

    L, i = teardrop(arc_points=121, roll=24)  # an even loop: its quadratic elements
    e = loop_edges(L)["edges"]
    k = int(np.argmin(np.hypot(e["mid"][:, 0] - L[i, 0], e["mid"][:, 1] - L[i, 1])))
    c = e["mid"][k] + np.array([0.3 * e["L"][k], 0.0])  # the curved element's centroid
    stop = {"reason": "interface-unresolved", "trigger": "curved-jacobian", "jacobian_bound": 0.09,
            "limit": 0.2, "r": float(c[0]), "z": float(c[1]), "on_interface": True,
            "elements_below": 1, "fresh": True}
    far_open = np.column_stack([np.linspace(0.0, 1e-3, 41), np.full(41, 4.7)])
    fake = _fake_driver(L, open_chain=far_open)
    got = CollapseDriver._ring_jacobian_reading(fake, dict(stop))
    assert got["chain"] == "ring" and got["index"] == int(e["index"][k])
    assert got["L"] == float(e["L"][k]) and got["turn"] == float(e["turn"][k])
    assert (got["r"], got["z"]) == (float(e["mid"][k, 0]), float(e["mid"][k, 1]))
    assert got["element"] == {"r": stop["r"], "z": stop["z"]} and got["jacobian_bound"] == 0.09
    assert {k_: got[k_] for k_ in stop if k_ not in ("r", "z")} == {
        k_: v for k_, v in stop.items() if k_ not in ("r", "z")}
    assert CollapseDriver._ring_jacobian_reading(fake, got) is got  # read once
    # met: at the floor (elements of 1e-6) by the fill of the wedge, above it by the refine
    req = CollapseDriver._floor_fill_request(fake, dict(stop))
    assert req is not None and req["contact"] == "floor"
    assert req["floor"]["guard"]["trigger"] == "curved-jacobian"
    assert req["floor"]["guard"]["index"] == got["index"]
    assert not CollapseDriver._ring_refine_due(fake, dict(stop))
    above = _fake_driver(L, open_chain=far_open, h_floor=1e-7)
    assert CollapseDriver._ring_refine_due(above, dict(stop))
    assert CollapseDriver._floor_fill_request(above, dict(stop)) is None
    # as it is: off the interface, beyond a chord of the ring, nearer the open chain, another
    # trigger, the ring removed or without recentred far kinks
    beyond = dict(stop, r=float(e["mid"][k, 0] - 3.0 * e["L"][k]))
    near_open = np.column_stack([np.full(401, stop["r"] + 1e-8),
                                 np.linspace(stop["z"] - 2e-5, stop["z"] + 2e-5, 401)])
    for fk, st in ((fake, dict(stop, on_interface=False)), (fake, beyond),
                   (_fake_driver(L, open_chain=near_open), dict(stop)),
                   (fake, dict(stop, trigger="absolute-quality")),
                   (_fake_driver(L, removed=True), dict(stop)),
                   (_fake_driver(L, jet="grow"), dict(stop)), (_fake_driver(L, loops=0), dict(stop))):
        assert CollapseDriver._ring_jacobian_reading(fk, st) == st and "chain" not in st
    assert CollapseDriver._ring_jacobian_reading(fake, None) is None


def _floor_request(L, i):
    return CollapseDriver._floor_fill_request(_fake_driver(L), _guard(L, i))


def test_the_driver_fills_the_wedge_with_the_floors_limits_and_checks_the_base(tmp_path,
                                                                                monkeypatch):
    """Through the run loop of a stand-in driver (``test_groove_driver``'s): the floor's request
    reaches the fill with the fold limits of the floor and a check of the base; the fill made, an
    event ``floor-fill`` and ``floor_fills`` 1 in the run's events; an ordinary groove's fill
    asked for without a base check, as before."""
    from test_bookkeeping import CutProblem
    from test_events_plumbing import _events
    from test_groove_driver import GROOVE_ON, _run, fake_fill, groove_request

    L, i = teardrop()
    for name, request in (("floor", _floor_request(L, i)),
                          ("plain", groove_request("ring", "RF"))):
        request = dict(request, handler=ev.GROOVE_HANDLER)
        calls, checks = [], []
        inner = fake_fill(calls)

        def groove_fill(*a, base_check=None, **kw):
            checks.append(base_check)
            return inner(*a, **kw)

        run = tmp_path / name
        drv = _run(run, CutProblem(), GROOVE_ON, request, monkeypatch, calls)
        monkeypatch.setattr(groovesurgery, "groove_fill", groove_fill)
        out = drv.run(1e-9)
        drv.close()
        assert len(calls) == 1 and out["stop"]["reason"] == "max-steps", out["stop"]
        assert out["events"]["by_kind"] == {"groove-fill": 1}
        floors = [e for e in _events(run) if e["event"] == "floor-fill"]
        if name == "floor":
            assert calls[0]["fold"] == request["floor"]["fold"]
            assert callable(checks[0]) and out["events"]["floor_fills"] == 1
            guard = checks[0]()  # the stand-in's base: none stood before, so none is checked
            assert guard["ok"] and guard["checked"] is (not any(
                math.isnan(v) for v in guard["before"].values()))
            assert len(floors) == 1 and floors[0]["k"] == 1
            assert floors[0]["wedge"]["pair"] == request["fold_pair"]
            assert floors[0]["guard"]["trigger"] == "edge-turn"
        else:
            assert calls[0]["fold"]["ratio"] == drv.s.fold_ratio and checks == [None]
            assert "floor_fills" not in out["events"] and floors == []


def corner(h=H, n=12, apex=(1.74e-4, 4.6116)):
    """A ring whose gas has a right-angled corner at ``apex`` (the root of the radial sheet where
    its top meets the column's wall): a wall down from above to the apex, then out along +r, the
    rest closed round counter-clockwise (its gas inside), and the apex's index."""
    a = np.asarray(apex, dtype=float)
    down = a + np.column_stack([np.zeros(n), (n - np.arange(n)) * h])  # from above to the apex
    out = a + np.column_stack([np.arange(1, n + 1) * h, np.zeros(n)])
    far = out[-1] + np.column_stack([np.zeros(2 * n), np.linspace(h, n * h, 2 * n)])
    back = np.column_stack([np.linspace(out[-1][0] - h, a[0] + h, 2 * n), np.full(2 * n,
                                                                              a[1] + (n + 1) * h)])
    # counter-clockwise with the gas inside: from the apex out along +r, up, back, down to it
    L = np.vstack([a[None, :], out, far, back, down[1:]])
    if signed_area(L) < 0.0:
        L = L[::-1].copy()
    i = int(np.argmin(np.hypot(L[:, 0] - a[0], L[:, 1] - a[1])))
    return L, i


def test_a_corner_of_gas_at_the_floor_is_read_with_the_corners_ratio():
    """``ring.FLOOR_CORNER_RATIO``: where no wedge is (a right-angled corner fails the wedge's
    1.5), the corner of gas open up to about 100 degrees is read with 1.3, the request's fold
    limits the corner's, ``corner`` in its wedge; a liquid corner (a right turn) none."""
    from faraday_singularity.ring import FLOOR_CORNER_RATIO

    L, i = corner()
    assert floor_wedge(L, i, stop_elements=3.0) is None
    w = floor_wedge(L, i, stop_elements=3.0, ratio=FLOOR_CORNER_RATIO)
    assert w is not None and w["angle"] == pytest.approx(90.0, abs=1.0)
    req = CollapseDriver._floor_fill_request(_fake_driver(L), _guard(L, i, turn=99.1))
    assert req is not None and req["floor"]["wedge"]["corner"] is True
    assert req["floor"]["fold"]["ratio"] == FLOOR_CORNER_RATIO
    trig = groovesurgery.fill_trigger(L, req["fold_pair"], loop=True, fold=req["floor"]["fold"],
                                      mouth=1.5)
    assert trig["holds"]
    # a wedge stays the wedge's: no corner flag, the wedge's ratio
    T, j = teardrop(roll=23)
    req = CollapseDriver._floor_fill_request(_fake_driver(T), _guard(T, j))
    assert "corner" not in req["floor"]["wedge"] and req["floor"]["fold"]["ratio"] == FLOOR_FOLD_RATIO
    # the corner of the liquid (the loop turning right) stays the stop
    R = L[::-1].copy()  # clockwise: its gas outside, the same corner a liquid one
    k = len(R) - 1 - i
    assert floor_wedge(R, k, stop_elements=3.0, ratio=FLOOR_CORNER_RATIO) is None
