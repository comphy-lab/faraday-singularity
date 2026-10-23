"""The side of a fold (``groove.side_folds``, ``groove.sheets`` ``log``): a fold whose pocket is
liquid is a thin liquid sheet, not an event, without a finite-element problem.

The synthetic states: an elliptic ring (``test_ring_stops``' at ``(0.01, 4.68)``) whose inner wall
carries a liquid sheet out into the ring's gas, straight (the replica of the sheet of event 5 of the
ring stage), thinned to the scale of the mesh, or curled back so that its tip comes up to the wall
across the gas; the ring with a narrow finger of gas (``test_groove_geometry.finger_ring``); a flat
open chain with a thin liquid fin standing in the gas above it, or with a slot of gas
(``test_groove_geometry.slot_open``), above a bubble.

* each fold is classed by its pocket's side: the sheet liquid, the finger, the slot and the gap
  under the curled sheet gas;
* with ``sheets`` ``event`` (the default) the stops are as at c9a2141: the sheet a ``ring-fold``
  (``RF``) or a ``groove`` (``A-groove``); with ``log`` the sheet is no event, the groove of gas
  still one (to the fill), a sheet hides no groove of gas (the curled sheet whose own faces lie
  closer than the gap at its tip), and a sheet whose faces close below ``sheet_contact`` element
  lengths is a ``sheet-contact`` without a handler;
* the driver (stand-ins): a sheet is a row of ``sheets.txt`` at most once per rebuild and the run
  goes on; a ``sheet-contact`` is a resumable stop, never the fill.
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

import test_groove_geometry as G  # noqa: E402
import test_ring_stops as rs  # noqa: E402

from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import groove  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402

C, A_E, B_E = (0.01, 4.68), 2e-3, 1e-3  # the ring
FOLD = {"ratio": 4.0, "stop": 0.05 * 1e-3, "stop_elements": 3.0, "search_factor": 4.0}
LOG = ev.EventSettings.from_dict({"ring": {"enabled": True},
                                  "groove": {"enabled": True, "sheets": "log"}})
EVENT = ev.EventSettings.from_dict({"ring": {"enabled": True}, "groove": {"enabled": True}})


def _ccw(L):
    L = np.asarray(L, dtype=float)
    return L if 0.5 * np.sum(L[:, 0] * np.roll(L[:, 1], -1) - np.roll(L[:, 0], -1) * L[:, 1]) \
        > 0 else L[::-1].copy()


def sheeted_ring(t=3e-5, L=3e-4, curl=None, gap=None, h=3e-6, c=C, a=A_E, b=B_E, round_root=0.0):
    """The ring with a liquid sheet of thickness ``t`` from its inner wall (its leftmost point) out
    into its gas, ``L`` long; with ``curl`` (a radius) the sheet turns up through a half circle and
    runs back towards the wall, its tip ``gap`` from it. The gas is the ellipse less the sheet
    (shapely), its boundary sampled ``h`` apart about the sheet, counter-clockwise, an even
    count. ``c``, ``a``, ``b`` the ellipse; ``round_root`` a radius of the rounding of the gas's
    corners where the sheet leaves the wall (none by default)."""
    from shapely.geometry import LineString, Polygon

    s = np.linspace(0.0, 2.0 * math.pi, 40001)[:-1]
    E = Polygon(np.column_stack([c[0] + a * np.cos(s), c[1] + b * np.sin(s)]))
    r_in, z0 = c[0] - a, c[1]
    line = [(r_in - 2e-5, z0), (r_in + L, z0)]
    if curl is not None:
        phi = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 400)[1:]
        line += [(r_in + L + curl * math.cos(p), z0 + curl + curl * math.sin(p)) for p in phi]
        z1 = z0 + 2.0 * curl
        r_wall = c[0] - a * math.sqrt(1.0 - ((z1 - c[1]) / b) ** 2)
        line.append((r_wall + gap + 0.5 * t, z1))
    sheet = LineString(line).buffer(0.5 * t, quad_segs=64)
    gas = E.difference(sheet)
    if round_root:
        gas = gas.buffer(-round_root, quad_segs=64).buffer(round_root, quad_segs=64)
    assert gas.geom_type == "Polygon" and not gas.interiors
    dense = np.asarray(gas.exterior.coords)
    dense = np.vstack([dense[:-1], dense[:1]])
    centre = (r_in + 0.5 * L, z0 + (curl or 0.0))

    def spacing(p):
        d = math.hypot(p[0] - centre[0], p[1] - centre[1])
        return h if d < L + 3 * (curl or 0) else min(4e-5 * a / A_E, h + 0.05 * (d - L))

    Lp = G.resample(dense, spacing)[:-1]
    Lp = _ccw(Lp)
    return Lp[:-1] if len(Lp) % 2 else Lp


def sides(P, loop):
    return groove.side_folds(P, loop=loop, **FOLD)


def ring_stops(L, events):
    return rs.stops([rs.flat_open()], [L], events=events, V_b=10.0 * rc.revolved_volume(L))


def test_each_fold_is_classed_by_the_side_of_its_pocket():
    sheet = sheeted_ring()
    got = sides(sheet, True)
    assert math.isfinite(got["liquid"]["fold"].gap) and got["liquid"]["fold"].gap < 5e-5
    assert not math.isfinite(got["gas"]["fold"].gap)
    assert got["liquid"]["fold"].gap == pytest.approx(3e-5, rel=0.05)
    assert got["liquid"]["h_fine"] == pytest.approx(2 * 3e-6, rel=0.3)
    finger = G.finger_ring()
    got = sides(finger, True)
    assert math.isfinite(got["gas"]["fold"].gap) and not math.isfinite(got["liquid"]["fold"].gap)
    slot = G.slot_open()
    got = sides(slot.points, False)
    assert math.isfinite(got["gas"]["fold"].gap) and not math.isfinite(got["liquid"]["fold"].gap)
    # the side of a fold agrees with the pocket's share of the gas (groove.gas_share)
    for P, loop, side in ((sheet, True, "liquid"), (finger, True, "gas"),
                          (slot.points, False, "gas")):
        a, b = sides(P, loop)[side]["fold"].pair
        gas = P if loop else rc.closed_section(P, ("axis", "fixed"), G.RESERVOIR)
        assert groove.side_of(groove.gas_share(P[a:b + 1], gas)) == side
    # the plain fold is the one of the two sides closer to its limit
    plain = __import__("faraday_singularity.ring", fromlist=["x"]).loop_fold_gap(sheet, **FOLD)
    assert plain.pair == got_pair(sides(sheet, True))


def got_pair(s):
    best = min(s.values(), key=lambda x: x["fold"].gap / max(FOLD["stop"], FOLD["stop_elements"]
                                                             * x["fold"].h_elem)
               if math.isfinite(x["fold"].gap) else math.inf)
    return best["fold"].pair


def test_a_sheet_of_the_ring_is_no_event_with_log_and_a_fold_as_before_without():
    sheet = sheeted_ring()
    before = ring_stops(sheet, EVENT)
    assert (before["kind"], before["class"], before["handler"]) == ("ring-fold", "RF",
                                                                    ev.GROOVE_HANDLER)
    assert "side" not in before
    assert ring_stops(sheet, LOG) is None
    # a finger of gas of the ring is still a fold, to the fill, its side recorded
    finger = G.finger_ring()
    req = ring_stops(finger, LOG)
    assert (req["kind"], req["class"], req["handler"], req["side"]) == (
        "ring-fold", "RF", ev.GROOVE_HANDLER, "gas")
    plain = ring_stops(finger, EVENT)
    assert {k: v for k, v in req.items() if k != "side"} == plain


def test_a_curled_sheet_coming_up_to_the_wall_is_the_groove_under_it():
    """The sheet curled back so that its tip lies 3e-5 from the wall across the gas: its own faces
    (2e-5 apart) are the fold closest to its limit, which hides the gap under it from the plain fold
    test; with ``log`` the gap, a pocket of gas, is the fold met, the fill's."""
    curled = sheeted_ring(t=2e-5, L=2e-4, curl=6e-5, gap=3e-5)
    got = sides(curled, True)
    assert got["liquid"]["fold"].gap == pytest.approx(2e-5, rel=0.1)
    assert got["gas"]["fold"].gap == pytest.approx(3e-5, rel=0.15)
    plain = ring_stops(curled, EVENT)
    assert plain["fold_pair"] == list(got["liquid"]["fold"].pair)
    req = ring_stops(curled, LOG)
    assert (req["kind"], req["class"], req["side"]) == ("ring-fold", "RF", "gas")
    assert req["fold_pair"] == list(got["gas"]["fold"].pair)
    # its pocket, the gas between the curl and the wall, is the fill's to take
    a, b = req["fold_pair"]
    assert groove.side_of(groove.gas_share(curled[a:b + 1], curled)) == "gas"


def test_a_sheet_whose_faces_meet_at_the_scale_of_the_mesh_is_a_contact_without_a_handler():
    thin = sheeted_ring(t=5e-6, L=2e-4, h=3e-6)
    got = sides(thin, True)["liquid"]
    assert got["fold"].gap < 1.0 * got["h_fine"]
    req = ring_stops(thin, LOG)
    assert (req["kind"], req["class"], req["handler"], req["side"]) == (
        ev.SHEET_CONTACT, "RF", None, "liquid")
    assert req["fold_gap"] < req["sheet_contact_limit"] == pytest.approx(got["h_fine"])
    # a stricter contact: below a fifth of an element, no event
    strict = ev.EventSettings.from_dict({"ring": {"enabled": True}, "groove": {
        "enabled": True, "sheets": "log", "sheet_contact": 0.2}})
    assert ring_stops(thin, strict) is None


def fin_open(z=4.69, r1=0.02, t=2e-5, H=3e-4, h=3e-6):
    """A flat open chain with a liquid fin standing in the gas above it at ``r1``: up the fin's left
    face, over its round top, down its right face; graded away from it."""
    x0, x1 = r1 - 0.5 * t, r1 + 0.5 * t
    top = G._arc_points((r1, z + H), 0.5 * t, math.pi, 0.0)
    parts = [np.column_stack([np.linspace(0.0, x0, 40001), np.full(40001, z)]),
             np.column_stack([np.full(4001, x0), np.linspace(z, z + H, 4001)])[1:], top[1:],
             np.column_stack([np.full(4001, x1), np.linspace(z + H, z, 4001)])[1:],
             np.column_stack([np.linspace(x1, G.R_C, 200001), np.full(200001, z)])[1:]]
    dense = np.vstack(parts)
    P = G.resample(dense, G.spacing((r1, z + 0.5 * H), fine=h, coarse=4e-3, reach=3e-4))
    return rc.Chain(P, ("axis", "fixed"))


def bubble_below():
    t = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 801)
    P = np.column_stack([2e-3 * np.cos(t), 4.4 + 1e-3 * np.sin(t)])
    P[0, 0] = P[-1, 0] = 0.0
    return rc.Chain(P, ("axis", "axis"))


def chain_stops(open_chain, events):
    chains = [bubble_below(), open_chain]
    s = rc.ReconnectSettings(delta=1e-3)
    cut = rc.cut_geometry(chains, 3.0, gas_before=1.0)
    return rc.post_cut_stops(chains, cut, s, s.tip_size * s.delta, events=events)


def test_on_the_open_chain_a_fin_is_no_event_with_log_and_a_slot_is_a_groove():
    fin = fin_open()
    got = sides(fin.points, False)
    assert got["liquid"]["fold"].gap == pytest.approx(2e-5, rel=0.05)
    g = got["gas"]["fold"]  # at most a pair across the fin's root, beyond the fold's limit
    assert not (g.gap < max(FOLD["stop"], FOLD["stop_elements"] * g.h_elem))
    before = chain_stops(fin, EVENT)
    assert (before["kind"], before["class"]) == ("groove", "A-groove") and "side" not in before
    assert chain_stops(fin, LOG) is None
    slot = G.slot_open()
    req = chain_stops(slot, LOG)
    assert (req["kind"], req["class"], req["handler"], req["side"]) == (
        "groove", "A-groove", ev.GROOVE_HANDLER, "gas")
    assert {k: v for k, v in req.items() if k != "side"} == chain_stops(slot, EVENT)
    thin = fin_open(t=5e-6, H=2e-4)
    req = chain_stops(thin, LOG)
    assert (req["kind"], req["class"], req["handler"]) == (ev.SHEET_CONTACT, "A-groove", None)


def test_the_sheet_keys_are_validated():
    ok = ev.GrooveEventSettings
    assert ok().sheets == "event" and ok().sheet_contact == 1.0
    ok(sheets="log", sheet_contact=0.5)
    for bad in ({"sheets": "ignore"}, {"sheet_contact": 0.0}, {"sheet_contact": math.nan}):
        with pytest.raises(ValueError, match="groove"):
            ok(**bad)


# --- the driver (stand-ins) ----------------------------------------------------------------------


def _sheet_loop():
    """The stand-in ring of ``test_ring_driver.RingProblem`` (about (0.01, 4.4)) with a liquid
    sheet from its inner wall into its gas, its roots rounded."""
    return sheeted_ring(t=4e-5, c=(0.01, 4.4), a=4e-3, b=2e-3, h=8e-6, round_root=2e-5)


def test_the_driver_logs_a_sheet_once_per_rebuild_and_a_contact_stops(tmp_path, monkeypatch):
    from test_groove_driver import fake_fill
    from test_events_plumbing import _events_driver
    from test_ring_driver import RingProblem, made_ring_cut

    from faraday_singularity import groovesurgery

    class SheetRing(RingProblem):
        def interface_loops(self):
            if not self.ringed:
                return []
            L = _sheet_loop()
            return [{"open": False, "end_types": ("loop", "loop"), "r": L[:, 0], "z": L[:, 1],
                     "u_r": np.zeros(len(L)), "u_z": np.full(len(L), self.w)}]

    recon = rc.ReconnectSettings(delta=1e-3, dt0=1e-6)
    calls = []
    monkeypatch.setattr(groovesurgery, "groove_fill", fake_fill(calls))
    pb = SheetRing()
    run = tmp_path / "log"
    drv = _events_driver(run, pb, recon, events=LOG, state_dt=0.0, state_keep=10)
    made_ring_cut(drv, pb, k=1)
    out = drv.run(1e-9)
    drv.close()
    assert out["stop"]["reason"] == "max-steps" and calls == []
    lines = (run / "sheets.txt").read_text().splitlines()
    head = lines[0][2:].split("\t")
    rows = [dict(zip(head, x.split("\t"))) for x in lines[1:]]
    rebuilds = [int(r["remesh_count"]) for r in rows]
    assert rows and len(set(rebuilds)) == len(rebuilds)  # at most one row per rebuild
    assert out["events"]["sheets"] == len(rows)
    for row in rows:
        assert row["chain"] == "ring" and float(row["thickness"]) < float(row["fold_limit"])
        assert float(row["length"]) > float(row["thickness"])
        assert float(row["thickness"]) > float(row["contact_limit"])
    # without log the same sheet is the fold met, by the fill (stood in for)
    pb2 = SheetRing()
    once = ev.EventSettings.from_dict({"ring": {"enabled": True}, "groove": {"enabled": True},
                                       "max_events": 2})  # the ring's cut, then one fill
    drv = _events_driver(tmp_path / "event", pb2, recon, events=once, state_dt=0.0)
    made_ring_cut(drv, pb2, k=1)
    out = drv.run(1e-9)
    drv.close()
    assert [c["class"] for c in calls] == ["RF"] and out["stop"]["reason"] == ev.EVENT_LIMIT
    assert not (tmp_path / "event" / "sheets.txt").exists()
    # a sheet's contact: a resumable stop without a handler, never the fill
    calls.clear()
    pb3 = SheetRing()
    drv = _events_driver(tmp_path / "contact", pb3, recon, events=LOG, state_dt=0.0)
    made_ring_cut(drv, pb3, k=1)
    request = {"reason": ev.EVENT_REQUEST, "kind": ev.SHEET_CONTACT, "class": "RF",
               "handler": None, "chain": "ring", "side": "liquid", "fold_pair": [3, 9],
               "fold_gap": 1e-6, "sheet_contact_limit": 2e-6}
    orig = drv._post_cut_stop
    drv._post_cut_stop = lambda: dict(request) if pb3.ring is not None else orig()
    out = drv.run(1e-9)
    drv.close()
    assert out["stop"]["reason"] == ev.EVENT_UNHANDLED and out["stop"]["kind"] == ev.SHEET_CONTACT
    assert calls == [] and out["steps"] == 0
