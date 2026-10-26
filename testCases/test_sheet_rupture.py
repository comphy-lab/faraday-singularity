"""The ring's thinning liquid lip, resolved and ruptured (decision 22), without a finite-element
problem (``sheetsurgery.lip_thinning``, ``sheetsurgery.sheet_rupture_geometry``; the lip's reading in
``ring.ring_stage_events``; ``driver.CollapseDriver._sheet_context``, ``_ring_sheet_reading`` and
``_meet_sheet_rupture`` on stand-ins, ``sheetsurgery.sheet_rupture`` stood in for). r1 and r1h of the
delta triple (6ef3c4d) stopped at tau/delta^2 25.35 on the ring's ``sheet-contact``: the lip curling
out from the top of the liquid tongue the downward head drives into the ring's gas, thinning from
4.7e-5 to 2.0e-5 at its thinnest section within 0.15 tau/delta^2.

* the thinning: a falling least-squares slope over at least three samples and the last below the
  first; a lip thickening, flat, or read too few times is not thinning;
* the geometry on the synthetic sheeted ring (``test_groove_sheets.sheeted_ring``): the rim beyond
  the section, liquid, removed and the remaining lip closed by a half circle, the new loop simple
  and counter-clockwise, its nodes those off the rim and the cap's, the volume of revolution of the
  region removed; refused for a pocket of gas, a rim within reach of the open chain, a rim too large;
* the stops: the lip's callback read once with the ring's liquid fold; a ``sheet-contact`` also
  where the faces lie within the refine's limit (not due) or the lip thins at ``h_s`` (due), none
  for a lip resolved and not thinning;
* the driver: the lip handled only in the ring stage with recentred far kinks, the groove fill with
  ``sheets`` ``log`` and the ring's calibrated noise; its samples once per time, restarting at a
  jump, the refined far kink that holds the lip kept alive; a contact not due read as the refine of
  the lip at its thickness over ``SHEET_ELEMENTS``; a rupture due met by the rupture (event
  ``sheet-rupture``, the lip's history cleared), a refusal the stop ``sheet-rupture-refused``;
* the key: ``jetcore.sheet_rupture_floors`` validated, written into the block and its hash only where
  set.
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

import test_groove_geometry as G  # noqa: E402
import test_groove_sheets as gsh  # noqa: E402
import test_ring_stops as rs  # noqa: E402

from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import groove  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402
from faraday_singularity import sheetsurgery as sh  # noqa: E402
from faraday_singularity.driver import (SHEET_ELEMENTS, SHEET_REFINE_ELEMENTS,  # noqa: E402
                                        SHEET_RUPTURE_FLOORS, CollapseDriver)

H_FLOOR = 7.8125e-7


def test_a_lip_thins_only_where_its_thickness_falls_over_its_recent_samples():
    falling = [[0.0, 3e-5], [1e-9, 2.9e-5], [2e-9, 2.7e-5]]
    got = sh.lip_thinning(falling)
    assert got["thinning"] and got["samples"] == 3 and got["slope"] < 0.0
    assert sh.lip_thinning([x + [1.0, 4.6] for x in falling])["thinning"]  # extra columns ignored
    assert not sh.lip_thinning([[0.0, 3e-5], [1e-9, 3.1e-5], [2e-9, 3.2e-5]])["thinning"]
    assert not sh.lip_thinning([[0.0, 3e-5], [1e-9, 3e-5], [2e-9, 3e-5]])["thinning"]
    assert not sh.lip_thinning(falling[:2])["thinning"]  # too few
    assert not sh.lip_thinning([[0.0, 3e-5], [0.0, 2.9e-5], [2e-9, 2.7e-5]])["thinning"]
    # falling on the whole but up again at the end, below the first: still thinning
    assert sh.lip_thinning([[0.0, 3e-5], [1e-9, 2.6e-5], [2e-9, 2.7e-5]])["thinning"]
    # falling then back above the first: not
    assert not sh.lip_thinning([[0.0, 3e-5], [1e-9, 2.0e-5], [2e-9, 3.05e-5]])["thinning"]
    assert not sh.lip_thinning([])["thinning"]


def _lip(t=3e-5):
    L = gsh.sheeted_ring(t=t, L=3e-4)
    f = gsh.sides(L, True)["liquid"]
    return L, f


def test_the_rim_beyond_the_lips_section_is_removed_and_the_lip_capped():
    L, f = _lip()
    a, b = f["fold"].pair
    n = len(L)
    g = sh.sheet_rupture_geometry(L, (a, b), open_points=rs.flat_open().points, V_b=1.0)
    assert g["checks"]["side"] == "liquid" and g["checks"]["failed"] == []
    assert g["w"] == pytest.approx(f["fold"].gap)
    rim = g["arc"]
    assert {rim[0], rim[-1]} == {a, b} and len(rim) == g["checks"]["rim_nodes"]
    # the pocket's arc, the shorter way round: the sheet's tip lies on it
    reading = groove.sheet_reading(L, (a, b), loop=True)
    assert len(rim) == reading["nodes"]
    from shapely.geometry import LinearRing, Polygon

    new = g["points"]
    assert LinearRing(new).is_simple
    assert 0.5 * np.sum(new[:, 0] * np.roll(new[:, 1], -1) - np.roll(new[:, 0], -1) * new[:, 1]) > 0
    assert len(new) == n - (len(rim) - 2) + (sh.CAP_POINTS - 2)
    # the cap: a half circle of diameter the section, from one end of the arc to the other,
    # bulging into the rim
    cap = g["cap"]
    c = 0.5 * (L[a] + L[b])
    assert np.allclose(np.hypot(*(cap - c).T), 0.5 * g["w"])
    assert {tuple(cap[0]), tuple(cap[-1])} == {tuple(L[a]), tuple(L[b])}
    tip = np.asarray(g["tip"])
    mid = cap[len(cap) // 2]
    assert np.dot(mid - c, tip - c) > 0.0
    # the removed region: the rim's arc and the cap back, its volume of revolution
    removed = Polygon(g["removed"])
    assert removed.is_valid and g["area"] == pytest.approx(removed.area)
    assert g["V_rim"] == pytest.approx(rc.revolved_volume(g["removed"]))
    # the rim's volume is the pocket's less the half disc the cap keeps
    r_c = float(c[0])
    half_disc = 0.5 * math.pi * (0.5 * g["w"]) ** 2
    assert g["V_rim"] == pytest.approx(reading["volume"] - 2 * math.pi * r_c * half_disc, rel=0.05)
    # the new loop's gas: the old gas and the removed region
    gained = Polygon(new).area - Polygon(L).area
    assert gained == pytest.approx(removed.area, rel=1e-6)


def test_the_rupture_is_refused_off_a_liquid_pocket_near_the_open_chain_or_for_a_large_rim():
    finger = G.finger_ring()
    f = gsh.sides(finger, True)["gas"]
    with pytest.raises(sh.SheetRuptureRefused) as e:
        sh.sheet_rupture_geometry(finger, f["fold"].pair, open_points=rs.flat_open().points,
                                  V_b=1.0)
    assert e.value.checks["failed"] == ["liquid"]
    L, f = _lip()
    near = np.column_stack([np.linspace(0.0, 0.02, 400), np.full(400, float(f["fold"].z) + 3e-5)])
    with pytest.raises(sh.SheetRuptureRefused) as e:
        sh.sheet_rupture_geometry(L, f["fold"].pair, open_points=near, V_b=1.0)
    assert e.value.checks["failed"] == ["open"]
    with pytest.raises(sh.SheetRuptureRefused) as e:
        sh.sheet_rupture_geometry(L, f["fold"].pair, open_points=rs.flat_open().points, V_b=1e-12)
    assert e.value.checks["failed"] == ["volume"]
    with pytest.raises(sh.SheetRuptureRefused):
        sh.sheet_rupture_geometry(L, (5, 5), open_points=rs.flat_open().points, V_b=1.0)


def _sheet_stops(L, callback):
    return rs.stops([rs.flat_open()], [L], events=gsh.LOG, V_b=10.0 * rc.revolved_volume(L),
                    sheet=callback)


def test_the_stops_read_the_lip_and_ask_for_its_refine_or_its_rupture():
    L, f = _lip()
    gap, h_fine = f["fold"].gap, f["h_fine"]
    seen = []

    def lip(h_s, refine, thinning):
        def read(fold, h):
            seen.append((fold.pair, h))
            return {"h_s": h_s, "refine_limit": refine, "thinning": {"thinning": thinning}}
        return read

    # resolved, not thinning, above h_s: no event, the lip read once with the ring's liquid fold
    assert _sheet_stops(L, lip(1e-6, 0.0, False)) is None
    assert seen == [(f["fold"].pair, pytest.approx(h_fine))]
    # fewer elements across than the refine asks for: a contact not due
    req = _sheet_stops(L, lip(1e-6, 2.0 * gap, False))
    assert (req["kind"], req["class"], req["handler"], req["side"]) == (
        ev.SHEET_CONTACT, "RF", None, "liquid")
    assert req["sheet_rupture"]["due"] is False and req["sheet_rupture"]["refine_limit"] == 2 * gap
    # thinning at h_s or below: due
    req = _sheet_stops(L, lip(1.01 * gap, 0.0, True))
    assert req["kind"] == ev.SHEET_CONTACT and req["sheet_rupture"]["due"] is True
    assert req["sheet_rupture"]["h_s"] == pytest.approx(1.01 * gap)
    # at h_s but thickening: no event; thinning above h_s: none either
    assert _sheet_stops(L, lip(1.01 * gap, 0.0, False)) is None
    assert _sheet_stops(L, lip(0.99 * gap, 0.0, True)) is None
    # without the callback: as before (the contact at one element only)
    assert rs.stops([rs.flat_open()], [L], events=gsh.LOG,
                    V_b=10.0 * rc.revolved_volume(L)) is None


# --- the driver, on stand-ins ---------------------------------------------------------------------


def _fake(*, ring=True, removed=False, jet="recentre", enabled=True, sheets="log", sigma=1e-20,
          floors=None, kinks=None, t=10.95):
    events = SimpleNamespace(groove=SimpleNamespace(enabled=enabled, sheets=sheets),
                             ring=SimpleNamespace(sigma_V=sigma, sigma_Pz=sigma), max_events=1000)
    f = SimpleNamespace(
        events=events, s=SimpleNamespace(refine_ratio=1.5),
        state=SimpleNamespace(lip_samples=[], kinks=kinks or {}, step=7767, dt_prev=4e-10),
        _ring=lambda: ({"removed": {"k": 9} if removed else None} if ring else None),
        _jetcore=lambda: (SimpleNamespace(far_kinks=jet, sheet_rupture_floors=floors)
                          if jet is not None else None),
        _h_floor=lambda: H_FLOOR, _t=lambda: t)
    f._sheet_handled = lambda: CollapseDriver._sheet_handled(f)
    f._sheet_rupture_thickness = lambda: CollapseDriver._sheet_rupture_thickness(f)
    f._lip_read = lambda: CollapseDriver._lip_read(f)
    f._holds_lip = CollapseDriver._holds_lip
    f._lip_cover = lambda previous, regions: CollapseDriver._lip_cover(f, previous, regions)
    return f


def test_the_lip_is_handled_only_in_the_ring_stage_with_recentred_kinks_the_fill_and_the_noise():
    assert CollapseDriver._sheet_handled(_fake())
    for kw in ({"ring": False}, {"removed": True}, {"jet": None}, {"jet": "grow"},
               {"enabled": False}, {"sheets": "event"}, {"sigma": 0.0}):
        assert not CollapseDriver._sheet_handled(_fake(**kw)), kw
    assert CollapseDriver._sheet_rupture_thickness(_fake()) == SHEET_RUPTURE_FLOORS * H_FLOOR
    assert CollapseDriver._sheet_rupture_thickness(_fake(floors=4.0)) == 4.0 * H_FLOOR
    assert CollapseDriver._sheet_context(_fake(jet="grow"), [np.zeros((8, 2))]) is None
    assert CollapseDriver._sheet_context(_fake(), []) is None


def _fold(gap, r=1.0e-3, z=4.6119):
    return SimpleNamespace(gap=gap, r=r, z=z)


def test_the_driver_samples_the_lip_once_per_time():
    kink = {"far": True, "chain": "ring", "refined": True, "released": False, "r": 1.0e-3,
            "z": 4.6119, "radius": 8e-5, "t_set": 10.0}
    other = dict(kink, r=2e-3)  # beyond the lip
    f = _fake(kinks={"ring_r1_far": kink, "ring_r2_far": other})
    ctx = CollapseDriver._sheet_context(f, [np.zeros((8, 2))])
    out = ctx(_fold(2.0e-5), 1.8e-5)
    assert f.state.lip_samples == [[10.95, 2.0e-5, 1.0e-3, 4.6119]]
    assert out["h_s"] == SHEET_RUPTURE_FLOORS * H_FLOOR == ctx.h_s  # the stage reads up to h_s
    assert out["refine_limit"] == SHEET_REFINE_ELEMENTS * 1.8e-5
    assert out["thinning"]["thinning"] is False
    # the hold of the lip's refinement is the measurement's (_lip_hold), not the reading's
    assert kink["t_set"] == 10.0 and other["t_set"] == 10.0
    ctx(_fold(1.9e-5), 1.8e-5)  # a rebuild at the same time: no new sample
    assert len(f.state.lip_samples) == 1
    for k, gap in enumerate((1.9e-5, 1.8e-5, 1.7e-5)):
        f._t = lambda k=k: 10.95 + (k + 1) * 1e-9
        out = CollapseDriver._sheet_context(f, [np.zeros((8, 2))])(_fold(gap), 1.8e-5)
    assert len(f.state.lip_samples) == 4 and out["thinning"]["thinning"] is True
    # at most LIP_SAMPLES kept
    for k in range(10):
        f._t = lambda k=k: 11.0 + k * 1e-9
        CollapseDriver._sheet_context(f, [np.zeros((8, 2))])(_fold(1e-5 - k * 1e-7), 1.8e-5)
    assert len(f.state.lip_samples) == sh.LIP_SAMPLES
    # a section far from the last: another feature, its own history
    f._t = lambda: 12.0
    CollapseDriver._sheet_context(f, [np.zeros((8, 2))])(_fold(1e-5, r=3e-3), 1.8e-5)
    assert f.state.lip_samples == [[12.0, 1e-5, 3e-3, 4.6119]]
    # elements at the refine's floor: no refine asked
    out = CollapseDriver._sheet_context(f, [np.zeros((8, 2))])(_fold(1e-5, r=3e-3), 1.1e-6)
    assert out["refine_limit"] == 0.0


def _contact(**kw):
    out = {"reason": ev.EVENT_REQUEST, "kind": ev.SHEET_CONTACT, "class": "RF", "handler": None,
           "chain": "ring", "side": "liquid", "contact": "fold", "fold_pair": [2721, 2757],
           "fold_gap": 2.0e-5, "fold_r": 1.033e-3, "fold_z": 4.611884, "h_fine": 2.16e-5,
           "sheet_contact_limit": 2.16e-5,
           "sheet_rupture": {"h_s": 1.5625e-6, "due": False, "refine_limit": 4.3e-5,
                             "thinning": {"thinning": True}}}
    out.update(kw)
    return out


def test_a_contact_not_due_is_read_as_the_refine_of_the_lip_at_its_thickness():
    f = _fake()
    got = CollapseDriver._ring_sheet_reading(f, _contact())
    assert (got["reason"], got["trigger"], got["chain"]) == ("interface-unresolved",
                                                            "sheet-contact", "ring")
    assert got["L"] == 2.16e-5 and got["target"] == pytest.approx(2.0e-5 / SHEET_ELEMENTS)
    assert (got["r"], got["z"]) == (1.033e-3, 4.611884)
    assert got["liquid_contact"]["kind"] == ev.SHEET_CONTACT  # the liquid refine's budgets
    json.dumps(got, default=str)
    # the target never below the floor
    thin = CollapseDriver._ring_sheet_reading(f, _contact(fold_gap=2e-6))
    assert thin["target"] == H_FLOOR
    # a rupture due, another chain or kind, a gas side, or the lip not handled: as it is
    for stop in (_contact(sheet_rupture={"due": True}), _contact(chain="open"),
                 _contact(kind="ring-fold"), _contact(side="gas"), None,
                 dict(_contact(), reason="interface-unresolved")):
        assert CollapseDriver._ring_sheet_reading(f, stop) == stop
    assert CollapseDriver._ring_sheet_reading(_fake(sigma=0.0), _contact()) == _contact()


def test_a_rupture_due_is_met_by_the_rupture_and_a_refusal_is_its_stop(tmp_path, monkeypatch):
    from faraday_singularity.surgery import CutRefused

    calls, logged = [], []

    def made(pb, events, reconnect, outdir, request, *, k, dt_prev, h_floor, lip, driver,
             on_made, base_check, log=print):
        calls.append({"k": k, "lip": lip, "h_floor": h_floor, "dt_prev": dt_prev})
        driver._event_after = {"m": "after"}
        rec = {"kind": ev.SHEET_RUPTURE, "geometry": {"w": 1.5e-6},
               "removed": {"V": 1e-15, "Pz": 2e-12, "KE": 3e-9, "centroid": [1e-3, 4.6]},
               "base_guard": {"ok": True}}
        on_made(rec)  # the driver's bookkeeping, as the surgery makes it once made
        return rec

    import faraday_singularity.sheetsurgery as mod

    monkeypatch.setattr(mod, "sheet_rupture", made)
    f = _fake()
    f.pb, f.reconnect, f.outdir = SimpleNamespace(), SimpleNamespace(delta=1e-3), tmp_path
    f.state.lip_samples = [[1.0, 2e-6], [2.0, 1.8e-6], [3.0, 1.5e-6]]
    f._made_events = lambda: [{}, {}]
    f._event = lambda name, m, **kw: logged.append((name, kw))
    f._write_ring_event_row = lambda rec: logged.append(("row", rec["kind"]))
    f._drop_event_dumps = f._prune_meshes = lambda: None
    f._cut_record = lambda: {"settings": {"dt0": 1e-9}}
    f.measure = lambda: {"m": "measured"}
    f._on_groove_made = lambda rec: None
    due = _contact(sheet_rupture={"h_s": 1.5625e-6, "due": True, "thinning": {"thinning": True}})
    m, stop = CollapseDriver._meet_sheet_rupture(f, {"m": "before"}, due)
    assert stop is None and m == {"m": "after"} and calls[0]["k"] == 3
    assert calls[0]["lip"]["samples"] == [[1.0, 2e-6], [2.0, 1.8e-6], [3.0, 1.5e-6]]
    assert calls[0]["lip"]["due"] is True and calls[0]["h_floor"] == H_FLOOR
    assert [x[0] for x in logged] == ["sheet-rupture", "row"] and f.state.lip_samples == []
    assert logged[0][1]["V"] == 1e-15 and logged[0][1]["w"] == 1.5e-6

    def refused(*a, **kw):
        e = CutRefused("no")
        e.record = {"failed_at": "geometry", "error": "CutRefused: no"}
        e.precut = "states/precut_e003.dump"
        raise e

    monkeypatch.setattr(mod, "sheet_rupture", refused)
    f._event_stop = lambda reason, request, **kw: {"reason": reason, **kw}
    f.state.lip_samples = [[1.0, 2e-6]]
    m, stop = CollapseDriver._meet_sheet_rupture(f, {"m": "before"}, due)
    assert stop["reason"] == ev.SHEET_RUPTURE_REFUSED and stop["failed_at"] == "geometry"
    assert m == {"m": "before"} and f.state.lip_samples == [[1.0, 2e-6]]
    assert ev.SHEET_RUPTURE_REFUSED in ev.EVENT_STOPS
    assert ev.SHEET_RUPTURE in ev.EVENT_CLASSES["ring"]


def test_the_rupture_key_is_validated_and_written_only_where_set():
    base = {"ring": {"enabled": True}, "groove": {"enabled": True, "sheets": "log"},
            "jetcore": {"enabled": True, "far_kinks": "recentre"}}
    plain = ev.EventSettings.from_dict(base)
    assert "sheet_rupture_floors" not in plain.to_dict()["jetcore"]
    set4 = ev.EventSettings.from_dict({**base, "jetcore": {**base["jetcore"],
                                                           "sheet_rupture_floors": 4.0}})
    assert set4.to_dict()["jetcore"]["sheet_rupture_floors"] == 4.0
    assert set4.sha256() != plain.sha256()
    assert set4.changed_keys(plain.to_dict()) == ["jetcore.sheet_rupture_floors"]
    assert plain.changed_keys(plain.to_dict()) == []
    # a block recorded before the key: the same hash as a block without it
    recorded = json.loads(json.dumps(plain.to_dict()))
    assert ev.EventSettings.from_dict(recorded).sha256() == plain.sha256()
    for bad in (0.5, 0.0, -2.0, math.nan, True, "2"):
        with pytest.raises((ValueError, TypeError)):
            ev.EventSettings.from_dict({**base, "jetcore": {**base["jetcore"],
                                                            "sheet_rupture_floors": bad}})
    with pytest.raises(ValueError):  # a jetcore key needs jetcore enabled
        ev.EventSettings.from_dict({"jetcore": {"sheet_rupture_floors": 2.0}})


# --- the review of 2026-10-26 (round 2: F1-F5) ---------------------------------------------------


def _graded_circle(dense=202, coarse=8, R=1.6, span=0.625):
    """A counter-clockwise circle whose arc of ``span`` radians holds ``dense`` nodes and the rest
    of it ``coarse`` intervals: the short arc (about 1) holds most of the nodes (about 9 long)."""
    a = np.linspace(0.0, span, dense)
    b = np.linspace(span, 2.0 * math.pi, coarse + 1)[1:-1]
    phi = np.concatenate([a, b])
    return np.column_stack([2.0 + R * np.cos(phi), R * np.sin(phi)])


def test_the_rims_arc_is_the_shorter_way_round_in_length_as_the_detector_reads_it():
    P = _graded_circle()
    n = len(P)
    assert n == 202 + 7
    # by length the dense arc, though the other way round takes fewer node steps
    assert sh._arc(P, 0, 201) == list(range(202))
    assert sh._arc(P, 201, 0) == list(range(201, -1, -1))
    assert n - 201 < 201
    # across the seam: the same arc, rolled
    k = 100
    Q = np.roll(P, -k, axis=0)
    a, b = (0 - k) % n, 201 - k
    walk = sh._arc(Q, a, b)
    assert walk == list(range(a, n)) + list(range(0, b + 1))
    assert sh._arc(Q, b, a) == walk[::-1]
    assert np.allclose(Q[walk], P[:202])
    # the detector's pocket on the sheeted ring is the surgery's rim
    L, f = _lip()
    a, b = f["fold"].pair
    reading = groove.sheet_reading(L, (a, b), loop=True)
    arc = sh._arc(L, a, b)
    assert arc[0] == a and arc[-1] == b and len(arc) == reading["nodes"]


def _refined_lip(m=12):
    """The sheeted ring with every element of the lip's pocket split into ``m`` (the points on
    their chords: the same polygon), so that the lip holds more nodes than the rest of the ring."""
    L, f = _lip()
    a, b = sorted(f["fold"].pair)
    n = len(L)
    out, pair = [], []
    for i in range(n):
        if i == a:
            pair.append(len(out))
        if i == b:
            pair.append(len(out))
        out.append(L[i])
        if a <= i < b:
            for j in range(1, m):
                out.append(L[i] + (L[i + 1] - L[i]) * j / m)
    return np.asarray(out), tuple(pair), f


def test_a_lip_graded_finer_than_the_rest_of_the_ring_is_ruptured_on_its_own_pocket():
    G2, pair, f = _refined_lip()
    n = len(G2)
    a, b = pair
    assert (b - a) > n - (b - a)  # the lip holds most of the nodes
    g = sh.sheet_rupture_geometry(G2, pair, open_points=rs.flat_open().points, V_b=1.0)
    assert g["checks"]["side"] == "liquid" and g["checks"]["failed"] == []
    L, _ = _lip()
    plain = sh.sheet_rupture_geometry(L, f["fold"].pair, open_points=rs.flat_open().points,
                                      V_b=1.0)
    assert g["V_rim"] == pytest.approx(plain["V_rim"], rel=1e-9)
    assert g["w"] == pytest.approx(plain["w"])
    # across the seam: the loop rolled so that the pocket runs through its first node
    k = (a + b) // 2
    R = np.roll(G2, -k, axis=0)
    g2 = sh.sheet_rupture_geometry(R, ((a - k) % n, (b - k) % n),
                                   open_points=rs.flat_open().points, V_b=1.0)
    assert g2["checks"]["failed"] == [] and g2["V_rim"] == pytest.approx(g["V_rim"], rel=1e-9)


def test_a_removal_is_held_to_the_tolerance_of_its_volumes_magnitude():
    from faraday_singularity.ringsurgery import RING_BUDGET_LIMITS, _ring_closures

    V, miss = 1e-12, 0.005
    box = {x: {"Pz": 0.0, "KE": 0.0} for x in ("before", "after_transfer", "after_projection")}
    scales = {"V_fill": V, "U": 1.0, "P_ref": V, "KE_ref": 0.5 * V}
    sigma = {"V": 1e-18, "Pz": 1e-30, "KE": 1e-30}
    got = {}
    for sign in (-1.0, 1.0):  # a removal (a lip's rupture) and a fill
        poly = {"before": {"V_liquid": 1e-6, "A": 0.0},
                "after": {"V_liquid": 1e-6 + sign * V * (1.0 + miss), "A": 0.0}}
        got[sign] = _ring_closures({"V": sign * V, "Pz": 0.0, "KE": 0.0}, poly, box, None, scales,
                                   {"A_cl": 0.0, "A_rem": 0.0}, sigma=sigma, prefix="S")
    for c in got.values():
        assert c["limits"]["S1"] == pytest.approx(RING_BUDGET_LIMITS["R1"] * V)
        assert c["relative"]["S1"] == pytest.approx(miss, rel=1e-3)
        assert "S1" not in c["hard"]
    # beyond the tolerance: hard either way
    poly = {"before": {"V_liquid": 1e-6, "A": 0.0}, "after": {"V_liquid": 1e-6 - 1.05 * V, "A": 0.0}}
    c = _ring_closures({"V": -V, "Pz": 0.0, "KE": 0.0}, poly, box, None, scales,
                       {"A_cl": 0.0, "A_rem": 0.0}, sigma=sigma, prefix="S")
    assert "S1" in c["hard"]


def _fields(U=800.0, lost=()):
    """A stand-in of ``surgery.sample_fields``: a uniform axial velocity ``U`` everywhere, the
    points of ``lost`` (their indices) not located."""
    def sample(pb, r, z, names):
        found = np.ones(len(r), dtype=bool)
        found[list(lost)] = False
        out = {"velocity_x": np.where(found, 0.0, math.nan), "velocity_y": np.where(found, U,
                                                                                   math.nan)}
        return {k: out[k] for k in names}, found
    return sample


def test_the_rims_budgets_are_its_own_quadratures_where_the_box_grid_misses_it(monkeypatch):
    from shapely import points
    from shapely.geometry import Polygon

    from faraday_singularity import surgery

    L = gsh.sheeted_ring(t=1.5625e-6, L=4e-6, h=1e-7)
    f = gsh.sides(L, True)["liquid"]
    g = sh.sheet_rupture_geometry(L, f["fold"].pair, open_points=rs.flat_open().points, V_b=1.0)
    # the box's grid at the ring's box (ring.box_radius delta, 4e-3 at delta 1e-3) holds no point
    # of the rim: its momentum and energy were logged as zero
    grid = surgery.site_grid(g["r_c"], g["z_c"], max(4e-3, 2.0 * g["L_rim"]))
    poly = Polygon(g["removed"])
    assert not poly.contains(points(np.column_stack([grid["r"], grid["zz"]]))).any()
    # the rim's own quadrature resolves it
    monkeypatch.setattr(surgery, "sample_fields", _fields(800.0))
    q = sh.rim_quadrature(None, g["removed"], g["V_rim"])
    assert q["checked"] and q["found"] == q["points"] > 0
    assert q["V_quad"] == pytest.approx(g["V_rim"], rel=1e-9)
    assert q["Pz"] == pytest.approx(800.0 * q["V_quad"], rel=1e-12)
    assert q["KE"] == pytest.approx(0.5 * 800.0**2 * q["V_quad"], rel=1e-12)
    # a point not located, or a volume not the rim's: unresolved, refused by the rupture
    monkeypatch.setattr(surgery, "sample_fields", _fields(800.0, lost=(3,)))
    q = sh.rim_quadrature(None, g["removed"], g["V_rim"])
    assert not q["checked"] and math.isnan(q["Pz"]) and q["found"] == q["points"] - 1
    monkeypatch.setattr(surgery, "sample_fields", _fields(800.0))
    assert not sh.rim_quadrature(None, g["removed"], 1.05 * g["V_rim"])["checked"]
    assert sh.rim_quadrature(None, g["removed"], 1.005 * g["V_rim"])["checked"]


def _kink(**kw):
    return {"name": "kink_ring_r1_far", "far": True, "chain": "ring", "refined": True,
            "released": False, "r": 1.0e-3, "z": 4.6119, "radius": 8e-5, "t_set": 10.0,
            "h_before": 2e-5, "target": 5e-6, "turn": 10.0, "yields": 0, "yield_t": None, **kw}


def _held_fake(kinks, *, t=11.0, t_prev=10.99, samples=((10.99, 5e-6, 1.0e-3, 4.6119),)):
    f = _fake(kinks=kinks, t=t)
    f.state.t_prev = t_prev
    f.state.lip_samples = [list(x) for x in samples]
    f._lip_hold = lambda previous: CollapseDriver._lip_hold(f, previous)
    return f


def test_the_lips_refinement_is_held_before_the_kinks_are_released():
    from faraday_singularity.jetcore import far_kink_regions

    args = dict(t=11.0, hold=lambda L: 0.5, h_far=1e-5, h_floor=H_FLOOR, yield_turn=75.0,
                sizes={})
    kinks = {"ring_r1_far": _kink(), "ring_r2_far": _kink(name="kink_ring_r2_far", r=2e-3),
             "ring_r3_far": _kink(name="kink_ring_r3_far", released=True)}
    f = _held_fake(kinks)
    held = CollapseDriver._lip_hold(f, dict(f.state.kinks))
    assert held["ring_r1_far"]["t_set"] == 11.0 and kinks["ring_r1_far"]["t_set"] == 10.0
    assert held["ring_r2_far"] is kinks["ring_r2_far"]  # its disc misses the lip
    assert held["ring_r3_far"] is kinks["ring_r3_far"]  # released
    # the release reads the held entries: the lip's refinement stays, the rest is released
    out = far_kink_regions(held, {}, {}, **args)
    assert out["ring_r1_far"]["released"] is False and out["ring_r2_far"]["released"] is True
    assert far_kink_regions(dict(kinks), {}, {}, **args)["ring_r1_far"]["released"] is True
    # a lip not read at the state the step started from holds nothing
    stale = _held_fake(kinks, samples=((10.98, 5e-6, 1.0e-3, 4.6119),))
    assert CollapseDriver._lip_hold(stale, dict(kinks)) == kinks
    # after an event (t_prev NaN) a lip read now holds
    after = _held_fake(kinks, t_prev=math.nan, samples=((11.0, 5e-6, 1.0e-3, 4.6119),))
    assert CollapseDriver._lip_hold(after, dict(kinks))["ring_r1_far"]["t_set"] == 11.0
    # no lip, or the lip not handled: nothing
    assert CollapseDriver._lip_hold(_held_fake(kinks, samples=()), dict(kinks)) == kinks
    unhandled = _held_fake(kinks)
    unhandled.events.ring.sigma_V = 0.0
    assert CollapseDriver._lip_hold(unhandled, dict(kinks)) == kinks


def test_the_measurement_holds_the_lip_through_the_remesh_checks_and_the_commit():
    """measure (``_split_kinks``) -> stop (the lip read) -> commit (``_commit_post_cut``): the
    refinement held at the measurement is what the commit keeps; the stop's reading of the lip
    adds its sample and changes no kink."""
    kinks = {"ring_r1_far": _kink()}
    f = _held_fake(kinks)
    f.s = SimpleNamespace(refine_ratio=1.5, coarsen_hold=1e-6)
    f.state.far_yields = 0
    f._jetcore = lambda: SimpleNamespace(far_kinks="recentre", sheet_rupture_floors=None,
                                         far_floor=1e-5, ring_kink_floor=False,
                                         far_yield_turn=75.0)
    f._column_size = lambda: math.nan
    m = {"time": 11.0, "kinks": CollapseDriver._split_kinks(
        f, {}, {"edges": {}}, {"whole": True, "R_K": math.inf, "mask": []}, 1.0, None)}
    assert m["kinks"]["ring_r1_far"]["released"] is False
    assert m["kinks"]["ring_r1_far"]["t_set"] == 11.0
    CollapseDriver._sheet_context(f, [np.zeros((8, 2))])(_fold(4.9e-6), 1.2e-6)  # the stop
    assert f.state.kinks["ring_r1_far"]["t_set"] == 10.0  # the reading changes no kink
    assert f.state.lip_samples[-1][0] == 11.0
    focus = SimpleNamespace(region=lambda name: SimpleNamespace(size=5e-6))
    f.pb = SimpleNamespace(active_mesh_template=lambda: SimpleNamespace(focus=focus))
    f.events.pinch = SimpleNamespace(enabled=False)
    f._cut_record = lambda: {"settings": {}}
    f._post_cut_state = lambda: {"apexes": {}}
    f._event = lambda *a, **kw: None
    CollapseDriver._commit_post_cut(f, m)
    assert f.state.kinks["ring_r1_far"]["released"] is False
    assert f.state.kinks["ring_r1_far"]["t_set"] == 11.0
    # without the lip the same sequence releases it
    g = _held_fake({"ring_r1_far": _kink()}, samples=())
    g.s, g.state.far_yields, g._jetcore, g._column_size = f.s, 0, f._jetcore, f._column_size
    released = CollapseDriver._split_kinks(g, {}, {"edges": {}}, {"whole": True, "R_K": math.inf,
                                                                   "mask": []}, 1.0, None)
    assert released["ring_r1_far"]["released"] is True


def test_the_state_after_a_rupture_holds_no_lip_history_as_the_live_run_goes_on(tmp_path,
                                                                                  monkeypatch):
    """The rupture keeps ``postcut_e<k>`` after the driver's bookkeeping (``on_made``): the lip's
    history is cleared there, so that the state written (``_write_state`` writes ``lip_samples``
    only where there are any) and restored (``restore`` reads none as none) is the live run's; the
    history stays in the record's ``lip``."""
    written = {}

    def made(pb, events, reconnect, outdir, request, *, k, dt_prev, h_floor, lip, driver,
             on_made, base_check, log=print):
        rec = {"kind": ev.SHEET_RUPTURE, "geometry": {"w": 1.5e-6}, "lip": lip,
               "removed": {"V": 1e-15, "Pz": 2e-12, "KE": 3e-9, "centroid": [1e-3, 4.6]},
               "base_guard": {"ok": True}}
        on_made(rec)  # as sheetsurgery._sheet_rupture: the bookkeeping, then postcut_e<k>
        written["postcut"] = [list(x) for x in driver.state.lip_samples]
        written["lip"] = lip
        driver._event_after = {"m": "after"}
        return rec

    import faraday_singularity.sheetsurgery as mod

    monkeypatch.setattr(mod, "sheet_rupture", made)
    f = _fake()
    f.pb, f.reconnect, f.outdir = SimpleNamespace(), SimpleNamespace(delta=1e-3), tmp_path
    history = [[1.0, 2e-6, 1e-3, 4.6], [2.0, 1.8e-6, 1e-3, 4.6], [3.0, 1.5e-6, 1e-3, 4.6]]
    f.state.lip_samples = [list(x) for x in history]
    f._made_events = lambda: []
    f._event = lambda *a, **kw: None
    f._write_ring_event_row = f._drop_event_dumps = f._prune_meshes = lambda *a: None
    f._cut_record = lambda: {"settings": {"dt0": 1e-9}}
    f.measure = lambda: {}
    booked = []
    f._on_groove_made = lambda rec: booked.append(rec["kind"])
    due = _contact(sheet_rupture={"h_s": 1.5625e-6, "due": True, "thinning": {"thinning": True}})
    m, stop = CollapseDriver._meet_sheet_rupture(f, {"m": "before"}, due)
    assert stop is None and booked == [ev.SHEET_RUPTURE]
    assert written["postcut"] == [] == f.state.lip_samples
    # the record keeps the history the rupture met
    assert written["lip"]["samples"] == history and written["lip"]["due"] is True


# --- the review of 2026-10-26 (core-sheet round 1: F1, F2) ----------------------------------------


def _ring_edges(points, L, turn):
    """A guard's reading of the ring (``pc["edges"]["ring"]``): elements at ``points``, each of
    length ``L`` and turning by ``turn`` degrees."""
    P = np.asarray(points, dtype=float).reshape(-1, 2)
    return {"edges": {"mid": P, "L": np.full(len(P), L), "turn": np.full(len(P), turn),
                      "index": np.arange(len(P))}}


def test_a_held_lip_stays_in_its_refinement_when_the_guards_elements_recentre_it():
    """The guard asks for the ring's elements 6e-5 from the lip (within the held region's reach):
    ``jetcore.far_kink_regions`` recentres the held region on them, four of their chords wide
    (2e-5), the lip outside it; the measurement widens it about its new centre to hold the ball
    about the lip that the held disc held, its target the refinement's, and the focus built from it
    holds the lip."""
    from faraday_singularity.jetcore import far_kink_regions
    from faraday_singularity.mesh import FocusRegion

    lip = (1.0e-3, 4.6119)
    kinks = {"ring_r1_far": _kink(r=lip[0], z=lip[1], radius=8e-5, target=1.25e-6)}
    f = _held_fake(kinks, samples=((10.99, 5e-6) + lip,))
    f.s = SimpleNamespace(refine_ratio=1.5, coarsen_hold=1.0)
    f.state.far_yields = 0
    f._jetcore = lambda: SimpleNamespace(far_kinks="recentre", sheet_rupture_floors=None,
                                         far_floor=1e-5, ring_kink_floor=False,
                                         far_yield_turn=75.0)
    f._column_size = lambda: math.nan
    edge = (lip[0] + 6e-5, lip[1])
    pc = {"edges": {"ring": _ring_edges([edge], 5e-6, 40.0)}}
    refine = {"ring": np.array([0])}
    core = {"whole": True, "R_K": math.inf, "mask": []}
    # the reading alone: recentred on the element, the lip outside its disc
    held = CollapseDriver._lip_hold(f, dict(kinks))
    plain = far_kink_regions(held, pc["edges"], refine, t=11.0, hold=lambda L: 1.0, h_far=1e-5,
                             h_floor=H_FLOOR, yield_turn=75.0, sizes={})["ring_r1_far"]
    assert plain["radius"] == pytest.approx(2e-5) and (plain["r"], plain["z"]) == edge
    assert math.hypot(lip[0] - plain["r"], lip[1] - plain["z"]) > plain["radius"]
    # the measurement: the same centre and target, widened to hold the lip's ball
    got = CollapseDriver._split_kinks(f, refine, pc, core, 1.0, None)["ring_r1_far"]
    assert (got["r"], got["z"]) == edge and got["target"] == plain["target"] <= 1.25e-6
    assert got["released"] is False
    clearance = 8e-5  # the held disc's about the lip: centred on it
    assert got["radius"] == pytest.approx(6e-5 + clearance)
    region = FocusRegion(got["name"], size=got["target"], r=got["r"], z=got["z"],
                         radius=got["radius"])
    for dr, dz in ((0.0, 0.0), (clearance, 0.0), (0.0, -clearance),
                   (-0.7 * clearance, 0.7 * clearance)):
        assert math.hypot(lip[0] + dr - region.r, lip[1] + dz - region.z) \
            <= region.radius * (1 + 1e-12)
    # a lip no longer read: as the reading made it
    stale = _held_fake(dict(kinks), samples=((10.98, 5e-6) + lip,))
    stale.s, stale.state.far_yields, stale._jetcore = f.s, 0, f._jetcore
    stale._column_size = f._column_size
    assert CollapseDriver._split_kinks(stale, refine, pc, core, 1.0, None)["ring_r1_far"][
        "radius"] == pytest.approx(2e-5)
    # a region still holding the lip's ball, or one released: unchanged
    near = {"ring_r1_far": dict(plain, radius=1.5e-4)}  # holds the ball (6e-5 + 8e-5)
    assert CollapseDriver._lip_cover(f, held, near) == near
    gone = {"ring_r1_far": dict(plain, released=True)}
    assert CollapseDriver._lip_cover(f, held, gone) == gone


def test_a_thinning_lip_beyond_the_fold_limit_is_read_up_to_its_rupture_thickness():
    """A lip thicker than the ring stage's fold limit (max(kappa_g delta, fold_stop_elements
    h_elem), 5e-5 here) but within ``h_s`` is read by the driver's callback and ruptured once
    thinning; beyond ``h_s`` it is not read; read and not thinning it is no event (and no kink)."""
    L, f = _lip(t=7e-5)
    gap = f["fold"].gap
    assert gap > 5e-5
    seen = []

    def lip(h_s, thinning):
        def read(fold, h):
            seen.append(fold.pair)
            return {"h_s": h_s, "refine_limit": 0.0, "thinning": {"thinning": thinning}}
        read.h_s = h_s
        return read

    assert _sheet_stops(L, lip(0.9 * gap, True)) is None and seen == []
    req = _sheet_stops(L, lip(1.1 * gap, True))
    assert seen == [f["fold"].pair]
    assert (req["reason"], req["kind"], req["chain"], req["side"]) == (
        ev.EVENT_REQUEST, ev.SHEET_CONTACT, "ring", "liquid")
    assert req["sheet_rupture"]["due"] is True and req["fold_gap"] == pytest.approx(gap)
    assert req["fold_gap"] > req["fold_limit"]
    assert _sheet_stops(L, lip(1.1 * gap, False)) is None
    # a callback without h_s (and no callback): the fold limit as before
    plain = lip(1.1 * gap, True)
    del plain.h_s
    assert _sheet_stops(L, plain) is None


def test_a_lip_beyond_the_fold_search_is_found_up_to_its_rupture_thickness():
    """The ring's liquid fold is searched for out to ``h_s`` (the core-sheet review, round 2, F2):
    a lip 3e-4 thick, beyond the fold search of the ring stage (four times max(kappa_g delta,
    2 fold_stop_elements spacing), 2e-4 here), below ``h_s`` = 4e-4 (512 floors), is read and,
    thinning, due."""
    L = gsh.sheeted_ring(t=3e-4, L=1.5e-3)
    plain = groove.side_folds(L, loop=True, **gsh.FOLD)
    assert not math.isfinite(plain["liquid"]["fold"].gap)  # beyond the stage's own search
    wide = groove.side_folds(L, loop=True, **dict(gsh.FOLD, stop=4e-4))["liquid"]["fold"]
    assert wide.gap == pytest.approx(3e-4, rel=1e-3)
    seen = []

    def lip(h_s, thinning):
        def read(fold, h):
            seen.append((fold.pair, fold.gap))
            return {"h_s": h_s, "refine_limit": 0.0, "thinning": {"thinning": thinning}}
        read.h_s = h_s
        return read

    req = _sheet_stops(L, lip(4e-4, True))
    assert req is not None and req["kind"] == ev.SHEET_CONTACT and req["side"] == "liquid"
    assert req["sheet_rupture"]["due"] is True and req["fold_gap"] == pytest.approx(wide.gap)
    assert seen[-1] == (wide.pair, wide.gap)
    assert _sheet_stops(L, lip(4e-4, False)) is None  # read, not thinning: no event
    # without h_s beyond the stage's own reach, the search is the stage's (no lip found)
    seen.clear()
    assert _sheet_stops(L, lip(2.9e-4, True)) is None
    assert all(g < 2.9e-4 for _, g in seen)
