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


def test_the_driver_samples_the_lip_once_per_time_and_keeps_its_refinement_alive():
    kink = {"far": True, "chain": "ring", "refined": True, "released": False, "r": 1.0e-3,
            "z": 4.6119, "radius": 8e-5, "t_set": 10.0}
    other = dict(kink, r=2e-3)  # beyond the lip
    f = _fake(kinks={"ring_r1_far": kink, "ring_r2_far": other})
    ctx = CollapseDriver._sheet_context(f, [np.zeros((8, 2))])
    out = ctx(_fold(2.0e-5), 1.8e-5)
    assert f.state.lip_samples == [[10.95, 2.0e-5, 1.0e-3, 4.6119]]
    assert out["h_s"] == SHEET_RUPTURE_FLOORS * H_FLOOR
    assert out["refine_limit"] == SHEET_REFINE_ELEMENTS * 1.8e-5
    assert out["thinning"]["thinning"] is False
    assert kink["t_set"] == 10.95 and other["t_set"] == 10.0
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
        return {"kind": ev.SHEET_RUPTURE, "geometry": {"w": 1.5e-6},
                "removed": {"V": 1e-15, "Pz": 2e-12, "KE": 3e-9, "centroid": [1e-3, 4.6]},
                "base_guard": {"ok": True}}

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
