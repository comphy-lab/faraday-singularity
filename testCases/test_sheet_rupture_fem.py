"""Finite-element test of the rupture of the ring's thinning liquid lip through the driver
(``driver.CollapseDriver._sheet_context``, ``_meet_sheet_rupture``; ``sheetsurgery.sheet_rupture``),
and of a restart from the state after it, on a synthetic state.

The synthetic pierced state of ``ring_fem_state.py`` after its gas cut is put in the ring stage
with an elliptic ring in the liquid (``test_groove_fem.py``'s) whose gas a tapered tongue of
liquid enters from its outer side, its end rounded (the lip), its root rounded into the wall, a
solenoidal flow about it. The events block resolves and ruptures the lip (``groove.sheets``
``log``, jetcore's recentred far kinks, the ring's noise floors measured on the state) at
``h_s`` = ``jetcore.sheet_rupture_floors h_floor`` above the tongue's thinnest section, and the
driver's history of the lip is given two earlier, thicker samples, so that the stops of the state
read the lip as thinning at ``h_s`` or below: a ``sheet-contact`` due for its rupture.

* the driver meets it with the rupture: the rim beyond the thinnest section removed, one ring
  kept, the liquid less by the rim's volume, the transfer as predicted, the rim's budgets those of
  its own quadrature (every point located, its volume the rim's), the hard closures S1-S3 and
  ``S3_box`` within their limits, the event ``sheet-rupture`` with the lip's history in its record;
* the state after it (``postcut_e001``) holds no history of the lip, as the run going on holds
  none; a restart from it in a new problem reads the same ring and the same measurement, and its
  next reading of the lip is the run's (the review of 2026-10-26, F5).

The module takes a few minutes.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

pytest.importorskip("pyoomph")

import ring_fem_state as S  # noqa: E402
import test_groove_fem as GF  # noqa: E402
import test_groove_geometry as G  # noqa: E402

from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import groove, surgery  # noqa: E402
from faraday_singularity.driver import CollapseDriver, CollapseSettings  # noqa: E402
from faraday_singularity.mesh import Focus, FocusRegion, RingSizing  # noqa: E402
from faraday_singularity.reconnect import revolved_volume  # noqa: E402
from faraday_singularity.ringsurgery import RING_BUDGET_LIMITS  # noqa: E402

R_ROOT, R_END, LENGTH = 0.004, 0.002, 0.04  # the tongue's half-width at its root and at its end
ROUND = 0.002  # the radius of the rounding of the gas's corners at the tongue's root
H_FLOOR = 6e-4  # the run's floor
FLOORS = 10.0  # h_s = 6e-3, above the tongue's thinnest section (about 4.6e-3)
FOLD = {"ratio": 4.0, "stop_elements": 3.0, "search_factor": 4.0}
SETTINGS = {"t_end": 10.0, "dt_max": 0.01, "max_steps": 2, "state_dt": 0.0, "state_keep": 10}


def tongue_loop(h=GF.H_FINE):
    """The ring: the ellipse of ``test_groove_fem.py`` less a tongue of liquid from its outermost
    point inwards, ``LENGTH`` long, its half-width from ``R_ROOT`` to ``R_END``, its end round, the
    gas's corners at its root rounded by ``ROUND``; counter-clockwise, an even count, sampled ``h``
    apart about the tongue. Returns the loop and the tongue's end."""
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union

    c, a, b = GF.RING_C, GF.RING_A, GF.RING_B
    s = np.linspace(0.0, 2.0 * math.pi, 40001)[:-1]
    E = Polygon(np.column_stack([c[0] + a * np.cos(s), c[1] + b * np.sin(s)]))
    r_out = c[0] + a + 0.002
    tongue = unary_union([Point(r_out - x, c[1]).buffer(R_ROOT + (R_END - R_ROOT) * x / LENGTH,
                                                         quad_segs=32)
                          for x in np.linspace(0.0, LENGTH, 400)])
    gas = E.difference(tongue).buffer(-ROUND, quad_segs=64).buffer(ROUND, quad_segs=64)
    assert gas.geom_type == "Polygon" and not gas.interiors
    end = np.array([r_out - LENGTH - R_END, c[1]])
    L = G.resample(np.asarray(gas.exterior.coords), G.spacing(end, fine=h, coarse=4e-3,
                                                              reach=0.05))[:-1]
    L = L[:-1] if len(L) % 2 else L
    if 0.5 * np.sum(L[:, 0] * np.roll(L[:, 1], -1) - np.roll(L[:, 0], -1) * L[:, 1]) < 0.0:
        L = L[::-1].copy()
    return L, end


def block(noise):
    sigma = {"sigma_V": 2.0 * noise["V"], "sigma_Pz": 2.0 * noise["Pz"],
             "sigma_KE": 2.0 * noise["KE"]}
    return ev.EventSettings.from_dict({
        "ring": {"enabled": True, **sigma}, "gap": {"kappa_g": GF.KAPPA},
        "groove": {"enabled": True, "max_fill": 1.0, "sheets": "log", **sigma},
        "jetcore": {"enabled": True, "far_kinks": "recentre", "ring_kink_floor": True,
                    "sheet_rupture_floors": FLOORS},
        "post": {"h_floor": H_FLOOR}})


def _lip_fold(pb):
    lp = pb.interface_loops()[0]
    P = np.column_stack([lp["r"], lp["z"]])
    return groove.side_folds(P, loop=True, stop=GF.KAPPA * S.DELTA, **FOLD)["liquid"]["fold"], P


@pytest.fixture(scope="module")
def ruptured(tmp_path_factory):
    out = tmp_path_factory.mktemp("lip")
    res = {}
    with S.pierce_problem(out / "pb") as pb:
        pb.initialise()
        S.gas_cut_record(pb)
        res["V_b"] = float(pb.cut["geometry"]["bubble_volume"])
        tmpl = pb.active_mesh_template()
        named = dict(surgery._named_chains(pb)[0])
        L, end = tongue_loop()
        focus = Focus(size=GF.H_FINE, radius=0.2, grading=0.3, regions=(
            FocusRegion("cavity_tip", size=0.004, r=0.0, z=0.95, radius=0.1),
            FocusRegion("tongue", size=GF.H_FINE, r=float(end[0]) + 0.02, z=float(end[1]),
                        radius=0.05)))
        tmpl.ring_stage = True
        tmpl.ring_sizing = RingSizing(h_ring=4e-3, grading=0.3)
        tmpl.focus = focus
        tmpl.pending_ring_geometry = {"chains": [named["open"]], "loops": [L]}
        pb._ring_transfer = {"axis_bulk": True, "reach": 0.1}
        try:
            pb.force_remesh({tmpl})
        finally:
            pb._ring_transfer = None
        pb._domains_to_remesh.clear()
        loops = pb.interface_loops()
        assert len(loops) == 1
        pb.ring = GF.ring_record(pb, np.column_stack([loops[0]["r"], loops[0]["z"]]))
        GF.impose(pb, (GF.RING_C[0] - 0.05, GF.RING_C[1]))
        fold, _ = _lip_fold(pb)
        noise = GF.site_noise(pb, (float(fold.r), float(fold.z)), 4.0 * S.DELTA)
        res["noise"] = noise
        fold, P = _lip_fold(pb)  # the noise's rebuilds moved the mesh's nodes
        res["fold"] = {"gap": fold.gap, "r": fold.r, "z": fold.z}
        res["V_ring"] = revolved_volume(P)
        events = block(noise)
        res["events"] = events
        drv = CollapseDriver(pb, CollapseSettings(**SETTINGS), out / "run", reconnect=S.RECON,
                             events=events)
        drv.state.dt_prev = GF.DT
        t = drv._t()
        history = [[t - 2e-5, 1.3 * fold.gap, fold.r, fold.z],
                   [t - 1e-5, 1.15 * fold.gap, fold.r, fold.z]]
        drv.state.lip_samples = [list(x) for x in history]
        res.update(history=history, handled=drv._sheet_handled(),
                   h_s=drv._sheet_rupture_thickness(), volume_before=pb.cut_observables()["volume"])
        m = drv.measure()
        request = drv._stop(m)
        res["request"] = request
        m2, after = drv._meet_event(m, request)
        res["after"] = after
        res["lip_live"] = [list(x) for x in drv.state.lip_samples]
        res["volume_after"] = pb.cut_observables()["volume"]
        res["loops"] = pb.interface_loops()
        res["record"] = json.loads((out / "run" / "events" / "event_001.json").read_text())
        res["log"] = [json.loads(x) for x in (out / "run" / "events.jsonl").read_text()
                      .splitlines()]
        res["summary"] = drv._events_summary({})
        # the run goes on: its next reading of the lip
        res["next_live"] = drv._stop(m2)
        res["lip_next_live"] = [list(x) for x in drv.state.lip_samples]
        res["loops_live"] = [np.column_stack([x["r"], x["z"]]) for x in pb.interface_loops()]
        res["m_live"] = m2
        postcut = out / "run" / "states" / "postcut_e001.dump"
        res["postcut"] = postcut
        res["postcut_meta"] = json.loads(postcut.with_suffix(".json").read_text())
        drv.close()
    # a restart from the state after the rupture, in a new problem
    with S.pierce_problem(out / "pb-restart") as pb:
        pb.initialise()
        pb.load_state(str(postcut))
        drv = CollapseDriver(pb, CollapseSettings(**SETTINGS), out / "restart", reconnect=S.RECON,
                             events=res["events"])
        drv.restore(res["postcut_meta"])
        res["lip_restored"] = [list(x) for x in drv.state.lip_samples]
        m = drv.measure()
        res["next_restart"] = drv._stop(m)
        res["lip_next_restart"] = [list(x) for x in drv.state.lip_samples]
        res["loops_restart"] = [np.column_stack([x["r"], x["z"]]) for x in pb.interface_loops()]
        res["m_restart"] = m
        drv.close()
    return res


def test_the_thinning_lip_is_ruptured_by_the_driver_its_budgets_closed(ruptured):
    req = ruptured["request"]
    assert ruptured["handled"] and ruptured["fold"]["gap"] <= ruptured["h_s"]
    assert (req["reason"], req["kind"], req["class"], req["chain"], req["side"]) == (
        ev.EVENT_REQUEST, ev.SHEET_CONTACT, "RF", "ring", "liquid")
    assert req["sheet_rupture"]["due"] is True and req["sheet_rupture"]["thinning"]["thinning"]
    assert ruptured["after"] is None  # made
    rec = ruptured["record"]
    assert (rec["kind"], rec["class"], rec["outcome"], rec["stage"]) == (
        ev.SHEET_RUPTURE, ev.SHEET_RUPTURE_CLASS, "made", "ring")
    geo = rec["geometry"]
    assert geo["checks"]["failed"] == [] and geo["w"] == pytest.approx(req["fold_gap"])
    assert 0.0 < geo["V_rim_over_V_b"] < 1e-3
    # the rim's budgets: its own quadrature, every point located, its volume the rim's
    q = rec["rim_quadrature"]
    assert q["checked"] and q["found"] == q["points"] > 0
    assert q["V_quad"] == pytest.approx(geo["V_rim"], rel=RING_BUDGET_LIMITS["R1"])
    removed = rec["removed"]
    assert removed["V"] == geo["V_rim"] and removed["Pz"] == q["Pz"] and removed["KE"] == q["KE"]
    assert removed["KE"] > 0.0 and removed["Pz"] != 0.0
    # the grid's reading of the rim beside it (this rim spans many of its cells)
    assert removed["Pz_grid"] == pytest.approx(removed["Pz"], rel=0.05)
    assert rec["filled"] == {"V": -removed["V"], "Pz": -removed["Pz"], "KE": -removed["KE"]}
    c = rec["closures"]
    assert c["hard"] == [] and "S_base" not in c["flags"]
    for x in ("S1", "S2", "S3", "S3_box"):
        assert c[x] <= c["limits"][x], x
    assert c["limits"]["S1"] >= RING_BUDGET_LIMITS["R1"] * removed["V"]  # its magnitude's (F2)
    assert c["relative"]["S1"] >= 0.0
    assert rec["base_guard"]["ok"]
    # one ring, the rim gone: the liquid less by the rim's volume
    assert len(ruptured["loops"]) == 1
    assert ruptured["volume_after"] - ruptured["volume_before"] == pytest.approx(
        -removed["V"], rel=0.05)
    P = np.column_stack([ruptured["loops"][0]["r"], ruptured["loops"][0]["z"]])
    assert revolved_volume(P) == pytest.approx(ruptured["V_ring"] + removed["V"],
                                               abs=0.05 * removed["V"]
                                               + 5 * ruptured["noise"]["V"])
    tr = rec["transfer"]
    assert abs(tr["reported_unlocated"] - tr["predicted_unlocated"]) <= surgery.UNLOCATED_SLACK
    assert tr["predicted_off_closure"] == 0 and tr["closure_nodes"] > 0
    assert tr["former_gas"]["bulk"] == 0 and tr["former_gas"]["axis"] == 0
    # its event, the lip's history in its record, and the run's count
    assert rec["lip"]["samples"][:2] == ruptured["history"] and len(rec["lip"]["samples"]) == 3
    assert rec["lip"]["h_s"] == pytest.approx(FLOORS * H_FLOOR)
    made = [e for e in ruptured["log"] if e["event"] == "sheet-rupture"]
    assert len(made) == 1 and made[0]["V"] == removed["V"]
    assert ruptured["summary"]["by_kind"] == {ev.SHEET_RUPTURE: 1}


def test_a_restart_from_the_state_after_the_rupture_goes_on_as_the_run_does(ruptured):
    assert ruptured["lip_live"] == []
    assert "lip_samples" not in ruptured["postcut_meta"]
    assert ruptured["lip_restored"] == ruptured["lip_live"]
    # the same ring and the same measurement
    live, restart = ruptured["loops_live"], ruptured["loops_restart"]
    assert len(live) == len(restart) == 1 and np.array_equal(live[0], restart[0])
    a, b = ruptured["m_live"], ruptured["m_restart"]
    same = [k for k in a if isinstance(a[k], float) and k in b
            and not (a[k] == b[k] or (math.isnan(a[k]) and math.isnan(b[k])))]
    assert same == []
    # the next reading of the lip: the remaining lip, its history begun afresh, in both
    assert ruptured["lip_next_restart"] == ruptured["lip_next_live"]
    assert len(ruptured["lip_next_live"]) == 1
    assert ruptured["lip_next_live"][0][1] > ruptured["fold"]["gap"]
    assert ruptured["next_restart"] == ruptured["next_live"]
