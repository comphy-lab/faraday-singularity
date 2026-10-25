"""Finite-element test of the ring's refinement on a freshly built mesh
(``driver.CollapseDriver._ring_refine``) on the synthetic pierced state of ``ring_fem_state.py``.

The ring cut's mesh of that state turns by about 63 degrees in one element of the ring, where the
downward jet's thread meets the roof's fillet (meshed at the interface's own size), and by about
1.3 degrees at most on the open chain. With the guard's limits lowered so that the ring alone meets
the fresh one (``edge_turn_refine`` 20, ``edge_turn_stop_fresh`` 50, ``edge_turn_stop`` 120) and
the production's jetcore sizing (far kinks recentred, the ring's kinks at the column's size):

* with the refine (this commit) the driver makes the ring cut, meets the ring's fresh stop by the
  rebuild with the element in a far kink of its own at ``min(1/2, edge_turn_refine/turn)`` of its
  chord, the very state the stop was read on, until the fresh mesh holds the turn within the fresh
  limit (at most ``RING_REFINE_TRIES`` rebuilds, an event ``ring-refine`` each, a yield each), the
  ring's volume kept to the rebuild's noise, and steps on to its limit; the dumps after it hold the
  refine's own far kink, marked, at most its size, and the far kink about a feature elsewhere
  (the column's corner) an unmarked one of its own;
* without it (``RING_REFINE_TRIES`` 0, e923ead's behaviour) the same run stops there,
  ``interface-unresolved`` on the ring on the fresh mesh, with the reading the first rebuild met.

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

from faraday_singularity import driver as dv  # noqa: E402
from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402
from faraday_singularity.driver import CollapseDriver, CollapseSettings  # noqa: E402
from faraday_singularity.reconnect import revolved_volume  # noqa: E402
from faraday_singularity.ring import loop_edges  # noqa: E402

RECON_LOW = rc.ReconnectSettings(delta=S.DELTA, dt0=1e-5, tip_size=0.25, bridge_size=0.25,
                                 focus_size=0.25, edge_turn_refine=20.0,
                                 edge_turn_stop_fresh=50.0, edge_turn_stop=120.0)
BLOCK = {"ring": {"enabled": True, "sigma_V": 1e-6, "sigma_Pz": 1e-6, "sigma_KE": 1e-6},
         "jetcore": {"enabled": True, "far_kinks": "recentre", "ring_kink_floor": True}}
STEPS = 3


def _events(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def _ring_profiles(run: Path, reason: str) -> list[tuple[dict, np.ndarray]]:
    out = []
    for f in sorted((run / "profiles").glob("profile_*_ring.txt")):
        head = json.loads(f.read_text().splitlines()[0][2:])
        if head["reason"] == reason:
            out.append((head, np.loadtxt(f, comments="#")[:-1, :2]))  # the seam not repeated
    return out


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("ring-refine")
    res = {}
    tries = dv.RING_REFINE_TRIES
    try:
        for name, n in (("refine", tries), ("off", 0)):
            dv.RING_REFINE_TRIES = n
            block = ev.EventSettings.from_dict(BLOCK)
            with S.pierce_problem(out / f"pb-{name}") as pb:
                pb.initialise()
                S.gas_cut_record(pb)
                S.impose(pb)
                drv = CollapseDriver(pb, CollapseSettings(t_end=10.0, dt_max=0.01,
                                                          max_steps=STEPS, state_dt=0.0,
                                                          state_keep=50),
                                     out / name, reconnect=RECON_LOW, events=block)
                drv.reset_after_cut(pb.cut)
                result = drv.run(1e-8)
                res[name] = {"result": result, "run": out / name,
                             "kinks": json.loads(json.dumps(drv.state.kinks)),
                             "far_yields": drv.state.far_yields, "ring": dict(pb.ring),
                             "h_floor": drv._h_floor(),
                             "events": _events(out / name / "events.jsonl")}
                drv.close()
    finally:
        dv.RING_REFINE_TRIES = tries
    return res


def test_without_the_refine_the_run_stops_on_the_rings_fresh_mesh(runs):
    off = runs["off"]["result"]
    stop = off["stop"]
    assert (stop["reason"], stop["trigger"], stop["chain"], stop["fresh"]) == (
        "interface-unresolved", "edge-turn", "ring", True), stop
    assert RECON_LOW.edge_turn_stop_fresh < stop["turn"] <= RECON_LOW.edge_turn_stop
    assert off["steps"] == 0 and off["events"]["ring"]["cut"] == 1
    assert "ring_refines" not in stop
    assert not any(e["event"] == "ring-refine" for e in runs["off"]["events"])


def test_the_refine_rebuilds_the_rings_far_kink_and_the_run_steps_on(runs):
    got, off = runs["refine"], runs["off"]["result"]["stop"]
    result = got["result"]
    assert result["stop"]["reason"] == "max-steps" and result["steps"] == STEPS, result["stop"]
    assert result["events"]["made"] == 1 and result["events"]["ring"]["cut"] == 1
    refines = [e for e in got["events"] if e["event"] == "ring-refine"]
    assert 1 <= len(refines) <= dv.RING_REFINE_TRIES
    assert [e["attempt"] for e in refines] == list(range(1, len(refines) + 1))
    assert all(e["step"] == 0 for e in refines)  # at the ring cut's fresh mesh
    first = refines[0]["reading"]
    # the very state the run without the refine stopped on
    for k in ("turn", "L", "r", "z", "index"):
        assert first[k] == pytest.approx(off[k], rel=1e-12, abs=1e-15), k
    for e in refines:
        x = e["reading"]
        f = min(0.5, RECON_LOW.edge_turn_refine / x["turn"])
        assert e["target"] == pytest.approx(max(got["h_floor"], f * x["L"]), rel=1e-12)
        assert e["radius"] >= 4.0 * x["L"] * (1 - 1e-12)
        assert e["base"]["ok"] and not e["base"]["checked"]  # the state has no born jet
    # each rebuild finer at the element; the last fresh mesh within the fresh limit
    profiles = _ring_profiles(got["run"], "ring-refine")
    assert len(profiles) == len(refines)
    turns = []
    for head, L in profiles:
        e = loop_edges(L)["edges"]
        near = np.hypot(e["mid"][:, 0] - first["r"], e["mid"][:, 1] - first["z"]) <= 2 * first["L"]
        turns.append(float(e["turn"].max()))
        assert near.any() and float(e["L"][near].min()) < first["L"]
    assert turns[-1] <= RECON_LOW.edge_turn_stop_fresh and turns[-1] < first["turn"]
    # the ring's volume kept to the rebuild's noise
    V0 = got["ring"]["V0"]
    for head, L in profiles:
        assert abs(revolved_volume(L) - V0) <= 1e-3 * V0
    # a yield each; the refine's own far kink, marked, at most its size (or released after its
    # hold), every other region as the rule makes it
    assert got["far_yields"] >= len(refines)
    names = {e["name"] for e in refines}
    assert names and all(n.startswith("kink_ring_r") and n.endswith("_far") for n in names)
    for n in names:
        k = got["kinks"][n[len("kink_"):]]
        assert k["name"] == n and k["far"] and k["refined"] and k["chain"] == "ring"
        if not k["released"]:
            assert k["target"] <= min(e["target"] for e in refines if e["name"] == n) * (1 + 1e-12)
    assert not any(g.get("refined") for key, g in got["kinks"].items()
                   if f"kink_{key}" not in names)
    rows = (got["run"] / "jetcore.txt").read_text().splitlines()
    cols = rows[0][2:].split("\t")
    assert int(float(rows[-1].split("\t")[cols.index("far_yields")])) == got["far_yields"]


def test_the_far_kinks_after_the_refine_follow_their_rule_in_the_dumps(runs):
    """In every dump after the refine the refine's own far kink (``kink_ring_r<n>_far``) is there,
    marked, at most its size or released after its hold, and every other far kink of the ring (the
    ring's own or one of the multi-feature mode's, about another feature: on this state the
    column's lower corner, far from the thread's fillet the refine was made at) unmarked, at the
    column's size unless it has yielded; the dumps hold the yields."""
    got = runs["refine"]
    run = got["run"]
    refines = [e for e in got["events"] if e["event"] == "ring-refine"]
    names = {e["name"] for e in refines}
    h_col = got["ring"]["sizing"]["h_col"]
    metas = [json.loads(p.read_text()) for p in sorted((run / "states").glob("state_*.json"))]
    assert metas
    for m in metas:
        kinks = m["kinks"]
        for n in names:
            g = kinks[n[len("kink_"):]]
            assert g["refined"] and g["far"] and g["yields"] >= 1
            if not g["released"]:
                assert g["target"] <= min(e["target"] for e in refines if e["name"] == n) * (
                    1 + 1e-12)
        others = {k: g for k, g in kinks.items() if g.get("far") and g.get("chain") == "ring"
                  and f"kink_{k}" not in names}
        for k, g in others.items():
            assert "refined" not in g and (k == "ring_far" or g.get("multi")), k
            if not g["released"] and not g["yields"]:
                assert g["target"] >= h_col * (1 - 1e-12), k
                first = refines[0]["reading"]
                assert math.hypot(g["r"] - first["r"], g["z"] - first["z"]) > 4 * first["L"], k
    assert metas[-1]["far_yields"] == got["far_yields"]
