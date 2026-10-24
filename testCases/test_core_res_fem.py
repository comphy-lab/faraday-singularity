"""Finite-element tests of the core of the upward jet meshed finer (``jetcore.core_elements_per_scale``).

* On the synthetic heads' state of ``test_jetcore_fem.py`` (a jet with a necked head on each chain,
  the open chain's base, stem and walls its core), with the far sizing of production (the far
  floor, the far kinks recentred) and the key at 16 against the run's 8: on the state's own mesh
  every region is the region without the key and each region whose ball holds core nodes has its
  twin at half its target; after the rule's rebuilds the elements of the core are half as long as
  without the key and those off it as long, the rebuilt focus holding each twin with the core's
  selection.
* A restart in a new process from a dump of the run after the cut of the synthetic necked cavity
  (``test_events_fem.py``'s script, with jetcore) under the key: the leg names the key, the rule
  rebuilds at the first step it may from the twins, and the core holds about twice the nodes of
  the same restart without the key, the far region as many.

Each problem compiles in seconds; the module takes a few minutes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

pytest.importorskip("pyoomph")

import test_jetcore_fem as J  # noqa: E402
from test_jetcore_identity import _run_fem, _spec  # noqa: E402

from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402
from faraday_singularity.driver import CollapseDriver, CollapseSettings  # noqa: E402

FAR = {"enabled": True, "far_floor": J.H_HEAD, "far_kinks": "recentre", "far_refine_ratio": 2.0}
RECON = rc.ReconnectSettings(delta=J.DELTA, dt0=1e-5, tip_size=0.25, bridge_size=0.25,
                             focus_size=0.25, courant=0.4, elements_per_scale=8)


def _spacing(drv) -> dict:
    """The element length at each node of the open chain (twice the shorter of its segments, as
    ``reconnect.courant_limit`` reads it) and the core's mask, of the driver's current state."""
    pc = drv._post_cut_state()
    o = next(c for c in pc["segs"] if c["open"])
    ds = np.hypot(np.diff(o["r"]), np.diff(o["z"]))
    h = 2.0 * np.minimum(np.r_[ds[0], ds], np.r_[ds, ds[-1]])
    return {"h": h, "mask": np.asarray(pc["core"]["mask"], dtype=bool), "core": pc["core"]}


@pytest.fixture(scope="module")
def heads(tmp_path_factory):
    """The heads' state with the far sizing, without the key and with it at 16: the regions on the
    state's own mesh, then the rule's rebuilds until none is asked for (at most four)."""
    out = tmp_path_factory.mktemp("core-res")
    res = {}
    for name, extra in (("plain", {}), ("fine", {"core_elements_per_scale": 16})):
        block = ev.EventSettings.from_dict({**J.BLOCK, "jetcore": {**FAR, **extra}})
        with J.heads_problem(out / f"pb-{name}", RECON) as pb:
            pb.initialise()
            J.gas_cut_record(pb, RECON)
            drv = CollapseDriver(pb, CollapseSettings(t_end=10.0, dt_max=0.01, max_steps=1,
                                                      state_dt=0.0, state_keep=10),
                                 out / name, reconnect=RECON, events=block)
            first = drv.reset_after_cut(pb.cut)
            before = _spacing(drv)
            rebuilds = 0
            for _ in range(4):
                m = drv.measure()
                drv.state.last_remesh_step = drv.state.refine_blocked_until = -10
                if not drv._maybe_remesh_features(m):
                    break
                rebuilds += 1
            res[name] = {"first": {g["name"]: dict(g) for g in first["regions"]},
                         "before": before, "after": _spacing(drv), "rebuilds": rebuilds,
                         "focus": pb.active_mesh_template().focus,
                         "equations": pb.ndof(), "out": out / name}
            drv.close()
    return res


def test_on_the_states_mesh_the_twins_halve_the_regions_that_hold_core_nodes(heads):
    """Every region without the key is a region with it, the same but for its spacing, read off the
    core where it has a twin; each twin at half its region's target; besides, a kink region of the
    core (``kink_open``), since the state's core elements turn by more than half the guard's limit
    (``jetcore.core_turns``)."""
    plain, fine = heads["plain"]["first"], heads["fine"]["first"]
    twins = {n: g for n, g in fine.items() if "twin_of" in g}
    others = {n for n in fine if n not in twins}
    assert twins and others >= set(plain) and others - set(plain) <= {"kink_open"}
    for name, g in plain.items():
        for k in ("target", "r", "z", "radius", "h_max", "class"):
            if k == "h_max" and name in {t["twin_of"] for t in twins.values()}:
                continue  # read off the core where it has a twin
            assert fine[name][k] == g[k] or (k == "h_max" and np.isnan(g[k])), (name, k)
    for name, t in twins.items():
        assert t["target"] == 0.5 * plain[t["twin_of"]]["target"] and t["class"] == "core"
    assert "core_jet_base_cavity_tip" in twins and "core_neck_open" in twins
    core = heads["fine"]["before"]["mask"]
    assert 0 < core.sum() < len(core)
    assert np.array_equal(core, heads["plain"]["before"]["mask"])


def test_after_the_rules_rebuilds_the_core_is_meshed_twice_as_finely_and_the_rest_as_before(heads):
    """The median element length of the core about half of that without the key, off the core
    within a few per cent of it; the rebuilt focus holds each twin with the core's selection."""
    p, f = heads["plain"]["after"], heads["fine"]["after"]
    assert heads["fine"]["rebuilds"] >= 1 and heads["plain"]["rebuilds"] >= 1
    ratio_core = np.median(f["h"][f["mask"]]) / np.median(p["h"][p["mask"]])
    ratio_far = np.median(f["h"][~f["mask"]]) / np.median(p["h"][~p["mask"]])
    assert 0.45 <= ratio_core <= 0.56, ratio_core
    assert 0.97 <= ratio_far <= 1.03, ratio_far
    assert f["mask"].sum() >= 1.7 * p["mask"].sum()
    assert heads["fine"]["equations"] > heads["plain"]["equations"]
    focus = heads["fine"]["focus"]
    twins = [g for g in focus.regions if g.name.startswith("core_")]
    assert twins and all(g.core is not None for g in twins)
    assert all(g.core is None for g in focus.regions if not g.name.startswith("core_"))
    assert all(g.core is None for g in heads["plain"]["focus"].regions)


def test_a_restart_with_the_key_rebuilds_the_core_at_the_new_sizes(tmp_path):
    """From the dump of step 3 of the run after the cut of the synthetic necked cavity with jetcore
    (the far floor and kinks; no jet, so the whole open chain is the core), restarts in new
    processes with and without the key: with it the leg names it, a rebuild asked by the twins
    comes at the restart's first step, the dumps hold the twins with the core's selection, and the
    twins' elements (the open chain's nodes in the cut's regions) are about half as long as their
    regions' in the restart without it, the far region's nodes as many."""
    jet = {"enabled": True, "far_floor": 2e-4, "far_kinks": "recentre"}
    run = _run_fem(ROOT / "src", tmp_path / "base", _spec(jetcore=jet))
    state = run / "states" / "state_00000003.dump"
    rows = {}
    for name, extra in (("plain", {}), ("fine", {"core_elements_per_scale": 16})):
        spec = dict(_spec(jetcore={**jet, **extra}), restart=str(state))
        out = _run_fem(ROOT / "src", tmp_path / name, spec)
        events = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
        leg = next(e for e in events if e["event"] == "leg")
        remesh = [e for e in events if e["event"] == "remesh"]
        rows[name] = {"leg": leg, "remesh": remesh, "out": out,
                      "jet": (out / "jetcore.txt").read_text().splitlines()}
    fine, plain = rows["fine"], rows["plain"]
    assert fine["leg"]["changed"] == ["jetcore.core_elements_per_scale"]
    assert plain["leg"]["changed"] == []
    first = fine["remesh"][0]
    assert any(a.startswith("core_") for a in first["asked_by"]), first["asked_by"]
    assert first["step"] == 4
    dumps = sorted((fine["out"] / "states").glob("state_*.json"))
    focus = json.loads(dumps[-1].read_text())["focus"]
    twins = [g for g in focus["regions"] if g["name"].startswith("core_")]
    assert twins and all(len(g["core"]) == 8 for g in twins)
    assert all("core" not in g for g in focus["regions"] if not g["name"].startswith("core_"))
    late = {g["name"]: g for g in fine["remesh"][-1]["regions"]}  # after the first rebuilds
    ref = {g["name"]: g for g in plain["remesh"][-1]["regions"]}
    pairs = [(g["h_max"], ref[g["twin_of"]]["h_max"]) for g in late.values() if "twin_of" in g
             and np.isfinite(g["h_max"]) and np.isfinite(ref.get(g["twin_of"], {}).get("h_max",
                                                                                     np.nan))]
    assert pairs and all(0.4 <= a / b <= 0.6 for a, b in pairs), pairs
    head = fine["jet"][0][2:].split("\t")
    last = {name: dict(zip(head, rows[name]["jet"][-1].split("\t"))) for name in rows}
    assert last["fine"]["far_nodes"] == last["plain"]["far_nodes"]
    assert int(last["fine"]["core_nodes"]) > int(last["plain"]["core_nodes"])
