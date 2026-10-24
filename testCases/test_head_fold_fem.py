"""Finite-element test of the cut of the upward jet's head folding onto the core
(``events.far_head_request``'s ``head_fold``, ``surgery.liquid_cut``'s ``base_check``,
``driver.CollapseDriver._head_fold_check``) on the heads' state of ``test_jetcore_fem.py`` (the
flow about the necks imposed, ``pinch.far_head`` and ``jetcore`` on, the groove part enabled): a
closing groove between the widest node of the open chain's head and a node of the stem below its
neck, in the core, is met by the cut of that head at its root neck, never filled; the base of the
upward jet read before and after the cut (``base_guard`` in the record) within
``HEAD_FOLD_BASE_TOL``, the hard closures closed, the run going on; under a tolerance no cut can
meet the same cut is the hard closure ``F_base``, the resumable stop ``event-unclosed`` on the state
before it.

Each problem compiles in seconds; the module takes about two minutes.
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

import test_liquid_cut_fem as lcf  # noqa: E402
from test_jetcore_fem import FAR_HEAD_BLOCK, _chains, gas_cut_record, heads_problem  # noqa: E402

from faraday_singularity import driver as drv_mod  # noqa: E402
from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import groovesurgery  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402
from faraday_singularity.driver import CollapseDriver, CollapseSettings  # noqa: E402

BLOCK = {**FAR_HEAD_BLOCK, "groove": {"enabled": True}}


def _run(tmp_path, monkeypatch, *, tol=None):
    def no_fill(*a, **kw):
        raise AssertionError("a groove of the head onto the core is not filled")

    monkeypatch.setattr(groovesurgery, "groove_fill", no_fill)
    if tol is not None:
        monkeypatch.setattr(drv_mod, "HEAD_FOLD_BASE_TOL", tol)
    recon = rc.ReconnectSettings(delta=lcf.DELTA, dt0=1e-5, tip_size=0.25, bridge_size=0.25,
                                 focus_size=0.25)
    block = ev.EventSettings.from_dict(BLOCK)
    out = {}
    with heads_problem(tmp_path / "pb", recon) as pb:
        pb.initialise()
        gas_cut_record(pb, recon)
        lcf.impose(pb, lcf.stream_velocity, list(_chains()[2]))
        drv = CollapseDriver(pb, CollapseSettings(t_end=10.0, dt_max=0.01, max_steps=2,
                                                  state_dt=0.0, state_keep=10),
                             tmp_path / "run", reconnect=recon, events=block)
        m = drv.reset_after_cut(pb.cut)
        named = drv._named_post_cut_chains()
        head = next(x for x in ev.liquid_necks(named, block.liquid)
                    if x["chain"] == "open" and x["status"] == "head" and x["head"][0] == 0)
        core = np.asarray(drv._post_cut_state()["core"]["mask"], dtype=bool)
        P = np.asarray(dict(named)["open"].points)
        widest = int(np.argmax(P[:head["index"], 0]))
        stem = int(np.flatnonzero(core & (np.arange(len(P)) > head["index"]))[3])
        groove = {"reason": ev.EVENT_REQUEST, "kind": "groove", "class": "A-groove",
                  "handler": ev.GROOVE_HANDLER, "chain": "open", "contact": "fold",
                  "fold_pair": [widest, stem], "fold_gap": 1e-4, "fold_limit": 2e-4,
                  "jet": "cavity_tip", "side": "gas"}
        out.update(head=head, widest=widest, stem=stem, core_neck=bool(core[head["index"]]),
                   head_in_core=int(core[:head["index"]].sum()), m=m)
        _, stop = drv._meet_event(m, groove)
        out["stop"] = stop
        out["events"] = [dict(x) for x in pb.events]
        if stop is None:
            out["result"] = drv.run(1e-8)
        drv.close()
    out["records"] = {p.name: json.loads(p.read_text())
                      for p in sorted((tmp_path / "run" / "events").glob("event_*.json"))}
    return out


def test_the_head_folding_onto_the_stem_is_cut_and_leaves_the_base(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch)
    assert out["head_in_core"] == 0
    assert out["stop"] is None, out["stop"]
    rec = out["events"][-1]
    assert (rec["kind"], rec["class"], rec["outcome"]) == ("far-head", "F-head", "made")
    assert rec["request"]["index"] == out["head"]["index"]
    assert rec["request"]["head_fold"] == {"head_node": out["widest"], "core_node": out["stem"]}
    assert not rec["closures"]["hard"] and rec["removed"]["V"] > 0.0
    g = rec["base_guard"]
    assert g["ok"] is True and g["tol"] == pytest.approx(drv_mod.HEAD_FOLD_BASE_TOL)
    assert all(abs(v) <= drv_mod.HEAD_FOLD_BASE_TOL for v in g["change"].values())
    assert g["before"]["r_j"] == pytest.approx(out["m"]["r_j"])
    assert out["result"]["events"]["by_kind"] == {"far-head": 1}
    assert out["result"]["stop"]["reason"] == "max-steps" and out["result"]["steps"] == 2


def test_a_cut_that_would_move_the_base_beyond_the_tolerance_is_a_stop(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch, tol=1e-15)
    stop = out["stop"]
    assert stop["reason"] == ev.EVENT_UNCLOSED and stop["hard"] == ["F_base"]
    assert Path(stop["precut"]).name == "precut_e001.dump"
    rec = out["records"]["event_001.json"]
    assert rec["outcome"] == "unclosed" and rec["failed_at"] == "base"
    assert rec["base_guard"]["ok"] is False and "F_base" in rec["closures"]["hard"]
