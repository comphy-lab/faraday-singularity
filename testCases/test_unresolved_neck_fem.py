"""Finite-element test of the cut of a far head whose neck a freshly built mesh does not resolve
(``events.unresolved_neck_request``, ``driver.CollapseDriver._unresolved_neck_request``) on the
heads' state of ``test_jetcore_fem.py`` (the flow about the necks imposed, ``pinch.far_head`` and
``jetcore`` on): under a ``liquid.resolution`` above the state's (its two tracked necks, the upward
jet's head on the open chain and the downward jet's on the bubble, have 7.7 element lengths across
their radii), the run's loop meets the stop ``liquid-neck-unresolved`` of each by the cut of its
head at that neck, the open chain's first (the smaller ratio), then the bubble's: kind
``far-head``, class ``F-head``, the stop as ``far_event``, the hard closures closed, the upward
jet's base read before and after each cut within ``HEAD_FOLD_BASE_TOL`` (``base_guard``), the
liquid's volume less the heads', and the run going on to its step limit. Under a tolerance no cut
can meet, the first cut is the hard closure ``F_base``, the resumable stop ``event-unclosed`` on
the state before it.

Each problem compiles in seconds; the module takes about two minutes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

pytest.importorskip("pyoomph")

import test_liquid_cut_fem as lcf  # noqa: E402
from test_events_plumbing import _events  # noqa: E402
from test_jetcore_fem import FAR_HEAD_BLOCK, _chains, gas_cut_record, heads_problem  # noqa: E402

from faraday_singularity import driver as drv_mod  # noqa: E402
from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402
from faraday_singularity.driver import CollapseDriver, CollapseSettings  # noqa: E402

RESOLUTION = 8.0  # above the 7.7 element lengths across both tracked necks of the state


def _run(tmp_path, monkeypatch, *, tol=None):
    if tol is not None:
        monkeypatch.setattr(drv_mod, "HEAD_FOLD_BASE_TOL", tol)
    recon = rc.ReconnectSettings(delta=lcf.DELTA, dt0=1e-5, tip_size=0.25, bridge_size=0.25,
                                 focus_size=0.25)
    block = json.loads(json.dumps(FAR_HEAD_BLOCK))
    block["liquid"]["resolution"] = RESOLUTION
    block = ev.EventSettings.from_dict(block)
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
        out["necks"] = ev.liquid_necks(named, block.liquid)
        out["stop0"] = drv._stop(m)
        out["V0"] = pb.cut_observables()["volume"]
        out["m"] = m
        out["result"] = drv.run(1e-8)
        out["V1"] = pb.cut_observables()["volume"]
        out["events"] = [dict(x) for x in pb.events]
        drv.close()
    out["records"] = {p.name: json.loads(p.read_text())
                      for p in sorted((tmp_path / "run" / "events").glob("event_*.json"))}
    out["log"] = _events(tmp_path / "run")
    return out


def test_the_unresolved_necks_of_the_far_heads_are_cut_and_the_run_goes_on(tmp_path,
                                                                            monkeypatch):
    out = _run(tmp_path, monkeypatch)
    heads = [x for x in out["necks"] if x["status"] == "head"]
    assert {x["chain"] for x in heads} == {"open", "bubble"}
    assert all(x["r_n"] < RESOLUTION * x["h_loc"] for x in heads)
    stop0 = out["stop0"]
    assert stop0["reason"] == ev.LIQUID_NECK_UNRESOLVED and stop0["chain"] == "open"
    made = [r for r in out["events"] if r.get("kind") == "far-head"]
    assert [r["chain"] for r in made] == ["open", "bubble"]
    for rec in made:
        assert (rec["class"], rec["outcome"]) == ("F-head", "made")
        far = rec["request"]["far_event"]
        assert far["kind"] == ev.LIQUID_NECK_UNRESOLVED and far["chain"] == rec["chain"]
        assert rec["request"]["unresolved"]["resolution"] == RESOLUTION
        assert rec["request"]["index"] == far["index"]
        assert not rec["closures"]["hard"] and rec["removed"]["V"] > 0.0
        g = rec["base_guard"]
        assert g["ok"] is True and g["tol"] == pytest.approx(drv_mod.HEAD_FOLD_BASE_TOL)
        assert all(abs(v) <= drv_mod.HEAD_FOLD_BASE_TOL for v in g["change"].values())
    assert made[0]["base_guard"]["before"]["r_j"] == pytest.approx(out["m"]["r_j"])
    removed = sum(r["removed"]["V"] for r in made)
    assert out["V1"] == pytest.approx(out["V0"] - removed, abs=10.0 * lcf.KEYED["sigma_V"])
    result = out["result"]
    assert result["events"]["by_kind"] == {"far-head": 2}
    assert result["events"]["by_tip"] == {"cavity_tip": 1, "bubble_top": 1}
    assert result["stop"]["reason"] == "max-steps" and result["steps"] == 2
    far = [e for e in out["log"] if e["event"] == "far-head"]
    assert [e["far_kind"] for e in far] == [ev.LIQUID_NECK_UNRESOLVED] * 2
    assert set(out["records"]) == {"event_001.json", "event_002.json"}


def test_a_cut_that_would_move_the_base_beyond_the_tolerance_is_a_stop(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch, tol=1e-15)
    stop = out["result"]["stop"]
    assert stop["reason"] == ev.EVENT_UNCLOSED and stop["hard"] == ["F_base"]
    assert Path(stop["precut"]).name == "precut_e001.dump"
    rec = out["records"]["event_001.json"]
    assert rec["outcome"] == "unclosed" and rec["failed_at"] == "base"
    assert rec["request"]["far_event"]["kind"] == ev.LIQUID_NECK_UNRESOLVED
    assert rec["base_guard"]["ok"] is False and "F_base" in rec["closures"]["hard"]
