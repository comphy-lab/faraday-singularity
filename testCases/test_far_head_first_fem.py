"""Finite-element test of a closing groove on a far head, met by the far-head cut of that head before
any fill (``driver.CollapseDriver._meet_event``, decision 22), on the heads' state of
``test_jetcore_fem.py`` (the flow about the necks imposed, ``pinch.far_head`` and ``jetcore`` on,
the groove part enabled): the groove on the head of the bubble's top is never filled (the fill
would fail the test), the head is cut at its neck (kind ``far-head``, class ``F-head``, the groove
its ``far_event``), its hard closures closed, the head's volume removed, and the run goes on from
the fresh mesh.

Each problem compiles in seconds; the module takes about a minute.
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
from test_jetcore_fem import FAR_HEAD_BLOCK, _chains, gas_cut_record, heads_problem  # noqa: E402

from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import groovesurgery  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402
from faraday_singularity.driver import CollapseDriver, CollapseSettings  # noqa: E402

BLOCK = {**FAR_HEAD_BLOCK, "groove": {"enabled": True}}


def test_a_groove_on_the_downward_head_is_cut_with_the_head_and_never_filled(tmp_path,
                                                                             monkeypatch):
    def no_fill(*a, **kw):
        raise AssertionError("a groove on a far head is not filled")

    monkeypatch.setattr(groovesurgery, "groove_fill", no_fill)
    recon = rc.ReconnectSettings(delta=lcf.DELTA, dt0=1e-5, tip_size=0.25, bridge_size=0.25,
                                 focus_size=0.25)
    block = ev.EventSettings.from_dict(BLOCK)
    with heads_problem(tmp_path / "pb", recon) as pb:
        pb.initialise()
        gas_cut_record(pb, recon)
        lcf.impose(pb, lcf.stream_velocity, list(_chains()[2]))
        drv = CollapseDriver(pb, CollapseSettings(t_end=10.0, dt_max=0.01, max_steps=2,
                                                  state_dt=0.0, state_keep=10),
                             tmp_path / "run", reconnect=recon, events=block)
        m = drv.reset_after_cut(pb.cut)
        named = drv._named_post_cut_chains()
        neck = next(x for x in ev.liquid_necks(named, block.liquid)
                    if x["chain"] == "bubble" and x["status"] == "head")
        site = neck["head"][1] - 6
        groove = {"reason": ev.EVENT_REQUEST, "kind": "groove", "class": "A-groove",
                  "handler": ev.GROOVE_HANDLER, "chain": "bubble", "contact": "fold",
                  "fold_pair": [site - 4, site], "fold_gap": 1e-4, "fold_limit": 2e-4}
        V0 = pb.cut_observables()["volume"]
        _, stop = drv._meet_event(m, groove)
        assert stop is None, stop
        rec = pb.events[-1]
        assert (rec["kind"], rec["class"], rec["outcome"]) == ("far-head", "F-head", "made")
        assert rec["request"]["index"] == neck["index"]
        assert rec["request"]["far_event"]["class"] == "A-groove"
        assert not rec["closures"]["hard"] and rec["removed"]["V"] > 0.0
        assert pb.cut_observables()["volume"] == pytest.approx(V0 - rec["removed"]["V"],
                                                               abs=5.0 * lcf.KEYED["sigma_V"])
        result = drv.run(1e-8)
        drv.close()
    assert result["events"]["made"] == 1 and result["events"]["by_kind"] == {"far-head": 1}
    assert result["stop"]["reason"] == "max-steps" and result["steps"] == 2
    events = [json.loads(x) for x in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    far = [e for e in events if e["event"] == "far-head"]
    assert len(far) == 1 and far[0]["far_class"] == "A-groove"
    assert not any(e["event"] == "groove-fill" for e in events)
