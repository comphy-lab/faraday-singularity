"""Finite-element test of the fall-back of a groove's fill on a far head to the far-head cut of
that head (``driver.CollapseDriver._groove_fallback``), on the heads' state of
``test_jetcore_fem.py`` (the flow about the necks imposed, ``pinch.far_head`` and ``jetcore`` on,
the groove part enabled).

* ``refused``: a groove requested on the bubble's tracked head whose fold pair does not close
  there (the state has no groove on the head), so that the fill refuses it before the problem
  changes: the far-head cut of that head is made, nothing is loaded back, the run steps on.
* ``unclosed``: the same request, its fill stood in for by a real change of the problem (the state
  before the event kept as the fill keeps it, a rebuild under a finer focus and a projection, then
  ``surgery.BudgetUnclosed`` with ``G3_box`` its miss): the state before the event is loaded back
  (``CollapseDriver._reload_event_state``), so that the cut starts from that state (its dofs and
  pinned values, its time, the focus, the rebuild count) and its record and the run after it
  equal those of ``refused`` to round-off.

Each problem compiles in seconds; the module takes a few minutes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
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

from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import groovesurgery, surgery  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402
from faraday_singularity.driver import CollapseDriver, CollapseSettings  # noqa: E402
from faraday_singularity.mesh import Focus  # noqa: E402

DELTA = lcf.DELTA
BLOCK = {**FAR_HEAD_BLOCK, "groove": {"enabled": True}}


def fingerprint(pb) -> tuple:
    """The time, the dof count and a hash of the dofs and pinned values (positions included)."""
    t, _, n, h = pb._fingerprint()
    return t, n, h


def stand_in_fill(seen: dict):
    """A fill that changes the problem and does not close: the state before the event kept as
    ``groovesurgery.groove_fill`` keeps it, a rebuild under the focus with every region at 0.8 of
    its size, a projection, then ``BudgetUnclosed`` (``G3_box``) with its record written."""

    def groove_fill(pb, events, reconnect, outdir, request, *, k, dt_prev, elements_per_scale,
                    h_floor, fold, core=None, driver=None, on_made=None, log=print):
        event = {**{x: v for x, v in request.items() if x != "reason"}, "k": k}
        precut = surgery._keep(pb, driver, Path(outdir), f"precut_e{k:03d}",
                               {"cut_stage": ev.PRECUT_EVENT, "event": event})["path"]
        seen["precut"] = fingerprint(pb)
        seen["focus"] = pb.active_mesh_template().focus
        tmpl = pb.active_mesh_template()
        old = tmpl.focus
        tmpl.focus = Focus(size=0.8 * old.size, radius=old.radius, grading=old.grading,
                           regions=tuple(replace(g, size=0.8 * g.size) for g in old.regions))
        pb.force_remesh({tmpl})
        pb._domains_to_remesh.clear()
        pb.invalidate_cached_mesh_data()
        surgery.project(pb, replace(reconnect, dt0=dt_prev),
                        pb.get_current_time(dimensional=False, as_float=True), {})
        seen["changed"] = fingerprint(pb)
        seen["remesh_count_changed"] = pb.remesh_count
        record = {"schema": ev.EVENT_SCHEMA, "k": k, "kind": groovesurgery.GROOVE_KIND,
                  "class": request["class"], "outcome": "unclosed", "failed_at": "postcut",
                  "error": "BudgetUnclosed: for the test",
                  "closures": {"hard": ["G3_box"], "G3_box": 5.8e-9,
                               "limits": {"G3_box": 2.28e-9}}}
        path = Path(outdir) / "events" / f"event_{k:03d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record) + "\n")
        raise surgery.BudgetUnclosed("unclosed for the test", precut, record)

    return groove_fill


@pytest.fixture(scope="module")
def fallbacks(tmp_path_factory):
    out = tmp_path_factory.mktemp("fallback")
    recon = rc.ReconnectSettings(delta=DELTA, dt0=1e-5, tip_size=0.25, bridge_size=0.25,
                                 focus_size=0.25)
    block = ev.EventSettings.from_dict(BLOCK)
    res = {}
    for name in ("refused", "unclosed"):
        seen = {}
        with pytest.MonkeyPatch.context() as mp, heads_problem(out / f"pb-{name}", recon) as pb:
            if name == "unclosed":
                mp.setattr(groovesurgery, "groove_fill", stand_in_fill(seen))
            real_cut = surgery.liquid_cut

            def liquid_cut(problem, *a, **kw):
                seen["at_cut"] = fingerprint(problem)
                seen["focus_at_cut"] = problem.active_mesh_template().focus
                seen["remesh_count_at_cut"] = problem.remesh_count
                return real_cut(problem, *a, **kw)

            mp.setattr(surgery, "liquid_cut", liquid_cut)
            pb.initialise()
            gas_cut_record(pb, recon)
            lcf.impose(pb, lcf.stream_velocity, list(_chains()[2]))
            run = out / name
            drv = CollapseDriver(pb, CollapseSettings(t_end=10.0, dt_max=0.01, max_steps=2,
                                                      state_dt=0.0, state_keep=10),
                                 run, reconnect=recon, events=block)
            m = drv.reset_after_cut(pb.cut)
            named = drv._named_post_cut_chains()
            neck = next(x for x in ev.liquid_necks(named, block.liquid)
                        if x["chain"] == "bubble" and x["status"] == "head")
            site = neck["head"][1] - 6
            groove = {"reason": ev.EVENT_REQUEST, "kind": "groove", "class": "A-groove",
                      "handler": ev.GROOVE_HANDLER, "chain": "bubble", "contact": "fold",
                      "fold_pair": [site - 4, site], "fold_gap": 1e-4, "fold_limit": 2e-4}
            seen["before"] = fingerprint(pb)
            seen["focus_before"] = pb.active_mesh_template().focus
            seen["remesh_count_before"] = pb.remesh_count
            m2, stop = drv._meet_event(m, groove)
            res[name] = {"seen": seen, "stop": stop, "events": [dict(x) for x in pb.events],
                         "obs_after_cut": pb.cut_observables(), "neck": neck, "run": run,
                         "m2_time": m2["time"]}
            res[name]["result"] = drv.run(1e-8)
            res[name]["obs_end"] = pb.cut_observables()
            res[name]["t_end"] = pb.get_current_time(dimensional=False, as_float=True)
            drv.close()
    return res


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


@pytest.mark.parametrize("name", ["refused", "unclosed"])
def test_a_failing_fill_on_the_far_head_falls_back_to_the_cut_of_that_head(fallbacks, name):
    res = fallbacks[name]
    assert res["stop"] is None, res["stop"]
    rec = res["events"][-1]
    assert (rec["kind"], rec["class"], rec["outcome"]) == ("far-head", "F-head", "made")
    assert rec["k"] == 1 and rec["chain"] == "bubble" and not rec["closures"]["hard"]
    assert rec["request"]["index"] == res["neck"]["index"]
    assert rec["request"]["far_event"]["class"] == "A-groove"
    fb = rec["request"]["fallback"]
    assert fb["reason"] == (ev.EVENT_UNCLOSED if name == "unclosed" else ev.GROOVE_FILL_REFUSED)
    assert fb["reloaded"] is (name == "unclosed")
    assert fb["record"] == "events/event_001_fill.json"
    run = res["run"]
    assert json.loads((run / "events" / "event_001_fill.json").read_text())["outcome"] == \
        fb["outcome"]
    assert json.loads((run / "events" / "event_001.json").read_text())["kind"] == "far-head"
    lines = [e for e in _jsonl(run / "events.jsonl") if e["event"] == "groove-fallback"]
    assert len(lines) == 1 and lines[0]["reloaded"] is fb["reloaded"]
    result = res["result"]
    assert result["events"]["made"] == 1 and result["events"]["by_kind"] == {"far-head": 1}
    assert result["stop"]["reason"] == "max-steps" and result["steps"] == 2


def test_the_cut_starts_from_the_state_before_the_event(fallbacks):
    """The refusal leaves the problem as it was; the unclosed fill changed it (a new mesh), and
    the state loaded back is the state before the event to the bit, with its focus and rebuild
    count."""
    for name in ("refused", "unclosed"):
        seen = fallbacks[name]["seen"]
        assert seen["at_cut"] == seen["before"], name
        assert seen["focus_at_cut"] == seen["focus_before"], name
        assert seen["remesh_count_at_cut"] == seen["remesh_count_before"], name
    seen = fallbacks["unclosed"]["seen"]
    assert seen["precut"] == seen["before"]
    assert seen["changed"] != seen["before"]
    assert seen["remesh_count_changed"] > seen["remesh_count_before"]


def test_the_cut_after_the_reload_equals_the_cut_without_a_reload(fallbacks):
    """The far-head cut and the two steps after it, from the reloaded state and from the state a
    refused fill left: the same removal, closures, observables and time, to round-off."""
    a, b = (fallbacks[x] for x in ("refused", "unclosed"))
    ra, rb = a["events"][-1], b["events"][-1]
    for key in ("V", "Pz", "KE"):
        assert rb["removed"][key] == pytest.approx(ra["removed"][key], rel=1e-9, abs=1e-300)
    for key, v in ra["closures"]["relative"].items():
        assert rb["closures"]["relative"][key] == pytest.approx(v, rel=1e-6, abs=1e-12), key
    assert rb["trigger"]["r_n"] == ra["trigger"]["r_n"]
    for key in ("volume", "momentum_z", "kinetic_energy"):
        assert b["obs_after_cut"][key] == pytest.approx(a["obs_after_cut"][key], rel=1e-9)
        assert b["obs_end"][key] == pytest.approx(a["obs_end"][key], rel=1e-8)
    assert b["t_end"] == pytest.approx(a["t_end"], rel=1e-12)
    assert np.isfinite(b["m2_time"])
